#!/usr/bin/env python3
"""
Aplicación ASGI del asistente del PSIAA. Es el entrypoint que carga Vercel.

Dos rutas:

    GET  /                 -> index.html
    POST /api/preguntar    -> flujo SSE con la respuesta

¿Por qué un solo ASGI en la raíz y no una función en api/?

Vercel tiene dos modos para Python. El de "framework" busca el entrypoint en
nombres concretos (app.py, index.py, main.py, server.py, wsgi.py, asgi.py) en la
raíz, src/ o app/, y es el que activa por defecto. El modo de ficheros en api/
existe pero la documentación lo describe como soporte a proyectos antiguos y no
se activó: el despliegue falló con PYTHON_ENTRYPOINT_NOT_FOUND aunque encontrara
api/preguntar.py con su variable `app`.

Así que este fichero está en un nombre y sitio que Vercel reconoce, y además el
entrypoint se declara explícito en pyproject.toml. En modo framework TODAS las
rutas llegan aquí, por eso la página también se sirve desde aquí en vez de como
fichero estático. A cambio la respuesta lleva Cache-Control, así que las
recargas no vuelven a invocar la función.

El SSE es imprescindible: sin streaming, una respuesta de 15 s se ve como una
pantalla congelada.

Toda la lógica del asistente vive en psiaa/motor.py. Aquí solo hay transporte.
"""

import json
from pathlib import Path

from psiaa import motor

AQUI = Path(__file__).resolve().parent

# La página se lee una vez por instancia, no en cada petición.
PAGINA = (AQUI / "index.html").read_bytes()

CAB_SSE = [
    (b"content-type", b"text/event-stream; charset=utf-8"),
    (b"cache-control", b"no-cache, no-transform"),
    (b"connection", b"keep-alive"),
    # Vercel respeta esta cabecera para no bufferizar la respuesta.
    (b"x-accel-buffering", b"no"),
]

CAB_HTML = [
    (b"content-type", b"text/html; charset=utf-8"),
    (b"cache-control", b"public, max-age=0, must-revalidate"),
]


def sse(evento: str, datos: dict) -> bytes:
    return (f"event: {evento}\n"
            f"data: {json.dumps(datos, ensure_ascii=False)}\n\n").encode("utf-8")


async def cuerpo_peticion(receive) -> bytes:
    datos, mas = b"", True
    while mas:
        mensaje = await receive()
        datos += mensaje.get("body", b"")
        mas = mensaje.get("more_body", False)
    return datos


async def responder_simple(send, estado: int, cabeceras: list, cuerpo: bytes):
    await send({"type": "http.response.start", "status": estado, "headers": cabeceras})
    await send({"type": "http.response.body", "body": cuerpo})


async def preguntar(scope, receive, send):
    try:
        peticion = json.loads(await cuerpo_peticion(receive) or b"{}")
    except json.JSONDecodeError:
        peticion = {}

    mensaje = (peticion.get("mensaje") or "").strip()[:2000]
    historial = [
        {"role": m.get("role", "user"), "content": str(m.get("content", ""))}
        for m in (peticion.get("historial") or [])
        if isinstance(m, dict) and isinstance(m.get("content"), str)
    ][-6:]

    await send({"type": "http.response.start", "status": 200, "headers": CAB_SSE})

    async def emitir(datos: bytes):
        await send({"type": "http.response.body", "body": datos, "more_body": True})

    # Comentario SSE inmediato: confirma al navegador que la conexión está viva
    # durante los segundos que tarda el enrutador, antes del primer token.
    await emitir(b": conectado\n\n")

    if not mensaje:
        await emitir(sse("error", {"t": "No has escrito ninguna pregunta."}))
    else:
        # motor.responder() entrega el texto ACUMULADO (es lo que espera Gradio).
        # Aquí se manda solo lo nuevo, para no reenviar la respuesta entera en
        # cada trozo.
        #
        # Si el acumulado deja de ser una extensión de lo ya enviado, el motor ha
        # rehecho la respuesta desde cero (un 503 a mitad obliga a descartar y
        # reintentar con otro modelo). En ese caso hay que decirle al cliente que
        # tire lo pintado, o acabaría con las dos versiones pegadas.
        enviado = ""
        try:
            for acumulado in motor.responder(mensaje, historial):
                if acumulado.startswith(enviado):
                    nuevo = acumulado[len(enviado):]
                else:
                    await emitir(sse("reinicio", {}))
                    enviado, nuevo = "", acumulado
                if nuevo:
                    await emitir(sse("delta", {"t": nuevo}))
                enviado = acumulado
        except Exception as e:                   # nunca debería llegar aquí
            await emitir(sse("error", {"t": motor.aviso_error(e)}))

    await emitir(sse("fin", {}))
    await send({"type": "http.response.body", "body": b"", "more_body": False})


async def app(scope, receive, send):
    if scope["type"] == "lifespan":
        # Vercel no lo usa, pero un servidor ASGI local sí puede enviarlo.
        while True:
            mensaje = await receive()
            if mensaje["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif mensaje["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return

    if scope["type"] != "http":
        return

    ruta = scope.get("path", "/").rstrip("/") or "/"
    metodo = scope["method"]

    if ruta == "/api/preguntar":
        if metodo == "OPTIONS":
            return await responder_simple(send, 204, [(b"allow", b"POST, OPTIONS")], b"")
        if metodo != "POST":
            return await responder_simple(
                send, 405, [(b"content-type", b"text/plain; charset=utf-8")],
                'Usa POST con {"mensaje": "..."}'.encode("utf-8"))
        return await preguntar(scope, receive, send)

    if ruta == "/":
        if metodo not in ("GET", "HEAD"):
            return await responder_simple(send, 405, CAB_HTML, b"")
        return await responder_simple(send, 200, CAB_HTML,
                                      b"" if metodo == "HEAD" else PAGINA)

    if ruta == "/favicon.ico":
        return await responder_simple(send, 204, [], b"")

    await responder_simple(send, 404, [(b"content-type", b"text/plain; charset=utf-8")],
                           "No existe esa ruta.".encode("utf-8"))
