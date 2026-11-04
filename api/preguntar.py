#!/usr/bin/env python3
"""
Función serverless de Vercel: POST /api/preguntar

Recibe  {"mensaje": "...", "historial": [{"role": "user"|"assistant", "content": "..."}]}
Devuelve un flujo SSE con eventos:

    event: delta   data: {"t": "<trozo de texto>"}
    event: fin     data: {}
    event: error   data: {"t": "<mensaje para la persona>"}

Se expone como aplicación ASGI (`app`), que es la forma que soporta Vercel y la
única que permite ir enviando la respuesta a medida que llega. Sin streaming, una
respuesta de 15 s se ve como una pantalla congelada.

Toda la lógica vive en motor.py. Aquí solo hay transporte.
"""

import json
import sys
from pathlib import Path

# El motor vive fuera de api/ a propósito: Vercel convierte cada fichero de api/
# en una función, y motor.py no es una función.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from psiaa import motor  # noqa: E402

CABECERAS = [
    (b"content-type", b"text/event-stream; charset=utf-8"),
    (b"cache-control", b"no-cache, no-transform"),
    (b"connection", b"keep-alive"),
    # Vercel respeta esta cabecera para no bufferizar la respuesta.
    (b"x-accel-buffering", b"no"),
]


def sse(evento: str, datos: dict) -> bytes:
    return (f"event: {evento}\n"
            f"data: {json.dumps(datos, ensure_ascii=False)}\n\n").encode("utf-8")


async def leer_cuerpo(receive) -> bytes:
    cuerpo, mas = b"", True
    while mas:
        mensaje = await receive()
        cuerpo += mensaje.get("body", b"")
        mas = mensaje.get("more_body", False)
    return cuerpo


async def app(scope, receive, send):
    if scope["type"] != "http":
        return

    if scope["method"] == "OPTIONS":
        await send({"type": "http.response.start", "status": 204,
                    "headers": [(b"allow", b"POST, OPTIONS")]})
        await send({"type": "http.response.body", "body": b""})
        return

    if scope["method"] != "POST":
        await send({"type": "http.response.start", "status": 405,
                    "headers": [(b"content-type", b"text/plain; charset=utf-8")]})
        await send({"type": "http.response.body",
                    "body": "Usa POST con {\"mensaje\": \"...\"}".encode()})
        return

    try:
        peticion = json.loads(await leer_cuerpo(receive) or b"{}")
    except json.JSONDecodeError:
        peticion = {}

    mensaje = (peticion.get("mensaje") or "").strip()[:2000]
    historial = [
        {"role": m.get("role", "user"), "content": str(m.get("content", ""))}
        for m in (peticion.get("historial") or [])
        if isinstance(m, dict)
    ][-6:]

    await send({"type": "http.response.start", "status": 200, "headers": CABECERAS})

    async def emitir(datos: bytes):
        await send({"type": "http.response.body", "body": datos, "more_body": True})

    # Un comentario SSE inmediato: confirma al navegador que la conexión está
    # viva durante los 1-3 s que tarda el enrutador, antes del primer token.
    await emitir(b": conectado\n\n")

    if not mensaje:
        await emitir(sse("error", {"t": "No has escrito ninguna pregunta."}))
        await emitir(sse("fin", {}))
        await send({"type": "http.response.body", "body": b"", "more_body": False})
        return

    # motor.responder() va entregando el texto ACUMULADO (es lo que espera
    # Gradio). Aquí se manda solo lo nuevo de cada paso para no reenviar la
    # respuesta entera en cada trozo.
    enviado = ""
    try:
        for acumulado in motor.responder(mensaje, historial):
            if acumulado.startswith(enviado):
                nuevo = acumulado[len(enviado):]
            else:
                # El motor ha reescrito el texto (p. ej. al añadir un aviso de
                # error sobre lo ya emitido): se reenvía completo.
                nuevo = "\n\n" + acumulado
            if nuevo:
                await emitir(sse("delta", {"t": nuevo}))
                enviado = acumulado
    except Exception as e:                      # nunca debería llegar aquí
        await emitir(sse("error", {"t": motor.aviso_error(e)}))

    await emitir(sse("fin", {}))
    await send({"type": "http.response.body", "body": b"", "more_body": False})
