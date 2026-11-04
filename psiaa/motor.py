#!/usr/bin/env python3
"""
Motor del asistente documental del PSIA-A. Sin interfaz: lo usan tanto la app
local de Gradio (../app.py) como la función serverless (../api/preguntar.py).

Arquitectura: enrutador + contexto completo. NO hay embeddings ni base vectorial.

  1. El enrutador recibe un índice de las 110 fuentes del programa (~10k tokens)
     y elige de 1 a 4 que puedan contener la respuesta.
  2. Esas fuentes se inyectan ENTERAS en el contexto y el modelo responde citando
     ficha y página, o declara que no tiene información.

El motivo de no trocear: el PSIA-A está hecho de tablas por franja de edad,
calendarios de cribado y algoritmos. Un chunk de 500 tokens separa el valor
("120-150 g", "a las 6 semanas") de su encabezado de edad, y la respuesta sale
mal aunque la recuperación "acierte". Cada ficha es ya la unidad editorial del
programa y la mayor ocupa 21k tokens, así que caben de sobra sin trocear.
"""

import json
import logging
import os
import re
import time
from pathlib import Path

from google import genai
from google.genai import types

# El SDK avisa en cada llamada de que no usamos "automatic function calling".
# No usamos herramientas, así que el aviso es ruido que ensucia los logs.
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

AQUI = Path(__file__).resolve().parent

MAX_FUENTES = 4
# Techo del contexto inyectado: ~80k tokens. Holgado para 4 de las fuentes más
# grandes (los guiones de las sesiones grupales rondan los 20k cada uno) y muy
# por debajo del límite de 250k tokens/minuto de la capa gratuita.
MAX_CARACTERES = 320_000

# Orden de preferencia de modelos. Google retira y renombra modelos cada pocos
# meses, así que se resuelven al arrancar probándolos de verdad.
#
# OJO: no sirve mirar models.list(). gemini-2.5-flash seguía apareciendo en el
# listado cuando ya devolvía 404 ("no longer available to new users") al usarlo.
#
# Se usan DOS modelos distintos, uno por paso, a propósito. La cuota gratuita que
# de verdad aprieta es:
#
#     GenerateRequestsPerMinutePerProjectPerModel-FreeTier = 5
#
# Es "per model": cada modelo tiene su propio cupo de 5 peticiones/minuto. Con un
# solo modelo, las 2 peticiones de cada pregunta compiten por el mismo cupo y
# salen 2,5 preguntas/minuto. Repartiéndolas en dos modelos, cada paso tiene su
# cupo entero y salen 5 preguntas/minuto, gratis.
#
# El reparto además encaja con la tarea: enrutar es clasificar (le basta un
# modelo "lite"), responder sobre tablas clínicas quiere el modelo más capaz.
MODELOS_RESPUESTA = [
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-flash-latest",
]
MODELOS_ENRUTADOR = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.8-flash",
    "gemini-flash-latest",
]

NO_INFO = (
    "No tengo información sobre eso en el Programa de Salud Infantil y "
    "Adolescente de Andalucía."
)

_clave = os.environ.get("GOOGLE_API_KEY")
if not _clave:
    raise SystemExit(
        "Falta la variable de entorno GOOGLE_API_KEY.\n"
        "En local: export GOOGLE_API_KEY=...\n"
        "En Vercel: Settings -> Environment Variables"
    )
# Timeout explícito: sin él, una petición estancada (típico en conexiones
# inestables) deja la app colgada para siempre, sin error y sin respuesta.
TIMEOUT_MS = int(os.environ.get("TIMEOUT_SEGUNDOS", "90")) * 1000

cliente = genai.Client(
    api_key=_clave,
    http_options=types.HttpOptions(timeout=TIMEOUT_MS),
)


# Las sondas de arranque son llamadas mínimas: 20 s de sobra. Así un servidor
# lento no retrasa el arranque un minuto y medio por cada candidato.
SONDA = types.HttpOptions(timeout=20_000)


def espera_sugerida(e: Exception) -> float | None:
    """Segundos que pide esperar un 429, o None si el error no es de cuota.

    La API devuelve el retardo exacto en el cuerpo del error ('retryDelay': '18s'
    y "Please retry in 19.02s"), así que no hay que adivinarlo con backoff ciego.
    """
    texto = str(e)
    if "RESOURCE_EXHAUSTED" not in texto and "429" not in texto:
        return None
    for patron in (r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'",
                   r"retry in (\d+(?:\.\d+)?)s"):
        m = re.search(patron, texto)
        if m:
            return float(m.group(1)) + 1.0      # un segundo de margen
    return 20.0


_retirados: set[str] = set()      # modelos que dieron 404: no se vuelven a probar


def con_pool(pool: list[str], hacer, rondas: int = 2):
    """Ejecuta hacer(modelo) probando los modelos del pool en orden.

    La clave está en qué se hace ante un 429: NO esperar, sino pasar al siguiente
    modelo. La cuota gratuita es por modelo
    (GenerateRequestsPerMinutePerProjectPerModel), así que un modelo agotado no
    dice nada del siguiente. Esperar sería desperdiciar cupo disponible.

    Solo si TODOS los modelos están agotados se espera el retardo que pide la API
    y se da otra ronda. Ventaja added: no hay que conocer el límite exacto de cada
    modelo (Google ya no los publica), el pool se adapta solo.
    """
    ultimo = None
    for ronda in range(rondas):
        for modelo in pool:
            if modelo in _retirados:
                continue
            try:
                return hacer(modelo)
            except Exception as e:
                ultimo = e
                texto = str(e)
                if "NOT_FOUND" in texto or "404" in texto:
                    print(f"[pool] {modelo} retirado, se descarta", flush=True)
                    _retirados.add(modelo)
                    continue
                if espera_sugerida(e) is not None:
                    print(f"[pool] {modelo}: cuota agotada, probando el siguiente",
                          flush=True)
                    continue
                if "503" in texto or "UNAVAILABLE" in texto:
                    print(f"[pool] {modelo}: sobrecargado, probando el siguiente",
                          flush=True)
                    continue
                raise                          # error real: no lo tapamos
        if ronda < rondas - 1:
            espera = espera_sugerida(ultimo) or 20.0
            print(f"[pool] todos agotados: esperando {espera:.0f}s", flush=True)
            time.sleep(min(espera, 60))
    raise ultimo


def usable(nombre: str) -> bool:
    """¿Responde de verdad este modelo con esta clave? Una llamada mínima."""
    print(f"[arranque] probando {nombre}...", flush=True)
    try:
        cliente.models.generate_content(
            model=nombre,
            contents="ok",
            config=types.GenerateContentConfig(
                max_output_tokens=16, http_options=SONDA),
        )
        return True
    except Exception as e:
        if "API_KEY_INVALID" in str(e):
            raise SystemExit(
                "La GOOGLE_API_KEY no es válida. Consigue una en "
                "https://aistudio.google.com/apikey\n"
                "En Vercel: Settings -> Environment Variables, y vuelve a desplegar."
            ) from None
        # Un fallo de red se repetiría igual en los cinco candidatos y el mensaje
        # final culparía a la clave, que es justo donde no hay que mirar.
        if isinstance(e, (ConnectionError, OSError)) or "ConnectError" in type(e).__name__ \
                or "nodename nor servname" in str(e):
            raise SystemExit(
                "No hay conexión con la API de Google (fallo de red o de DNS).\n"
                "No es la clave ni el modelo: revisa la conexión y vuelve a intentarlo.\n"
                f"Detalle: {type(e).__name__}: {str(e)[:120]}"
            ) from None
        print(f"[aviso] {nombre} no sirve: {type(e).__name__}: {str(e)[:150]}")
        return False


def anunciados() -> list[str]:
    """Flash que anuncia la cuenta, del más reciente al más antiguo. Red de seguridad
    por si los nombres de MODELOS_* quedan obsoletos."""
    try:
        return sorted(
            (m.name.removeprefix("models/") for m in cliente.models.list()
             if "generateContent" in (m.supported_actions or [])
             and "flash" in m.name and "preview" not in m.name),
            reverse=True,
        )
    except Exception as e:
        print(f"[aviso] no se ha podido listar modelos: {e}")
        return []


_verificados: dict[str, bool] = {}

# Sondear al arrancar está bien en local (confirma que la clave y el modelo
# funcionan antes de preguntar nada) y es un ERROR en serverless: no hay proceso
# persistente, así que las sondas se repiten en cada arranque en frío y se comen
# la cuota de 5 peticiones/minuto antes de atender a nadie.
#
# Por eso el valor por defecto es NO sondear: se coge el primer modelo de la
# lista y, si no sirve, con_pool() lo descubre en la primera petición real y pasa
# al siguiente. Cuesta 0 peticiones de arranque y se autocorrige igual.
#
# En local, SONDEAR=1 recupera el diagnóstico temprano.
SONDEAR = os.environ.get("SONDEAR") == "1"


def resolver_modelo(preferidos: list[str], evitar: str | None = None) -> str | None:
    candidatos = [n for n in preferidos if n != evitar]
    if not SONDEAR:
        return candidatos[0] if candidatos else None

    for nombre in dict.fromkeys(candidatos + [n for n in anunciados() if n != evitar]):
        if nombre not in _verificados:
            _verificados[nombre] = usable(nombre)
        if _verificados[nombre]:
            return nombre
    return None


MODELO = os.environ.get("MODELO") or resolver_modelo(MODELOS_RESPUESTA)
if not MODELO:
    raise SystemExit(
        "Ningún modelo compatible responde con esta clave. Revisa que la clave sea "
        "válida y mira los avisos de arriba.\n"
        "Puedes forzar uno con: export MODELO=... (y MODELO_ENRUTADOR=...)"
    )

# Un modelo distinto para enrutar, para no compartir el cupo con las respuestas.
MODELO_ROUTER = os.environ.get("MODELO_ENRUTADOR") or \
    resolver_modelo(MODELOS_ENRUTADOR, evitar=MODELO) or MODELO


def construir_pool(verificado: str, preferidos: list[str]) -> list[str]:
    """El modelo ya verificado va primero; detrás, los demás como reserva de cupo.

    No se verifican los de reserva al arrancar: gastaría cupo en comprobar algo
    que quizá no se use nunca. Si alguno está retirado, con_pool lo descarta en
    cuanto dé 404, una sola vez.
    """
    return list(dict.fromkeys([verificado] + preferidos))


POOL_RESPUESTA = construir_pool(MODELO, MODELOS_RESPUESTA)
POOL_ROUTER = construir_pool(MODELO_ROUTER, MODELOS_ENRUTADOR)

print(f"[info] respuesta: {' > '.join(POOL_RESPUESTA)}")
print(f"[info] enrutado:  {' > '.join(POOL_ROUTER)}")
print(f"[info] cada modelo tiene su propio cupo por minuto, así que el pool "
      f"multiplica la capacidad")

# El enrutador solo tiene que clasificar y devolver ids: no necesita razonar, y
# pagar "thinking" en cada pregunta es tirar tokens y latencia. Pero el control
# cambió entre generaciones:
#   - Gemini 2.x:  thinking_budget=0      (entero)
#   - Gemini 3.x:  thinking_level="low"   (el entero da error, y "minimal" ya no
#                                          existe en 3.8-flash)
# y pasar los dos a la vez también falla. En lugar de deducirlo del nombre del
# modelo, se prueba cuál acepta. La sonda lleva además el esquema JSON, así que
# valida de una vez todo lo que el enrutador necesita.
ESQUEMA_SONDA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
}


PENSAMIENTOS = {
    "low": ("thinking_level='low'", types.ThinkingConfig(thinking_level="low")),
    "budget0": ("thinking_budget=0", types.ThinkingConfig(thinking_budget=0)),
    "none": ("sin configurar", None),
}


def resolver_pensamiento():
    # En serverless NO hay proceso persistente: cada arranque en frío repetiría
    # las sondas y se comería la cuota de 5 peticiones/minuto. Fijando MODELO,
    # MODELO_ENRUTADOR y PENSAMIENTO por entorno, el arranque cuesta 0 peticiones.
    fijado = os.environ.get("PENSAMIENTO")
    if fijado:
        if fijado not in PENSAMIENTOS:
            raise SystemExit(
                f"PENSAMIENTO={fijado!r} no vale. Opciones: {sorted(PENSAMIENTOS)}"
            )
        return PENSAMIENTOS[fijado]

    if not SONDEAR:
        # Sin sondear: se asume el control de Gemini 3.x, que es el vigente. Si
        # resultara equivocado, enrutar() lo corrige en la primera petición real
        # y lo recuerda. Coste de arranque: 0 peticiones.
        return PENSAMIENTOS["low"]

    opciones = [
        ("thinking_level='low'", types.ThinkingConfig(thinking_level="low")),
        ("thinking_budget=0", types.ThinkingConfig(thinking_budget=0)),
        ("sin configurar", None),
    ]
    for etiqueta, cfg in opciones:
        print(f"[arranque] probando {etiqueta}...", flush=True)
        try:
            cliente.models.generate_content(
                model=MODELO_ROUTER,
                contents="Devuelve {\"ok\": true}",
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                    response_schema=ESQUEMA_SONDA,
                    thinking_config=cfg,
                    http_options=SONDA,
                ),
            )
            return etiqueta, cfg
        except Exception as e:
            print(f"[aviso] {etiqueta} no vale: {type(e).__name__}: {str(e)[:120]}")
    raise SystemExit(
        f"{MODELO_ROUTER} no acepta salida JSON con esquema, que es imprescindible "
        "para el enrutador. Prueba otro con: export MODELO_ENRUTADOR=..."
    )


_etiqueta, _cfg = resolver_pensamiento()

# En lista de un elemento para poder corregirlo desde enrutar() sin `global`.
_PENSAMIENTO = [_cfg]
_ORDEN_DEGRADACION = ["low", "budget0", "none"]
print(f"[info] pensamiento del enrutador: {_etiqueta}")


def _degradar_pensamiento(e: Exception) -> bool:
    """Si el error viene del thinking_config, pasa a la siguiente opción.

    Devuelve True si ha cambiado algo y merece la pena reintentar.
    """
    texto = str(e)
    if "thinking" not in texto.lower() or "INVALID_ARGUMENT" not in texto:
        return False
    actual = next((k for k in _ORDEN_DEGRADACION
                   if PENSAMIENTOS[k][1] is _PENSAMIENTO[0]), None)
    siguiente = _ORDEN_DEGRADACION[_ORDEN_DEGRADACION.index(actual) + 1] \
        if actual and actual != _ORDEN_DEGRADACION[-1] else None
    if not siguiente:
        return False
    etiqueta, cfg = PENSAMIENTOS[siguiente]
    print(f"[info] thinking rechazado; se pasa a {etiqueta}", flush=True)
    _PENSAMIENTO[0] = cfg
    return True

CORPUS = json.loads((AQUI / "corpus.json").read_text(encoding="utf-8"))
POR_ID = {d["id"]: d for d in CORPUS}

INDICE = "\n".join(
    f"- id: {d['id']}\n"
    f"  tema: {d['titulo']}\n"
    f"  bloque: {d['bloque']} | dirigido a: {d['publico']}\n"
    f"  cubre: {d['claves']}"
    for d in CORPUS
)

# ---------------------------------------------------------------------------
# Paso 1 — enrutador
# ---------------------------------------------------------------------------

ENRUTADOR = f"""Eres el enrutador documental del Programa de Salud Infantil y Adolescente \
de Andalucía (PSIA-A). No respondes preguntas: solo decides qué fichas del programa hay \
que leer para responderlas.

ÍNDICE DE FICHAS DISPONIBLES
{INDICE}

Clasifica el último mensaje del usuario en un "tipo":

- "consulta": pregunta sobre salud infantil o adolescente, sobre la práctica clínica
  pediátrica en atención primaria, o sobre el programa en sí. Devuelve en "ids" entre 1 y
  {MAX_FUENTES} ids del índice, de más a menos probable.
- "saludo": saludos, agradecimientos o preguntas sobre qué eres y qué puedes consultar.
  "ids" vacío.
- "fuera_de_alcance": la pregunta no tiene nada que ver con salud infantil/adolescente ni
  con el programa (política, deportes, programación, recetas de cocina, adultos...).
  "ids" vacío.

Reglas:
- Copia los ids EXACTAMENTE como aparecen en el índice. Nunca inventes un id.
- Ante la duda entre dos fichas, incluye las dos: es más barato leer una ficha de más
  que fallar la recuperación.
- Una pregunta clínica de pediatría que crees que el programa podría tratar es
  "consulta", aunque no estés seguro de que la respuesta esté ahí. Que la información
  exista o no lo decide el siguiente paso, no tú. No uses "fuera_de_alcance" por duda.
- Si la pregunta es de seguimiento ("¿y a qué edad?", "¿y en prematuros?"), resuélvela
  con el historial de la conversación antes de elegir.

Cómo elegir cuando varias fichas hablan de la MISMA edad (muy importante):
El índice contiene cinco familias de documentos que cubren las mismas edades con
propósitos distintos. Fíjate en el campo "dirigido a" y en el bloque:

- Preguntan QUÉ HACE EL PROFESIONAL en una visita o control de salud ("¿qué se hace
  a los 6 meses?", "¿qué toca en la revisión de los 2 años?") -> la ficha de la
  revisión correspondiente del bloque "PSIA individual (visitas)". Añade
  `psiaa_visitas_individuales_actividades` si la pregunta es transversal o pide un
  resumen por edades.
- Preguntan A QUÉ EDAD o EN QUÉ VISITAS se hace un cribado o actividad concretos
  -> `psiaa_visitas_individuales_actividades` (la tabla tema × visita) y además la
  ficha temática de ese cribado.
- Preguntan QUÉ SE ENTREGA a la familia o piden el material/hoja de consejos
  -> bloque "Guías anticipatorias (entrega)" de esa edad. Si la pregunta es sobre
  material para el propio chico o chica, prefiere el dirigido a "Chicos y chicas".
- Preguntan por un TEMA CLÍNICO concreto sin referirse a una visita ("dosis de
  vitamina D", "maniobras de cribado de cadera") -> la ficha temática del tema, no
  la ficha de la visita.
- Preguntan por el desarrollo esperable o cómo estimularlo en un tramo de edad
  -> bloque "Parentalidad positiva (fichas de desarrollo)".
- Preguntan cómo evaluar el riesgo parental o qué preguntar a los padres
  -> bloque "Competencias parentales (instrumento)" de la visita más próxima."""

ESQUEMA_ENRUTADOR = {
    "type": "object",
    "properties": {
        "tipo": {"type": "string", "enum": ["consulta", "saludo", "fuera_de_alcance"]},
        "ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["tipo", "ids"],
}


def enrutar(pregunta: str, historial: list[dict]) -> tuple[str, list[str]]:
    ultimos = [m for m in historial[-4:] if isinstance(m.get("content"), str)]
    contexto = ""
    if ultimos:
        contexto = "CONVERSACIÓN PREVIA\n" + "\n".join(
            f"{m['role']}: {m['content'][:400]}" for m in ultimos
        ) + "\n\n"

    def llamar(modelo):
        return cliente.models.generate_content(
            model=modelo,
            contents=f"{contexto}MENSAJE DEL USUARIO\n{pregunta}",
            config=types.GenerateContentConfig(
                system_instruction=ENRUTADOR,
                temperature=0,
                response_mime_type="application/json",
                response_schema=ESQUEMA_ENRUTADOR,
                thinking_config=_PENSAMIENTO[0],
            ),
        )

    try:
        respuesta = con_pool(POOL_ROUTER, llamar)
    except Exception as e:
        # El control de "thinking" cambió de forma incompatible entre Gemini 2.x
        # (thinking_budget entero) y 3.x (thinking_level textual). Si el valor
        # asumido no vale, se corrige aquí y se RECUERDA, en vez de gastar una
        # sonda en cada arranque en frío para averiguarlo.
        if not _degradar_pensamiento(e):
            raise
        respuesta = con_pool(POOL_ROUTER, llamar)

    datos = json.loads(respuesta.text)
    # dict.fromkeys deduplica conservando el orden, y va ANTES del recorte: si el
    # modelo repite una ficha (lo hace cuando la considera relevante por dos
    # motivos distintos), sin deduplicar se inyectaba el mismo texto dos veces
    # —el doble de contexto pagado por nada— y aparecía repetida en el pie de
    # fuentes. Deduplicando primero, los 4 huecos son 4 fichas DISTINTAS.
    ids = list(dict.fromkeys(
        i for i in datos.get("ids", []) if i in POR_ID
    ))[:MAX_FUENTES]
    return datos.get("tipo", "consulta"), ids


# ---------------------------------------------------------------------------
# Paso 2 — respuesta fundamentada
# ---------------------------------------------------------------------------

RESPONDEDOR = f"""Eres un asistente documental del Programa de Salud Infantil y \
Adolescente de Andalucía (PSIA-A). Respondes a profesionales de pediatría y enfermería \
de atención primaria.

REGLA ABSOLUTA: tu única fuente son los FRAGMENTOS que se te entregan en cada turno. No
tienes ni usas ningún otro conocimiento médico. Si un dato no está literalmente en los
fragmentos, no existe para ti, por evidente que te parezca.

Si los fragmentos no contienen la respuesta, responde exactamente esto y nada más:
"{NO_INFO}"
Y añade, si procede, una línea indicando qué temas próximos sí cubre el programa.

Si contienen la respuesta solo en parte, responde la parte que esté documentada y di con
claridad qué parte no aparece en el programa. Nunca rellenes el hueco.

Cómo responder:
- En español, directo y clínico, sin preámbulos ni cortesías de relleno.
- Cita siempre la fuente de cada dato: [Título de la ficha, p.N], usando el número de
  página que aparece como [p.N] en el fragmento.
- Reproduce los valores numéricos, dosis, edades y periodicidades EXACTAMENTE como
  figuran. No redondees, no conviertas unidades, no "actualices" nada.
- Si un dato depende de la franja de edad o de pertenecer a población de riesgo, dilo
  siempre con su condición. Nunca des una cifra suelta sin su franja de edad.
- Si los fragmentos recogen un dato y además remiten a un recurso externo (por ejemplo
  Andavac para vacunación), menciona ambas cosas.
- Usa listas y tablas en Markdown cuando el contenido original sea una tabla.
- No inventes anexos, escalas ni enlaces que no estén en los fragmentos.

Cada fragmento indica a quién va dirigido ("Dirigido a"). Respétalo:
- "Profesional": contenido de la práctica clínica. Es lo que debes usar por defecto.
- "Familias" o "Chicos y chicas": son hojas de consejos que se ENTREGAN. Puedes citar
  su contenido, pero di siempre que viene del material de entrega a la familia o al
  menor, nunca lo presentes como protocolo de actuación profesional.

Si dos fragmentos se contradicen (por ejemplo, el cronograma sitúa una visita a los 12
meses y la ficha de esa revisión dice 9-12 meses), expón las dos versiones con su
fuente y señala la discrepancia. No elijas una ni promedies.

Los guiones de las sesiones grupales incluyen anexos gráficos (transparencias, fichas
de trabajo, diagramas como la «flor del desarrollo»). Su texto llega como etiquetas
sueltas y sin orden fiable. Cuando una respuesta dependa de uno de esos anexos, di que
existe, cita su nombre y página, y advierte de que hay que consultarlo en el PDF
original. No intentes reconstruir su disposición ni inventes el contenido que falta."""


def construir_contexto(ids: list[str]) -> tuple[str, list[str]]:
    """Devuelve (contexto, ids_descartados_por_tamaño).

    Los descartados se devuelven para avisar en la respuesta: un recorte
    silencioso se lee igual que una respuesta completa, y aquí eso significaría
    omitir una fuente sin que nadie se entere.
    """
    bloques, usados, fuera, total = [], [], [], 0
    for ident in dict.fromkeys(ids):      # red de seguridad: nunca inyectar dos veces lo mismo
        d = POR_ID[ident]
        trozo = (
            f"<<< FICHA: {d['titulo']}\n"
            f"Bloque del programa: {d['bloque']}\n"
            f"Dirigido a: {d['publico']}\n"
            f"Edición: {d['fecha']} | Archivo: {d['archivo']} | Páginas: {d['paginas']}\n\n"
            f"{d['texto']}\n>>> FIN DE LA FICHA\n"
        )
        # La primera fuente entra siempre, aunque sea enorme: sin ella no hay
        # nada que fundamentar.
        if usados and total + len(trozo) > MAX_CARACTERES:
            fuera.append(ident)
            continue
        bloques.append(trozo)
        usados.append(ident)
        total += len(trozo)
    return "\n".join(bloques), fuera


SUGERENCIA_SALUDO = (
    f"Soy un asistente que responde **solo** con el contenido del Programa de Salud "
    f"Infantil y Adolescente de Andalucía (PSIA-A). Tengo cargadas {len(CORPUS)} "
    "fuentes del programa:\n\n"
    "- Las **12 revisiones del niño sano** y el cronograma de visitas\n"
    "- Las fichas por tema: consejos preventivos, cribados universales y en "
    "situaciones de riesgo, grupos específicos\n"
    "- Las **guías anticipatorias** que se entregan a familias y adolescentes\n"
    "- Los materiales de parentalidad positiva y de competencias parentales\n\n"
    "Puedes preguntarme cosas como:\n"
    "- ¿Qué se hace en la revisión de los 6 meses?\n"
    "- ¿En qué visitas se criba la displasia de cadera?\n"
    "- ¿Qué dosis de vitamina D se recomienda para prevenir el raquitismo?\n\n"
    "Si algo no está en el programa, te lo diré en lugar de inventármelo."
)


def aviso_error(e: Exception) -> str:
    """Mensaje legible para el usuario. El cuerpo del error de la API es un JSON de
    40 líneas: útil en los logs, inservible dentro de un chat."""
    texto = str(e)
    if espera_sugerida(e) is not None:
        return (
            "⏳ **Demasiadas consultas a la vez.** La capa gratuita permite unas 5 "
            "preguntas por minuto entre todas las personas que usen esta página.\n\n"
            "Espera unos segundos y vuelve a preguntar; no se ha perdido nada."
        )
    if "503" in texto or "UNAVAILABLE" in texto:
        return ("⚠️ El modelo está sobrecargado ahora mismo (problema de Google, no "
                "tuyo). Vuelve a intentarlo en un minuto.")
    if "API_KEY_INVALID" in texto:
        return "⚠️ Problema de configuración: la clave de la API no es válida."
    return (f"⚠️ No he podido consultar el programa ahora mismo.\n\n"
            f"`{type(e).__name__}: {texto[:300]}`")


def responder(mensaje: str, historial: list[dict]):
    mensaje = (mensaje or "").strip()
    if not mensaje:
        return

    try:
        tipo, ids = enrutar(mensaje, historial)
    except Exception as e:                      # cuota agotada, red, API caída
        yield aviso_error(e)
        return

    if tipo == "saludo":
        yield SUGERENCIA_SALUDO
        return

    # Sin fichas candidatas no hay nada que fundamentar: se rechaza sin gastar
    # una segunda llamada a la API (la capa gratuita limita peticiones, no tokens).
    if tipo == "fuera_de_alcance" or not ids:
        yield (
            f"{NO_INFO}\n\n"
            "Solo puedo responder sobre el contenido del PSIA-A: promoción de la salud y "
            "consejos preventivos, cribados, grupos específicos y organización del programa."
        )
        return

    contexto, descartadas = construir_contexto(ids)
    usadas = [i for i in ids if i not in descartadas]
    contenido = (
        f"FRAGMENTOS DEL PROGRAMA\n\n{contexto}\n\n"
        f"PREGUNTA\n{mensaje}"
    )
    # Gemini exige que la conversación empiece por un turno "user" y alterne. El
    # recorte del historial puede dejar fuera el primer "user" de un par, así que
    # se descartan los mensajes del asistente que queden al principio.
    previos = [m for m in historial[-4:] if isinstance(m.get("content"), str)]
    while previos and previos[0]["role"] != "user":
        previos.pop(0)

    conversacion = [
        {"role": "user" if m["role"] == "user" else "model",
         "parts": [{"text": m["content"]}]}
        for m in previos
    ]
    conversacion.append({"role": "user", "parts": [{"text": contenido}]})

    # Un 503 puede llegar con el flujo YA ABIERTO, a mitad de la respuesta. No
    # existe forma de retomar la generación donde se quedó, así que lo que se
    # hace es DESCARTAR lo emitido y rehacer la respuesta con el siguiente modelo
    # del pool. Es más lento, pero una lista de actuaciones clínicas cortada por
    # la mitad se lee como completa, y eso es mucho peor que esperar.
    #
    # Por eso este bucle no usa con_pool(): hay que distinguir el fallo al abrir
    # el flujo (reintento transparente) del fallo a mitad (hay que avisar al
    # consumidor de que descarte lo ya pintado).
    vivos = [m for m in POOL_RESPUESTA if m not in _retirados] or POOL_RESPUESTA
    texto = ""
    fallo = None

    for n, modelo in enumerate(vivos):
        es_ultimo = n == len(vivos) - 1
        texto = ""
        try:
            flujo = cliente.models.generate_content_stream(
                model=modelo,
                contents=conversacion,
                config=types.GenerateContentConfig(
                    system_instruction=RESPONDEDOR,
                    temperature=0,
                ),
            )
            for trozo in flujo:
                if trozo.text:
                    texto += trozo.text
                    yield texto
            fallo = None
            break
        except Exception as e:
            fallo = e
            aviso = str(e)
            if "NOT_FOUND" in aviso or "404" in aviso:
                _retirados.add(modelo)
            recuperable = (espera_sugerida(e) is not None
                           or "503" in aviso or "UNAVAILABLE" in aviso)
            if es_ultimo or not recuperable:
                break
            if texto:
                print(f"[pool] {modelo} se cortó tras {len(texto)} caracteres; "
                      f"se rehace la respuesta con el siguiente modelo", flush=True)
                texto = ""
                yield ""        # cadena vacía = señal de reinicio al consumidor
            else:
                print(f"[pool] {modelo} no pudo abrir el flujo, probando el siguiente",
                      flush=True)
            espera = espera_sugerida(e)
            if espera:
                time.sleep(min(espera, 30))

    if fallo is not None:
        if texto:
            # El aviso va DELANTE, no al pie: si se queda abajo, la respuesta
            # truncada se lee como si estuviera entera.
            yield (
                "⚠️ **Respuesta incompleta.** Se cortó a mitad y no he conseguido "
                "rehacerla, así que falta contenido. No la uses como si estuviera "
                f"completa: vuelve a preguntar.\n\n{aviso_error(fallo)}\n\n---\n\n"
                f"{texto}"
            )
        else:
            yield aviso_error(fallo)
        return

    pie = "\n".join(
        f"- {POR_ID[i]['titulo']} · ed. {POR_ID[i]['fecha']} · `{POR_ID[i]['archivo']}`"
        for i in usadas
    )
    if descartadas:
        omitidas = ", ".join(POR_ID[i]["titulo"] for i in descartadas)
        pie += (
            f"\n\n⚠️ No ha cabido en el contexto, así que **no** se ha leído: "
            f"{omitidas}. Pregúntalo por separado si te interesa."
        )
    yield f"{texto}\n\n---\n**Fichas consultadas**\n{pie}"
