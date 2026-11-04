#!/usr/bin/env python3
"""
Construye corpus.json a partir de los PDFs del PSIA-A.

Se ejecuta UNA VEZ en local (o cada vez que cambie algún PDF).
La app solo lee corpus.json: no hay embeddings, ni base vectorial, ni troceado ciego.

    pip install pypdf
    python build_corpus.py

Criterio de selección (decidido tras comparar versiones):
  - Se usan las 47 fichas temáticas de Repetidos/ (son la unidad editorial real
    del programa: cada una es un tema completo y autocontenido).
  - Se usan los 50 documentos de estructura/ que aportan contenido nuevo: las 12
    revisiones del niño sano, el cronograma, las guías anticipatorias por edad y
    los instrumentos de competencias parentales.
  - Se DESCARTAN los 5 PDFs macro psiaa_temas_* / psiaa_aspectos_generales como
    bloque indivisible: son la versión 2014-2018 del mismo contenido y, si se
    indexan junto a las fichas de 2024-2026, el modelo mezcla criterios caducados.
  - EXCEPCIÓN: "Aspectos generales" y la "Guía de parentalidad positiva para
    profesionales" no tienen ficha equivalente, así que se incorporan troceados
    por sus propios marcadores de capítulo.
  - Se DESCARTA estructura/salud_mental.pdf: no es un duplicado de
    SaludMental_Nov24.pdf (solo 45 % de solape) sino su versión anterior, que
    todavía diagnostica el TDAH con criterios DSM-IV-TR.
  - Se omiten 13 PDF de estructura/ que son byte a byte idénticos a los de
    Repetidos/ (comprobado por hash), y el macro psiaa_temas_intervencion_
    cribado_universales.pdf.
"""

import json
import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader

RAIZ = Path(__file__).resolve().parent.parent
FICHAS = RAIZ / "Repetidos"
ESTRUC = RAIZ / "estructura"
GRUPAL = RAIZ / "Actividades_grupales"
SALIDA = Path(__file__).resolve().parent / "psiaa" / "corpus.json"

# ---------------------------------------------------------------------------
# Índice curado: título legible + bloque del programa + palabras clave.
#
# Las claves son el único "motor de búsqueda" que hay aquí: el enrutador ve
# esta tabla (y nada más) para elegir fichas. Incluyen sinónimos clínicos,
# nombres de test/escalas y la forma en que un profesional haría la pregunta,
# porque es justo donde fallaba la similitud vectorial.
# ---------------------------------------------------------------------------
META = {
    # --- Actividades de promoción de salud y consejos preventivos ---
    "PromocionLactanciaMaterna_Mar25": (
        "Promoción de la lactancia materna",
        "Promoción y consejo",
        "lactancia materna, amamantamiento, pecho, calostro, agarre, posturas, grietas, "
        "ingurgitación, mastitis, hipogalactia, extracción y conservación de leche, "
        "banco de leche, destete, lactancia artificial, fórmula, biberón, relactación",
    ),
    "Alimentacion4-24-meses_Fichafeb26": (
        "Alimentación de 4 a 24 meses",
        "Promoción y consejo",
        "alimentación complementaria, beikost, introducción de alimentos, papillas, "
        "cereales, gluten, purés, baby led weaning, BLW, triturados, atragantamiento, "
        "lactante, 6 meses, 12 meses, hierro, alergias alimentarias, leche de vaca",
    ),
    "Alimentacion_2_18_anos_Fichafeb26ok": (
        "Alimentación de 2 a 18 años",
        "Promoción y consejo",
        "alimentación saludable, dieta mediterránea, raciones, menú, desayuno, "
        "frutas y verduras, azúcares, ultraprocesados, bebidas azucaradas, "
        "vegetarianismo, veganismo, escolar, adolescente, pirámide alimentaria",
    ),
    "Asesoria_Comedores_Escolares_FichaEne26": (
        "Asesoría de comedores escolares",
        "Promoción y consejo",
        "comedor escolar, menú escolar, colegio, guardería, escuela infantil, "
        "dietas especiales en el comedor, alérgenos, celiaquía en el colegio, "
        "programa de alimentación escolar, asesoría de menús",
    ),
    "Tabaquismopasivo_Mar25": (
        "Tabaquismo pasivo",
        "Promoción y consejo",
        "tabaco pasivo, humo ambiental, fumadores en casa, tabaquismo de los padres, "
        "exposición al humo, espacios sin humo, vapeo en el hogar, nicotina de tercera mano",
    ),
    "ConsejoActividadFisicayDeporte_Mar25": (
        "Consejo sobre actividad física y deporte",
        "Promoción y consejo",
        "actividad física, ejercicio, deporte, sedentarismo, pantallas y movimiento, "
        "minutos de ejercicio recomendados, juego activo, deporte escolar, "
        "aptitud deportiva, reconocimiento deportivo",
    ),
    "ConsejoFotoProteccion_Mar25": (
        "Consejo sobre fotoprotección",
        "Promoción y consejo",
        "fotoprotección, protector solar, crema solar, FPS, sol, quemadura solar, "
        "radiación ultravioleta, melanoma, exposición solar, sombra, gorra, playa",
    ),
    "ConsejoSuenn_Mar25": (
        "Consejo preventivo sobre el sueño",
        "Promoción y consejo",
        "sueño, insomnio infantil, higiene del sueño, despertares nocturnos, siesta, "
        "colecho, rutina de acostarse, horas de sueño por edad, pesadillas, terrores "
        "nocturnos, sonambulismo, melatonina, apnea del sueño, ronquido",
    ),
    "SiindromeMuerteSubitaLactante_Jun25": (
        "Prevención del síndrome de muerte súbita del lactante (SMSL)",
        "Promoción y consejo",
        "muerte súbita del lactante, SMSL, posición para dormir, decúbito supino, boca "
        "arriba, cuna, colecho seguro, superficie firme, chupete y SMSL, sobrecalentamiento",
    ),
    "PrevencionAccidentes_Mar25": (
        "Prevención de accidentes",
        "Promoción y consejo",
        "accidentes, lesiones no intencionadas, seguridad infantil, caídas, "
        "intoxicaciones, quemaduras, ahogamiento, atragantamiento, asfixia, "
        "silla de coche, sistema de retención infantil, casco, ventanas, piscina, "
        "seguridad vial, mordeduras",
    ),
    "HigieneCorporal_Mar25": (
        "Higiene corporal",
        "Promoción y consejo",
        "higiene, baño, cuidado del cordón umbilical, piel del recién nacido, pañal, "
        "dermatitis del pañal, uñas, oídos, cerumen, higiene genital, fimosis, "
        "sinequias, pediculosis, piojos",
    ),
    "ParentalidadPositiva_Fichafeb26": (
        "Promoción del buen trato: parentalidad positiva y apego seguro",
        "Promoción y consejo",
        "buen trato, parentalidad positiva, apego seguro, vínculo, crianza, límites, "
        "disciplina positiva, rabietas, castigo físico, competencias parentales, "
        "bienestar emocional, estilos educativos",
    ),
    "atencion_sexualidad": (
        "Atención afectivo-sexual (infancia)",
        "Promoción y consejo",
        "educación afectivo-sexual en la infancia, desarrollo psicosexual, identidad de "
        "género, curiosidad sexual infantil, juegos sexuales, masturbación infantil, "
        "preguntas sobre sexo, diversidad familiar",
    ),
    "uso_responsable_medicamentos": (
        "Uso responsable de los medicamentos",
        "Promoción y consejo",
        "uso de medicamentos, automedicación, antibióticos, resistencia antibiótica, "
        "antitérmicos, paracetamol, ibuprofeno, jarabes, dosificación por peso, "
        "conservación de fármacos, botiquín, adherencia",
    ),
    "AtencionAdolescente_FichaEne26": (
        "Atención al adolescente",
        "Promoción y consejo",
        "adolescencia, consulta joven, entrevista al adolescente, confidencialidad, "
        "menor maduro, HEADSSS, cribado psicosocial, autonomía del menor, "
        "acompañamiento familiar, consentimiento del adolescente",
    ),
    "SexualidadAdolescencia_FichaEne26": (
        "Sexualidad y salud afectivo-sexual y reproductiva en la adolescencia",
        "Promoción y consejo",
        "sexualidad adolescente, anticoncepción, preservativo, píldora, anticoncepción de "
        "urgencia, postcoital, ITS, infecciones de transmisión sexual, VIH, embarazo "
        "adolescente, IVE, orientación sexual, identidad de género, forum clinic",
    ),
    "PrevencionConsumoAlcoholTabacoDrogas_FichaNov24": (
        "Prevención y detección del consumo de alcohol, tabaco y drogas",
        "Promoción y consejo",
        "alcohol, tabaco, drogas, cannabis, porros, botellón, binge drinking, cocaína, "
        "consumo de riesgo, CRAFFT, AUDIT, cigarrillo electrónico, vapeo, cachimba, "
        "deshabituación, intervención breve",
    ),
    "PrevencionAbusoTRIC_FichaEne24": (
        "Prevención del abuso de TRIC (tecnologías de la relación, información y comunicación)",
        "Promoción y consejo",
        "pantallas, uso de tecnologías, TRIC, móvil, videojuegos, redes sociales, "
        "tiempo de pantalla, adicción a internet, ciberacoso, ciberbullying, grooming, "
        "sexting, control parental, nomofobia, juego de azar online",
    ),
    "metodologia_consejo": (
        "Metodología del consejo",
        "Temas generales",
        "metodología del consejo, entrevista motivacional, counselling, comunicación "
        "con la familia, cómo dar consejo, educación para la salud, "
        "modelo de cambio de conducta, estadios de Prochaska",
    ),
    "la_familia_espacio_educativo": (
        "La familia como espacio educativo y de crecimiento saludable",
        "Temas generales",
        "familia, espacio educativo, funciones de la familia, tipos de familia, "
        "crecimiento saludable, clima familiar, relación familia-escuela",
    ),

    # --- Actividades de intervención y cribado universales ---
    "CribadoNeonatalEnfermedadesMetabolicas_Mar25": (
        "Cribado neonatal de enfermedades metabólicas congénitas",
        "Cribado universal",
        "cribado neonatal, prueba del talón, metabolopatías, hipotiroidismo congénito, "
        "fenilcetonuria, fibrosis quística neonatal, anemia falciforme, MCADD, "
        "papel de filtro, 48 horas de vida, resultado alterado, repetición de muestra",
    ),
    "CrecimientoFisicoyDesarrolloPuberal_Mar25": (
        "Supervisión del crecimiento físico y desarrollo puberal",
        "Cribado universal",
        "peso, talla, perímetro craneal, percentiles, curvas de crecimiento, "
        "velocidad de crecimiento, talla baja, pubertad, Tanner, telarquia, "
        "pubarquia, menarquia, pubertad precoz, pubertad retrasada, ginecomastia",
    ),
    "obesidad_sobrepeso": (
        "Cribado y prevención del sobrepeso y la obesidad",
        "Cribado universal",
        "obesidad, sobrepeso, IMC, índice de masa corporal, perímetro de cintura, "
        "percentil de IMC, rebote adiposo, tablas de referencia, comorbilidad, "
        "síndrome metabólico, intervención en obesidad infantil",
    ),
    "PrevencionRaquitismo_Mar25": (
        "Prevención del raquitismo",
        "Cribado universal",
        "raquitismo, vitamina D, profilaxis con vitamina D, 400 UI, colecalciferol, "
        "suplementación en lactantes, déficit de vitamina D, craneotabes",
    ),
    "prevencion_deficit_yodo": (
        "Prevención del déficit de yodo",
        "Cribado universal",
        "yodo, déficit de yodo, sal yodada, yoduro potásico, embarazo y lactancia, "
        "bocio, yodoprofilaxis",
    ),
    "SaludBucodental_Mar25": (
        "Salud bucodental",
        "Cribado universal",
        "salud bucodental, caries, flúor, fluoruros, cepillado, pasta dentífrica, "
        "dentición, erupción dentaria, PADI, maloclusión, chupete y dientes, "
        "traumatismo dental, higiene oral, selladores",
    ),
    "CribadoDisplasiaCaderas_marz2025": (
        "Cribado de la displasia evolutiva de cadera",
        "Cribado universal",
        "displasia de cadera, DEC, luxación congénita de cadera, Ortolani, Barlow, "
        "abducción limitada, pliegues asimétricos, Galeazzi, ecografía de cadera, "
        "factores de riesgo de displasia, presentación podálica",
    ),
    "deteccion_anomalias_aparato_locomotor": (
        "Detección precoz de anomalías en el desarrollo del aparato locomotor",
        "Cribado universal",
        "aparato locomotor, escoliosis, test de Adams, cifosis, pie plano, pie zambo, "
        "metatarso aducto, genu valgo, genu varo, marcha, rotación tibial, "
        "deformidades de los pies, asimetría de miembros",
    ),
    "CribadoCriptorquidia_FichaJun25": (
        "Cribado de criptorquidia",
        "Cribado universal",
        "criptorquidia, testículo no descendido, teste en ascensor, testículo retráctil, "
        "palpación testicular, orquidopexia, derivación a cirugía",
    ),
    "DeteccionPrecozAnomaliasOculares_Mar25": (
        "Cribado de ambliopía, estrabismo y trastornos de la refracción",
        "Cribado universal",
        "ambliopía, estrabismo, refracción, miopía, hipermetropía, astigmatismo, "
        "agudeza visual, optotipos, test de Hirschberg, cover test, reflejo rojo, "
        "leucocoria, retinoblastoma, obstrucción lagrimal, visión binocular",
    ),
    "DeteccionPrecozHipoacusia_Mar25": (
        "Detección precoz de la hipoacusia infantil",
        "Cribado universal",
        "hipoacusia, sordera, cribado auditivo neonatal, otoemisiones acústicas, "
        "potenciales evocados auditivos, PEATC, audiometría, factores de riesgo "
        "de hipoacusia, retraso del lenguaje por audición, implante coclear",
    ),
    "EvaluacionDesarrolloPsicomotorAfectivo_Dic24": (
        "Evaluación del desarrollo psicomotor y afectivo",
        "Cribado universal",
        "desarrollo psicomotor, hitos del desarrollo, retraso psicomotor, Haizea-Llevant, "
        "tabla de desarrollo, signos de alarma, autismo, TEA, M-CHAT, señales de alerta "
        "del espectro autista, tono muscular, reflejos primitivos, atención temprana, "
        "sedestación, marcha autónoma, sonrisa social",
    ),
    "DeteccionTrastornosLenguajeyAprendizaje_Nov24": (
        "Detección precoz de trastornos del lenguaje y del aprendizaje",
        "Cribado universal",
        "lenguaje, retraso del lenguaje, TEL, trastorno del desarrollo del lenguaje, "
        "dislalia, tartamudez, disfemia, dislexia, discalculia, TDAH y aprendizaje, "
        "fracaso escolar, logopedia, lectoescritura, bilingüismo",
    ),
    "SaludMental_Nov24": (
        "Salud mental",
        "Cribado universal",
        "salud mental infantil, ansiedad, depresión, TDAH, trastorno de conducta, "
        "autolesiones, ideación suicida, trastornos de la conducta alimentaria, TCA, "
        "anorexia, bulimia, acoso escolar, bullying, duelo, SDQ, derivación a USMIJ",
    ),
    "ValoracionyPlanAtencionSocial_Mar25": (
        "Valoración social y plan de atención social en salud",
        "Cribado universal",
        "valoración social, trabajo social sanitario, riesgo social, vulnerabilidad, "
        "pobreza infantil, absentismo escolar, plan de atención social, "
        "determinantes sociales, exclusión, coordinación con servicios sociales",
    ),

    # --- Actividades de intervención y cribado en situaciones de riesgo ---
    "CribadoDepresionPostparto_Mar25": (
        "Cribado de la depresión posparto",
        "Cribado en situaciones de riesgo",
        "depresión posparto, puerperio, Edinburgh, EPDS, escala de Edimburgo, "
        "salud mental materna, tristeza posparto, maternity blues, vínculo madre-bebé",
    ),
    "CribadoHipercolesterolemia_FichaEne26": (
        "Cribado y prevención de hipercolesterolemia",
        "Cribado en situaciones de riesgo",
        "colesterol, hipercolesterolemia familiar, LDL, HDL, triglicéridos, perfil "
        "lipídico, dislipemia, antecedentes familiares de cardiopatía precoz, "
        "cribado en cascada, estatinas en pediatría",
    ),
    "CribadoHipertensionArterial__Mar25": (
        "Cribado y prevención de hipertensión arterial",
        "Cribado en situaciones de riesgo",
        "hipertensión arterial, tensión arterial, presión arterial, percentiles de TA, "
        "manguito, toma de tensión en niños, HTA secundaria, MAPA",
    ),
    "CribadoyPrevencionFerropenia": (
        "Cribado y prevención de ferropenia",
        "Cribado en situaciones de riesgo",
        "ferropenia, anemia ferropénica, hierro, ferritina, hemoglobina, "
        "suplementos de hierro, prematuro y hierro, dieta rica en hierro",
    ),
    "CribadoTuberculosis_FichaJun25": (
        "Cribado de tuberculosis (TBC)",
        "Cribado en situaciones de riesgo",
        "tuberculosis, TBC, Mantoux, prueba de tuberculina, PPD, IGRA, "
        "infección tuberculosa latente, contacto con enfermo de TBC, "
        "quimioprofilaxis, isoniazida, BCG",
    ),
    "atencion_malos_tratos_menores": (
        "Atención a situaciones de malos tratos a menores",
        "Cribado en situaciones de riesgo",
        "maltrato infantil, abuso sexual infantil, negligencia, desamparo, "
        "indicadores de maltrato, hoja de notificación SIMIA, parte judicial, "
        "fiscalía de menores, protección del menor, violencia de género y menores, "
        "síndrome del niño zarandeado, mutilación genital femenina",
    ),

    # --- Actividades en grupos específicos ---
    "PrematurosMenos32semanas_Mar25": (
        "Programa de actividades para prematuros de menos de 32 semanas o menos de 1.500 g",
        "Grupos específicos",
        "prematuro, gran prematuro, bajo peso, menos de 1500 gramos, edad corregida, "
        "seguimiento del prematuro, retinopatía de la prematuridad, displasia "
        "broncopulmonar, palivizumab, crecimiento del prematuro, atención temprana",
    ),
    "ActividadesSindromeDown_Mar25": (
        "Actividades preventivas en niños y niñas con síndrome de Down",
        "Grupos específicos",
        "síndrome de Down, trisomía 21, seguimiento del síndrome de Down, "
        "cardiopatía congénita, hipotiroidismo en Down, inestabilidad atlantoaxoidea, "
        "celiaquía en Down, hipoacusia en Down, curvas específicas de Down",
    ),
    "ActividadesMenoresOtrosPaises_Mar25": (
        "Actividades preventivas en menores adoptados procedentes de otros países e inmigrantes",
        "Grupos específicos",
        "adopción internacional, menor inmigrante, menor migrante, niño procedente de "
        "otro país, cribado al llegar, parásitos, parasitosis intestinal, hepatitis B, "
        "VIH, sífilis, tuberculosis en inmigrantes, calendario vacunal incompleto, "
        "MENA, mutilación genital femenina",
    ),
    "AtencionMenorconDiscapacidad_Mar25": (
        "Atención al menor con discapacidad",
        "Grupos específicos",
        "discapacidad, menor con discapacidad, parálisis cerebral, dependencia, "
        "certificado de discapacidad, atención temprana, necesidades especiales, "
        "accesibilidad, cuidador principal, respiro familiar",
    ),
    "VacunaciooSituacionesRiesgo_Dic24": (
        "Vacunación en situaciones de riesgo, calendario acelerado y viajes internacionales",
        "Grupos específicos",
        "vacunas, vacunación, calendario vacunal, calendario acelerado, "
        "vacunación de rescate, grupos de riesgo, inmunodeprimido, asplenia, "
        "viajes internacionales, vacunas del viajero, Andavac, Plan Estratégico de "
        "Vacunaciones, pauta incompleta",
    ),
    "Act_EnfCeliaca_FibrosisQuistica_FichaMar25-1": (
        "Actividades recomendadas en enfermedad celíaca y fibrosis quística",
        "Grupos específicos",
        "enfermedad celíaca, celiaquía, gluten, anticuerpos antitransglutaminasa, "
        "biopsia intestinal, dieta sin gluten, fibrosis quística, FQ, cloruro en sudor, "
        "insuficiencia pancreática, seguimiento de celíaco",
    ),
}

# ---------------------------------------------------------------------------
# Índice curado de estructura/ : bloques 3 y 5 del programa.
#
# (título, bloque, público, claves)
#
# El campo "público" es decisivo para el enrutado. Aquí conviven cinco familias
# de documentos que hablan de la MISMA edad con propósitos distintos: lo que el
# profesional hace en la visita, lo que se entrega a la familia, lo que se
# entrega al propio adolescente, y los instrumentos de evaluación. Sin esa
# distinción, "¿qué se hace a los 2 meses?" tiene cuatro respuestas plausibles.
# ---------------------------------------------------------------------------
EDADES_VISITA = {
    "01": ("1ª revisión", "primera semana de vida",
           "primera semana, recién nacido, primera visita, 1a revision, visita del RN, "
           "apertura de historia, 7 días"),
    "02": ("2ª revisión", "entre los 15 días y el primer mes de vida",
           "15 días, 1 mes, primer mes, mes de vida, segunda visita, 2a revision, quince días"),
    "03": ("3ª revisión", "2 meses",
           "2 meses, dos meses, tercera visita, 3a revision"),
    "04": ("4ª revisión", "4 meses",
           "4 meses, cuatro meses, cuarta visita, 4a revision"),
    "05": ("5ª revisión", "6 meses",
           "6 meses, seis meses, quinta visita, 5a revision, medio año"),
    "06": ("6ª revisión", "9-12 meses",
           "9 meses, 10 meses, 11 meses, 12 meses, un año, nueve a doce meses, "
           "sexta visita, 6a revision, primer cumpleaños"),
    "07": ("7ª revisión", "15 meses",
           "15 meses, quince meses, año y tres meses, séptima visita, 7a revision"),
    "08": ("8ª revisión", "2 años",
           "2 años, dos años, 24 meses, octava visita, 8a revision"),
    "09": ("9ª revisión", "3-4 años",
           "3 años, 4 años, tres años, cuatro años, novena visita, 9a revision, preescolar"),
    "10": ("10ª revisión", "6 años",
           "6 años, seis años, décima visita, 10a revision, inicio de primaria"),
    "11": ("11ª revisión", "9-11 años",
           "9 años, 10 años, 11 años, nueve a once años, undécima visita, 11a revision, preadolescente"),
    "12": ("12ª revisión", "12-14 años",
           "12 años, 13 años, 14 años, doce a catorce años, duodécima visita, 12a revision, adolescente"),
}

META_ESTRUCTURA = {
    # --- Bloque 3: PSIA individual. Tablas maestras ---
    "psiaa_visitas_individuales_cronograma": (
        "Cronograma de las visitas individuales: edad de cada revisión y profesional que la realiza",
        "PSIA individual (visitas)", "Profesional",
        "cronograma de visitas, calendario de revisiones, cuántas visitas tiene el "
        "programa, a qué edades son las revisiones, qué profesional hace cada visita, "
        "matrona, pediatra, enfermera, visita prenatal, educación maternal, "
        "listado de revisiones del niño sano",
    ),
    "psiaa_visitas_individuales_actividades": (
        "Tabla de contenidos: qué actividad del programa corresponde a cada visita (tema × edad)",
        "PSIA individual (visitas)", "Profesional",
        "qué cribados toca en cada visita, en qué visitas se hace un cribado concreto, "
        "tabla de actividades por edad, a qué edades se hace cada tema, hasta qué edad "
        "se criba algo, desde qué edad, resumen de todo el programa por edad, "
        "qué temas se revisan a cada edad, qué actividades son universales y cuáles "
        "solo en situación de riesgo, actividades específicas no ligadas a una visita",
    ),

    # --- Bloque 5: guías anticipatorias que se ENTREGAN a la familia ---
    "consejos_0_meses_nov18": (
        "Guía anticipatoria y consejos para entregar: recién nacido (0 meses)",
        "Guías anticipatorias (entrega)", "Familias",
        "consejos para entregar al nacer, recién nacido, 0 meses, primera semana, "
        "buenos motivos para dar el pecho, hoja de consejos del recién nacido",
    ),
    "consejos_2_meses_nov18": (
        "Guía anticipatoria y consejos para entregar: 2 meses",
        "Guías anticipatorias (entrega)", "Familias",
        "consejos para entregar a los 2 meses, segundo mes de vida, hoja de consejos 2 meses",
    ),
    "consejos_4_meses_nov18": (
        "Guía anticipatoria y consejos para entregar: 4 meses",
        "Guías anticipatorias (entrega)", "Familias",
        "consejos para entregar a los 4 meses, cuarto mes de vida, hoja de consejos 4 meses",
    ),
    "consejos_6_meses_nov18": (
        "Guía anticipatoria y consejos para entregar: 6 meses",
        "Guías anticipatorias (entrega)", "Familias",
        "consejos para entregar a los 6 meses, sexto mes de vida, hoja de consejos 6 meses",
    ),
    "consejos_9_12_meses_nov18": (
        "Guía anticipatoria y consejos para entregar: 9-12 meses",
        "Guías anticipatorias (entrega)", "Familias",
        "consejos para entregar a los 9-12 meses, al año de vida, hoja de consejos 12 meses",
    ),
    "consejos_18_meses_oct15": (
        "Guía anticipatoria y consejos para entregar: 18 meses",
        "Guías anticipatorias (entrega)", "Familias",
        "consejos para entregar a los 18 meses, año y medio, hoja de consejos 18 meses",
    ),
    "consejos_2_anos_oct15": (
        "Guía anticipatoria y consejos para entregar: 2 años",
        "Guías anticipatorias (entrega)", "Familias",
        "consejos para entregar a los 2 años, hoja de consejos 2 años",
    ),
    "consejos_3_4_anos_oct15": (
        "Guía anticipatoria y consejos para entregar: 3-4 años",
        "Guías anticipatorias (entrega)", "Familias",
        "consejos para entregar a los 3-4 años, hoja de consejos 3 años, 4 años",
    ),
    "consejos_6_anos_oct15": (
        "Guía anticipatoria y consejos para entregar: 6 años",
        "Guías anticipatorias (entrega)", "Familias",
        "consejos para entregar a los 6 años, hoja de consejos 6 años",
    ),
    "consejos_9_11_anos_familias": (
        "Guía anticipatoria y consejos para entregar a la familia: 9-11 años",
        "Guías anticipatorias (entrega)", "Familias",
        "consejos para la familia 9-11 años, hábitos, autonomía, hoja de consejos 9 a 11 años",
    ),
    "consejos_9_11_anos_adolescentes": (
        "Consejos para entregar al propio chico o chica: 9-11 años",
        "Guías anticipatorias (entrega)", "Chicos y chicas",
        "consejos para el niño de 9-11 años, material para el menor, "
        "hoja dirigida al chico o chica de 9 a 11 años",
    ),
    "consejos_12_14_anos_familias": (
        "Guía anticipatoria y consejos para entregar a la familia: 12-14 años",
        "Guías anticipatorias (entrega)", "Familias",
        "consejos para la familia del adolescente, 12-14 años, hábitos del adolescente",
    ),
    "consejos_12_14_anos_adolescentes": (
        "Consejos para entregar al propio adolescente: 12-14 años",
        "Guías anticipatorias (entrega)", "Chicos y chicas",
        "consejos para el adolescente, material para el propio adolescente, "
        "pubertad explicada al adolescente, hoja dirigida al chico o chica de 12 a 14 años",
    ),

    # --- Bloque 5: fichas de desarrollo por tramo (ideas para trabajar con la familia) ---
    "1_2_meses_padres_madres": (
        "Ficha de desarrollo y estimulación: del primer al segundo mes de vida",
        "Parentalidad positiva (fichas de desarrollo)", "Profesional",
        "desarrollo del primer al segundo mes, estimulación 1-2 meses, "
        "ideas para trabajar con padres y madres 1-2 meses",
    ),
    "2_4_meses_padres_madres": (
        "Ficha de desarrollo y estimulación: del segundo al cuarto mes de vida",
        "Parentalidad positiva (fichas de desarrollo)", "Profesional",
        "desarrollo del segundo al cuarto mes, estimulación 2-4 meses",
    ),
    "4_6_meses_padres_madres": (
        "Ficha de desarrollo y estimulación: del cuarto al sexto mes de vida",
        "Parentalidad positiva (fichas de desarrollo)", "Profesional",
        "desarrollo del cuarto al sexto mes, estimulación 4-6 meses",
    ),
    "6_12_meses_padres_madres": (
        "Ficha de desarrollo y estimulación: del sexto mes al primer cumpleaños",
        "Parentalidad positiva (fichas de desarrollo)", "Profesional",
        "desarrollo del sexto mes al año, estimulación 6-12 meses",
    ),
    "12_15_meses_padres_madres": (
        "Ficha de desarrollo y estimulación: del primer año a los quince meses",
        "Parentalidad positiva (fichas de desarrollo)", "Profesional",
        "desarrollo de 12 a 15 meses, estimulación del primer año a los quince meses",
    ),
    "15_24_meses_padres_madres": (
        "Ficha de desarrollo y estimulación: de los quince meses al segundo cumpleaños",
        "Parentalidad positiva (fichas de desarrollo)", "Profesional",
        "desarrollo de 15 a 24 meses, estimulación de quince meses a dos años",
    ),
    "2_4_anos_padres_madres": (
        "Ficha de desarrollo y estimulación: de los dos a los cuatro años",
        "Parentalidad positiva (fichas de desarrollo)", "Profesional",
        "desarrollo de 2 a 4 años, estimulación de dos a cuatro años",
    ),
    "4_6_anos_padres_madres": (
        "Ficha de desarrollo y estimulación: de los cuatro a los seis años",
        "Parentalidad positiva (fichas de desarrollo)", "Profesional",
        "desarrollo de 4 a 6 años, estimulación de cuatro a seis años",
    ),
    "6_10_anos_padres_madres": (
        "Ficha de desarrollo y estimulación: de los seis a los diez años",
        "Parentalidad positiva (fichas de desarrollo)", "Profesional",
        "desarrollo de 6 a 10 años, estimulación de seis a diez años",
    ),
    "guia_adolescencia": (
        "Ideas fundamentales de la guía temática «Llegó la adolescencia»",
        "Parentalidad positiva (fichas de desarrollo)", "Profesional",
        "ideas fundamentales llegó la adolescencia, resumen de la guía de adolescencia "
        "para trabajar con padres y madres, límites en la adolescencia",
    ),

    # --- Bloque 5: guías temáticas completas para familias ---
    "cuidadores_bebe_a_bordo": (
        "Guía temática «Bebé a bordo: una nueva vida familiar»",
        "Guías temáticas (entrega)", "Familias",
        "bebé a bordo, nueva vida familiar, llegada del bebé, cambios en la pareja tras "
        "el nacimiento, reparto de tareas, cansancio de los primeros meses, celos del hermano",
    ),
    "cuidadores_estimulacion_desarrollo": (
        "Guía temática «La estimulación del desarrollo»",
        "Guías temáticas (entrega)", "Familias",
        "estimulación del desarrollo, cómo estimular al bebé, juego y aprendizaje, "
        "el desarrollo está predeterminado, qué pueden hacer los padres por el desarrollo",
    ),
    "cuidadores_apego": (
        "Guía temática «Cómo establecer un buen apego»",
        "Guías temáticas (entrega)", "Familias",
        "cómo establecer un buen apego, vínculo afectivo, apego seguro e inseguro, "
        "llanto del bebé, separaciones, angustia de separación, figura de referencia",
    ),
    "cuidadores_autonomia": (
        "Guía temática «El logro de la autonomía»",
        "Guías temáticas (entrega)", "Familias",
        "logro de la autonomía, independencia del niño, control de esfínteres, "
        "vestirse solo, responsabilidades en casa, sobreprotección",
    ),
    "cuidadores_arte_de_educar": (
        "Guía temática «El arte de educar»",
        "Guías temáticas (entrega)", "Familias",
        "el arte de educar, normas y límites, disciplina, castigos, premios, "
        "rabietas, estilos educativos, cómo poner límites, coherencia educativa",
    ),
    "cuidadores_aprender_divertirse_salir_clase": (
        "Guía temática «Aprender y divertirse al salir de clase»",
        "Guías temáticas (entrega)", "Familias",
        "tiempo libre, actividades extraescolares, deberes, ocio, juego, "
        "aprender y divertirse al salir de clase, agenda del niño, descanso",
    ),
    "cuidadores_adolescencia": (
        "Guía temática «Llegó la adolescencia» (guía completa para familias)",
        "Guías temáticas (entrega)", "Familias",
        "llegó la adolescencia, convivir con un adolescente, conflictos con "
        "adolescentes, comunicación con el adolescente, supervisión, "
        "es la adolescencia una etapa problemática",
    ),

    # --- Bloque 5: instrumento de evaluación de capacidades parentales ---
    "visitas_4_6mes_competencias_parentales": (
        "Evaluación del riesgo en capacidades parentales: visitas de 4-6 meses (registro, indicadores y preguntas)",
        "Competencias parentales (instrumento)", "Profesional",
        "evaluación de competencias parentales 4-6 meses, capacidades parentales, "
        "estrés parental, registro de evaluación, qué preguntar a los padres, "
        "riesgo psicosocial, red de apoyo, vinculación afectiva",
    ),
    "visitas_12_15mes_competencias_parentales": (
        "Evaluación del riesgo en capacidades parentales: visitas de 12-15 meses (registro, indicadores y preguntas)",
        "Competencias parentales (instrumento)", "Profesional",
        "evaluación de competencias parentales 12-15 meses, capacidades parentales, "
        "estrés parental, registro de evaluación, riesgo psicosocial",
    ),
    "visita_2a_competencias_parentales": (
        "Evaluación del riesgo en capacidades parentales: visita de 2 años (registro, indicadores y preguntas)",
        "Competencias parentales (instrumento)", "Profesional",
        "evaluación de competencias parentales 2 años, capacidades parentales, "
        "estrés parental, registro de evaluación, riesgo psicosocial",
    ),
    "visita_4a_competencias_parentales": (
        "Evaluación del riesgo en capacidades parentales: visita de 4 años (registro, indicadores y preguntas)",
        "Competencias parentales (instrumento)", "Profesional",
        "evaluación de competencias parentales 4 años, capacidades parentales, "
        "estrés parental, registro de evaluación, riesgo psicosocial",
    ),
    "visita_6a_competencias_parentales": (
        "Evaluación del riesgo en capacidades parentales: visita de 6 años (registro, indicadores y preguntas)",
        "Competencias parentales (instrumento)", "Profesional",
        "evaluación de competencias parentales 6 años, capacidades parentales, "
        "estrés parental, registro de evaluación, riesgo psicosocial",
    ),
    "visitas_10_12a_competencias_parentales": (
        "Evaluación del riesgo en capacidades parentales: visitas de 10-12 años (registro, indicadores y preguntas)",
        "Competencias parentales (instrumento)", "Profesional",
        "evaluación de competencias parentales 10-12 años, capacidades parentales, "
        "estrés parental, registro de evaluación, riesgo psicosocial, preadolescente",
    ),
}

# Las 12 revisiones se generan desde EDADES_VISITA: el texto del índice es
# idéntico salvo la edad, y escribirlo doce veces a mano solo invita a erratas.
for _n, (_ord, _edad, _claves) in EDADES_VISITA.items():
    META_ESTRUCTURA[f"psiaa_visitas_individuales_{_n}revision"] = (
        f"{_ord} del niño sano: {_edad} — actividades completas de la visita",
        "PSIA individual (visitas)", "Profesional",
        f"qué se hace en la visita de {_edad}, contenido de la {_ord}, {_claves}, "
        "revisión del niño sano, control de salud, examen de salud, en qué consiste la "
        "visita, evaluación de cuidados familiares, cribados de la visita, "
        "inmunizaciones de la visita, consejos de la visita, actividades en situaciones "
        "de riesgo de la visita",
    )

# ---------------------------------------------------------------------------
# Índice curado de Actividades_grupales/ : bloque 4 del programa.
#
# Se descarta psiaa_actividades_grupales_comunitarias.pdf (244 páginas): es el
# volumen completo y estos 7 ficheros son sus secciones (similitud 0,94-0,98 por
# tramo de páginas). Lo único que se pierde son las páginas 1-13 de portada,
# autoría e índice.
#
# Cada sesión se deja entera (12-22k tokens) y no se trocea: son guiones de
# taller donde objetivos, dinámica, duración y anexos solo tienen sentido juntos.
# ---------------------------------------------------------------------------
META_GRUPALES = {
    "actividades_grupales_generalidades": (
        "Actividades grupales y comunitarias: generalidades (población diana, periodicidad, tipología de intervenciones, activos de salud y catálogo de actuaciones)",
        "Actividades grupales", "Profesional",
        "actividades grupales, actividad comunitaria, intervención grupal, cómo organizar "
        "un grupo, población diana de las actividades grupales, periodicidad de los "
        "grupos, tipología de intervenciones, técnicas grupales, enfoque de activos de "
        "salud, mapa de activos, catálogo de actuaciones grupales, promoción comunitaria, "
        "educación para la salud en grupo, requisitos para hacer un grupo",
    ),
    "sesion_grupal_00_introduccion": (
        "Sesiones grupales: introducción y consideraciones metodológicas para el trabajo en grupo",
        "Actividades grupales", "Profesional",
        "cómo dinamizar un grupo, metodología del trabajo en grupo, papel del "
        "coordinador o coordinadora, número de participantes, duración de las sesiones, "
        "cómo empezar una sesión grupal, dificultades en el grupo, qué sesiones existen",
    ),
    "sesion_grupal_01_nueva_vida_familiar": (
        "Sesión grupal 1: «Una nueva vida familiar» (guion completo con actividades, anexos y transparencias)",
        "Actividades grupales", "Profesional",
        "sesión grupal 1, taller nueva vida familiar, llegada del bebé, desarrollo "
        "personal de los progenitores, relación de pareja tras el nacimiento, actividad "
        "laboral y conciliación, relaciones sociales, primera sesión del grupo de padres",
    ),
    "sesion_grupal_02_apego": (
        "Sesión grupal 2: «Nuestro bebé: adaptación y vínculo de apego» (guion completo con actividades, anexos y transparencias)",
        "Actividades grupales", "Profesional",
        "sesión grupal 2, taller de apego, vínculo de apego, apego seguro, dinámica "
        "sobre el apego, organización familiar, conciliación vida familiar y laboral, "
        "cuidadores alternativos, guardería, necesidades de apoyo, "
        "cómo promover un apego seguro en grupo",
    ),
    "sesion_grupal_03_desarrollo": (
        "Sesión grupal 3: «Cómo estimular el desarrollo infantil» (guion completo con actividades, anexos y transparencias)",
        "Actividades grupales", "Profesional",
        "sesión grupal 3, taller de estimulación del desarrollo, cómo estimular el "
        "desarrollo infantil, flor del desarrollo, dinámica sobre desarrollo, "
        "juego y estimulación en grupo",
    ),
    "sesion_grupal_04_arte_educar": (
        "Sesión grupal 4: «El arte de educar» (guion completo con actividades, anexos y transparencias)",
        "Actividades grupales", "Profesional",
        "sesión grupal 4, taller el arte de educar, establecer normas, establecer "
        "consecuencias, límites, disciplina, dinámica sobre normas y castigos, "
        "ingredientes de la educación, cómo trabajar los límites con padres y madres",
    ),
    "sesion_grupal_05_adolescencia": (
        "Sesión grupal 5: «La llegada de la adolescencia» (guion completo con actividades, anexos y transparencias)",
        "Actividades grupales", "Profesional",
        "sesión grupal 5, taller la llegada de la adolescencia, grupo de familias de "
        "adolescentes, dinámica sobre adolescencia, conflictos con adolescentes en grupo, "
        "materiales de la sesión de adolescencia",
    ),
}

# Fuentes extra troceadas desde PDFs grandes que no tienen ficha equivalente.
# (ruta, rango de páginas 1-indexado inclusive, id, título, bloque, claves)
EXTRAS = [
    (
        ESTRUC / "psiaa_estructura.pdf", (1, 2),
        "psiaa_estructura", "Estructura del PSIA-A", "Aspectos generales",
        "estructura del programa, secciones del PSIA-A, cómo se organiza el programa, "
        "apartados, índice general, esquema de exposición de los temas, "
        "población diana, periodicidad, actividades, herramientas y anexos, "
        "qué bloques tiene el programa",
    ),
    (
        FICHAS / "psiaa_aspectos_generales.pdf", (13, 25),
        "aspectos_generales_introduccion",
        "Aspectos generales: presentación e introducción (justificación, marco legislativo, metodología)",
        "Aspectos generales",
        "justificación del programa, marco legislativo, marco normativo, derechos de la "
        "infancia, objetivos del programa, metodología del programa, enfoque de activos, "
        "atención primaria, cartera de servicios, equidad",
    ),
    (
        FICHAS / "psiaa_aspectos_generales.pdf", (29, 35),
        "aspectos_generales_evaluacion",
        "Aspectos generales: evaluación e indicadores del programa",
        "Aspectos generales",
        "evaluación del programa, indicadores, cobertura, cumplimiento, "
        "indicadores de proceso y resultado, monitorización, cómo se evalúa el PSIA-A",
    ),
    # La guía de parentalidad (pp.498-538) se trocea en tres por sus propios
    # capítulos. Antes entraba como una sola fuente de 24k tokens, que se comía
    # el presupuesto de contexto entero; además sus anexos A y B han quedado
    # superados por las fichas individuales de estructura/, salvo las cuatro
    # guías de situaciones familiares específicas, que solo están aquí.
    (
        RAIZ / "psiaa_temas_nov18.pdf", (498, 510),
        "parentalidad_guia_metodologia",
        "Guía de parentalidad positiva para profesionales: presentación, metodología y catálogo de actuaciones",
        "Parentalidad positiva (guía profesional)",
        "guía para profesionales de parentalidad, cómo usar la guía de parentalidad, "
        "consideraciones metodológicas, catálogo de actuaciones, qué fichas y guías "
        "existen, cómo organizar el trabajo con familias",
    ),
    (
        RAIZ / "psiaa_temas_nov18.pdf", (511, 532),
        "parentalidad_ideas_fundamentales",
        "Ideas fundamentales de todas las fichas de desarrollo y guías temáticas (incluidas las de situaciones familiares específicas)",
        "Parentalidad positiva (guía profesional)",
        "ideas fundamentales de cada ficha, resumen de todas las guías temáticas, "
        "ficha de adolescencia de diez a diecisiete años, guía para familias en proceso "
        "de separación o divorcio, familias combinadas o reconstituidas, familias "
        "homoparentales, bebés con necesidades especiales",
    ),
    (
        RAIZ / "psiaa_temas_nov18.pdf", (533, 538),
        "parentalidad_sesiones_grupales",
        "Resumen de las 5 sesiones grupales: objetivos y actividades de cada una (vista general; el guion completo está en el bloque «Actividades grupales»)",
        "Parentalidad positiva (guía profesional)",
        "resumen de las sesiones grupales, qué sesiones grupales hay, objetivos "
        "generales de cada sesión, vista general de los talleres, "
        "listado de actividades de las cinco sesiones",
    ),
]


# ---------------------------------------------------------------------------
# Caso especial: la matriz "actividad × visita".
#
# psiaa_visitas_individuales_actividades.pdf es una tabla de 42 actividades por
# 12 visitas donde cada marca solo significa algo junto a su fila Y su columna.
# El PDF codifica la columna con ESPACIOS: dos espacios antes de la primera 'x'
# quiere decir que la columna 1 está vacía y las marcas empiezan en la 2.
#
#   'Cribado de displasia evolutiva de la cadera  x x x x x x x     '
#                                               ^^ la columna 1s va vacía
#
# limpiar() destruía esa información dos veces: colapsaba los espacios y, peor,
# su regex de recomposición de palabras cortadas convertía 'x x x x x x x' en
# 'xxxxxxx'. El modelo recibía siete marcas sin posición y adivinaba, con lo que
# desplazaba la ventana de cribado una visita entera.
#
# Así que esta tabla se decodifica aquí, de forma determinista, y se escribe en
# texto explícito. Lo que el modelo no tiene que interpretar, no lo puede fallar.
# ---------------------------------------------------------------------------
COLUMNAS = ["1s", "15-1m", "2m", "4m", "6m", "12m", "15m", "2a", "3-4a", "6a",
            "9-11a", "12-14a"]
EDADES = {
    "1s": "1ª semana", "15-1m": "15 días-1 mes", "2m": "2 meses", "4m": "4 meses",
    "6m": "6 meses", "12m": "12 meses", "15m": "15 meses", "2a": "2 años",
    "3-4a": "3-4 años", "6a": "6 años", "9-11a": "9-11 años", "12-14a": "12-14 años",
}
MARCAS = {"", "x", "sr", "sr1"}

# Filas de control: si la extracción cambia, el build debe fallar en vez de
# publicar una tabla corrida en silencio. Verificadas contra el PDF original.
CONTROLES = {
    "Cribado neonatal de enfermedades metabólicas congénitas": ["1s", "15-1m"],
    "Cribado de displasia evolutiva de la cadera":
        ["15-1m", "2m", "4m", "6m", "12m", "15m", "2a"],
    "Prevención del déficit de yodo": COLUMNAS,
    "Sexualidad, salud afectivo-sexual y reproductiva en el adolescente": ["12-14a"],
    "Atención al adolescente.": ["9-11a", "12-14a"],
}


def parsear_fila(linea: str):
    """('etiqueta', ['x','', 'sr', ...12 ranuras]) o None si la línea no es de la tabla."""
    trozos = linea.rstrip().split(" ")
    i = len(trozos)
    while i > 0 and trozos[i - 1] in MARCAS:
        i -= 1
    ranuras = trozos[i:]
    etiqueta = " ".join(trozos[:i]).strip()
    if not etiqueta or not any(r for r in ranuras):
        return None                      # cabecera, nota al pie, pie de página
    if len(ranuras) > len(COLUMNAS):
        raise SystemExit(f"matriz: fila con {len(ranuras)} ranuras: {linea!r}")
    ranuras += [""] * (len(COLUMNAS) - len(ranuras))
    return etiqueta, ranuras


def matriz_a_texto(ruta: Path, paginas=(1, 2)) -> str:
    lector = PdfReader(str(ruta))
    filas = []
    for n in paginas:
        for linea in (lector.pages[n - 1].extract_text() or "").split("\n"):
            fila = parsear_fila(linea)
            if fila:
                filas.append(fila)

    universal = {}      # etiqueta -> [columnas con 'x']
    riesgo = {}         # etiqueta -> [columnas con 'sr']
    for etiqueta, ranuras in filas:
        universal[etiqueta] = [c for c, m in zip(COLUMNAS, ranuras) if m == "x"]
        riesgo[etiqueta] = [c for c, m in zip(COLUMNAS, ranuras)
                            if m in ("sr", "sr1")]

    for etiqueta, esperado in CONTROLES.items():
        obtenido = universal.get(etiqueta)
        if obtenido != esperado:
            raise SystemExit(
                "La matriz actividad × visita se ha decodificado mal.\n"
                f"  fila:      {etiqueta}\n"
                f"  esperado:  {esperado}\n"
                f"  obtenido:  {obtenido}\n"
                "Revisa parsear_fila(): el PDF habrá cambiado de formato."
            )

    def edades(cols):
        return ", ".join(EDADES[c] for c in cols)

    partes = [
        "CONTENIDOS DEL PROGRAMA DE ACTIVIDADES INDIVIDUALES",
        "Qué actividad del programa corresponde a cada visita.",
        "",
        "Las 12 visitas, en orden: " + ", ".join(EDADES[c] for c in COLUMNAS) + ".",
        "«Universal» = se hace a toda la población en esa visita.",
        "«Solo en situación de riesgo» = únicamente si hay factores de riesgo.",
        "",
        "== VISTA POR ACTIVIDAD: en qué visitas se hace cada tema ==",
        "",
    ]
    for etiqueta, _ in filas:
        partes.append(etiqueta)
        if universal[etiqueta]:
            partes.append(f"  - Universal: {edades(universal[etiqueta])}")
        if riesgo[etiqueta]:
            partes.append(f"  - Solo en situación de riesgo: {edades(riesgo[etiqueta])}")
        partes.append("")

    partes += ["== VISTA POR VISITA: qué temas se revisan en cada una ==", ""]
    for i, col in enumerate(COLUMNAS):
        uni = [e for e, _ in filas if universal[e] and col in universal[e]]
        rie = [e for e, _ in filas if riesgo[e] and col in riesgo[e]]
        partes.append(f"Visita de {EDADES[col]} (columna «{col}»)")
        partes.append("  Universal: " + ("; ".join(uni) if uni else "ninguna"))
        if rie:
            partes.append("  Solo en situación de riesgo: " + "; ".join(rie))
        partes.append("")

    # Lo que la tabla dice sobre actividades no ligadas a una visita concreta.
    cola = []
    for n in paginas:
        texto = lector.pages[n - 1].extract_text() or ""
        if "Actividades específicas:" in texto:
            cola = texto.split("Actividades específicas:")[1].split("Programa de")[0]
            cola = [l.strip() for l in cola.split("\n") if l.strip()]
    if cola:
        partes += ["== ACTIVIDADES ESPECÍFICAS (no ligadas a una visita concreta) =="]
        partes += [f"  - {l}" for l in cola]

    return "[p.1-2]\n" + "\n".join(partes)


def limpiar(texto: str) -> str:
    """Normaliza el texto extraído sin destruir la estructura de tablas.

    pypdf deja ligaduras, guiones de corte y espacios intercalados dentro de
    palabras ("P r o g r a m a"). Eso hundía la recuperación por embeddings y
    también despista al modelo, así que se arregla aquí una sola vez.
    """
    texto = unicodedata.normalize("NFKC", texto)
    # Encabezado repetido en cada página: puro ruido, se repite 500 veces.
    texto = re.sub(
        r"Programa\s+de\s*\n?\s*Salud\s+Infantil\s*\n?\s*y\s+Adolescente\s*\n?\s*"
        r"de\s+Andaluc[ií]a\s*\n?\s*(TEMAS)?",
        "",
        texto,
    )
    # Palabras con una letra y un espacio alternados -> se recomponen.
    texto = re.sub(
        r"\b(?:[A-Za-zÁÉÍÓÚÜÑáéíóúüñ] ){3,}[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]\b",
        lambda m: m.group(0).replace(" ", ""),
        texto,
    )
    texto = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", texto)   # guion de corte de línea
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


def leer(ruta: Path, rango=None):
    """Devuelve el texto con marcas [p.N] por página, usando la numeración real del PDF."""
    lector = PdfReader(str(ruta))
    ini, fin = rango if rango else (1, len(lector.pages))
    partes = []
    for n in range(ini, min(fin, len(lector.pages)) + 1):
        bruto = lector.pages[n - 1].extract_text() or ""
        txt = limpiar(bruto)
        if len(txt) < 15:          # portadas y páginas en blanco
            continue
        partes.append(f"[p.{n}]\n{txt}")
    return "\n\n".join(partes), (ini, min(fin, len(lector.pages)))


def fecha_de(nombre: str) -> str:
    """Extrae la fecha de edición del nombre del fichero ('Mar25' -> '03/2025')."""
    meses = {
        "ene": "01", "feb": "02", "mar": "03", "marz": "03", "abr": "04",
        "may": "05", "jun": "06", "jul": "07", "ago": "08", "sep": "09",
        "oct": "10", "nov": "11", "dic": "12",
    }
    m = re.search(r"(ene|feb|marz|mar|abr|may|jun|jul|ago|sep|oct|nov|dic)\.?(\d{2,4})",
                  nombre, re.I)
    if not m:
        return "2014-2018 (versión original del programa)"
    mes = meses[m.group(1).lower()]
    anio = m.group(2)
    anio = f"20{anio}" if len(anio) == 2 else anio
    return f"{mes}/{anio}"


# Esta fuente NO pasa por leer()/limpiar(): se decodifica con matriz_a_texto().
ID_MATRIZ = "psiaa_visitas_individuales_actividades"


# ---------------------------------------------------------------------------
# Fichas que necesitan transcripción manual.
#
# Sus PDF usan fuentes sin tabla de caracteres (sin ToUnicode CMap): los glifos
# se dibujan pero el texto extraído son índices de glifo (\x01\x02\x03…), no
# letras. Ni pypdf ni PyMuPDF pueden recuperarlo; haría falta OCR.
#
# En vez de meter OCR en el build, la transcripción vive como fichero de texto
# en transcripciones/: así es revisable, versionable y corregible a mano, que es
# lo que corresponde a contenido clínico.
#
# Si el fichero no existe, la fuente se OMITE con un aviso en vez de romper el
# build ni colar ruido en el corpus.
# ---------------------------------------------------------------------------
TRANSCRIPCIONES = Path(__file__).resolve().parent / "transcripciones"
TRANSCRITOS = {
    "CribadoCriptorquidia_FichaJun25": "CribadoCriptorquidia_FichaJun25.md",
    "SiindromeMuerteSubitaLactante_Jun25": "SiindromeMuerteSubitaLactante_Jun25.md",
}

# Proporción mínima de letras sobre el total de caracteres.
#
# Hay PDF cuyas fuentes no llevan tabla de caracteres (sin ToUnicode CMap): los
# glifos se dibujan pero no hay forma de saber qué letra es cada uno, y la
# extracción devuelve signos de puntuación y espacios. El texto "existe" y no
# está vacío, así que una comprobación de longitud no lo detecta — y el modelo,
# al recibir ruido, responde correctamente que no tiene información, con lo que
# el fallo parece del asistente y no de la extracción.
#
# Medido sobre las 110 fuentes: las sanas van del 70 % al 82 % de letras; las
# ilegibles daban 0 % y 9,5 %. El corte en 50 % es inequívoco.
MIN_LETRAS = 0.50
LETRA = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]")


def proporcion_letras(texto: str) -> float:
    limpio = re.sub(r"\[p\.\d+\]", "", texto)
    return len(LETRA.findall(limpio)) / max(1, len(limpio))


omitidas: list[str] = []


def añadir(corpus, ruta, ident, titulo, bloque, publico, claves, rango=None):
    if not ruta.exists():
        raise SystemExit(f"FALTA el PDF esperado: {ruta}")

    if ident in TRANSCRITOS:
        fichero = TRANSCRIPCIONES / TRANSCRITOS[ident]
        if not fichero.exists():
            omitidas.append(ident)
            return
        texto = fichero.read_text(encoding="utf-8").strip()
        if "[p.1]" not in texto:
            raise SystemExit(
                f"TRANSCRIPCIÓN SIN MARCAS DE PÁGINA: {fichero.name}\n"
                "  Hace falta una línea '[p.1]', '[p.2]'… al principio de cada\n"
                "  página. Sin ellas la app no puede citar la página y las citas\n"
                "  son el punto entero de este asistente."
            )
        paginas = [int(n) for n in re.findall(r"\[p\.(\d+)\]", texto)]
        ini, fin = min(paginas), max(paginas)
    elif ident == ID_MATRIZ:
        texto, (ini, fin) = matriz_a_texto(ruta), (1, 2)
    else:
        texto, (ini, fin) = leer(ruta, rango)

    ratio = proporcion_letras(texto)
    if texto and ratio < MIN_LETRAS:
        raise SystemExit(
            f"EXTRACCIÓN ILEGIBLE: {ruta.name}\n"
            f"  solo el {ratio:.1%} del texto son letras (lo normal es 70-82 %).\n"
            f"  El PDF usa fuentes sin tabla de caracteres: no se puede extraer su\n"
            f"  texto y haría falta OCR. Añádelo a TRANSCRITOS o quítalo de META.\n"
            f"  Muestra: {texto[:120]!r}"
        )
    if not texto:
        raise SystemExit(f"SIN TEXTO extraíble (¿necesita OCR?): {ruta}")
    corpus.append({
        "id": ident,
        "titulo": titulo,
        "bloque": bloque,
        "publico": publico,
        "claves": claves,
        "fecha": fecha_de(ruta.name),
        "archivo": ruta.name,
        "paginas": f"{ini}-{fin}",
        "texto": texto,
    })


def main():
    corpus = []

    # Fichas temáticas (Repetidos/): todas dirigidas al profesional.
    for ident, (titulo, bloque, claves) in META.items():
        añadir(corpus, FICHAS / f"{ident}.pdf", ident, titulo, bloque,
               "Profesional", claves)

    # Visitas, guías anticipatorias y materiales de parentalidad (estructura/).
    for ident, (titulo, bloque, publico, claves) in META_ESTRUCTURA.items():
        añadir(corpus, ESTRUC / f"{ident}.pdf", ident, titulo, bloque,
               publico, claves)

    # Actividades grupales y comunitarias (Actividades_grupales/).
    for ident, (titulo, bloque, publico, claves) in META_GRUPALES.items():
        añadir(corpus, GRUPAL / f"{ident}.pdf", ident, titulo, bloque,
               publico, claves)

    for ruta, rango, ident, titulo, bloque, claves in EXTRAS:
        añadir(corpus, ruta, ident, titulo, bloque, "Profesional", claves, rango)

    ids = [d["id"] for d in corpus]
    if len(ids) != len(set(ids)):
        repes = {i for i in ids if ids.count(i) > 1}
        raise SystemExit(f"ids duplicados: {repes}")

    SALIDA.write_text(
        json.dumps(corpus, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    total = sum(len(d["texto"]) for d in corpus)
    indice = sum(len(d["titulo"]) + len(d["bloque"]) + len(d["claves"]) + 40
                 for d in corpus)
    if omitidas:
        print(f"\n AVISO: {len(omitidas)} fuente(s) OMITIDAS por falta de transcripción:")
        for i in omitidas:
            print(f"   - {i}  ->  falta transcripciones/{TRANSCRITOS[i]}")
        print("   El asistente dirá 'no tengo información' sobre esos temas.\n")

    print(f"{len(corpus)} fuentes -> {SALIDA}")
    print(f"{total:,} caracteres (~{total // 4:,} tokens)")
    print(f"índice del enrutador: ~{indice // 4:,} tokens")
    print(f"fuente más grande: ~{max(len(d['texto']) for d in corpus) // 4:,} tokens")

    print("\nFuentes por bloque:")
    bloques = {}
    for d in corpus:
        bloques.setdefault(d["bloque"], []).append(d)
    for b, ds in sorted(bloques.items(), key=lambda x: -len(x[1])):
        toks = sum(len(d["texto"]) for d in ds) // 4
        print(f"  {len(ds):>3}  {toks:>7,} tok  {b}")

    print("\n10 fuentes más grandes (tokens aprox.):")
    for d in sorted(corpus, key=lambda x: -len(x["texto"]))[:10]:
        print(f"  {len(d['texto']) // 4:>7,}  {d['id']}")


if __name__ == "__main__":
    main()
