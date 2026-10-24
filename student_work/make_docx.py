"""Generate the Spanish explanation of the TEAM_01 model as a .docx.

Needs python-docx, which is not in the project venv. Use the side venv:
  outputs\\venv_torch22\\Scripts\\python.exe student_work\\make_docx.py student_work\\Explicacion_modelo_TEAM_01.docx
"""
import sys

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

out = sys.argv[1]
doc = Document()
st = doc.styles["Normal"]
st.font.name = "Calibri"
st.font.size = Pt(11)


def h(text, level):
    doc.add_heading(text, level=level)


def p(text, bold_prefix=None):
    para = doc.add_paragraph()
    if bold_prefix:
        para.add_run(bold_prefix).bold = True
    para.add_run(text)
    return para


def bullets(items, style="List Bullet"):
    for it in items:
        para = doc.add_paragraph(style=style)
        if isinstance(it, tuple):
            para.add_run(it[0]).bold = True
            para.add_run(it[1])
        else:
            para.add_run(it)


def table(header, rows, bold_row=None):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    for i, c in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = ""
        cell.paragraphs[0].add_run(c).bold = True
    for r_i, r in enumerate(rows):
        cells = t.add_row().cells
        for i, c in enumerate(r):
            run = cells[i].paragraphs[0].add_run(c)
            if bold_row is not None and r_i == bold_row:
                run.bold = True
    doc.add_paragraph()


title = doc.add_heading("Agente de trading con aprendizaje por refuerzo — TEAM_01", 0)
sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.LEFT
r = sub.add_run("Equipo: Team 1 · Integrantes: Daniel Palacios, Sofia Ariza\n"
                "Qué hicimos para mejorar el modelo, por qué elegimos el modelo final y qué hace")
r.italic = True
r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

h("1. Objetivo", 1)
p("Entrenar con aprendizaje por refuerzo una política de trading que recibe 16 variables del mercado "
  "y elige una de 9 acciones: mantener, o comprar/vender con orden de mercado o límite, en tamaño "
  "pequeño (10 acciones) o grande (50). El modelo se entrega como TorchScript en TEAM_01.zip.")
p("El profesor lo evalúa en ABIDES-NG con semillas y escenarios ocultos: mercado normal, crisis, baja "
  "liquidez y competencia entre equipos. La nota se reparte así:")
table(["Criterio", "Peso"], [
    ["Rentabilidad media", "50 %"],
    ["Control del riesgo (máximo drawdown)", "20 %"],
    ["Robustez entre escenarios", "20 %"],
    ["Comportamiento operativo (órdenes inválidas, exposición excesiva, insolvencia)", "10 %"],
])
p("Por eso no buscamos el modelo que más gana en una simulación, sino el que gana de forma consistente "
  "y controla el riesgo.")

h("2. Punto de partida: el DQN inicial", 1)
p("El DQN del kit tenía tres problemas:")
bullets([
    ("Sesgo comprador. ", "Los tres mercados de práctica nunca bajan, así que aprendió a comprar siempre: "
                          "ganaba +13 % en tendencia alcista y perdía −6 % en caída y −2 % en tendencia bajista."),
    ("Choque constante con el límite de posición. ", "Unas 40 órdenes rechazadas por cada 100 pasos, lo que "
                                                     "penaliza el 10 % operativo."),
    ("Mucha varianza. ", "Con la misma receta y otra semilla, una versión perdía −7 % en crisis y otra ganaba +14 %."),
])

h("3. Qué hicimos", 1)
h("Paso 0: infraestructura de evaluación", 2)
p("Antes de mejorar nada creamos un protocolo fijo:")
bullets([
    ("Escenarios: ", "los 3 oficiales más 4 de estrés propios (crash, tendencia bajista, baja liquidez y "
                     "mercado errático), con episodios de 300 y 1000 pasos."),
    ("Tres conjuntos de semillas separados: ", "tuning (para comparar experimentos entre sí), validation "
                                               "(para decidir entre rondas) y holdout (se usa una sola vez, "
                                               "para la elección final)."),
    ("Una puntuación que imita la rúbrica: ", "50/20/20/10."),
])
p("Comparamos siempre contra dos referencias: \"no hacer nada\" (HOLD) y una regla de momentum. También "
  "construimos un simulador que ejecuta nuestro modelo dentro de ABIDES-NG, porque el test del kit no lo hacía.")

h("Ronda 1: cuatro ideas en paralelo", 2)
p("En cada ronda probamos varias ideas a la vez y las comparamos con el mismo protocolo.")
table(["Idea", "Resultado"], [
    ["Recompensa con penalización de riesgo (drawdown, rotación, inventario)", "Mejora moderada"],
    ["Entrenar en mercados aleatorios (tendencia al alza y a la baja, volatilidad, spread, liquidez, "
     "rellenos de órdenes límite menos favorables)",
     "La mayor mejora: puntuación 0,63 frente a 0,31; ganancias en crash (+5,2 %) y en tendencia bajista (+4,1 %)"],
    ["DQN mejorado (Double, Dueling, n-step)", "Sin mejora"],
    ["PPO", "Fracasó: acabó comprando el 98 % del tiempo y perdiendo −15 % en crisis"],
], bold_row=1)
p("lo importante era la variedad de mercados en el entrenamiento, no el algoritmo.", bold_prefix="Conclusión: ")

h("Ronda 2: combinaciones", 2)
p("Mercados aleatorios más recompensa de riesgo, con 5 semillas cada una. Descubrimos algo clave:")
bullets([
    ("El mejor modelo individual no generaliza. ", "El mejor en tuning bajó de 0,84 a 0,52 con semillas nuevas."),
    ("Un ensamble sí generaliza. ", "Promediar varias semillas buenas llegó a 0,79 en validación."),
])

h("Ronda 3: ABIDES-NG reveló riesgos que la práctica ocultaba", 2)
bullets([
    "La acción cuesta unos 1000 $ en vez de 100 $. Con 100 000 $ de capital, 400 acciones suponen cuatro "
    "veces el patrimonio en apalancamiento.",
    "Los mercados tranquilos son 5–10 veces más calmados y el spread es mucho menor.",
    "En stress_test el libro de órdenes puede vaciarse: el precio llegó a caer de 1000 $ a 0,24 $.",
])
p("Para estos riesgos diseñamos una capa de seguridad que se puso a prueba en ABIDES:")
table(["Protección", "Efecto medido"], [
    ["Exposición máxima ≈ 30 % del patrimonio, medida con el efectivo y válida a cualquier precio",
     "Rentabilidad en ABIDES de −5,9 % a +2,3 %; drawdown máximo de 107 % a 7,6 %"],
    ["Freno de pérdidas: con el patrimonio por debajo de −5 % solo reduce posición",
     "Un caso de insolvencia (−428 %) quedó en −25 %"],
    ["Libro roto: con spread > 100 pb no abre más exposición",
     "Drawdown máximo en estrés de 189 % a 21,6 %"],
])
p("También entrenamos modelos con mercados \"tipo ABIDES\". Solos no mejoraron, pero dos de ellos (los más "
  "estables) sumaron diversidad al ensamble final.")

h("4. Por qué elegimos v1 en la primera etapa", 1)
p("Al final de la primera etapa elegimos un ensamble de 7 DQN con la capa de seguridad (v1). La segunda etapa "
  "(secciones 8 y 9) lo mejoró hasta v2, que es el modelo entregado. Las razones para elegir v1 fueron:")
bullets([
    ("Ganó en las semillas holdout, ", "que nunca se usaron para elegir."),
], style="List Number")
table(["Modelo (holdout práctica, 280 episodios)", "Puntuación", "Rentab.", "Drawdown", "Peor escenario", "Ep. rentables"], [
    ["Modelo final (7 miembros)", "0,713", "+2,23 %", "1,35 %", "−0,17 %", "76 %"],
    ["Regla de momentum", "0,688", "+3,15 %", "1,80 %", "−1,49 %", "60 %"],
    ["Versión de 5 miembros", "0,588", "+2,07 %", "1,09 %", "−0,59 %", "75 %"],
    ["HOLD (no hacer nada)", "0,488", "0 %", "0 %", "0 %", "0 %"],
    ["DQN inicial", "0,025", "−0,17 %", "4,39 %", "−7,26 %", "41 %"],
], bold_row=0)
p("Además, 0 órdenes inválidas.")
bullets([
    ("Fue el mejor en los tres escenarios normales de ABIDES ", "(estable, volátil y baja liquidez)."),
    ("Pasó una última prueba en semillas de ABIDES nunca vistas.", ""),
    ("Carga y da resultados idénticos en PyTorch 2.2, ", "por si el profesor usa una versión antigua."),
], style="List Number")
table(["ABIDES, semillas nunca vistas", "Rentabilidad", "Drawdown medio", "Drawdown máximo"], [
    ["Modelo final (individual)", "+2,19 %", "1,2 %", "8,9 %"],
    ["Regla de momentum (individual)", "−5,4 %", "10,3 %", "88 %"],
    ["Modelo final, copia A (competitivo)", "+4,7 %", "2,9 %", "18,3 %"],
    ["Modelo final, copia B (competitivo)", "+3,1 %", "3,0 %", "15,8 %"],
    ["Regla de momentum (competitivo)", "+2,4 %", "6,1 %", "43 %"],
], bold_row=0)
p("el DQN inicial obtuvo de media más rentabilidad bruta en ABIDES. Pero eran apuestas apalancadas en libros "
  "rotos: mediana ≈ 0 %, drawdown de hasta 96 % y unas 135 órdenes inválidas por simulación. Con la rúbrica, "
  "eso es justo lo que se penaliza.", bold_prefix="Nota de honestidad: ")

h("5. Qué hace el modelo entregado (v2), paso a paso", 1)
bullets([
    ("Recibe 16 variables. ", "Del mercado: rentabilidades a 1, 5 y 20 pasos, volatilidad, spread, "
                              "desequilibrio del libro, profundidad y precio frente a sus medias. De la cartera "
                              "propia: inventario, efectivo, rentabilidad del patrimonio, tiempo transcurrido y "
                              "órdenes abiertas."),
    ("Neutraliza los datos del libro de órdenes. ", "A las redes les llega desequilibrio 0 y profundidad media "
                                                    "de práctica, porque en ABIDES esos valores están fuera de lo "
                                                    "que vieron al entrenar. La capa de seguridad sí ve los datos "
                                                    "reales."),
    ("Nueve redes neuronales (16→64→64→9) ", "estiman el valor (Q) de cada una de las 9 acciones, y se "
                                            "promedian: las 7 de v1 más 2 nuevas elegidas en la segunda etapa."),
    ("La capa de seguridad anula las acciones que ", "superarían 400 acciones de inventario, llevarían la "
                                                     "exposición a más del 30 % del patrimonio, aumentarían la "
                                                     "posición estando en −5 % (en ese caso obliga a reducirla), "
                                                     "abrirían posición con el libro roto o añadirían una orden "
                                                     "grande mientras hay otras en espera."),
    ("La acción final es la de mayor valor (argmax), ", "siempre válida."),
], style="List Number")
doc.add_picture("student_work/figures/fig5_como_decide.png", width=Inches(6.3))
p("En la práctica, el modelo opera de forma moderada a ambos lados del mercado: gana en caídas y no depende "
  "de que el mercado suba.")

h("6. Limitaciones", 1)
bullets([
    ("Sacrifica la ganancia en tendencias alcistas: ", "+0,5 % frente a +9 % de la regla. Es el precio de "
                                                      "eliminar el sesgo comprador."),
    ("El escenario stress_test de ABIDES sigue siendo muy aleatorio. ", "Las protecciones limitan la pérdida "
                                                                       "pero no la hacen predecible."),
    ("No conocemos el adaptador exacto del profesor: ", "cómo mide la profundidad, cada cuánto actúa el agente "
                                                        "y la escala de precios."),
    ("El freno de pérdidas sigue siendo un punto débil: ", "con el libro vacío, la orden a mercado que obliga a "
                                                          "cerrar la posición puede ejecutarse a un precio "
                                                          "absurdo. Le pasó una vez a v2 (−126 %) y una vez a v1 "
                                                          "(−97 %) en la prueba final."),
])

h("7. Proceso", 1)
p("Usamos varios agentes de IA en paralelo, cada uno con un experimento. Un orquestador comparaba todos los "
  "resultados con el mismo protocolo y decidía la siguiente ronda. Todo, incluidos los experimentos fallidos, "
  "está registrado en student_work/EXPERIMENTS.md.")

h("8. Segunda etapa: rondas 5 a 7", 1)
p("Con más tiempo antes de la presentación repetimos el proceso partiendo del modelo v1. Antes de mirar ningún "
  "resultado nuevo fijamos semillas nuevas (tuning, validación y una prueba final holdout2 que solo se usa una vez) "
  "y las reglas para sustituir a v1.")

h("Ronda 5: diagnóstico en ABIDES con más escenarios", 2)
p("Probamos v1 en 11 escenarios de ABIDES en vez de 4, incluidos un día con tendencia, mercados líquidos y poco "
  "profundos, y un agente que compra o vende 1,2 millones de acciones (presión compradora o vendedora). "
  "Resultado: v1 quedaba por debajo de no hacer nada (rentabilidad media −0,31 %). Las pérdidas venían de la "
  "ejecución, no de la predicción:")
bullets([
    ("Órdenes en espera: ", "varias órdenes límite grandes se ejecutaban a la vez y la posición llegaba a 120–170 "
                            "acciones (más del 100 % del patrimonio), aunque la capa de seguridad apuntaba al 30 %."),
    ("Ventas forzadas en un libro vacío: ", "con presión vendedora, el freno de pérdidas vendía a mercado sin "
                                           "compradores y perdía un −31 % aunque el precio acababa igual."),
    ("Rasgos fuera de rango: ", "en ABIDES el desequilibrio del libro llega a ±0,6–0,85, cuando en práctica es ±0,15, "
                                "así que las redes extrapolaban."),
])

h("Ronda 6: cuatro agentes en paralelo", 2)
table(["Idea", "Resultado"], [
    ["Redes que siguen tendencias en ambos sentidos", "Solas sí siguen la tendencia, pero dentro del ensamble "
                                                      "las demás redes las anulan. Descartada."],
    ["30 redes nuevas con las recetas que ya funcionaban", "14 de 30 pasan el filtro; dos de ellas se añaden al "
                                                           "ensamble (9 redes)."],
    ["Guardia de órdenes en espera", "Arregla los dos casos de pérdida grande en ABIDES; cuesta algo de "
                                     "rentabilidad en el mercado de práctica."],
    ["Neutralizar los rasgos del libro de órdenes", "Igual en práctica y mucho mejor en ABIDES."],
])

h("Ronda 7: combinación y validación", 2)
p("Ninguna pieza sola es suficiente. En la validación de ABIDES (55 simulaciones con semillas nuevas), solo la "
  "combinación de las dos mejoras es a la vez rentable y segura:")
doc.add_picture("student_work/figures/fig3_abides_ablacion.png", width=Inches(6.3))
p("El candidato v2 es: 9 redes (las 7 de v1 más 2 nuevas) + guardia de órdenes en espera (bloquea las órdenes "
  "grandes que aumentan la posición mientras hay órdenes sin ejecutar) + rasgos del libro neutralizados. "
  "Las protecciones de v1 se mantienen.")

h("9. Prueba final y decisión", 1)
p("Las reglas para sustituir a v1 se escribieron antes de ver ningún resultado de la prueba final. v2 tenía que "
  "cumplir las cuatro:")
table(["Criterio", "v2", "v1", "¿Cumple?"], [
    ["1. Práctica (holdout2): puntuación ≥ v1", "0,6375", "0,6375", "Sí (empate exacto)"],
    ["2a. ABIDES, 110 simulaciones: rentabilidad media ≥ v1", "+3,25 %", "+0,60 %", "Sí"],
    ["2b. ABIDES: drawdown medio ≤ 1,25 × v1", "4,9 %", "4,1 %", "Sí"],
    ["2c. ABIDES: drawdown máximo ≤ el de v1", "116 %", "98 %", "No"],
    ["3. ABIDES competitivo: rentabilidad ≥ v1 y drawdown máx. ≤ 20 %", "+3,31 % / 15 %", "+0,75 % / 16 %", "Sí"],
    ["4. Compatible con PyTorch 2.2", "idéntico", "idéntico", "Sí"],
])
doc.add_picture("student_work/figures/fig2_abides_prueba_final.png", width=Inches(6.3))
doc.add_picture("student_work/figures/fig4_abides_competitivo.png", width=Inches(5.5))
p("v2 no cumplió una de las condiciones: en una sola simulación (volátil + momentum) perdió un −126 %, más "
  "que la peor caída de v1 (−98 %).", bold_prefix="Resultado de la prueba: ")
p("aun así, el equipo decidió entregar v2. Es mejor que v1 en casi todo lo que puntúa la rúbrica: más "
  "rentabilidad en ABIDES (+3,25 % frente a +0,60 %), más rentabilidad y más simulaciones con ganancia en el "
  "mercado compartido (+3,31 % frente a +0,75 %; 73 % frente a 50 %), menos simulaciones con caídas de más del "
  "20 % (3,6 % frente a 5,5 %) y menor caída en el mercado de práctica. TEAM_01.zip contiene ahora v2 y el "
  "validador lo da por VÁLIDO; v1 queda guardado como copia de seguridad.", bold_prefix="Decisión del equipo: ")
p("esta decisión se aparta de la regla que fijamos antes de la prueba, y lo dejamos registrado. La regla "
  "protegía contra el riesgo de un episodio catastrófico, y ese riesgo sigue existiendo: el drawdown máximo es "
  "un 20 % de la nota. Además, la condición de práctica solo se cumplió por empate exacto, después de cambiar "
  "\"mejor que v1\" por \"igual o mejor que v1\" antes de la prueba.", bold_prefix="Transparencia: ")
p("en esa simulación el libro de órdenes se vació (spread de 87 $). El freno de pérdidas obliga a cerrar la "
  "posición con una orden a mercado, y sin contrapartida esa orden se ejecuta a precios absurdos. v1 tiene el "
  "mismo punto débil (−97 % con presión compradora). La mejora pendiente es que el freno espere (HOLD) mientras "
  "el libro está roto, y probarla con semillas nuevas.", bold_prefix="Por qué falló esa simulación: ")
p("las medias y los peores casos pueden apuntar a modelos distintos. Fijar las reglas antes de la prueba nos "
  "obligó a ver ese conflicto con claridad y a tomar la decisión de forma consciente y documentada.",
  bold_prefix="Lo que aprendimos: ")

doc.save(out)
print("saved", out)
