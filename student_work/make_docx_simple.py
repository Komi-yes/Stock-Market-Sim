"""Generate the plain-language Spanish explanation (no technical background needed) as a .docx.

Needs python-docx, which is not in the project venv. Use the side venv:
  outputs\\venv_torch22\\Scripts\\python.exe student_work\\make_docx_simple.py student_work\\Explicacion_sencilla_TEAM_01.docx
Run `python -m student_work.make_figures` first so the charts exist.
"""
import sys

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

out = sys.argv[1]
doc = Document()
st = doc.styles["Normal"]
st.font.name = "Calibri"
st.font.size = Pt(11.5)
FIG = "student_work/figures/"
GREY = RGBColor(0x52, 0x51, 0x4E)


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


def table(header, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    for i, c in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = ""
        cell.paragraphs[0].add_run(c).bold = True
    for r in rows:
        cells = t.add_row().cells
        for i, c in enumerate(r):
            run = cells[i].paragraphs[0].add_run(c)
            if i == 0:
                run.bold = True
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    doc.add_paragraph()


def figure(name, caption, width=6.3):
    doc.add_picture(FIG + name, width=Inches(width))
    para = doc.add_paragraph()
    run = para.add_run("Cómo leer este gráfico: " + caption)
    run.italic = True
    run.font.size = Pt(10)
    run.font.color.rgb = GREY


def callout(text):
    para = doc.add_paragraph()
    run = para.add_run(text)
    run.italic = True
    run.font.color.rgb = RGBColor(0x2A, 0x78, 0xD6)


# ---------------------------------------------------------------------------
doc.add_heading("Nuestro agente de bolsa, explicado sin tecnicismos", 0)
sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.LEFT
r = sub.add_run("Equipo TEAM_01 (Team 1) · Daniel Palacios y Sofia Ariza\n"
                "Para cualquier lector: las palabras técnicas se explican la primera vez que aparecen "
                "y están reunidas en el glosario del final.")
r.italic = True
r.font.color.rgb = GREY

h("1. La idea en pocas palabras", 1)
p("Hemos creado un programa que compra y vende acciones por sí solo dentro de una bolsa simulada (una bolsa "
  "de mentira que funciona en el ordenador, con precios que suben y bajan y con otros participantes que "
  "también compran y venden). El programa no sigue reglas escritas por nosotros: aprendió a operar "
  "practicando miles de veces y recibiendo una \"nota\" según ganaba o perdía dinero.")
p("El profesor pondrá a competir los programas de todos los equipos en un simulador más realista llamado "
  "ABIDES-NG, con situaciones que no conocemos de antemano: días tranquilos, crisis, mercados con pocos "
  "participantes y mercados donde operan los programas de los demás equipos.")
callout("Resumen de una línea: preferimos un programa que gane poco pero casi nunca tenga pérdidas grandes, "
        "porque así es como se puntúa.")

h("2. Cómo se puntúa", 1)
p("La nota no premia solo ganar dinero. Se reparte así:")
table(["Qué se mide", "Peso", "En palabras sencillas"], [
    ["Rentabilidad media", "50 %", "Cuánto dinero gana de media, en todas las situaciones."],
    ["Control del riesgo", "20 %", "La peor caída del dinero desde su punto más alto (el \"drawdown\"). "
                                   "Cuanto menor, mejor."],
    ["Robustez", "20 %", "Que funcione bien en todo tipo de mercados, no solo en uno."],
    ["Buen comportamiento", "10 %", "No enviar órdenes imposibles, no arriesgar de más y no quedarse "
                                    "sin dinero."],
], widths=[1.6, 0.7, 4.0])
p("Por eso un programa que gana mucho en un día pero se arruina otro día saca peor nota que uno prudente.")

h("3. Cómo aprende un programa: aprendizaje por refuerzo", 1)
p("El método se llama aprendizaje por refuerzo (en inglés, Reinforcement Learning o RL). Funciona como enseñar "
  "a un perro con premios, o como aprender un videojuego a base de partidas:")
bullets([
    ("Observa: ", "el programa mira 16 datos del mercado y de su propia cartera (por ejemplo, si el precio "
                  "ha subido en los últimos minutos, cuánto dinero le queda y cuántas acciones tiene)."),
    ("Decide: ", "elige una de 9 acciones: esperar sin hacer nada, o comprar o vender una cantidad pequeña "
                 "(10 acciones) o grande (50), de forma inmediata o poniendo un precio y esperando."),
    ("Recibe una nota: ", "si la decisión le hizo ganar dinero, la nota es buena; si perdió o se arriesgó "
                          "demasiado, es mala."),
    ("Repite: ", "tras miles de intentos aprende qué decisiones suelen salir bien en cada situación."),
], style="List Number")
p("La parte del programa que \"piensa\" es una red neuronal: una fórmula matemática con muchos números "
  "ajustables que se van corrigiendo con la práctica. No es un cerebro; es una calculadora que aprende de "
  "la experiencia. El tipo concreto que usamos se llama DQN (Deep Q-Network, \"red profunda de valores Q\"): "
  "para cada una de las 9 acciones calcula una puntuación de lo buena que parece, y elige la más alta.")
figure("fig5_como_decide.png",
       "de izquierda a derecha, el programa recibe los datos, siete redes dan su puntuación a cada acción, una "
       "capa de seguridad tacha las acciones peligrosas y se ejecuta la mejor de las que quedan.")

h("4. El punto de partida y sus problemas", 1)
p("El kit del curso traía un programa de ejemplo. Al probarlo encontramos tres problemas:")
bullets([
    ("Solo sabía comprar. ", "Los mercados de práctica del kit casi siempre suben, así que aprendió que "
                             "comprar siempre funciona. Cuando el mercado caía, perdía dinero."),
    ("Intentaba hacer cosas prohibidas. ", "Unas 40 de cada 100 órdenes chocaban con el límite de 500 "
                                           "acciones, y eso resta en la nota de buen comportamiento."),
    ("Dependía de la suerte. ", "Entrenando dos veces el mismo programa, una versión perdía un 7 % en una "
                                "crisis y otra ganaba un 14 %. Así no se puede confiar en él."),
])

h("5. Cómo lo mejoramos (primera etapa)", 1)
h("Probar de forma justa: el examen con preguntas nuevas", 2)
p("Antes de mejorar nada, preparamos un sistema de evaluación fijo. Cada simulación depende de una "
  "\"semilla\": un número que decide cómo se mueven los precios. Usamos tres grupos de semillas separados, "
  "como en un examen:")
bullets([
    ("Semillas de práctica (tuning): ", "para comparar ideas entre sí; es como hacer ejercicios del libro."),
    ("Semillas de validación: ", "para decidir entre rondas; es como un simulacro de examen."),
    ("Semillas de examen final (holdout): ", "se usan una sola vez al final y nunca para elegir; así el "
                                             "resultado no está \"maquillado\"."),
])
p("Si eligiéramos el modelo mirando el examen final, sería como aprenderse las respuestas: el resultado "
  "parecería bueno pero no diría nada de cómo funcionará en la competición real.")

h("Las cuatro ideas que funcionaron", 2)
table(["Idea", "Analogía", "Efecto"], [
    ["Entrenar en mercados muy variados", "Un piloto que practica con sol, lluvia y niebla, no solo con sol",
     "Dejó de depender de que el mercado suba; empezó a ganar también en caídas. Fue la mayor mejora."],
    ["Castigar el riesgo durante el entrenamiento", "Restar puntos por conducir rápido aunque se llegue antes",
     "Menos caídas grandes y menos órdenes imposibles."],
    ["Un comité de redes", "Preguntar a siete expertos y hacer la media, en vez de fiarse de uno",
     "Mucho menos dependiente de la suerte. Un comité de 7 redes se comporta mejor que la mejor red sola."],
    ["Una capa de seguridad", "El cinturón y los frenos de un coche",
     "Impide arriesgar más del 30 % del dinero, frena si pierde más de un 5 % y no opera si el mercado "
     "se queda sin compradores o vendedores."],
], widths=[1.7, 2.2, 2.4])
p("También probamos ideas que no funcionaron (otros algoritmos de aprendizaje, como PPO, siglas de Proximal "
  "Policy Optimization). Las dejamos anotadas: saber qué no funciona también es un resultado.")
p("El resultado de esta etapa fue el modelo que llamamos v1 (versión 1): el comité de 7 redes con la capa "
  "de seguridad. Es el modelo que está entregado.")

h("6. Segunda etapa: buscar los puntos débiles", 1)
p("Con más tiempo, pusimos a prueba v1 en 11 tipos de mercado del simulador ABIDES-NG, incluidos algunos que "
  "nunca habíamos probado: un día con tendencia, mercados con muchos y con pocos participantes, y mercados "
  "donde un gran inversor compra o vende 1,2 millones de acciones a lo largo del día (lo que empuja el precio "
  "con fuerza en una dirección).")
p("Descubrimos que v1 perdía algo de dinero de media (−0,31 %), y no porque se equivocara al predecir, sino "
  "por cómo se ejecutaban sus órdenes:")
bullets([
    ("Órdenes que se acumulaban. ", "Cuando el programa pone un precio y espera, la orden puede quedarse "
                                    "\"en espera\" varios minutos. Si se acumulan varias y se cumplen a la vez, "
                                    "acaba con muchas más acciones de las que la capa de seguridad permitía."),
    ("Ventas en un mercado vacío. ", "Cuando el freno de pérdidas se activaba, vendía de inmediato aunque no "
                                     "hubiera compradores, y la venta salía a precios ridículos."),
    ("Datos que nunca había visto. ", "Algunos datos del mercado realista tenían valores muy distintos a los "
                                      "del mercado de práctica, y las redes no sabían interpretarlos."),
])
p("Para atacar estos problemas trabajamos con varios asistentes de inteligencia artificial en paralelo, "
  "cada uno probando una idea, y un coordinador que comparaba todos los resultados con las mismas reglas. "
  "De las ideas probadas, dos funcionaron: una regla que impide órdenes grandes mientras hay otras en espera, "
  "y \"neutralizar\" los datos raros (darle a las redes valores normales en lugar de los extraños). Con ellas "
  "y dos redes nuevas formamos un candidato: v2 (versión 2).")
figure("fig3_abides_ablacion.png",
       "cada barra es una versión del programa probada en 55 simulaciones nuevas. A la izquierda, la ganancia "
       "media (mejor cuanto más a la derecha); a la derecha, la peor caída (mejor cuanto más corta). Solo la "
       "combinación de las dos mejoras (barra azul, v2) gana dinero y a la vez tiene caídas pequeñas.")

h("7. El examen final y la decisión", 1)
p("Antes de hacer el examen final escribimos las condiciones que v2 debía cumplir para sustituir a v1. "
  "Escribirlas antes es importante: si las decidiéramos después de ver el resultado, podríamos ajustarlas "
  "sin darnos cuenta para que ganara el modelo que nos gusta.")
table(["Condición", "v2", "v1", "¿Cumple v2?"], [
    ["Nota en el mercado de práctica igual o mejor", "0,6375", "0,6375", "Sí (empate exacto)"],
    ["Ganancia media en ABIDES igual o mejor", "+3,25 %", "+0,60 %", "Sí"],
    ["Caída media en ABIDES parecida o menor", "4,9 %", "4,1 %", "Sí"],
    ["Peor caída en ABIDES no mayor que la de v1", "116 %", "98 %", "No"],
    ["Con todos los programas en el mismo mercado: ganar al menos lo mismo", "+3,31 %", "+0,75 %", "Sí"],
    ["Funciona igual en una versión antigua del programa (PyTorch 2.2)", "idéntico", "idéntico", "Sí"],
], widths=[3.1, 0.9, 0.9, 1.4])
figure("fig2_abides_prueba_final.png",
       "cada fila es un tipo de mercado; azul es v2 y naranja es v1. A la izquierda, la ganancia media; a la "
       "derecha, la peor caída. v2 gana más en la mayoría de mercados, pero en \"Volátil + momentum\" tuvo "
       "una caída enorme.")
figure("fig4_abides_competitivo.png",
       "cada punto es un programa operando en el mismo mercado que los demás. Arriba es ganar más; a la "
       "izquierda es arriesgar menos. v2 (azul) es el más equilibrado: gana algo y casi no sufre caídas. Los "
       "programas de arriba a la derecha ganan más de media, pero con caídas de hasta el 93 %.", width=5.6)
p("v2 fue mejor en casi todas las medias, pero en una sola simulación perdió un 126 % del dinero. La causa: "
  "el mercado se quedó sin compradores, el freno de pérdidas obligó a cerrar la posición de inmediato y la "
  "operación salió a un precio absurdo. Como no cumplió la condición de la peor caída, mantuvimos v1 como "
  "modelo entregado.", bold_prefix="Decisión. ")
p("Una condición (la nota de práctica) solo se cumplió por un empate exacto, y eso fue posible porque antes "
  "del examen cambiamos la regla de \"mejor que v1\" a \"igual o mejor que v1\". Lo contamos porque es parte "
  "de hacer las cosas de forma honesta.", bold_prefix="Transparencia. ")
p("v1 tiene el mismo punto débil (perdió un 97 % en una simulación parecida). La mejora pendiente es que el "
  "freno espere cuando el mercado está vacío en lugar de vender a cualquier precio, y probarla con "
  "simulaciones nuevas.", bold_prefix="Siguiente paso. ")

h("8. Lo que hace el programa entregado, paso a paso", 1)
bullets([
    "Una vez por minuto de mercado, mira 16 datos: cómo se ha movido el precio, lo agitado que está el "
    "mercado, la diferencia entre el precio de compra y el de venta, cuánto dinero tiene y cuántas acciones.",
    "Siete redes neuronales dan una puntuación a cada una de las 9 acciones posibles, y se hace la media.",
    "La capa de seguridad tacha las acciones peligrosas: superar 400 acciones, arriesgar más del 30 % del "
    "dinero, aumentar la apuesta si ya pierde más de un 5 % (en ese caso, además, reduce la posición) u "
    "operar cuando el mercado está roto.",
    "Ejecuta la acción con mejor puntuación de las que quedan. Esperar sin hacer nada siempre está permitido.",
], style="List Number")

h("9. Limitaciones, en palabras sencillas", 1)
bullets([
    ("Gana poco cuando todo sube. ", "Al quitarle la costumbre de comprar siempre, aprovecha menos las subidas "
                                     "fuertes (en práctica, +0,5 % frente a +9 % de una regla sencilla)."),
    ("Las crisis extremas son casi una lotería. ", "En el escenario de estrés del simulador los precios pueden "
                                                   "desplomarse sin motivo; las protecciones limitan la pérdida "
                                                   "pero no la hacen predecible."),
    ("No conocemos todos los detalles de la competición. ", "Por ejemplo, cada cuánto tiempo decidirá el "
                                                            "programa o cómo se medirá el mercado."),
])

h("10. Lo que aprendimos", 1)
bullets([
    "Practicar en situaciones variadas importa más que usar el algoritmo más sofisticado.",
    "Varias opiniones promediadas (un comité) son más fiables que la mejor opinión individual.",
    "Hay que probar pronto en el entorno real: allí aparecieron problemas que la práctica escondía.",
    "Fijar las reglas antes del examen nos protege de engañarnos a nosotros mismos, aunque a veces nos haga "
    "rechazar un modelo que parece mejor.",
])

h("Glosario", 1)
p("Siglas y palabras técnicas que aparecen en el proyecto, explicadas sin tecnicismos.")
table(["Término", "Qué significa"], [
    ["ABIDES-NG", "Simulador de bolsa muy realista (las siglas vienen del inglés Agent-Based Interactive "
                  "Discrete Event Simulation; \"NG\" es Next Generation, nueva generación). Simula una bolsa con "
                  "cientos de participantes automáticos. Es donde se hace la competición."],
    ["Acción (de bolsa)", "Una pequeña parte de una empresa que se compra y se vende. En este documento también "
                          "decimos \"acción\" para cada decisión del programa; el contexto aclara cuál."],
    ["Agente", "Programa que toma decisiones por sí mismo dentro de un entorno, en este caso la bolsa."],
    ["Aprendizaje por refuerzo (RL)", "Reinforcement Learning: forma de aprender a base de probar y recibir "
                                      "premios o castigos, como enseñar a un perro con premios."],
    ["Capa de seguridad", "Reglas fijas que se ponen encima del programa para impedir decisiones peligrosas, "
                          "como el cinturón y los frenos de un coche."],
    ["Comité o ensamble", "Varias redes que opinan a la vez; se hace la media de sus puntuaciones."],
    ["Drawdown (caída máxima)", "La mayor bajada del dinero desde su punto más alto. Si se llega a 110 € y "
                                "luego se baja a 99 €, el drawdown es del 10 %."],
    ["DQN", "Deep Q-Network (red profunda de valores Q): el tipo de red neuronal que usamos. Calcula una "
            "puntuación (\"valor Q\") para cada acción posible y elige la más alta."],
    ["Escenario", "Un tipo de mercado: tranquilo, volátil, en crisis, con tendencia, etc."],
    ["Exposición", "Qué parte del dinero está invertida en acciones y, por tanto, en riesgo."],
    ["Libro de órdenes", "La lista de todas las ofertas de compra y de venta que hay en un momento dado."],
    ["Liquidez", "Facilidad para comprar o vender rápido sin mover el precio. Poca liquidez: hay pocos "
                 "compradores y vendedores."],
    ["Momentum", "Tendencia de un precio a seguir moviéndose en la misma dirección. La \"regla de momentum\" "
                 "es una estrategia sencilla: comprar si sube, vender si baja."],
    ["Orden a mercado", "Comprar o vender ya mismo, al mejor precio disponible en ese momento."],
    ["Orden límite", "Comprar o vender solo a un precio fijado; puede quedarse en espera hasta que alguien "
                     "acepte ese precio."],
    ["Patrimonio", "Todo el dinero del programa: el efectivo más lo que valen sus acciones. Empieza con "
                   "100 000 $."],
    ["PPO", "Proximal Policy Optimization: otro método de aprendizaje por refuerzo que probamos y que no "
            "funcionó en este proyecto."],
    ["PyTorch", "Herramienta de programación con la que se construyen y entrenan las redes neuronales."],
    ["Red neuronal", "Fórmula matemática con muchos números ajustables que aprende de los ejemplos. No piensa: "
                     "calcula."],
    ["Rentabilidad", "Cuánto se gana o se pierde, en porcentaje del dinero inicial."],
    ["Semilla", "Número que fija cómo se mueven los precios en una simulación. Con la misma semilla, la "
                "simulación se repite igual; con otra, sale distinta."],
    ["Sobreajuste", "Cuando un programa aprende \"de memoria\" los ejemplos de práctica y luego falla con "
                    "situaciones nuevas."],
    ["Spread", "Diferencia entre el precio al que alguien vende y el precio al que alguien compra. Un spread "
               "enorme significa que el mercado está \"roto\"."],
    ["Tendencia", "Movimiento sostenido del precio hacia arriba (alcista) o hacia abajo (bajista)."],
    ["TorchScript", "Formato en el que se guarda el programa entrenado para que el profesor pueda ejecutarlo "
                    "sin nuestro código."],
    ["Validación / holdout", "Pruebas con simulaciones que el programa no ha visto. Validación sirve para "
                             "decidir entre opciones; el holdout (\"apartado\") es el examen final y se usa "
                             "una sola vez."],
    ["Venta en corto", "Vender acciones que no se tienen (prestadas) esperando que el precio baje para "
                       "recomprarlas más baratas. Así se puede ganar cuando el mercado cae."],
    ["Volatilidad", "Cuánto y con qué rapidez sube y baja el precio. Alta volatilidad: mercado agitado."],
    ["v1 / v2", "Versión 1 (el modelo entregado) y versión 2 (el candidato de la segunda etapa)."],
], widths=[1.8, 4.5])

doc.save(out)
print("saved", out)
