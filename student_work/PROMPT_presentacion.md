# Presentation prompt (paste into the slide tool together with the five PNG figures)

Create a **16-slide presentation in Spanish** for a university workshop on reinforcement learning in a simulated stock exchange.
Team: **TEAM_01 ("Team 1")**, members **Daniel Palacios and Sofia Ariza**. Audience: classmates and the lecturer, mixed technical level.
Talk length: about 12–15 minutes. It must be easy to follow for someone who has never studied machine learning or finance.

## 1. Base material
I attach five charts. They are the **visual base** of the deck: copy their colours, typography feel and flat style everywhere.
- `fig1_practica_holdout2.png`: practice market, return per scenario
- `fig2_abides_prueba_final.png`: final test, return and max drawdown per market type
- `fig3_abides_ablacion.png`: what each piece of the new model contributes
- `fig4_abides_competitivo.png`: risk vs return with all agents in the same market
- `fig5_como_decide.png`: how the model decides, in 4 steps

If the tool can draw native **vector** charts, rebuild each chart as an editable vector chart with the exact numbers in section 6 and the same colours.
If not, place the PNGs as they are: never stretch them, and never recolour them.

## 2. Visual style (follow strictly)
- **Flat, minimal, editorial look**, like a clean data report. **No emojis anywhere.** No stock photos, no 3D, no shadows, no gradients, no clip-art, no decorative backgrounds.
- **Vector graphics only** for diagrams and icons: simple line icons (2 px stroke, rounded ends, one colour, no fill), arrows, boxes with 12 px rounded corners, timelines.
  Draw them as SVG or native shapes so they stay sharp. Use icons sparingly, at most one per idea, to support a word, never to replace it.
- **Colour palette** (the same as the charts, and each colour always means the same thing):

  | Hex | Meaning |
  |---|---|
  | `#2A78D6` (blue) | v2, the new candidate |
  | `#EB6834` (orange) | v1, the submitted model |
  | `#1BAF7A` (green-aqua) | the momentum rule (simple strategy) |
  | `#8A8985` (grey) | other agents / neutral |
  | `#0B0B0B` | titles and main text |
  | `#52514E` | secondary text |
  | `#E4E3DF` | thin lines and grids |
  | `#FCFCFB` | slide background (near-white) |

  Use red only for one "rejected / fails" mark, paired with the word "No", never colour alone. Green only for "passes", always with the word "Sí".
- **Typography**: one sans-serif family (Inter, Source Sans 3 or Calibri).
  - Titles 32–36 pt bold.
  - Body 18–22 pt.
  - Chart labels at least 14 pt.
  - Left-aligned text, generous white space, at most 3 bullets per slide and at most about 25 words of body text per slide.
- **Action titles**: every slide title is a full sentence that states the conclusion. For example, "Entrenar en mercados variados fue la mayor mejora", not "Resultados ronda 1".
- **One idea per slide.** Put every important number in large type (48–60 pt) with a short label under it ("hero numbers").
- **Consistency**: v1 always orange and on the left, v2 always blue and on the right, when they are compared side by side.
- **Abbreviations**: the first time any abbreviation appears, write its meaning in brackets in Spanish. Repeat the meaning in a small footnote on any later slide where it is essential.
  Do not invent new abbreviations. Write numbers in Spanish format (decimal comma: 3,25 %).
- **Speaker notes**: for every slide, write 3–5 sentences of simple spoken Spanish that explain the slide to a non-expert.

## 3. Abbreviations and terms to explain (use these exact explanations)

| Term | Meaning |
|---|---|
| RL | Reinforcement Learning, aprendizaje por refuerzo: aprender probando y recibiendo premios o castigos |
| DQN | Deep Q-Network, red neuronal que puntúa cada acción posible y elige la mejor |
| PPO | Proximal Policy Optimization, otro método de RL que probamos y no funcionó |
| ABIDES-NG | Agent-Based Interactive Discrete Event Simulation, Next Generation: el simulador de bolsa realista donde se compite |
| Drawdown (DD) | Caída máxima del dinero desde su punto más alto |
| Ensamble / comité | Varias redes que opinan y se promedian |
| Holdout | Simulaciones apartadas que solo se usan una vez, como examen final |
| Semilla | Número que fija cómo se mueven los precios en una simulación |
| Orden a mercado / orden límite | Comprar o vender ya, frente a comprar o vender solo a un precio fijado (puede quedar en espera) |
| Spread | Diferencia entre precio de compra y de venta; si es enorme, el mercado está "roto" |
| Momentum | Tendencia del precio a seguir en la misma dirección |
| v1 / v2 | Versión 1 (entregada) y versión 2 (candidata) |
| TorchScript | Formato del archivo entregado; PyTorch es la herramienta con la que se entrenan las redes |

## 4. Slide-by-slide content
1. **Title.** "Un agente de bolsa que aprende solo: qué hicimos y por qué".
   - Subtitle: TEAM_01, Daniel Palacios y Sofia Ariza.
   - Visual: a minimal vector line chart of a price with a small decision marker. No photo.
2. **El reto.** A program that buys and sells by itself in a simulated stock exchange and competes against other teams in ABIDES-NG, in unknown markets.
   - Visual: vector diagram "mercado → programa → orden de compra/venta → mercado" as a loop.
3. **Cómo se puntúa.** Four blocks sized by weight:
   - Rentabilidad media 50 %
   - Control del riesgo (drawdown) 20 %
   - Robustez 20 %
   - Buen comportamiento 10 %

   Key message: ganar mucho una vez no basta; hay que ganar sin caídas grandes.
4. **Cómo aprende: aprendizaje por refuerzo (RL).** Four-step cycle as a vector loop: observa 16 datos → elige 1 de 9 acciones → recibe una nota → repite miles de veces.
   - Analogy line: "como aprender un videojuego a base de partidas".
5. **Cómo decide nuestro modelo.** Use `fig5_como_decide.png`, or rebuild it as 4 vector boxes:
   - Mira el mercado (grey)
   - Siete redes opinan (orange)
   - Capa de seguridad (blue)
   - Decide (green-aqua)
6. **El punto de partida tenía tres problemas.** Three columns, each with one line icon:
   - solo sabía comprar
   - 40 de cada 100 órdenes chocaban con el límite
   - dependía de la suerte (−7 % frente a +14 % con la misma receta)
7. **Probamos de forma justa.** Vector graphic of three separate boxes:
   - práctica (tuning): ejercicios
   - validación: simulacro
   - examen final (holdout): se usa una sola vez

   Message: "si miras el examen al elegir, te engañas a ti mismo".
8. **Lo que funcionó en la primera etapa.** Four rows, each with icon, idea and analogy:
   - mercados variados = piloto con sol, lluvia y niebla (la mayor mejora)
   - castigo al riesgo
   - comité de 7 redes = media de 7 expertos
   - capa de seguridad = cinturón y frenos: máx. 30 % del dinero en riesgo, freno al −5 %, no operar con el mercado roto

   One small line: "también registramos lo que no funcionó (PPO)".
9. **El resultado: v1, el modelo entregado.** Hero numbers from the practice final exam:
   - +2,23 % ganancia media
   - 1,35 % caída media
   - 0 órdenes inválidas

   Mark the model in orange.
10. **Segunda etapa: buscar los puntos débiles en ABIDES-NG.** Tested v1 in 11 market types; mean −0,31 %. Three causes, each with a small vector pictogram:
    - órdenes en espera que se acumulan
    - ventas forzadas en un mercado sin compradores
    - datos que nunca había visto
11. **Qué aporta cada mejora.** Use `fig3_abides_ablacion.png`. Highlight that only the combination (blue bar, v2) is both profitable and safe: +1,31 % with a 11,7 % worst drop, against −0,38 % and 52,8 % for v1.
12. **Las reglas del examen final se escribieron antes.** Table with 6 rows and columns Condición / v2 / v1 / ¿Cumple? (data in section 6). Mark the failed row with the word "No" and a thin red outline. Mark the practice row "Sí (empate exacto)".
13. **El examen final en ABIDES-NG.** Use `fig2_abides_prueba_final.png`. Callout: "v2 gana más de media (+3,25 % frente a +0,60 %), pero una simulación cayó un 126 %".
14. **Con todos los programas en el mismo mercado.** Use `fig4_abides_competitivo.png`. Message: v2 is the most balanced (+3,31 % with a 15 % worst drop). Others earn more on average but with drops of up to 93 %.
15. **Decisión: mantenemos v1.**
    - Two short paragraphs: why (we respect rules fixed in advance; one catastrophic drop weighs 20 % of the grade).
    - The root cause: in an empty market the loss brake sold at an absurd price. v1 has the same weakness: −97 % once.
    - Next step: make the brake wait while the market is broken, and test on new simulations.
    - Honesty note: the practice condition passed only as an exact tie, because before the exam we changed "mejor que v1" to "igual o mejor que v1".
16. **Lo que aprendimos.** Four short lines with line icons:
    - variedad de práctica mayor que algoritmo sofisticado
    - un comité mayor que un experto
    - probar pronto en el entorno real
    - reglas previas nos protegen de engañarnos

    Close with a small glossary strip of the abbreviations used.

## 5. Things to avoid
- Emojis, cartoons, photos, 3D charts, pie charts and dual-axis charts.
- Rainbow colours, and colours that change meaning between slides.
- Paragraphs longer than three lines, and text on top of charts.
- Abbreviations left unexplained, and English jargon where a Spanish word exists.

## 6. Exact data (for rebuilding charts or tables)

**Fig 1, practice market, final exam (holdout2), mean return % per scenario** (v2 / v1 / regla):

| Scenario | v2 | v1 | regla |
|---|---|---|---|
| Estable | +0,02 | +0,11 | −0,18 |
| Tendencia alcista | +0,33 | +0,55 | +9,11 |
| Volátil | +0,40 | +0,68 | −0,33 |
| Crash | +4,65 | +6,73 | +7,57 |
| Tendencia bajista | +4,22 | +5,75 | +6,97 |
| Baja liquidez | +0,28 | +0,71 | −1,20 |
| Errático | −0,64 | −1,24 | −0,17 |

**Fig 2, ABIDES-NG final test, 11 markets × 10 new seeds** (mean return % v2 / v1; max drawdown % v2 / v1):

| Market | Return v2 | Return v1 | Max DD v2 | Max DD v1 |
|---|---|---|---|---|
| Día estable | +2,37 | +0,01 | 7,4 | 0,2 |
| Día volátil | −1,85 | −1,32 | 17,7 | 10,6 |
| Baja liquidez | −0,04 | −0,10 | 0,3 | 0,5 |
| Estrés | +7,91 | +3,24 | 18,2 | 9,0 |
| Día con tendencia | +23,53 | +2,99 | 11,5 | 20,1 |
| Referencia (rmsc04) | −0,02 | −0,02 | 0,1 | 0,1 |
| Mercado líquido | −0,65 | +4,15 | 23,3 | 5,9 |
| Mercado poco profundo | −0,03 | −0,07 | 0,2 | 0,3 |
| Presión compradora | +5,53 | −9,93 | 21,8 | 98,2 |
| Presión vendedora | +4,19 | +7,76 | 16,3 | 17,3 |
| Volátil + momentum | −5,19 | −0,11 | 116,4 | 45,7 |

**Fig 3, ABIDES-NG validation, 55 runs each** (mean return %; max drawdown %):

| Version | Mean return | Max drawdown |
|---|---|---|
| v1 (7 redes) | −0,38 | 52,8 |
| 9 redes | −0,97 | 140,4 |
| 9 redes + guardia de órdenes en espera | −1,32 | 113,9 |
| 9 redes + libro neutralizado | +1,26 | 20,7 |
| v2 = 9 redes + ambas mejoras | +1,31 | 11,7 |

**Fig 4, competitive ABIDES-NG, same market** (mean return %; max drawdown %):

| Agent | Mean return | Max drawdown |
|---|---|---|
| v2 | +3,31 | 15,3 |
| v1 | +0,75 | 15,5 |
| Regla de momentum | +5,53 | 79,4 |
| DQN inicial semilla 7 | +8,25 | 92,9 |
| DQN inicial semilla 8 | +17,81 | 88,1 |

**Final-exam table (slide 12)** (Condición / v2 / v1 / ¿Cumple?):

| Condición | v2 | v1 | ¿Cumple? |
|---|---|---|---|
| Nota en práctica igual o mejor | 0,6375 | 0,6375 | Sí (empate exacto) |
| Ganancia media en ABIDES igual o mejor | +3,25 % | +0,60 % | Sí |
| Caída media no mayor que 1,25 × v1 | 4,9 % | 4,1 % | Sí |
| Peor caída no mayor que la de v1 | 116 % | 98 % | No |
| Mercado compartido: ganar al menos lo mismo, peor caída ≤ 20 % | +3,31 % / 15 % | +0,75 % / 16 % | Sí |
| Funciona igual en PyTorch 2.2 | idéntico | idéntico | Sí |
