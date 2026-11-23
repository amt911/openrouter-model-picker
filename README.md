# openrouter-model-picker

Programa interactivo de terminal para elegir **el mejor modelo y el más barato** de
[OpenRouter](https://openrouter.ai) según la tarea:

1. **Programar**
2. **Reescribir documentos**
3. **Cotejar información de papers**

Además compara modelos concretos lado a lado y exporta todo a CSV.

Un solo fichero en Python, **sin dependencias** (solo la librería estándar, Python ≥ 3.10).

## Uso

```bash
git clone https://github.com/amt911/openrouter-model-picker
cd openrouter-model-picker
export OPENROUTER_API_KEY=sk-or-...   # o pégala cuando la pida
python3 ormp.py --gui                 # interfaz gráfica en el navegador
python3 ormp.py                       # menú interactivo en la terminal
```

### Recomendación automática

Pestaña **★ Recomendación** en la GUI, o `python3 ormp.py --recommend --baseline <tu modelo> --per-month <horas de sesión>`
en la terminal. Para cada sección:

1. Puntúa los modelos con los benchmarks y busca el **proveedor fiable más barato** de cada uno
   (uptime 24 h ≥ 99%, sin 4 bits y, para programar, con herramientas).
2. Propone **Ahorro** (tu mismo modelo con ese proveedor), **Mejora** (el mejor que supera al tuyo y
   cuesta menos de lo que pagas hoy) y **Calidad** (el más barato con ≥ 90% de la mejor calidad).
3. Elige: si «Calidad» cuesta menos de $20 al mes con tu volumen, gana la calidad (no compensa
   ahorrar céntimos); si no, el ahorro y después la mejora.

Usa el volumen y el gasto real de «Tu uso» de cada pestaña, y muestra el total antes y después
y cómo fijar cada proveedor.

### Interfaz gráfica

`python3 ormp.py --gui` abre la GUI en el navegador (servidor local en
`127.0.0.1:8765`, solo accesible desde tu equipo; `--port 0` elige un puerto libre).

- Una pestaña por tarea con la recomendación, un gráfico **calidad vs. coste** (la línea
  verde es la frontera calidad/precio) y una tabla ordenable y filtrable.
- Controles en vivo: tokens por tarea, contexto mínimo, cobertura mínima, incluir gratis.
- **Tu uso**: tu modelo actual (para ver qué es mejor *y* más barato que él), sesiones en
  paralelo × horas/día × días/mes en «Programar» (tareas al mes en el resto) y, opcional, lo
  que pagas hoy al mes: con eso el volumen se calibra a tu factura real y todos los costes
  pasan a $/mes. En el gráfico tu modelo sale en naranja y la zona verde de arriba a la
  izquierda es "mejor y más barato".
- **Proveedores**: haz clic en un modelo (tabla o gráfico) para ver todos los proveedores
  que lo sirven en OpenRouter, ordenados por coste con tu perfil: precio, caché,
  cuantización (fp4/fp8), nivel *flex*, uptime de 24 h y soporte de herramientas. El mismo
  modelo puede costar la mitad según quién lo sirva; el panel te dice cómo fijar el proveedor
  (`"provider": {"order": [...]}` o `{"sort": "price"}`).
- **Tabla simple / Tabla completa**: la simple muestra solo calidad, coste y contexto; la
  completa tiene una fila por modelo y proveedor con benchmarks, precios, caché, cuantización
  y uptime, con filtros (más barato de cada modelo, con herramientas, sin 4 bits, uptime
  ≥ 99%) y descarga en CSV. La primera vez descarga los proveedores de OpenRouter (unos
  segundos; se guardan 1 h).
- **Filtros de tabla**: por autor (deepseek, google…), solo frontera, y «mostrar con pocos
  datos» para ver los modelos que no llegan a la cobertura mínima (salen marcados y no
  entran en las recomendaciones).
- **Límites de precio**: presupuesto máximo al mes, máximo $ por tarea (por pestaña), máximo $ de entrada y de salida
  por millón de tokens (para todas las pestañas) y tareas al mes para ver el gasto
  mensual estimado. Se recuerdan entre sesiones y la línea naranja del gráfico marca tu
  límite. Las recomendaciones pasan a ser "lo mejor dentro de tu presupuesto".
- Pestaña **Comparar modelos** para ver varios modelos lado a lado.
- Puedes pegar la API key en la propia GUI; se queda solo en memoria.

Sin menú:

```bash
python3 ormp.py coding                # también: rewrite, papers
python3 ormp.py papers --free         # incluye variantes :free (con límites de uso)
python3 ormp.py coding --max-cost 0.03 --max-out 5   # límites: $/tarea y $/M tokens de salida
python3 ormp.py coding --baseline deepseek/deepseek-v4.1-flash-1017 --per-month 3000   # vs. tu modelo, $/mes
python3 ormp.py --compare "claude,gpt-5,deepseek"
python3 ormp.py --csv modelos.csv
python3 ormp.py coding --refresh      # ignora la caché de 12 h
```

Sin API key funciona igual el catálogo (precios, contexto, comparación), pero no los
benchmarks, porque `/api/v1/benchmarks` exige autenticación.

## Cómo puntúa

- **Datos**: `/api/v1/models` (precio y contexto reales) y `/api/v1/benchmarks`
  (Artificial Analysis, Design Arena y las evals propias de OpenRouter: GPQA, tau-bench,
  búsqueda). Se hacen 4 peticiones y se cachean 12 h en `~/.cache/openrouter-model-picker`
  (el endpoint permite 500 peticiones/día).
- **Calidad (0-100)**: cada métrica se convierte en percentil entre los modelos que la
  tienen y se hace la media ponderada según la tarea:

  | Tarea | Métricas (peso) | Perfil de coste por tarea | Contexto mín. |
  |---|---|---|---|
  | Programar | Design Arena coding ELO 45%, AA Intelligence 35%, tau-bench 20% | 1 hora de sesión de agente: 4M entrada (90% cacheada) + 50k salida | 64k |
  | Reescribir | AA Intelligence 45%, AA Multilingual 40%, AA Legal 15% | 6k + 6k | 32k |
  | Papers | AA Intelligence 30%, GPQA Diamond 25%, Search evals 20%, AA Healthcare 15%, AA Engineering 10% | 80k + 2.5k | 128k |

- **Coste realista**: la entrada cacheada se cobra al precio de caché del modelo (los
  modelos sin caché pagan toda la entrada), y los modelos con precio por horario (⏱) usan
  la media de una semana de uso continuo.
- **Cobertura**: qué parte del peso tiene datos para ese modelo. Por defecto se exige
  ≥ 50% para que un modelo con una sola métrica no salga primero por casualidad.
- **Recomendación**: mejor calidad, y el modelo más barato que llega al 90% y al 75% de
  esa calidad.
- **Frontera calidad/precio (★)**: modelos que ningún otro supera a la vez en calidad y
  en precio. Ahí está siempre la respuesta a "el mejor y más barato".

Todo es ajustable desde el menú (opción 5): tokens por tarea, contexto mínimo,
cobertura, incluir gratis. Las variantes `:batch` no se puntúan (son API asíncrona).

**Limitaciones honestas**: no existe un benchmark directo de "reescribir documentos" ni
de "cotejar papers" en esta API; los pesos son una aproximación razonable, no una verdad.
Artificial Analysis retiró sus índices de Coding y Agentic, por eso programar se apoya en
Design Arena y tau-bench.

## Atribución

Los benchmarks de Artificial Analysis se muestran con su cita
("Source: Artificial Analysis (artificialanalysis.ai) via OpenRouter"), como pide la API.

## Tests

```bash
python3 -m unittest discover tests
```

## Licencia

MIT
