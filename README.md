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
python3 ormp.py                       # menú interactivo
```

Sin menú:

```bash
python3 ormp.py coding                # también: rewrite, papers
python3 ormp.py papers --free         # incluye variantes :free (con límites de uso)
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
  | Programar | Design Arena coding ELO 45%, AA Intelligence 35%, tau-bench 20% | 20k entrada + 3k salida | 64k |
  | Reescribir | AA Intelligence 45%, AA Multilingual 40%, AA Legal 15% | 6k + 6k | 32k |
  | Papers | AA Intelligence 30%, GPQA Diamond 25%, Search evals 20%, AA Healthcare 15%, AA Engineering 10% | 80k + 2.5k | 128k |

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
