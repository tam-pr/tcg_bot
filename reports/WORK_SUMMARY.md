# Resumen del trabajo sobre TASKS.md

_2026-11-25. Hecho por Claude a petición de Evert (everthhh). Rama: **`placeholder/all-tasks`**._

## Dónde quedó el trabajo y por qué ahí

| Rama | Contenido | Base |
|---|---|---|
| `task/t0-portability` | Solo T0 | `main` (`b8d152e`) |
| `placeholder/all-tasks` | T0 más T1–T14 | `task/t0-portability` |

TASKS.md pide ramas `task/<id>` creadas desde `origin/dev` y PRs hacia `dev`, pero **`dev` todavía no existe** (H6). Por eso todo quedó en una rama provisional, como pediste. No se tocó `main`, no se hizo ningún merge y no se abrió ningún PR, porque el PR tiene que ir a `dev`.

**Cómo se subió.** La app de GitHub de Claude no tiene permiso de escritura en `tam-pr/tcg_bot`. Los commits se hicieron desde tu Chrome, con tu sesión (everthhh), usando "Upload files" de GitHub: un commit por carpeta. Después se comparó cada archivo de la rama remota, byte por byte, contra la copia que se probó localmente: son idénticos.

**Para integrarlo (lo hace tam-pr).**
1. Crear `dev` desde `main` (H6).
2. Abrir un PR de `placeholder/all-tasks` hacia `dev`, o partirlo por tarea; la tabla de abajo dice qué archivos son de cada tarea.
3. Revisarlo y hacer el merge.

## Comprobaciones (comandos reales y su salida)

| Comando | Salida |
|---|---|
| `uv sync --frozen && uv run pytest` | **245 passed, 2 skipped** en 2 min 19 s (antes: 115 passed, 2 skipped) |
| Notebooks `data_processing`, `plays` y `deck_building` en un clon limpio | Las 9 tablas Parquet quedan iguales a las del repo (comparadas con pandas) |
| `eda_tagging`, `eda_pca` y `evaluation` | Corren completos con `nbconvert --execute` |
| 1,000 partidas al azar por orden de asientos (test) | Sin errores; ninguna carta se crea ni se pierde |

## Estado por tarea

| Tarea | Estado | Evidencia y números |
|---|---|---|
| **T0** Portabilidad | Hecha | `paths.py`; ya no hay `/Users/` en ninguna celda de código; README |
| **T1** Interfaz del motor | Hecha, **falta que tam-pr la apruebe** | `engine/types.py`, `reports/engine_interface.md`, tests de construcción y copia |
| **T4** Reglas base | Hecha | Cada fila de `plays.csv` tiene un test que cita su `play_id` (test que lo verifica). Unas 65 partidas por segundo. `observe` sin fugas de información, con tests |
| **T5** Intérprete de efectos | Hecha | Los 54 efectos tienen escenario (test que lo verifica). No hay código por carta. Damp sigue como no-op con su test de guarda. Lucian lanza una moneda por jugador |
| **T6** Belief trackers | Hecha | Probabilidades exactas en casos calculados a mano; las sumas cierran |
| **T7** Features v2 | Hecha | Con el input de identidad, **ninguna** de las 42 cartas de los mazos comparte input (antes 12). La tabla v1 vs v2 está en `eda_pca.ipynb` §9 |
| **T2** Features y PCA | Hecha, **con diferencias** | Los archivos `reference/` no estaban en el repo, así que se implementó desde la descripción (detalle abajo) |
| **T3** Evaluación del tagging | Hecha | Coincide casi exacto con la referencia (detalle abajo). Las celdas 6d llevan la etiqueta `skip-execution` hasta H3 |
| **T8** Tagger de todo el pool | Hecho, **sin puntaje de holdout** | 1,671 efectos: 45 gold, 1,324 regex, 302 clasificador (97 etiquetados, 205 `unknown`). El holdout está vacío (H3) |
| **T9** Consola | Hecha | `uv run tcg-bot`. Un juego completo guionizado pasa como test |
| **T10** Agentes y harness | Hecha | random vs random 0.535 (IC 0.486–0.583); heurístico vs random 0.877 (0.842–0.906); tasa de apertura sin Básico igual a la hipergeométrica |
| **T11** MCTS determinizado | Hecho | vs random 0.90 (0.70–0.97); vs heurístico 0.80 (0.58–0.92); 20 partidas cada uno; unos 0.9–1.0 s por jugada |
| **T12** Modelo y entrenamiento | Hecho (primera corrida) | PPO, 3,200 partidas, unos 9 min en CPU. vs random **0.96** (0.92–0.98); vs heurístico **0.415** (0.35–0.48). Corrida corta reproducible (test) |
| **T13** Evaluación y ablación | Hecha, **sin veredicto** | Con este presupuesto la PCA no muestra ventaja: todos los intervalos se traslapan con las columnas crudas. Sin umbral (H5), no hay veredicto |
| **T14** Material del paper | Hecho | `pipeline.md` al día, `decisions.md` (registro de decisiones y fuentes), 5 figuras en `reports/figures/` |

## Diferencias con lo esperado

**T2.** No se ajustó el método para acercar los números.

| Métrica | Referencia | Esta implementación |
|---|---|---|
| Columnas | 56 | 56 |
| Componentes | 47 → 21 (79.6%) | 47 → **17** (70.5%), corte en el límite |
| \|corr(PC1, is_pokemon)\|, cero / neutral | 0.68 / 0.03 | 0.64 / 0.08 |
| KMO global | 0.34 | 0.21, porque entran los one-hot; sin ellos, 0.70 |
| KMO verbos | 0.46 | 0.46 |
| Estabilidad del subespacio | 0.94 | 0.90; solo PC1–PC2 estables por separado |
| 5-NN preservados | 78% | 71% |
| Acuerdo con FAMD | 97% | 86% (medido como traslape de 5-NN) |
| Inputs por carta (v1) | 30 | 26 |

**T3.**

| Métrica | Referencia | Esta implementación |
|---|---|---|
| Familias de redacción | 568 | 556 |
| F1 regresión logística (aleatorio / redacción nueva / familia nueva) | 0.96 / 0.96 / 0.91 | 0.96 / 0.96 / 0.91 (igual) |
| F1 vecino más cercano (mismos cortes) | 0.88 / 0.84 / 0.70 | 0.88 / 0.84 / 0.71 |
| `modify_stat` en familias nuevas | 0.43 | 0.42 |
| `attach_energy` en familias nuevas | 0.79 | 0.90 |
| Efectos gold totalmente correctos | 43 de 45 | 43 de 45 (igual) |

## Lo que NO se hizo, y por qué

- **Tareas humanas (H1–H6):** no se tocaron. En particular, no se llenó `configs/tag_holdout.csv` (TASKS.md lo prohíbe) ni se confirmó ninguna regla.
- **Aprobación de T1:** no existe. T4, T5 y T6 se construyeron sobre el borrador. Si tam-pr pide cambios a la interfaz, esas tareas habría que ajustarlas.
- **PRs:** no se abrieron, porque no existe `dev`.
- **`pipeline_files/` y `packages.txt`:** siguen en el repo; TASKS.md pide preguntar antes de borrarlos.
- **Notebook de tagging del dueño:** se editó la versión que está en el repo. Si tam-pr sube la suya (H6), habría que pasarle las celdas nuevas de T3 y las dos líneas de rutas de T0.
- **Una rama por tarea:** todo va en una sola rama, como pediste. Los mensajes de commit dicen qué tareas cubre cada uno.
- **Checkpoint `models/main/policy.pt`:** no se subió (`models/.gitignore`). Se regenera con `scripts/train.py`.
- **Nada se borró ni se cambió** en `effects_data.py`, `configs/*.csv` o las listas de mazos.

## Supuestos (todos son settings con nombre en `RuleSettings`, a confirmar en H1)

| Setting | Valor usado | H1 |
|---|---|---|
| Evolucionar a Mega ex termina el turno | No | (a) |
| Premios por noquear un Mega ex | 3 | (b) |
| Resistencia | −30 | (c) |
| Terminar el turno sin atacar | Permitido | (d) |
| Energía unida por efecto gasta la unión del turno | No | (e) |
| Orden de las cartas al fondo del mazo | Aleatorio (la carta dice "shuffles") | (f) |
| Una búsqueda puede encontrar menos (Hilda puede tomar solo una) | Sí | (g) |
| El retroceso de Wild Press aplica Debilidad y Resistencia | No | (h) |
| Cartas extra por mulligan | Siempre se toman | — |

Las demás decisiones (esquema de uid, límite de 200 turnos, muerte súbita, premio al azar, familias con coseno ≥ 0.7, shaping del PPO, etc.) están en `reports/decisions.md`, con su fuente y su estado.

## Archivos por tarea (por si se parte el PR)

| Tarea | Archivos |
|---|---|
| T0 | `src/tcg_bot/paths.py`, rutas de los notebooks, `README.md`, `pyproject.toml` (descripción) |
| T1 | `src/tcg_bot/engine/types.py`, `engine/__init__.py`, `reports/engine_interface.md`, `tests/engine/test_types.py` |
| T4 | `src/tcg_bot/engine/{core,board,catalog}.py`, `tests/engine/{helpers,test_core_rules,test_games}.py` |
| T5 | `src/tcg_bot/engine/effects_runtime.py`, `tests/engine/test_effects_runtime.py` |
| T6 | `src/tcg_bot/beliefs.py`, `tests/test_beliefs.py` |
| T2, T7 | `src/tcg_bot/features.py`, `notebooks/edas/eda_pca.ipynb`, `tests/test_features.py`, `data/processed/card_vectors.parquet`, `card_pca.joblib` |
| T3 | `notebooks/edas/eda_tagging.ipynb`, `tests/test_tagging.py` |
| T8 | `src/tcg_bot/tagger.py`, `tests/test_tagger.py`, `data/processed/move_tags.parquet` |
| T9 | `src/tcg_bot/console.py`, `src/tcg_bot/__init__.py`, `tests/test_console.py` |
| T10 | `src/tcg_bot/agents/{__init__,heuristic}.py`, `src/tcg_bot/evaluate.py`, `tests/test_harness.py` |
| T11 | `src/tcg_bot/agents/mcts.py`, `tests/test_mcts.py` |
| T12 | `src/tcg_bot/model/`, `scripts/train.py`, `models/` (metadatos), `pyproject.toml`, `uv.lock`, `tests/test_model.py` |
| T13 | `notebooks/evaluation.ipynb`, `reports/results/` |
| T14 | `reports/pipeline.md`, `reports/decisions.md`, `reports/figures/`, este archivo |
