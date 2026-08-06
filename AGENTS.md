# AIUS Agent Notes

Research codebase for a study of pre-trained model (PTM) reuse in natural science
publications. `aius/` is the pipeline package; `figures/`, `scripts/`, and
`statistics/` are one-off analysis code. Any `*_old/` directory is archival — do
not fix, lint, or import from it.

## Source Of Truth

- Trust `pyproject.toml`, `Makefile`, `.pre-commit-config.yaml`, and `ruff.toml`
  over `README.md`.
- `ruff.toml` targets `py310` even though the project requires `~=3.13`. Ruff
  behavior follows `ruff.toml`.
- There is no CI (`.github/` does not exist). Pre-commit is the only gate.
- Work happens on the `dev` branch; `main` is the published branch.

## The `aius` Console Script Is Not Editable

`make build` installs a *tarball* into `.venv`, so `.venv/bin/aius` runs the
snapshot in `site-packages/aius/`, **not** your working tree. Editing `aius/**`
has no effect on the `aius` command until you re-run `make build`.

To exercise working-tree code directly:

```bash
.venv/bin/python -m aius.main <subcommand> ...
```

## Lint / Format Reality

- `ruff format` is clean repo-wide. Keep it that way.
- `ruff check` reports ~1500 pre-existing errors (mostly `D*`, `CPY001`, `ANN*`,
  `G004`). The `ruff-check` pre-commit hook fails on almost any file you touch.
  This is the baseline — do not attempt a repo-wide cleanup, and do not treat a
  failing `ruff-check` on an untouched rule as your regression.
- `ruff.toml` sets `fixable = []`, so `ruff check --fix` is a deliberate no-op.
  Fix findings by hand.
- `isort` and `ruff-format` disagree on import wrapping. Both hooks report
  "files were modified" on the same file, but the round trip is a net no-op
  **when run from the repo root** (isort only detects `aius` as first-party
  there). After a pre-commit run, check `git diff` before believing the failure.

Useful invocations:

```bash
pre-commit run --files <paths>          # scoped check
uvx ruff@0.15.4 check <path>            # matches the pinned hook version
```

## Tests

There are effectively none. Bare `pytest` **errors during collection** on
`scripts/_old/really_old/0-pilot-study/test_pilot_study.py`, which imports a
module that no longer exists. Use:

```bash
.venv/bin/python -m pytest --ignore=scripts/_old -q   # collects 0 tests
```

Verify changes by running the relevant CLI step against a scratch database
instead.

## CLI Wiring

`aius/main.py` -> `aius/cli/argparse.py` -> `aius/factory.py:runner_factory` ->
a `Runner` subclass under `aius/<step>/runner.py`.

The argparse layer is namespaced by convention, not by argparse groups: every
argument uses `dest="<subcommand>.<name>"`, and
`Argparse.identify_subcommand()` recovers the subcommand by splitting the *first*
parsed key on `.`. Consequences:

- A new argument **must** set `dest="<subcommand>.<name>"` or subcommand
  dispatch breaks.
- `runner_factory` reads those exact dotted keys, so adding an argument means
  editing `aius/factory.py` too.

Every invocation writes `aius_<unix_timestamp>.log` into the **current working
directory** (gitignored). Expect stray log files after running anything.

## Pipeline Gotchas

Order: `init` -> `search` -> `openalex` -> `jats` -> `pandoc` -> `analyze`.

- **`init` is not idempotent.** All writes use `if_exists="append"`, so a second
  `init` against the same database fails with
  `UNIQUE constraint failed: _llm_prompts._id`. Always start from a fresh DB.
- `jats` and later steps read the `natural_science_article_dois` view, which
  requires `cited_by_count > 0` and an OpenAlex field match. It is dropped and
  recreated on every `DB` construction, so it silently returns nothing until
  `openalex` has run.
- `analyze` chooses its **input table** from `--system-prompt-id`:
  - `uses_dl` <- `markdown`
  - `uses_ptms` <- `uses_dl_analysis`
  - `identify_ptms` <- `identify_ptms_analysis` joined back to `markdown`
    (reads its own output table; the prior stage must be loaded first)
  - `identify_ptm_reuse`, `identify_ptm_impact` <- `uses_ptms_analysis`

  Running a stage before its input table is populated yields an empty run, not
  an error.
- **`analyze` does not write to the database.** It writes
  `aius_<backend>_<prompt>_index-<i>_stride-<s>.parquet` into the cwd. Loading
  is a separate manual step:
  `python scripts/data_loading/load_parquet_2_db.py --parquet-dir ... --db-path ... --db-table ...`
- `--index` / `--stride` shard the document list via `islice` for parallel
  workers. See `scripts/loyola/dijkstra.bash` (GNU parallel) and
  `scripts/alcf/*.pbs` (PBS array jobs + Ollama) for real usage.
- Backend name mismatch: the CLI choice is `openai-batch`, but
  `OpenAIBatchBackend.name` is `openaibatch`, and `AnalysisRunner.execute`
  compares against `"openaibatch"` to skip the parquet write. Keep both spellings
  in sync if you rename.
- The `openai-batch` path uploads JSONL shards and drops `doc_map.txt` in the
  cwd; results come back through `scripts/data_loading/jsonl_2_parquet.py`.
- OpenAI helper scripts under `scripts/openai/` require `OPENAI_API_KEY`.

## Analysis Scripts

- Databases are gitignored (`*.db`, `data/*`) and distributed via Zenodo, not
  the repo. Nothing under `data/` is reproducible from a clean checkout.
- `figures/*.py` are Click CLIs whose `--db` default is a relative
  `../data/aius*.db`, so **run them from inside `figures/`** or pass `--db`.
- `statistics/` is inconsistent: only 5 of 17 scripts use Click. The rest
  hardcode a module-level `DB_PATH = Path("../data/aius_12-17-2025.db")` that you
  must edit before running.
- `figures/`, `scripts/`, and `statistics/` are not packages (no `__init__.py`)
  and are not shipped by the build; run them as scripts, never `python -m`.

## Dependencies

`pydantic` is imported by `aius/analyze/data_models.py` but is **not** declared
in `pyproject.toml` — it resolves only transitively through `openai`. Declare it
if you touch that area.
