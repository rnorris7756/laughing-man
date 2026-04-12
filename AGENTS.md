# Agent and contributor notes

## Cursor cloud and headless environments

This is a **Python CLI** app (webcam face overlay) managed with **uv** — no Docker, databases, or external services required for development.

### Git branches and pull requests

- **Do not push directly to `master`.** The default branch is protected; CI and the release flow expect work to land via pull request.
- **Use a topic branch** from current `master`, for example `feature/…`, `fix/…`, `docs/…`, or `ci/…`.
- **Open a PR**, let CI pass, then merge (or follow the maintainer’s merge process).

### Quick reference

| Action | Command |
| --- | --- |
| Install / sync deps | `uv sync --group dev` |
| Install git hooks (once per clone) | With **direnv**, `.envrc` runs `pre-commit install` when the hook is missing. Otherwise: `uv run pre-commit install` |
| Lint + typecheck (same as CI) | `uv run pre-commit run --all-files` |
| Lint (Ruff) | `uv run ruff check src tests scripts` |
| Typecheck (ty) | `uv run ty check` |
| Run tests | `uv run pytest` (add `-v` for verbose) |
| CLI help | `uv run laughing-man --help` |
| Run the app | `uv run laughing-man` (needs a webcam) |

### Environment caveats

- **Webcam at runtime.** The main command opens a camera (often `/dev/video0` on Linux). In a **headless cloud VM** without a camera device, run **unit tests** and **import checks** only; the live overlay loop fails when opening the camera.
- **Model download.** BlazeFace (`.tflite`) and YuNet (`.onnx`) may be fetched to `~/.cache/laughing-man/` on first use — **network access** may be required once.
- **`tool.uv.link-mode = "copy"`** in `pyproject.toml` avoids broken OpenCV wheels when the uv cache and `.venv` sit on different filesystems. Do not change this without a good reason.
- **Python version.** `.python-version` pins **3.13** for local/CI convenience; `requires-python` in `pyproject.toml` remains **>=3.10** for compatibility.
- **direnv.** If you use [direnv](https://direnv.net/), loading the repo runs `.envrc`: it creates `.venv` with `uv sync --group dev` on first use and installs the **pre-commit** Git hook when it is missing.

## Conventional commits

Use [Conventional Commits](https://www.conventionalcommits.org/) for commit messages so history stays readable and tooling (changelog generators, semantic versioning) can work if adopted later.

**Format:** `type(scope): short description`

Common **types:** `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`.

**Guidelines:**

- Keep the subject line in imperative mood (e.g. “add ruff config”, not “added”).
- One logical change per commit when practical (config vs. code fixes vs. CI).
- Use a body for non-obvious rationale or breaking changes.

**Examples:**

- `feat(cli): add virtual camera option`
- `fix(overlay): correct alpha compositing on custom images`
- `chore(dev): add ruff and ty for lint and type checking`

## Planning work and commits

When planning a change set, **sketch the commits in advance**: what each commit will contain and its conventional message. That reduces the risk of **partial-file staging** (`git add -p` or staging only some hunks) mixing unrelated edits in the same file, which is easy to get wrong and produces confusing history.

**Practical habits:**

- **Order work** so one logical concern is finished before another touches the same files, when you can.
- If one file must hold multiple concerns, **finish and commit the first concern entirely** (or split into a separate file in a first commit), then apply the second change set.
- Use **interactive staging** only when you are deliberately splitting a file; double-check `git diff --staged` before committing.

## Linting and type checking

This project uses **Ruff** for linting/formatting and **ty** (Astral) for static type checking. **ty** is chosen for speed, alignment with Ruff/uv in the Astral ecosystem, and solid diagnostics; it is configured in `pyproject.toml` under `[tool.ty]`.

**Pre-commit** runs the same checks as CI: one hook runs `uv run --locked bash -ec 'ruff … && ty …'`, so Ruff and ty versions always come from `uv.lock` (same as GitHub Actions). After `uv sync --group dev`, run `uv run pre-commit install` once per clone unless **direnv** already installed the hook via `.envrc`; commits then run lint and typecheck automatically. [uv](https://github.com/astral-sh/uv) must be on your `PATH` when Git invokes the hooks. On Windows, use **Git Bash** (or another environment where `bash` is available), which matches typical `pre-commit` setups.

From the repo root (with dev dependencies installed, e.g. `uv sync --group dev`):

```bash
uv run pre-commit run --all-files
```

Individual tools (equivalent to the hooks):

```bash
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run ty check
```

Auto-fix import/order issues and apply formatting:

```bash
uv run ruff check --fix src tests scripts
uv run ruff format src tests scripts
```

Run tests:

```bash
uv run pytest
```
