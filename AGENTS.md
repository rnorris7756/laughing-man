# Agent and contributor notes

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

## Linting and type checking

This project uses **Ruff** for linting/formatting and **ty** (Astral) for static type checking. **ty** is chosen for speed, alignment with Ruff/uv in the Astral ecosystem, and solid diagnostics; it is configured in `pyproject.toml` under `[tool.ty]`.

From the repo root (with dev dependencies installed, e.g. `uv sync --group dev`):

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
