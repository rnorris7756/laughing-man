# AGENTS.md

## Cursor Cloud specific instructions

This is a Python CLI application (webcam face overlay) managed entirely by **uv** — no Docker, no external services, no databases.

### Quick reference

| Action | Command |
|---|---|
| Install / sync deps | `uv sync` |
| Run tests | `uv run pytest -v` |
| CLI help | `uv run laughing-man --help` |
| Run the app | `uv run laughing-man` (requires a webcam) |

### Important caveats

- **Webcam required at runtime.** The main `laughing-man` command captures from `/dev/video0`. In a headless Cloud VM without a webcam, you can only run unit tests and verify imports — the live overlay loop will fail with a camera-open error.
- **No lint tool is configured** in `pyproject.toml` or the repo. There is no `ruff`, `flake8`, or `mypy` config to run.
- **Python 3.13** is specified in `.python-version`. `uv` will auto-install it if missing when you run `uv sync`.
- **`tool.uv.link-mode = "copy"`** is set in `pyproject.toml` to work around cross-device hardlink issues with OpenCV. Do not change this.
- On first run, face-detection models (BlazeFace `.tflite`, YuNet `.onnx`) are auto-downloaded to `~/.cache/laughing-man/` — this requires internet access.
