# laughing-man

Webcam **Laughing Man** face overlay (Ghost in the Shell style) using MediaPipe BlazeFace, OpenCV, and Pillow.

## Install

### From PyPI

Stable releases are published on [PyPI](https://pypi.org/project/laughing-man/):

```bash
pip install laughing-man
```

With [uv](https://github.com/astral-sh/uv) you can use:

```bash
uv pip install laughing-man
```

After installing, run `laughing-man` or `laughing-man --help` (the `laughing-man` console script is on your `PATH`).

### From source (development)

You need [uv](https://github.com/astral-sh/uv) on your `PATH`. Clone the repository:

```bash
git clone https://github.com/rnorris7756/laughing-man.git
cd laughing-man
```

#### Using direnv (recommended)

[direnv](https://direnv.net/) loads the repo’s `.envrc` automatically when you `cd` into the project (after you **allow** it once per machine). That keeps the development environment consistent without manually activating a venv each time:

1. Install **direnv** and [hook it into your shell](https://direnv.net/docs/hook.html) (Bash, Zsh, Fish, etc.).
2. In the repo root, run **`direnv allow`** so `.envrc` is trusted.
3. The next time the environment loads, `.envrc` will:
   - create **`.venv`** and run **`uv sync --group dev`** if the venv does not exist yet (installs runtime + dev dependencies from **`uv.lock`**);
   - **`source`** that venv whenever you are inside the directory;
   - run **`uv run pre-commit install`** if the Git **pre-commit** hook is missing (so Ruff and ty match CI before each commit).

After that, use `laughing-man` or `uv run laughing-man` as usual; commands see the activated environment while you stay in the tree.

#### Without direnv

Install dependencies and the pre-commit hook yourself:

```bash
uv sync --group dev
uv run pre-commit install
uv run laughing-man
```

Run `uv run pre-commit install` once per clone so Ruff and ty run before each commit (same checks as CI). Git invokes hooks with **`uv`** on your `PATH`.

## Virtual webcam (Discord, OBS, browsers)

Use `--virtual-cam` to expose the composited video as a camera other apps can select.

### Linux: v4l2loopback

The virtual device is provided by the **v4l2loopback** kernel module. Load it before starting the app (often required once per boot):

```bash
sudo modprobe v4l2loopback
```

You can create a named device on a fixed number (example):

```bash
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="LaughingMan"
```

Then point the CLI at that device if needed:

```bash
laughing-man --virtual-cam --v4l2-device /dev/video10
```

Use `--no-preview` to stream only to the virtual camera (no OpenCV window); stop with Ctrl+C.

If the module loads at boot and grabs `/dev/video0`, the real webcam may move to `/dev/video1`. The CLI defaults to `--camera auto`, which on Linux picks the first `/dev/video*` that is **not** a v4l2loopback output (opening loopback for capture can crash OpenCV). You can also pin the real device with `--camera /dev/video1` or load loopback on a fixed high number: `sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="LaughingMan"`.

Package names vary by distro (e.g. `v4l2loopback-dkms` on Debian/Ubuntu).

### Other platforms

On Windows and macOS, [pyvirtualcam](https://github.com/letmaik/pyvirtualcam) may use OBS Virtual Camera or other backends when available.
