# README demo recording harness

This directory contains the source files for a short README demo that starts with terminal usage and then cuts to a local recording of the face overlay.

## What other GitHub projects commonly do

- Script terminal demos with [VHS](https://github.com/charmbracelet/vhs) so the terminal portion is repeatable from a committed `.tape` file.
- Assemble the terminal segment with a short product clip using `ffmpeg`.
- Keep generated media out of git while iterating, then add either:
  - a small GIF for inline autoplay; or
  - a poster image that links to a GitHub-hosted MP4 uploaded through the README/PR editor.
- Keep README media short and compressed. Large GIFs and MP4s slow down clones and page loads.

## Prerequisites

Install project dependencies first:

```bash
uv sync --group dev
```

Install local recording tools:

- [VHS](https://github.com/charmbracelet/vhs) for `docs/demo/terminal.tape`
- `ffmpeg` for capture, assembly, poster extraction, and GIF generation
- Linux only for the automated overlay capture command below: `v4l2loopback`

Check the harness can find the external tools:

```bash
uv run python scripts/record_readme_demo.py check
```

## 1. Render the terminal segment

```bash
uv run python scripts/record_readme_demo.py terminal
```

This writes:

```text
docs/demo/build/terminal.mp4
```

## 2. Record the overlay segment

On Linux, create a virtual camera device once per boot:

```bash
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="LaughingMan"
```

Then record a short overlay clip:

```bash
uv run python scripts/record_readme_demo.py overlay --virtual-device /dev/video10 --duration 8
```

This starts `laughing-man`, waits briefly for the model/camera pipeline to warm up, records `/dev/video10` with `ffmpeg`, and writes:

```text
docs/demo/build/overlay.mp4
```

Useful variants:

```bash
# Show the OpenCV preview window while recording.
uv run python scripts/record_readme_demo.py overlay --preview

# Pin a physical camera if auto-selection picks the wrong device.
uv run python scripts/record_readme_demo.py overlay --camera /dev/video1

# Use YuNet for a lighter-weight demo capture.
uv run python scripts/record_readme_demo.py overlay --face-backend yunet

# If ffmpeg cannot infer the v4l2 frame size.
uv run python scripts/record_readme_demo.py overlay --video-size 1280x720
```

On macOS or Windows, record the live overlay with OBS, QuickTime, or your preferred recorder, save it as `docs/demo/build/overlay.mp4`, and continue with the assembly step.

## 3. Assemble the final demo

```bash
uv run python scripts/record_readme_demo.py assemble
uv run python scripts/record_readme_demo.py poster
uv run python scripts/record_readme_demo.py gif
```

Expected outputs:

```text
docs/demo/build/laughing-man-demo.mp4
docs/demo/build/laughing-man-demo-poster.jpg
docs/demo/build/laughing-man-demo.gif
```

## README insertion options

For a small inline animation:

```markdown
![Laughing Man terminal and overlay demo](docs/demo/build/laughing-man-demo.gif)
```

For a clickable MP4 thumbnail after uploading the MP4 to GitHub-hosted assets:

```markdown
[![Laughing Man terminal and overlay demo](docs/demo/build/laughing-man-demo-poster.jpg)](https://github.com/user-attachments/assets/REPLACE-ME)
```

Before merging the README demo, replace `REPLACE-ME` with the GitHub asset URL or commit a suitably small GIF/poster asset in the chosen documentation path.
