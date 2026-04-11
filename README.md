# laughing-man

Webcam **Laughing Man** face overlay (Ghost in the Shell style) using MediaPipe BlazeFace, OpenCV, and Pillow.

## Setup

Install dependencies (e.g. with [uv](https://github.com/astral-sh/uv)):

```bash
uv sync
```

Run the CLI:

```bash
uv run laughing-man
```

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

Package names vary by distro (e.g. `v4l2loopback-dkms` on Debian/Ubuntu).

### Other platforms

On Windows and macOS, [pyvirtualcam](https://github.com/letmaik/pyvirtualcam) may use OBS Virtual Camera or other backends when available.
