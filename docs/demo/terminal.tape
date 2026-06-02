Output docs/demo/build/terminal.mp4

Require uv

Set Shell "bash"
Set FontSize 24
Set Width 1280
Set Height 720
Set Padding 32
Set Framerate 30
Set TypingSpeed 45ms
Set Theme "Dracula"
Set WindowBar Colorful
Set CursorBlink false

Type "uv run laughing-man --help"
Enter
Sleep 500ms
Wait /Usage:/
Sleep 2s

Type "uv run laughing-man --face-backend yunet --camera auto --virtual-cam --v4l2-device /dev/video10 --no-preview"
Sleep 1800ms
