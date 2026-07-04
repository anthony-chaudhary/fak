# Guard TUI Capture

Source fixture: `visuals/guard-tui-live.jsonl`.

The checked-in media show the real guard console pane rendered from a guard
decision journal. The recording was built by appending the fixture rows one at a
time, running the console command after each append, and encoding the captured
terminal frames.

Render the final terminal state:

```powershell
go run ./cmd/fak console guard --journal visuals/guard-tui-live.jsonl --at 2026-07-04T14:30:00Z --width 130 --color never
```

Checked-in outputs:

- `visuals/guard-tui-screenshot.png`
- `visuals/guard-tui-video-poster.png`
- `visuals/guard-tui-video.gif`
- `visuals/guard-tui-video.mp4`

Public-scrub note: the fixture is payload-free and uses only generic tool names,
verdicts, reasons, and placeholder hash-chain values. It contains no local user
path, account tag, credential, hostname, or server-specific detail.
