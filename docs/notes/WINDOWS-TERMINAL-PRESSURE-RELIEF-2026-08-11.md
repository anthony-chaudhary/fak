# Windows Terminal pressure relief

`fak terminal-relief` is the reversible actuator paired with `fak stallscan`'s
10,000-handle / 500-thread leak signal. It requires three consecutive pressured
observations and a one-hour cooldown by default. Dry-run is the default; `--apply`
is required to act.

The actuator fails closed when a recognized stateful editor/document process is a terminal
descendant, and unless at least one `fak info` dashboard is restorable.
On apply it starts each dashboard as a detached process first, then replaces only
the leaking Windows Terminal process tree. Processes not known to carry unsaved document state are reported in the JSON witness but do not block replacement.

The durable monitor invokes it after the existing background-daemon mitigation:

```powershell
pwsh -File tools/fak_stall_monitor.ps1 -AutoMitigate
```

Read back the decision and state:

```powershell
fak terminal-relief --json
Get-Content "$env:LOCALAPPDATA\Fleet\terminal-relief.json"
fak stallscan --json
```

To disable automatic relief, run the monitor without `-AutoMitigate`. To reset
only the consecutive-run/cooldown memory, stop the monitor and remove
`%LOCALAPPDATA%\Fleet\terminal-relief.json`; this does not touch a terminal or a
session.
