# Industry Scorecard Freshness Cadence

Recurring freshness check for the industry scorecard — keeps stale SOTA bars from silently regrowing.

## What this does

Runs the industry scorecard freshness checks on a weekly cadence:
- `--stale`: Lists SOTA bars past the 365-day review window
- `--verify-sources`: Verifies fak numbers still match their committed artifacts
- Full scorecard run for grade/coverage tracking

Output is captured to `docs/industry-scorecard/cadence-output/freshness-cadence-YYYY-MM-DD.jsonl` so you can audit the history of runs.

## Install the scheduled task

The schedule is defined in `tools/loop-registry.json` (weekly, catch-up policy, 1-hour jitter).

### macOS (launchd)

Create `~/Library/LaunchAgents/com.fak.industry-freshness-cadence.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.fak.industry-freshness-cadence</string>
    <key>ProgramArguments</key>
    <array>
        <string>python</string>
        <string>tools/industry_freshness_cadence.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/path/to/fak/repo</string>
    <key>StartInterval</key>
    <integer>604800</integer>
    <key>StandardOutPath</key>
    <string><tmp>/fak-industry-freshness-cadence.log</string>
    <key>StandardErrorPath</key>
    <string><tmp>/fak-industry-freshness-cadence.err</string>
</dict>
</plist>
```

Then:
```bash
# Install
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.fak.industry-freshness-cadence.plist

# Verify
launchctl list | grep industry-freshness
```

### Linux (systemd)

Create `/etc/systemd/system/fak-industry-freshness-cadence.service`:

```ini
[Unit]
Description=fak industry scorecard freshness cadence

[Service]
Type=oneshot
WorkingDirectory=/path/to/fak/repo
ExecStart=python tools/industry_freshness_cadence.py
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Create `/etc/systemd/system/fak-industry-freshness-cadence.timer`:

```ini
[Unit]
Description=fak industry scorecard freshness cadence weekly timer

[Timer]
OnCalendar=weekly
RandomizedDelaySec=3600
Persistent=true

[Install]
WantedBy=timers.target
```

Then:
```bash
sudo systemctl daemon-reload
sudo systemctl enable fak-industry-freshness-cadence.timer
sudo systemctl start fak-industry-freshness-cadence.timer

# Verify
sudo systemctl status fak-industry-freshness-cadence.timer
```

### Windows (Task Scheduler)

Use Task Scheduler to create a weekly task:
- Trigger: Weekly, every 7 days, with a 1-hour random delay
- Action: Start a program
  - Program: `python`
  - Arguments: `tools\industry_freshness_cadence.py`
  - Start in: `C:\path\to\fak\repo`

## Manual run

To run the freshness check manually without the schedule:

```bash
python tools/industry_freshness_cadence.py
```

## Review captured output

Each run writes a JSONL entry to `docs/industry-scorecard/cadence-output/freshness-cadence-YYYY-MM-DD.jsonl`:

```json
{
  "timestamp": "2026-06-27T12:34:56.789Z",
  "run_type": "freshness-cadence",
  "checks": [
    {"name": "stale", "output": "...\n"},
    {"name": "verify-sources", "output": "...\n"},
    {"name": "scorecard", "output": "...\n"}
  ]
}
```

## Handling stale bars

When `--stale` reports bars past the window:

1. Re-check the bar on the web (new papers, vendor docs, leaderboards)
2. If the bar **changed**: update `sota_bar`, `sota_systems`, `notes`, and set both `source_date` AND `last_reviewed` to the re-check date
3. If the bar **is still accurate**: update ONLY `last_reviewed` to the re-check date (keep `source_date` unchanged)

This is the "re-confirmed" case that prevents silent regrowth — the bar stays current without fabricating a new source date.

See `docs/industry-scorecard/UPDATE-PROCESS.md` for the full process.