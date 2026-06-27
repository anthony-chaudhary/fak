# Industry Scorecard Freshness Cadence

Recurring freshness check for the industry scorecard — keeps stale SOTA bars from silently regrowing.

## What this does

Runs the industry scorecard freshness checks on a weekly cadence:
- `--stale`: Lists SOTA bars past the 365-day review window
- `--verify-sources`: Verifies fak numbers still match their committed artifacts
- Full scorecard run for grade/coverage tracking

Output is captured to `docs/industry-scorecard/cadence-output/freshness-cadence-YYYY-MM-DD.jsonl` so you can audit the history of runs.

## Install the scheduled task

The schedule is registered in `.dos/loop-registry.json` (weekly, catch-up policy, 1-hour jitter).

### macOS (launchd)

```bash
# Emit the .plist
./fak cron emit --registry .dos/loop-registry.json industry-freshness-cadence \
  --target launchd --fak-bin ./fak \
  -- python tools/industry_freshness_cadence.py > ~/Library/LaunchAgents/com.fak.industry-freshness-cadence.plist

# Install
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.fak.industry-freshness-cadence.plist

# Verify
launchctl list | grep industry-freshness
```

### Linux (systemd)

```bash
# Emit the unit
./fak cron emit --registry .dos/loop-registry.json industry-freshness-cadence \
  --target systemd --fak-bin ./fak \
  -- python tools/industry_freshness_cadence.py > /tmp/fak-industry-freshness-cadence.service

# Install (copy to /etc/systemd/system/ or ~/.config/systemd/user/)
sudo cp /tmp/fak-industry-freshness-cadence.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable fak-industry-freshness-cadence.service
sudo systemctl start fak-industry-freshness-cadence.service

# Verify
sudo systemctl status fak-industry-freshness-cadence.service
```

### Windows (Task Scheduler)

```powershell
# Emit the XML
.\fak.exe cron emit --registry .dos\loop-registry.json industry-freshness-cadence `
  --target taskscheduler --fak-bin .\fak.exe `
  -- python tools\industry_freshness_cadence.py > $env:TEMP\fak-industry-freshness-cadence.xml

# Import into Task Scheduler (GUI or PowerShell)
# GUI: Open Task Scheduler -> Import Task -> select the XML file
# PowerShell (optional):
#   Register-ScheduledTask -Xml (Get-Content $env:TEMP\fak-industry-freshness-cadence.xml | Out-String) `
#     -TaskName "fak-industry-freshness-cadence"
```

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