# Five whys: why desktop-window spam kept returning (2026-08-17)

**Verdict:** window suppression existed at individual launch sites, but the installed-state default remained unsafe. On the audited workstation, 28 FAK/fleet tasks used an Interactive principal and 28 used S4U. The migration tool selected only tasks with historical result `0x800710E0` unless an operator remembered `-All`, while `fak windowgate --live-tasks` treated hidden Interactive tasks as advisory. A healthy recurring task could therefore keep reaching the desktop indefinitely.

## Five whys

1. **Why did windows keep appearing?** Enabled recurring Scheduled Tasks launched console-prone actions under an Interactive principal, which gives their process trees access to the logged-in desktop.
2. **Why did hidden flags not settle it?** `-WindowStyle Hidden` and `conhost --headless` suppress only the wrapper they decorate. Descendants can allocate another console, and later edits can remove the wrapper. S4U removes desktop access at the principal boundary.
3. **Why were Interactive tasks still installed after the July migration?** `tools/migrate_fleet_tasks_to_s4u.ps1` defaulted to tasks whose last result was `0x800710E0`; converting every Interactive task required the easy-to-miss `-All` opt-in. The live host had 28 Interactive tasks, many reporting success.
4. **Why did the guard not force that drift to closure?** Static `fak windowgate` checks source launch syntax, not installed Task Scheduler state. Its opt-in `--live-tasks` audit classified Interactive tasks with hidden wrappers as watchlist rows rather than violations.
5. **Why did fixes keep landing launcher by launcher?** The invariant was expressed as advice and local suppression, not as the default migration selection and hard installed-state policy. New and legacy task installers could be individually “hidden” while preserving the unsafe principal.

## Durable correction

- Migration now selects **all** matching Interactive tasks by default; narrow incident-only selection is explicitly named `-FailingOnly` and warns that it preserves desktop-visible drift.
- The live-task read-back already reports enabled console-prone Interactive tasks that lack a wrapper as violations; hidden wrappers remain defense in depth rather than the migration target.
- The existing safe installers already use S4U plus hidden wrapper flags; this change makes the shared repair path select the safe principal without an opt-in.

## Witness

- Pre-fix host census (2026-08-17): 56 matching recurring tasks, split 28 S4U / 28 Interactive.
- Runtime proof after rebuilding: `fak windowgate --live-tasks` must return `ACTION` until the elevated migration clears every enabled Interactive console-prone task.
- Repair command (requires elevated PowerShell): `tools\migrate_fleet_tasks_to_s4u.ps1 -Apply -VerifyRun`.

## Shift-left closure for new development

The migration command is recovery, not the policy boundary. New development is now stopped at the candidate-index boundary:

- `internal/windowgate.PSInstallerViolation` requires executable PowerShell task installers to declare an off-desktop principal (`S4U` or `SYSTEM`). Its sole interactive classification is the exact `FakHostRelaunchBroker` adapter, which must enter the current user's desktop to activate `wt.exe`; the classifier pins that script path, one `InteractiveToken` current-user/Limited principal, action, trigger, limits, and install read-backs.
- `conhost --headless`, `-WindowStyle Hidden`, and `pythonw.exe` remain useful defense in depth, but no longer excuse an Interactive or omitted principal.
- `internal/hooks.CheckDesktopPopup` invokes that rule from the staged pre-commit gate, so an unsafe installer cannot become a commit even when the developer never runs the migration or live audit.
- The tracked-tree witness cross-checks every committed `.ps1` installer, and ordinary Interactive tasks still fail even when they resemble the broker.

This makes the safe state a source-level default for a new developer: copy an existing installer and it is S4U; author an unsafe one and the commit gate names `INTERACTIVE_TASK_POPUP` with the fix.
