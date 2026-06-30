// Package windowgate is the NO-DESKTOP-POPUP ratchet: the durable gate that keeps
// always-on fleet automation from flashing console windows on the interactive
// desktop — the "random terminal popups".
//
// # Three ways a scheduled tick pops a window, all guarded here
//
// 1. TASK LEVEL (PowerShell installers). A Scheduled Task that runs in the user's
// INTERACTIVE session (-LogonType Interactive, schtasks ... /IT, or the
// Register-ScheduledTask default principal) renders every console window — the
// task's own and every child it spawns — ON THE DESKTOP. -WindowStyle Hidden does
// NOT suppress the conhost flash (the maintainers learned this the hard way; see
// tools/register_worktree_doctor.ps1 / tools/register_runaway_reaper.ps1). The fix
// every repo installer already uses is an OFF-DESKTOP principal — S4U / a service
// account — OR a fully headless launcher
// (conhost.exe --headless ..., the FleetResumeWatchdog pattern). ScanTree FAILS any
// task-creating .ps1 that is neither.
//
// 2. CODE LEVEL (Python helper completeness). A windowless parent (pythonw, or any
// task whose own window is hidden) that spawns a CONSOLE child (git, gh, taskkill,
// tasklist, cmd) WITHOUT creationflags gets a BRAND-NEW visible console for that
// child — capture_output / PIPE redirect the streams but NOT the window; only
// CREATE_NO_WINDOW suppresses it. The dispatch family routes every spawn through the
// shared suppressor dispatch_worker.no_window_creationflags(). The original bug was
// PARTIAL coverage: the worker Popen passed the flag but the taskkill / mklink / git
// helper calls did not. ScanTree FAILS any module that opts into the suppressor
// (references no_window_creationflags) yet leaves a subprocess spawn without a
// creationflags hint (a **kwargs splat counts as provided; a POSIX-only tool such as
// pgrep is exempt since it can never run on Windows).
//
// 3. CODE LEVEL (Go dispatch helpers). The native `fak dispatch ...` path shells out
// to `gh`, `git`, `dos`, `powershell`, and Python helpers before it launches a worker.
// Those short helper commands need the same Windows no-window hook as the worker
// spawn, otherwise moving a tick from Python to Go reopens the desktop-flash bug.
// ScanTree FAILS cmd/fak/dispatch*.go helper execs that reach Output/CombinedOutput/
// Run/Start before configureDispatchHelperCommand(cmd) or configureDispatchSpawn(cmd).
//
// # Why Go, scanning Python and PowerShell
//
// The de-Python ratchet (internal/pythongate) bans NEW tools/*.py, so this gate — the
// checking layer over the fleet's launch hygiene — lives in Go and analyzes the
// tracked .ps1 / .py / dispatch .go text directly. It scans the git-tracked tree so
// untracked peer scratch files never affect the verdict.
package windowgate
