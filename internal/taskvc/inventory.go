package taskvc

// This file is the version-controlled record of the fleet's always-on Scheduled
// Task posture — the artifact that lets a reimaged host be rebuilt from the repo,
// and the ratchet that keeps installer names bound to the loops they register
// (#3323).
//
// Scope matches the reboot-survival audit: ENABLED tasks at the ROOT task path (\)
// that belong to THIS repo's fleet. A task is in scope when its action runs a
// script tracked in this repo, or when it drives this repo's fleet loops (the
// Fak*/Fleet* families and the campaign one-shots they spawn). The criterion is
// deliberately by ACTION, not by name prefix: ClaudeAccountBackup runs
// tools/claude_account_backup.py from this very tree and would be silently missed
// by a Fak*/Fleet* name filter, which is exactly the orphan class #3323 exists to
// catch.
//
// Deliberately excluded, and NOT gaps:
//   - Other automation trees that merely share the root task path: DOS-cleanup-sweep
//     (C:\work\dos-kernel-public), HostMaintenance-* (C:\work\host-maintenance),
//     JobSearchApplyProfileFleet (C:\work\job) and the \JobSearch\ tasks
//     (FleetHeartbeat, FleetPipelineTick). Each is reproducible from its own repo,
//     not this one.
//   - "Claude state cleanup" — the Claude harness's own housekeeping of
//     ~/.claude, owned by the harness install, not by this repo.
//   - \Microsoft\Windows\... vendor and OS tasks (UserTask, OneDrive*, AMD*, ...) —
//     OS-owned; some only match the fleet's name filter by coincidence.
//   - Disabled tasks (FakFleetJanitor, FleetSlackBeat, the FleetIssueDispatch*
//     campaign variants, ...) — they are not part of the live posture, and
//     several are deliberately-parked one-shots.
//
// Measured live on 2026-07-26 (`Get-ScheduledTask`, root path, State != Disabled:
// 48 tasks, of which 29 are in scope by the rule above): 29 covered and 0 orphans,
// split 16 rebuildable from a versioned installer and 13 covered by a scrubbed
// task-XML export under tools/scheduled-tasks/. The #5409 reconcile has since
// promoted FleetWorktreeDoctor from the capture tier to its installer, so the split
// now reads 17 / 12 against the same 29 tasks; its export stays tracked but no row
// leans on it. The 2026-07-08 audit counted ~35 orphans; most were closed by
// installers landed in the intervening weeks, and the residual 12 are captured here.
//
// The two coverage tiers are NOT equivalent, and the difference is the honest part.
// An installer rebuilds the loop from the repo outright. An XML export versions the
// task — schedule, principal shape, exact command line — but not the script that
// command line points at. For a task whose action lives outside the tree, restoring
// the XML restores a task pointing at a file a fresh host does not have; the Reason
// on each StatusXML row names exactly what is still missing, and an installer
// remains the preferred remedy.
//
// Adding a loop to the fleet? Add its row here. A StatusInstaller row is checked
// against the tree by TestFleetTaskInventoryBindsToVersionedInstallers, so a
// coverage claim can never outrun the installer that backs it; a StatusXML row is
// checked against the tracked capture, so it cannot outrun the file either.

// Inventory returns the declared coverage of every enabled fak-fleet Scheduled
// Task. It is a value copy, so callers cannot mutate the shared record.
func Inventory() []Coverage {
	return append([]Coverage(nil), inventory...)
}

var inventory = []Coverage{
	// ---- Rebuildable from a versioned installer -----------------------------
	// Each of these installers declares its task name as a literal default, so
	// re-running it updates the live task in place rather than duplicating it.
	{Task: "FakLogvaultCapture", Status: StatusInstaller, Installer: "tools/register_logvault_backup.ps1"},
	{Task: "FakLogvaultVerify", Status: StatusInstaller, Installer: "tools/register_logvault_backup.ps1"},
	{Task: "FakSelfUpdate", Status: StatusInstaller, Installer: "tools/install_self_update_schedule.ps1"},
	{Task: "FleetBenchPlanDoc", Status: StatusInstaller, Installer: "tools/register_bench_plan_doc.ps1"},
	{Task: "FleetDOSDispatchWatchdog", Status: StatusInstaller, Installer: "tools/register_dos_dispatch_watchdog.ps1"},
	{Task: "FleetDispatchStatusDoc", Status: StatusInstaller, Installer: "tools/register_dispatch_status_doc.ps1"},
	{Task: "FleetIdeaScout", Status: StatusInstaller, Installer: "tools/register_idea_scout.ps1"},
	{Task: "FleetIssueDispatch", Status: StatusInstaller, Installer: "tools/register_issue_dispatch.ps1"},
	{Task: "FleetProcResourceGuard", Status: StatusInstaller, Installer: "tools/register_proc_resource_guard.ps1"},
	{Task: "FleetPushLagPusher", Status: StatusInstaller, Installer: "tools/register_push_lag_pusher.ps1"},
	{Task: "FleetResolveProgress", Status: StatusInstaller, Installer: "tools/register_resolve_progress.ps1"},
	{Task: "FleetResumeWatchdog", Status: StatusInstaller, Installer: "tools/register_resume_watchdog.ps1"},
	{Task: "FleetRunawayReaper", Status: StatusInstaller, Installer: "tools/register_runaway_reaper.ps1"},
	{Task: "FleetScoutLoop", Status: StatusInstaller, Installer: "tools/register_scout_loop.ps1"},
	// The #3323 reconcile: this installer's -TaskName default was realigned to the
	// live task, so a reinstall updates FleetStaleWorkGarden instead of spawning a
	// second stale-work loop. This row is what keeps it from drifting again.
	{Task: "FleetStaleWorkGarden", Status: StatusInstaller, Installer: "tools/register_stale_work_watchdog.ps1"},
	{Task: "FleetSupervisorWatchdog", Status: StatusInstaller, Installer: "tools/register_supervisor_watchdog.ps1"},
	// The same reconcile, one class later (#5409). register_worktree_doctor.ps1 used
	// to interpolate "FleetWorktreeDoctor-$repoSlug", which no static parse can bind,
	// so this row sat at StatusDrift leaning on a task-XML capture as a stopgap. Its
	// -TaskName default is now the bare literal that the live task and the versioned
	// capture (tools/scheduled-tasks/FleetWorktreeDoctor.xml, whose <URI> reads
	// \FleetWorktreeDoctor) already agreed on, so a reinstall updates the live task in
	// place and the binder can prove it. The capture stays tracked as a historical
	// export, but it is no longer this row's coverage claim — the installer is.
	{Task: "FleetWorktreeDoctor", Status: StatusInstaller, Installer: "tools/register_worktree_doctor.ps1"},

	// ---- Captured: scrubbed task XML under tools/scheduled-tasks/ ------------
	// Each Reason names what the XML alone does NOT restore, because that residual
	// is the difference between this tier and a real installer.
	{
		Task:    "ClaudeAccountBackup",
		Status:  StatusXML,
		Capture: "tools/scheduled-tasks/ClaudeAccountBackup.xml",
		Reason: "runs tools/claude_account_backup.py, which IS tracked here — the strongest promotion " +
			"candidate in this tier, and the row a Fak*/Fleet* name filter would have missed entirely. " +
			"The captured action pins the absolute clone path (C:\\work\\fak) and a specific python.exe, " +
			"so a restore into a differently-located clone must edit both; an installer resolving the " +
			"script relative to $PSScriptRoot (as register_stale_work_watchdog.ps1 does) would fix that.",
	},
	{
		Task:    "July4CacheValueAutospawn",
		Status:  StatusXML,
		Capture: "tools/scheduled-tasks/July4CacheValueAutospawn.xml",
		Reason: "launches %USERPROFILE%/fak-armed/july4_cachevalue_autospawn.ps1, outside the repo, so " +
			"the capture versions a pointer to a file the repo does not carry. A dated cache-value " +
			"campaign one-shot (Interactive principal, 5 h repetition window), not durable boot " +
			"posture; retire the row with the campaign.",
	},
	{
		Task:    "FakFleetJanitorHeadless",
		Status:  StatusXML,
		Capture: "tools/scheduled-tasks/FakFleetJanitorHeadless.xml",
		Reason: "runs scripts/gcp-fleet-janitor.sh, which IS tracked — the one row a straightforward " +
			"register_gcp_fleet_janitor.ps1 would promote to a full installer. The capture redacts " +
			"the GCP project id its action pins (%GCP_PROJECT%), so that must be substituted on restore.",
	},
	{
		Task:    "FakBenchmarkFleetLoop",
		Status:  StatusXML,
		Capture: "tools/scheduled-tasks/FakBenchmarkFleetLoop.xml",
		Reason: "launches a generated %TEMP%/fak-bench-fleet-tick.cmd; the tick script is ephemeral " +
			"and never existed in version control, so a restore rebuilds the schedule but not the " +
			"script — the loop needs a tracked entrypoint before an installer is possible.",
	},
	{
		Task:    "FakMetaSuperloopNight100",
		Status:  StatusXML,
		Capture: "tools/scheduled-tasks/FakMetaSuperloopNight100.xml",
		Reason: "launches _scratch/run-meta-superloop-100.ps1 (gitignored), so the capture versions a " +
			"pointer to a file the repo does not carry. A parked overnight campaign one-shot, not " +
			"durable boot posture; retire the row with the campaign.",
	},
	{
		Task:    "FakOvernightMixedProfiles100",
		Status:  StatusXML,
		Capture: "tools/scheduled-tasks/FakOvernightMixedProfiles100.xml",
		Reason: "launches _scratch/overnight-mixed-profiles.ps1 (gitignored) with campaign-specific " +
			"-Baseline/-Target arguments; the capture preserves those arguments but not the script. " +
			"A one-shot campaign loop, not durable boot posture.",
	},
	{
		Task:    "FleetGLM52CampaignStop",
		Status:  StatusXML,
		Capture: "tools/scheduled-tasks/FleetGLM52CampaignStop.xml",
		Reason: "an inline one-shot kill switch that disables a named set of campaign dispatch tasks. " +
			"There is no script to version, so the capture IS the whole loop — this row is fully " +
			"restorable, and retires with the campaign it stops.",
	},
	{
		Task:    "FakReapOrphanTails",
		Status:  StatusXML,
		Capture: "tools/scheduled-tasks/FakReapOrphanTails.xml",
		Reason: "launches %LOCALAPPDATA%/fak-reaper/reap-orphan-tails.ps1, outside the repo — a restore " +
			"recreates the task pointing at a file a reimage does not have. Promoting this to an " +
			"installer means moving the script into tools/ first.",
	},
	{
		Task:    "FleetOwnerSeatResume",
		Status:  StatusXML,
		Capture: "tools/scheduled-tasks/FleetOwnerSeatResume.xml",
		Reason: "launches %LOCALAPPDATA%/Fleet/resume_on_owner_after_reset.ps1, outside the repo. Same " +
			"residual as FakReapOrphanTails: move the script into tools/, then add an installer.",
	},
	{
		Task:    "FleetStrandedRecovery",
		Status:  StatusXML,
		Capture: "tools/scheduled-tasks/FleetStrandedRecovery.xml",
		Reason: "launches %LOCALAPPDATA%/Fleet/stranded_recovery.ps1, outside the repo. Same residual " +
			"as FakReapOrphanTails.",
	},
	{
		Task:    "FleetWatchdogWatchdogAudit",
		Status:  StatusXML,
		Capture: "tools/scheduled-tasks/FleetWatchdogWatchdogAudit.xml",
		Reason: "launches %LOCALAPPDATA%/fak-watchdog-audit/run-watchdog-audit.ps1, outside the repo. " +
			"Same residual as FakReapOrphanTails.",
	},
	{
		Task:    "UserSeatDrain-1010",
		Status:  StatusXML,
		Capture: "tools/scheduled-tasks/UserSeatDrain-1010.xml",
		Reason: "launches tools/_watchdog/operator_user_seat_drain.py, which is gitignored " +
			"(.gitignore: tools/_watchdog/), so the capture points at an untracked script. Promoting " +
			"it means tracking that script — and porting it off Python, since a new tools/*.py trips " +
			"the pythongate ratchet.",
	},
}
