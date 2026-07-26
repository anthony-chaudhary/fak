package taskvc

// This file is the version-controlled record of the fleet's always-on Scheduled
// Task posture — the artifact that lets a reimaged host be rebuilt from the repo,
// and the ratchet that keeps installer names bound to the loops they register
// (#3323).
//
// Scope matches the reboot-survival audit: ENABLED fak-fleet tasks registered at
// the ROOT task path (\). Deliberately excluded, and NOT gaps:
//   - \JobSearch\ tasks (FleetHeartbeat, FleetPipelineTick) — a separate,
//     out-of-scope automation tree.
//   - \Microsoft\Windows\... vendor tasks (UserTask, UserTask-Roam) — OS-owned;
//     they only match the fleet's name filter by coincidence.
//   - Disabled tasks (FakFleetJanitor, FleetSlackBeat, the FleetIssueDispatch*
//     campaign variants, ...) — they are not part of the live posture, and
//     several are deliberately-parked one-shots.
//
// Measured live on 2026-07-26 (`Get-ScheduledTask`, root path, State != Disabled):
// 27 tasks — 16 rebuildable from a versioned installer, 1 name-drifted, 10 orphans.
// The 2026-07-08 audit counted ~35 orphans; most were closed by installers landed
// in the intervening weeks, and this inventory is the first thing that keeps the
// remainder from silently re-growing.
//
// Adding a loop to the fleet? Add its row here. A StatusInstaller row is checked
// against the tree by TestFleetTaskInventoryBindsToVersionedInstallers, so a
// coverage claim can never outrun the installer that backs it.

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

	// ---- Name drift: installer exists, but cannot bind the live task ---------
	{
		Task:   "FleetWorktreeDoctor",
		Status: StatusDrift,
		Reason: "tools/register_worktree_doctor.ps1 defaults -TaskName to the interpolated " +
			`"FleetWorktreeDoctor-$repoSlug", so it cannot be proven to update the bare live ` +
			"FleetWorktreeDoctor in place; a reinstall registers FleetWorktreeDoctor-fak alongside it. " +
			"Same class as the FleetStaleWorkGarden mismatch (#3323); reconcile needs an operator " +
			"decision on whether the live task or the per-clone naming scheme is canonical.",
	},

	// ---- Orphans: no installer in version control ---------------------------
	// Split by WHY, because the remedy differs. Exporting task XML would capture
	// the schedule but not the script it launches, so for the out-of-tree rows an
	// XML capture alone would restore a task pointing at a missing file.
	{
		Task:   "FakFleetJanitorHeadless",
		Status: StatusOrphan,
		Reason: "runs scripts/gcp-fleet-janitor.sh, which IS tracked — this is the one orphan a " +
			"straightforward register_gcp_fleet_janitor.ps1 would fully close. Its action also " +
			"pins a GCP project id that must stay out of the public tree (parameterize it).",
	},
	{
		Task:   "FakBenchmarkFleetLoop",
		Status: StatusOrphan,
		Reason: "launches a generated %TEMP%/fak-bench-fleet-tick.cmd; the tick script is ephemeral " +
			"and never existed in version control, so there is nothing to capture until the bench " +
			"loop grows a tracked entrypoint.",
	},
	{
		Task:   "FakMetaSuperloopNight100",
		Status: StatusOrphan,
		Reason: "launches _scratch/run-meta-superloop-100.ps1 (gitignored). A parked overnight " +
			"campaign one-shot, not durable boot posture; capturing it would version a pointer " +
			"to a gitignored file.",
	},
	{
		Task:   "FakOvernightMixedProfiles100",
		Status: StatusOrphan,
		Reason: "launches _scratch/overnight-mixed-profiles.ps1 (gitignored) with campaign-specific " +
			"-Baseline/-Target arguments; a one-shot campaign loop, not durable boot posture.",
	},
	{
		Task:   "FleetGLM52CampaignStop",
		Status: StatusOrphan,
		Reason: "an inline one-shot kill switch that disables a named set of campaign dispatch " +
			"tasks. No script to version; it retires with the campaign it stops.",
	},
	{
		Task:   "FakReapOrphanTails",
		Status: StatusOrphan,
		Reason: "launches %LOCALAPPDATA%/fak-reaper/reap-orphan-tails.ps1, which lives outside the " +
			"repo. Closing this means moving the script into tools/ first; an XML capture alone " +
			"would restore a task pointing at a file a reimage does not recreate.",
	},
	{
		Task:   "FleetOwnerSeatResume",
		Status: StatusOrphan,
		Reason: "launches %LOCALAPPDATA%/Fleet/resume_on_owner_after_reset.ps1, outside the repo. " +
			"Same remedy as FakReapOrphanTails: move the script into tools/, then add an installer.",
	},
	{
		Task:   "FleetStrandedRecovery",
		Status: StatusOrphan,
		Reason: "launches %LOCALAPPDATA%/Fleet/stranded_recovery.ps1, outside the repo. Same remedy " +
			"as FakReapOrphanTails.",
	},
	{
		Task:   "FleetWatchdogWatchdogAudit",
		Status: StatusOrphan,
		Reason: "launches %LOCALAPPDATA%/fak-watchdog-audit/run-watchdog-audit.ps1, outside the repo. " +
			"Same remedy as FakReapOrphanTails.",
	},
	{
		Task:   "UserSeatDrain-1010",
		Status: StatusOrphan,
		Reason: "launches tools/_watchdog/operator_user_seat_drain.py, which is gitignored " +
			"(.gitignore: tools/_watchdog/). Closing this means tracking the script — and porting " +
			"it off Python, since a new tools/*.py trips the pythongate ratchet.",
	},
}
