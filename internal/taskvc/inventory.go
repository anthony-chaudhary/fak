package taskvc

// Inventory records version-control coverage for enabled fleet Scheduled Tasks (#3323).
// Scope covers root-path tasks driving fleet loops or scripts in this repository.

// Inventory returns a copy of declared coverage for all enabled fleet Scheduled Tasks.
func Inventory() []Coverage {
	return append([]Coverage(nil), inventory...)
}

var inventory = []Coverage{
	// Tasks rebuildable from versioned installers.
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
	// Realigned installer default for FleetStaleWorkGarden (#3323).
	{Task: "FleetStaleWorkGarden", Status: StatusInstaller, Installer: "tools/register_stale_work_watchdog.ps1"},
	{Task: "FleetSupervisorWatchdog", Status: StatusInstaller, Installer: "tools/register_supervisor_watchdog.ps1"},
	// FleetWorktreeDoctor bound to literal installer default (#5409).
	{Task: "FleetWorktreeDoctor", Status: StatusInstaller, Installer: "tools/register_worktree_doctor.ps1"},

	// Tasks covered by scrubbed task XML exports under tools/scheduled-tasks/.
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
		Reason: "runs tools/scheduled-tasks/fak-bench-fleet-tick.cmd, which IS tracked since #6503 " +
			"promoted the generated %TEMP% payload into the tree, so a restore rebuilds both the " +
			"schedule and the script. The capture pins the workspace and fak binary as %FAK_WORKSPACE% " +
			"/%FAK_BIN%, which must be substituted on restore, and stays disabled until one real node " +
			"yields a witnessed numeric benchmark — `fak bench-loop install` refuses to arm it before that.",
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
