package envconfiglint

// admittedPostFreeze is the hand-maintained, explicitly-reasoned list of non-secret env
// reads that landed on the trunk AFTER baseline.go was frozen, while this ratchet was
// red and therefore not actually gating anything.
//
// It is deliberately a SEPARATE file from baseline.go. baseline.go is generated ("DO NOT
// EDIT by hand") and re-running its recipe would silently absorb any post-freeze arrival,
// turning a re-admission into an invisible one. Splitting them keeps every re-admission a
// recorded decision with a name attached — the internal/ctxknobs rule that a frozen
// baseline "may only be EXTENDED with an explicit, reviewed reason, never to re-admit
// [something] a cleaner default could retire."
//
// This list may only SHRINK. Each entry is behavioral configuration that genuinely belongs
// on a config surface; none can move there yet because that surface does not exist — it is
// #2862's deliverable. When #2862 lands, each read relocates and its line is deleted here.
// That makes this list the ratchet's own debt ledger: an empty admittedPostFreeze means the
// env/config boundary holds with no exceptions outstanding.
//
// Tracked for relocation by the follow-up issue filed alongside #2863.
//
// SECOND OCCURRENCE (#6215) — why this ledger tripled. Everything below the original nine
// entries landed during the ratchet's SECOND unwatched red: the gate went red again after the
// first cleanup and stayed red for 1050 trunk advances (FAK_FLEET_BUS, the oldest), so sixteen
// further non-secret reads shipped against a gate that refused nothing. They are admitted here
// rather than relocated because the destination surface (#2862) still does not exist, and
// because the alternative — regenerating baseline.go — would absorb all sixteen silently and
// destroy the ratchet, the exact failure this file's header warns about. The age itself was
// the defect, not the reads, so liveness.go now measures it: TestRatchetIsStillGating reds
// when an offense survives more than one trunk advance, which is the condition every entry
// below was created under and which nothing previously observed.
//
// SCANNER VISIBILITY — why four entries left WITHOUT relocating. ScanGoEnvReads is a regex
// over os.Getenv/os.LookupEnv taking a STRING-LITERAL argument, so it only sees a read whose
// name is spelled at the call site. 5b56bfdb8 deduped four such call sites into helpers that
// take the name as a parameter — resolveChatopsEnv(name) in cmd/fak/chatops.go and
// resolveBudgetMS(name, fallback) in cmd/fak/hooks.go — which made FAK_CHATOPS_BOT_USER,
// FAK_CHATOPS_CHANNEL, FAK_PRECOMMIT_CHECK_BUDGET_MS and FAK_PRECOMMIT_TOTAL_BUDGET_MS
// invisible to the scan. Those four are still READ at runtime; they did NOT move to a config
// surface, so their #2862 debt is unpaid. TestAdmittedPostFreezeStaysHonest reds on any entry
// the scan cannot corroborate, so the lines are deleted below — but record what that costs:
// the reads are now UNGATED, and nothing stops a new env read from being added behind either
// helper. Re-add the entry, and prefer a literal call site, if a helper is ever inlined.
var admittedPostFreeze = []string{
	// cmd/fak/dispatch_tick_lease_beat.go — overrides os.Hostname() as the identity stamped on
	// the DOS lane leases the dispatch tick beats, so two workers on one box can hold distinct
	// lanes. A machine label, not a credential: it authenticates nothing, and its siblings
	// DISPATCH_LANE / DISPATCH_POOL / DISPATCH_ACCOUNT are already grandfathered.
	// Relocates to: a `--host-id` flag on `fak dispatch tick`, threaded into
	// newDispatchLaneBeater instead of re-read from the process env at construction.
	"DISPATCH_HOST_ID",

	// cmd/fak/guard_negframe_summary.go — the comma-separated ablation-token list that turns
	// named features OFF for an A/B arm; the negframe token here selects #3546's control arm
	// by disabling the emit-time reframe pass (default-on treatment otherwise). An experiment
	// arm selector, not a credential — it authenticates nothing and gates no access.
	// Relocates to: a persistent `--ablate <token,...>` flag on the fak root command, sharing
	// the token vocabulary `fak ablate` already owns via internal/ablate.
	"FAK_ABLATE",

	// internal/archreport/usage.go — filesystem path override for the durable per-user
	// architecture-usage ledger, with the literal value "off" disabling recording entirely. A
	// location on disk plus a kill switch; it grants access to nothing the process could not
	// already reach, which is the FAK_FLEET_DIR judgment applied unchanged.
	// Relocates to: a path argument to UsagePath (a Config field the caller sets), the same
	// shape FAK_SERVICE_LEDGER_DIR takes below.
	"FAK_ARCHITECTURE_USAGE_FILE",

	// cmd/fak/chatops.go — the admin roster allowed to drive the door. An identity list, not a
	// credential (the chatops TOKEN is separate and secret-shaped). Its two siblings
	// FAK_CHATOPS_BOT_USER and FAK_CHATOPS_CHANNEL are absent because the scan can no longer
	// see them, not because they relocated — see SCANNER VISIBILITY above. This one stays: it
	// is still spelled literally, at resolveChatopsAdmins.
	"FAK_CHATOPS_ADMINS",

	// internal/launchshim/launchshim.go — the launch shim's bypass switch (1/true/yes/on):
	// when on, EffectiveDirect runs the real binary and skips the shim. A behavior toggle, not
	// an authorization boundary — the same judgment FAK_SHAREDTASK gets below.
	// Relocates to: nothing new is needed. EffectiveDirect ALREADY takes both a `flag bool`
	// argument and a Config.Disabled field; the env arm is a third, redundant input that can
	// simply be deleted once its callers set one of the two that are passed IN.
	"FAK_DIRECT",

	// cmd/fak/fleet_control.go — directory override for the fleet control bus, consulted ahead
	// of the grandfathered FLEET_STATE_DIR sibling it otherwise derives from. A path. This is
	// the OLDEST outstanding entry (1050 trunk advances at admission) and therefore the read
	// that dates the second unwatched red; see the SECOND OCCURRENCE note above.
	// Relocates to: a `--bus-dir` flag on `fak fleet`, beside `--registry-dir`.
	"FAK_FLEET_BUS",

	// cmd/fak/garden.go — two filesystem paths for the garden tick's growth collect:
	// FAK_FLEET_DIR overrides the %LOCALAPPDATA%/Fleet root it censuses beyond the repo, and
	// FAK_GARDEN_GROWTH_LEDGER overrides where it appends its would-reap/reaped JSONL soak
	// ledger. Both are locations on disk, not credentials — neither grants access to anything
	// the process could not already reach.
	// Relocate to: `--fleet-dir` and `--growth-ledger` flags on `fak garden`, the same shape
	// FAK_SERVICE_LEDGER_DIR takes below.
	"FAK_FLEET_DIR",
	"FAK_GARDEN_GROWTH_LEDGER",

	// The launch shim's three filesystem locations — cmd/fak/launch.go launchBinDir() for the
	// directory holding installed shim binaries, internal/launchshim/launchshim.go Path() for
	// the shim's config file, and internal/launchshim/stats.go StatsPath() for its counter file
	// (which defaults beside Path()). Three paths on disk, no credentials.
	// Relocate to: a `--bin-dir` flag on `fak launch`, and an explicit path argument to
	// launchshim.Path/StatsPath set by the launch command — so the shim is TOLD where it lives
	// once rather than each helper re-deriving it from the process env.
	"FAK_LAUNCH_BIN",
	"FAK_LAUNCH_CONFIG",
	"FAK_LAUNCH_STATS",

	// cmd/fak/learning_observation.go — filesystem path for the learning-observation JSON
	// store, ahead of its %APPDATA%/fak default. A path.
	// Relocates to: a `--store` flag on `fak learning-observation`, which already parses flags
	// per subcommand.
	"FAK_LEARNING_OBSERVATION_STORE",

	// internal/headroom/lingua.go — the LLMLingua compression sidecar's two settings: the
	// endpoint URL, and the target compression ratio (0 < r <= 1, default 0.5). An address and
	// a tuning number; neither authenticates anything. Note LinguaCompressor already carries a
	// URL FIELD and the env read is only its fallback, so half the relocation exists already.
	// Relocate to: a TargetRatio field on LinguaCompressor beside URL, and then delete
	// LinguaURL() so an unset URL is the caller's error rather than a silent env lookup.
	"FAK_LINGUA_TARGET_RATIO",
	"FAK_LINGUA_URL",

	// cmd/fak/mcp_filter_proof_live.go — the model id for the live MCP-filter proof run, read
	// in liveProofDefaults() beside OPENAI_API_KEY. The key is a declared secret and stays in
	// the environment; a model SELECTOR is a setting and does not.
	// Relocates to: a `--model` flag on the proof subcommand, defaulting in code.
	"FAK_MCP_FILTER_PROOF_MODEL",

	// internal/gateway/observer.go — filesystem path for the observer journal.
	"FAK_OBSERVER_JOURNAL",

	// cmd/fak/hooks.go — FAK_PRECOMMIT_CHECK_BUDGET_MS and FAK_PRECOMMIT_TOTAL_BUDGET_MS, the
	// two wall-clock caps that keep the #5335 pre-commit wedge from stalling the commit lane,
	// were listed here until 5b56bfdb8 routed both through resolveBudgetMS(name, fallback).
	// They are still read; the scan just cannot corroborate them. Their relocation target is
	// unchanged: `--check-budget-ms` and `--total-budget-ms` flags on `fak hooks pre-commit`.

	// cmd/fak/service.go — env default backing the `fak service status --ledger-dir` flag.
	"FAK_SERVICE_LEDGER_DIR",

	// internal/sessionregistry/registry.go — path for the child-registration JSONL, ahead of
	// its %APPDATA%/fak default. A path; its sibling FAK_SESSION_LEDGER_DIR is grandfathered.
	// Relocates to: Store.Path, which ALREADY EXISTS as a field — the env read only backs the
	// package-level DefaultPath(), so relocation is "make the caller name the store".
	"FAK_SESSION_REGISTRY",

	// cmd/fak/sharedtask_endpoint.go — the request-time on/off switch (1/true/yes/on) for the
	// shared-task co-editing subtree /v1/fak/sharedtask/ under `fak serve`; default off keeps
	// the endpoint inert for every subcommand. A feature toggle, not an authorization boundary:
	// what actually guards the surface is the caller's reader scope and sharedtask.Policy's
	// MaxScope, both of which still apply once it is on.
	// Relocates to: a `--sharedtask` opt-in flag on `fak serve`.
	"FAK_SHAREDTASK",

	// cmd/fak/guard_sessionstart.go — set to "off" to suppress the independent-tool width hint
	// appended to the SessionStart additionalContext. A presentation toggle on injected prose;
	// it gates no access and changes no policy.
	// Relocates to: a `--tool-width-hint` flag on `fak guard session-start`, beside the
	// `--managed` flag whose branch already guards this read.
	"FAK_TOOL_WIDTH_HINT",

	// internal/gateway/messages_stream_warmcontinue.go — arms the #3353 warm-continue resume
	// path, which replays a mid-stream-died turn's already-delivered text as an assistant
	// prefill instead of failing the turn; OFF by default so only a deploy or dogfood session
	// opts in. A retry-strategy toggle, not a credential.
	// Relocates to: a field on the gateway passthrough's config struct, set by the serve
	// command, so the arming decision is passed IN rather than re-read from the process env.
	"FAK_WARM_CONTINUE",

	// internal/fleetaccounts/freshprobe.go — the freshness window in MINUTES for an
	// entitlement-class (auth/access/credit) probe verdict before it ages out and the fold
	// falls back to the registry's own status; default 1440 (24h), deliberately far longer
	// than the usage-class window (FLEET_PROBE_FRESH_MIN, grandfathered, default 20) because
	// aging out an entitlement block INVERTS it — a dead seat reads as full headroom and the
	// allocator then prefers it. A duration, not a credential.
	// Relocates to: an exported package default plus a Config field alongside
	// ProbeLedgerFreshMin's, so both freshness windows move to the config surface together.
	"FLEET_PROBE_ENTITLEMENT_FRESH_MIN",

	// cmd/fak-dev/windows_setup_plan.go — the Go module-cache root, read (with a $HOME/go
	// fallback) only to NAME a directory in the Windows Defender exclusion plan. This one is
	// admitted on a different ground from every entry above: it is the Go toolchain's OWN
	// ambient variable, the same class as PATH, HOME and LOCALAPPDATA — all grandfathered, and
	// LOCALAPPDATA is read five lines above it in the same function. fak does not own it and no
	// fak config surface should claim it.
	// Relocates to: `go env GOPATH`, which asks the toolchain for its real answer instead of
	// re-implementing the toolchain's own default. That is a correctness upgrade, not just a
	// config move — the env var being empty does not mean the toolchain uses $HOME/go.
	"GOPATH",

	// cmd/fak/mcp_filter_proof_live.go — provider endpoint override for the live proof run,
	// read in liveProofDefaults() beside OPENAI_API_KEY. Exact sibling of the grandfathered
	// ANTHROPIC_BASE_URL, and judged the same way: an address is not a credential, and the key
	// that IS one sits next to it and stays in the environment.
	// Relocates to: whenever ANTHROPIC_BASE_URL does — the provider endpoints belong on one
	// config surface together, or neither moves.
	"OPENAI_BASE_URL",

	// Third unwatched-red cleanup (#7211): these reads predate the live liveness failure and
	// cannot remain unclassified while #2862 builds the destination config surface. Each is
	// admitted by its actual role rather than absorbed into the generated baseline.
	// FAK_ROOT_REGISTRATION_ID is lineage correlation metadata, not authentication material.
	"FAK_ROOT_REGISTRATION_ID",
	// ComSpec and SystemDrive are host-owned Windows discovery inputs.
	"ComSpec",
	"SystemDrive",
	// FAK_MICRO_TASK and FAK_LEASE_ID carry explicit worker-launch context.
	"FAK_MICRO_TASK",
	"FAK_LEASE_ID",
	// FAK_PROVIDER_ACCOUNT_IDENTITY selects an account-scoped benchmark observation.
	"FAK_PROVIDER_ACCOUNT_IDENTITY",
	// FAK_GUARD_REFUSAL_STATE_DIR and the tool-control pair override local state paths/mode.
	"FAK_GUARD_REFUSAL_STATE_DIR",
	"FAK_TOOLCALL_CONTROL_DIR",
	"FAK_TOOLCALL_CONTROL_MODE",
	// FAK_CLAUDE_SPEED selects the declared dispatch speed profile.
	"FAK_CLAUDE_SPEED",
	// FAK_WORK_EFFECT_CALIBRATION_JSON and FAK_EP_COORDINATED_DECODE are explicit
	// experimental runtime overrides pending typed gateway/serve config in #2862.
	"FAK_WORK_EFFECT_CALIBRATION_JSON",
	"FAK_EP_COORDINATED_DECODE",
	// FAK_DEV_EXE and FLEET_CODEX_EXE are executable discovery overrides for recovery.
	"FAK_DEV_EXE",
	"FLEET_CODEX_EXE",

	// FAK_CA_BUNDLE is resolved through secretload as the declared trust-source config;
	// the default loader may source that key from the process environment.
	// Relocates to: typed host trust configuration passed to httptrust.ResolveWith.
	"FAK_CA_BUNDLE",

	// FAK_CHILD_REGISTRY carries the private guard-to-child registry path across a process
	// boundary until child launch accepts the path as an explicit argument.
	"FAK_CHILD_REGISTRY",

	// FAK_STREAM_Q4K selects the experimental bounded Q4_K loader on CUDA hosts.
	// Relocates to: a typed fak serve setting passed into the GGUF load helpers.
	"FAK_STREAM_Q4K",

	// internal/goalregistry/goalregistry.go — selects the durable goal-registry JSON file.
	// The CLI already exposes --registry; the native harness still consumes DefaultPath(),
	// so removing the ambient override now would split the two operator views.
	// Relocates to: an explicit registry path passed into every goalregistry.Store caller.
	"FAK_GOAL_REGISTRY",

	// internal/servicewatchdog/systemd.go — systemd injects these three variables as the
	// inherited sd_notify/watchdog process protocol. They are not fak behavioral settings:
	// NOTIFY_SOCKET is the manager-owned datagram endpoint, WATCHDOG_USEC is the negotiated
	// heartbeat interval, and WATCHDOG_PID scopes that lease to this process. Copying them to a
	// fak flag/config file would sever the manager handshake and create a second authority.
	// Owner: internal/servicewatchdog. Sunset: remove together if fak retires sd_notify support
	// or systemd replaces this inherited protocol; review whenever that protocol integration changes.
	"NOTIFY_SOCKET",
	"WATCHDOG_PID",
	"WATCHDOG_USEC",

	// cmd/fak/release_status_ci.go shares this workflow filename contract with
	// tools/release_context.py and tools/release_decide.py. Keeping the same inherited override
	// in all three consumers preserves atomic release selection until the Python release pipeline
	// itself gains a typed config input. Owner: release pipeline. Sunset: remove from all three
	// consumers together when release context/decision accepts an explicit workflow argument.
	"FAK_RELEASE_FAST_CI_WORKFLOW",

	// cmd/fak/serve_load_helpers.go — opts into the experimental bounded Metal Q4_K loader.
	// This changes model-loading strategy rather than carrying a credential.
	// Relocates to: a typed fak serve setting passed into the GGUF load helpers.
	"FAK_METAL_STREAM_Q4K",

	// tools/videogen/trailer/main.go — the ffmpeg executable path, used as the DEFAULT VALUE of
	// the tool's own `-ffmpeg` flag. A path to a binary; the flag that overrides it is already
	// the config surface.
	// Relocates to: the trailer JSON the tool loads via `-config`, so the renderer's executable
	// is declared beside the render settings it belongs with.
	"VIDEOGEN_FFMPEG",

	// cmd/fak/launchguard.go — directory where launchguard state is stored.
	// Relocates to: a config setting or flag on launchguard initialization.
	"FAK_LAUNCHGUARD_DIR",

	// internal/selfupdate/cmd/selfupdate.go — update installer type: native or msix.
	// Relocates to: a config setting or flag on selfupdate.
	"FAK_SELF_UPDATE_INSTALLER",

	// cmd/fak/guard_prompt_transport.go — workspace directory for guarded codex prompt fuel.
	// Relocates to: launch plan workspace or flag.
	"DISPATCH_WORKSPACE",
}
