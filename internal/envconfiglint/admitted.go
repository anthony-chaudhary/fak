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
	// cmd/fak/guard_negframe_summary.go — the comma-separated ablation-token list that turns
	// named features OFF for an A/B arm; the negframe token here selects #3546's control arm
	// by disabling the emit-time reframe pass (default-on treatment otherwise). An experiment
	// arm selector, not a credential — it authenticates nothing and gates no access.
	// Relocates to: a persistent `--ablate <token,...>` flag on the fak root command, sharing
	// the token vocabulary `fak ablate` already owns via internal/ablate.
	"FAK_ABLATE",

	// cmd/fak/chatops.go — the admin roster allowed to drive the door. An identity list, not a
	// credential (the chatops TOKEN is separate and secret-shaped). Its two siblings
	// FAK_CHATOPS_BOT_USER and FAK_CHATOPS_CHANNEL are absent because the scan can no longer
	// see them, not because they relocated — see SCANNER VISIBILITY above. This one stays: it
	// is still spelled literally, at resolveChatopsAdmins.
	"FAK_CHATOPS_ADMINS",

	// cmd/fak/garden.go — two filesystem paths for the garden tick's growth collect:
	// FAK_FLEET_DIR overrides the %LOCALAPPDATA%/Fleet root it censuses beyond the repo, and
	// FAK_GARDEN_GROWTH_LEDGER overrides where it appends its would-reap/reaped JSONL soak
	// ledger. Both are locations on disk, not credentials — neither grants access to anything
	// the process could not already reach.
	// Relocate to: `--fleet-dir` and `--growth-ledger` flags on `fak garden`, the same shape
	// FAK_SERVICE_LEDGER_DIR takes below.
	"FAK_FLEET_DIR",
	"FAK_GARDEN_GROWTH_LEDGER",

	// internal/gateway/observer.go — filesystem path for the observer journal.
	"FAK_OBSERVER_JOURNAL",

	// cmd/fak/hooks.go — FAK_PRECOMMIT_CHECK_BUDGET_MS and FAK_PRECOMMIT_TOTAL_BUDGET_MS, the
	// two wall-clock caps that keep the #5335 pre-commit wedge from stalling the commit lane,
	// were listed here until 5b56bfdb8 routed both through resolveBudgetMS(name, fallback).
	// They are still read; the scan just cannot corroborate them. Their relocation target is
	// unchanged: `--check-budget-ms` and `--total-budget-ms` flags on `fak hooks pre-commit`.

	// cmd/fak/service.go — env default backing the `fak service status --ledger-dir` flag.
	"FAK_SERVICE_LEDGER_DIR",

	// cmd/fak/sharedtask_endpoint.go — the request-time on/off switch (1/true/yes/on) for the
	// shared-task co-editing subtree /v1/fak/sharedtask/ under `fak serve`; default off keeps
	// the endpoint inert for every subcommand. A feature toggle, not an authorization boundary:
	// what actually guards the surface is the caller's reader scope and sharedtask.Policy's
	// MaxScope, both of which still apply once it is on.
	// Relocates to: a `--sharedtask` opt-in flag on `fak serve`.
	"FAK_SHAREDTASK",

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
}
