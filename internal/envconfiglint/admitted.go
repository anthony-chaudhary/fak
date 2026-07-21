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
var admittedPostFreeze = []string{
	// cmd/fak/chatops.go — chatops deployment wiring: which channel to post in, which bot
	// identity to post as, and the admin roster allowed to drive it. Identities and a
	// channel name, not credentials (the chatops TOKEN is separate and secret-shaped).
	"FAK_CHATOPS_ADMINS",
	"FAK_CHATOPS_BOT_USER",
	"FAK_CHATOPS_CHANNEL",

	// internal/gateway/observer.go — filesystem path for the observer journal.
	"FAK_OBSERVER_JOURNAL",

	// cmd/fak/service.go — env default backing the `fak service status --ledger-dir` flag.
	"FAK_SERVICE_LEDGER_DIR",
}
