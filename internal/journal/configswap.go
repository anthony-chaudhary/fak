package journal

// Config-swap telemetry — the durable witness for a capability-floor or
// route-manifest HOT-SWAP (#3959).
//
// A hot-swap of the capability floor (the `--policy` manifest reloaded through
// reloadPolicy / POST /v1/fak/policy/reload) or the model-routing manifest
// (reloaded by the serve route-watcher) changes the SECURITY BOUNDARY of a live
// session, yet before this rung it left no durable record: reloadPolicy swapped
// adjudicator.Default.SetPolicy and returned, the gateway handler only logged,
// and the route watcher only printed to stderr. The tamper-evident decision
// journal — built so an auditor can reconstruct "what did the kernel decide,
// over which bytes, and why" — had no row kind for "the RULES THEMSELVES
// changed", so every decision after a swap chained onto an unrecorded policy.
//
// A CONFIG_SWAP row closes that hole the same way RESTART_HOP and CHILD_CRASH
// closed theirs: one first-class chained row per swap, written DIRECTLY through
// the chain (AppendConfigSwap → appendLocked), NOT through Emit(abi.Event) — a
// config swap is not a kernel decision, and routing a synthetic event through
// the frozen ABI would fan it out to every decision-stream folder that assumes
// an event IS an adjudication. The chained forensic identity — Kind, the swapped
// surface (Tool), and the closed outcome class (Reason) — rides the frozen
// decision fields, so it verifies end-to-end with every existing row; the full
// correlated record (schema fak.config.swap.v1, carrying the manifest source
// path and the sha256 of the installed bytes) rides the non-chained ConfigSwap
// payload field, the same layering RestartHop uses.

// KindConfigSwap marks a capability-floor / route-manifest hot-swap row. It is a
// genuine chained row (it consumes the next Seq and chains onto the prior head)
// that carries no verdict, so decision-folding consumers skip it; readers that
// key on Kind (an auditor reconstructing the policy timeline) fold it into the
// session's config-change history.
const KindConfigSwap = "CONFIG_SWAP"

// ConfigSwapSchema names the correlated per-swap record carried on a CONFIG_SWAP
// row's ConfigSwap field. Versioned like every fak wire schema: additive-only;
// never edit a shipped /vN in place.
const ConfigSwapSchema = "fak.config.swap.v1"

// Config-swap surfaces — the CLOSED vocabulary for WHICH live boundary swapped,
// mirrored onto the chained Tool field so the row is self-describing.
const (
	// ConfigSwapFloor is the capability-floor manifest: the `--policy` file
	// swapped by reloadPolicy (startup apply and POST /v1/fak/policy/reload).
	ConfigSwapFloor = "floor"
	// ConfigSwapRoute is the model-routing manifest: swapped by the serve
	// route-watcher's OnEvent on a validated edit.
	ConfigSwapRoute = "route"
)

// Config-swap outcomes — the CLOSED vocabulary for the Reason field of a
// CONFIG_SWAP row (and the Outcome field of its payload). A closed set keeps
// audit buckets stable instead of exploding into free-text.
const (
	// ConfigSwapOK is a swap that validated and is now the live boundary.
	ConfigSwapOK = "ok"
	// ConfigSwapRejected is a swap that was REFUSED (a malformed edit): the
	// last-good config is kept, but the attempt is recorded because a rejected
	// edit — an operator trying to widen or break the floor — is exactly what an
	// auditor asks about.
	ConfigSwapRejected = "rejected"
)

// ConfigSwapRow is the correlated per-swap record (ConfigSwapSchema): every axis
// of one capability-floor / route-manifest hot-swap tied together in a single
// value, instead of a stderr line the auditor cannot query after the fact.
type ConfigSwapRow struct {
	Schema  string `json:"schema"`            // ConfigSwapSchema
	Kind    string `json:"kind"`              // ConfigSwapFloor | ConfigSwapRoute
	Source  string `json:"source,omitempty"`  // the manifest path that was (re)installed
	Digest  string `json:"digest,omitempty"`  // sha256 of the installed bytes ("" when unreadable)
	Outcome string `json:"outcome"`           // ConfigSwapOK | ConfigSwapRejected
	Reason  string `json:"reason,omitempty"`  // rejection detail (error text); "" on ok
}

// AppendConfigSwap records one capability-floor or route-manifest hot-swap as a
// durable, chained CONFIG_SWAP row and returns the committed row (with its
// stamped Seq/hash). kind is the swapped surface (ConfigSwapFloor |
// ConfigSwapRoute, recorded on Tool so the row is self-describing), source is the
// manifest path, digest is the sha256 of the installed bytes, outcome is the
// closed class (ConfigSwapOK | ConfigSwapRejected, mirrored onto the chained
// Reason field), and reason carries the rejection detail on a refused edit.
// It is a no-op returning the zero Row on a nil receiver, so a caller that
// guarded the journal on may call it unconditionally — an unjournaled run stays
// byte-identical.
//
// Like AppendRestartHop, the row is written directly through the chain (not the
// ABI fan-out): a config swap is supervision, not a kernel decision. The write
// is flushed per row by appendLocked, so the swap survives whatever the
// reloaded session does next.
func (j *Journal) AppendConfigSwap(kind, source, digest, outcome, reason string) Row {
	if j == nil {
		return Row{}
	}
	payload := ConfigSwapRow{
		Schema:  ConfigSwapSchema,
		Kind:    kind,
		Source:  source,
		Digest:  digest,
		Outcome: outcome,
		Reason:  reason,
	}
	row := Row{
		Kind:       KindConfigSwap,
		Tool:       kind,
		Reason:     outcome,
		By:         "config-supervisor",
		ConfigSwap: &payload,
	}
	j.mu.Lock()
	j.appendLocked(row)
	committed := j.recent[len(j.recent)-1]
	j.mu.Unlock()
	return committed
}
