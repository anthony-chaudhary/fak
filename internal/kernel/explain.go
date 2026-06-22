// explain.go — the OFF-HOT-PATH dual of Fold: it folds the SAME adjudicator
// chain to the SAME winning verdict, but additionally records a per-rung
// Decision trace — the answer to the single most common debugging question in
// the whole kernel: "why did fak give THIS verdict for THIS tool call?"
//
// Fold (the hot path) keeps only the winning verdict; every rung that ran, what
// each returned, and which one won are discarded. That is correct for the
// nanosecond budget of a served call, but it leaves `fak preflight` — the
// canonical 60-second proof command — printing a single opaque line
// (verdict=X reason=Y by=Z) with no way to see the eight rungs it actually
// folded. FoldExplain is the additive answer: callers that only need the verdict
// use Fold; callers answering "why" (fak preflight --explain/--json, the gateway
// decision trace, the agent run report) use this.
//
// The Decision is deliberately SAFE TO LOG: it carries an args DIGEST and byte
// count, never the raw (possibly secret) args, and it surfaces only the
// bounded-disclosure witness the verdict already chose to disclose — the same
// deny-channel-is-not-a-policy-oracle discipline the adjudicator enforces.
package kernel

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// RungVerdict is one adjudicator's contribution to a folded decision — the
// per-rung detail Fold throws away. Rung is the concrete adjudicator type (a
// stable identity even when the rung DEFERS and sets no By), By is the verdict's
// self-reported decider, and Winner marks the rung whose verdict the lattice
// fold selected.
type RungVerdict struct {
	Index    int    `json:"index"`             // position in the folded chain (rank order)
	Rung     string `json:"rung"`              // concrete adjudicator type, e.g. "adjudicator.Adjudicator"
	By       string `json:"by,omitempty"`      // verdict.By: who claims the decision ("" on a bare defer)
	Kind     string `json:"kind"`              // ALLOW/DENY/TRANSFORM/QUARANTINE/WITNESS/DEFER/...
	Reason   string `json:"reason,omitempty"`  // reason name, omitted when NONE
	Claim    string `json:"claim,omitempty"`   // bounded-disclosure witness, if this rung disclosed one
	Rank     int    `json:"rank"`              // FoldRank(kind): restrictiveness-lattice position
	Deferred bool   `json:"deferred"`          // Kind==DEFER: this rung abstained, did not participate
	Winner   bool   `json:"winner"`            // this rung's verdict was the folded result
}

// Decision is the full, explainable trace of one adjudication fold. It is the
// structured dual of the one-line verdict: every rung consulted, what each
// returned, which won, the bounded-disclosure witness, the loopback
// disposition, and a one-line human explanation. Built only off the hot path.
type Decision struct {
	Tool        string        `json:"tool"`
	ArgsDigest  string        `json:"args_digest,omitempty"` // sha256[:12] of the args bytes — never the raw args
	ArgsBytes   int           `json:"args_bytes"`            // size of the args payload
	Verdict     string        `json:"verdict"`               // final verdict kind name
	Reason      string        `json:"reason,omitempty"`      // final reason name (omitted when NONE)
	By          string        `json:"by,omitempty"`          // winning rung's By (or synthesized: empty-policy/all-defer)
	Claim       string        `json:"claim,omitempty"`       // final bounded-disclosure witness
	Disposition string        `json:"disposition,omitempty"` // deny loopback: RETRYABLE/WAIT/ESCALATE/TERMINAL
	Posture     string        `json:"posture,omitempty"`     // verdict Meta: e.g. admit_and_log
	WouldDeny   string        `json:"would_deny,omitempty"`  // verdict Meta: the reason a posture downgrade suppressed
	Redacted    []string      `json:"redacted,omitempty"`    // TRANSFORM: arg keys whose value the rung rewrote
	Rungs       []RungVerdict `json:"rungs"`                 // every rung consulted, in fold order
	Explanation string        `json:"explanation"`           // one-line human summary
}

// FoldExplain folds an Adjudicator chain EXACTLY as Fold does (same winning
// verdict, same lattice resolution) and additionally returns the per-rung
// Decision trace. The returned Verdict is byte-identical to Fold(ctx, chain, c):
// the trace is pure forensic surplus, never a behavior change.
func FoldExplain(ctx context.Context, chain []abi.Adjudicator, c *abi.ToolCall) (abi.Verdict, Decision) {
	d := Decision{Tool: c.Tool}
	if b := refBytesK(ctx, c.Args); len(b) > 0 {
		sum := sha256.Sum256(b)
		d.ArgsDigest = hex.EncodeToString(sum[:])[:12]
		d.ArgsBytes = len(b)
	}

	// Mirror Fold's lattice resolution, capturing each rung as we go. The winning
	// verdict is selected identically: the highest-FoldRank non-Defer verdict, ties
	// won by the FIRST rung to reach that rank (strict-greater update).
	var v abi.Verdict
	switch {
	case len(chain) == 0:
		v = abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonDefaultDeny, By: "empty-policy"}
	default:
		best := abi.Verdict{Kind: abi.VerdictDefer, By: "no-link"}
		bestRank, bestIdx := -1, -1
		sawNonDefer := false
		d.Rungs = make([]RungVerdict, 0, len(chain))
		for i, a := range chain {
			rv := a.Adjudicate(ctx, c)
			rank := abi.FoldRank(rv.Kind)
			rung := RungVerdict{
				Index:    i,
				Rung:     rungType(a),
				By:       rv.By,
				Kind:     kindName(rv.Kind),
				Reason:   reasonOrEmpty(rv.Reason),
				Claim:    claimOf(rv),
				Rank:     rank,
				Deferred: rv.Kind == abi.VerdictDefer,
			}
			d.Rungs = append(d.Rungs, rung)
			if rv.Kind == abi.VerdictDefer {
				continue
			}
			sawNonDefer = true
			if rank > bestRank {
				bestRank, best, bestIdx = rank, rv, i
			}
		}
		switch {
		case !sawNonDefer:
			v = abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonDefaultDeny, By: "all-defer"}
		default:
			v = best
			d.Rungs[bestIdx].Winner = true
		}
	}

	d.populate(ctx, c, v)
	return v, d
}

// populate fills the final-verdict summary fields and the human explanation from
// the resolved verdict v.
func (d *Decision) populate(ctx context.Context, c *abi.ToolCall, v abi.Verdict) {
	d.Verdict = kindName(v.Kind)
	d.Reason = reasonOrEmpty(v.Reason)
	d.By = v.By
	d.Claim = claimOf(v)
	if v.Kind == abi.VerdictDeny {
		d.Disposition = Disposition(v.Reason)
	}
	d.Posture = v.Meta["posture"]
	d.WouldDeny = v.Meta["would_deny"]
	if v.Kind == abi.VerdictTransform {
		if tp, ok := v.Payload.(abi.TransformPayload); ok {
			d.Redacted = changedKeys(refBytesK(ctx, c.Args), refBytesK(ctx, tp.NewArgs))
		}
	}
	d.Explanation = d.explain()
}

// explain renders the one-line human summary from the populated fields.
func (d *Decision) explain() string {
	switch d.Verdict {
	case "ALLOW":
		switch {
		case d.By == "vdso":
			return d.Tool + " allowed: served by the vDSO fast path (deduplicated; no adjudication ran)."
		case d.By == "witness":
			return d.Tool + " allowed: a require-witness gate was corroborated by independent evidence."
		case d.Posture == "admit_and_log":
			return d.Tool + " allowed under the admit-and-log posture (would otherwise be " + or(d.WouldDeny, "denied") + "); forensic metadata recorded."
		default:
			return d.Tool + " allowed: an affirmative policy rung permitted it."
		}
	case "DENY":
		b := d.Tool + " denied by " + or(d.By, "the floor") + ": " + or(d.Reason, "DENY")
		if d.Disposition != "" {
			b += " (" + d.Disposition + ")"
		}
		b += "."
		if d.Claim != "" {
			b += " offending set: " + d.Claim + "."
		}
		if d.By == "empty-policy" || d.By == "all-defer" {
			b += " No rung affirmatively allowed it — fail-closed default deny."
		}
		return b
	case "TRANSFORM":
		if len(d.Redacted) > 0 {
			return d.Tool + " transformed by " + or(d.By, "a rung") + ": rewrote " + strings.Join(d.Redacted, ", ") + " before dispatch (e.g. secret redaction)."
		}
		return d.Tool + " transformed by " + or(d.By, "a rung") + " before dispatch."
	case "WITNESS":
		return d.Tool + " held pending an independent witness read-back" + claimSuffix(d.Claim) + "."
	case "QUARANTINE":
		return d.Tool + " result quarantined: held out of the model's context window."
	default:
		return d.Tool + ": verdict " + d.Verdict + " by " + or(d.By, "a rung") + "."
	}
}

// Text renders the Decision as a human-readable multi-line trace for
// `fak preflight --explain`. It leads with the verdict summary, then the full
// rung chain with the winner marked.
func (d Decision) Text() string {
	var b strings.Builder
	fmt.Fprintf(&b, "tool: %s", d.Tool)
	if d.ArgsBytes > 0 {
		fmt.Fprintf(&b, "   args: %d bytes (sha %s)", d.ArgsBytes, d.ArgsDigest)
	}
	b.WriteByte('\n')
	fmt.Fprintf(&b, "verdict: %s", d.Verdict)
	if d.Reason != "" {
		fmt.Fprintf(&b, "   reason: %s", d.Reason)
	}
	if d.By != "" {
		fmt.Fprintf(&b, "   by: %s", d.By)
	}
	if d.Disposition != "" {
		fmt.Fprintf(&b, "   disposition: %s", d.Disposition)
	}
	b.WriteByte('\n')
	if d.Posture != "" {
		fmt.Fprintf(&b, "posture: %s", d.Posture)
		if d.WouldDeny != "" {
			fmt.Fprintf(&b, " (would_deny: %s)", d.WouldDeny)
		}
		b.WriteByte('\n')
	}
	if d.Claim != "" {
		fmt.Fprintf(&b, "witness: %s\n", d.Claim)
	}
	if len(d.Redacted) > 0 {
		fmt.Fprintf(&b, "redacted: %s\n", strings.Join(d.Redacted, ", "))
	}
	fmt.Fprintf(&b, "explanation: %s\n\n", d.Explanation)

	if len(d.Rungs) == 0 {
		b.WriteString("decision chain: empty policy — fail-closed default deny.\n")
		return b.String()
	}
	fmt.Fprintf(&b, "decision chain (%d rung(s), most-restrictive wins):\n", len(d.Rungs))
	for _, r := range d.Rungs {
		marker := "  "
		if r.Winner {
			marker = "=>"
		}
		fmt.Fprintf(&b, "%s [%d] %-26s %-9s", marker, r.Index, r.Rung, r.Kind)
		if r.Reason != "" {
			fmt.Fprintf(&b, " %s", r.Reason)
		}
		if r.By != "" && r.By != r.Rung {
			fmt.Fprintf(&b, " by=%s", r.By)
		}
		if r.Claim != "" {
			fmt.Fprintf(&b, " {%s}", r.Claim)
		}
		if r.Winner {
			fmt.Fprintf(&b, "   <- winner (rank %d)", r.Rank)
		}
		b.WriteByte('\n')
	}
	return b.String()
}

// JSON renders the Decision as indented JSON for `fak preflight --json`.
func (d Decision) JSON() string {
	b, err := json.MarshalIndent(d, "", "  ")
	if err != nil {
		return "{}"
	}
	return string(b)
}

// --- small helpers (off the hot path) -------------------------------------

func rungType(a abi.Adjudicator) string {
	return strings.TrimPrefix(fmt.Sprintf("%T", a), "*")
}

func kindName(k abi.VerdictKind) string {
	switch k {
	case abi.VerdictAllow:
		return "ALLOW"
	case abi.VerdictDeny:
		return "DENY"
	case abi.VerdictTransform:
		return "TRANSFORM"
	case abi.VerdictQuarantine:
		return "QUARANTINE"
	case abi.VerdictRequireWitness:
		return "WITNESS"
	case abi.VerdictDefer:
		return "DEFER"
	}
	return fmt.Sprintf("KIND_%d", uint16(k))
}

func reasonOrEmpty(r abi.ReasonCode) string {
	if r == abi.ReasonNone {
		return ""
	}
	return abi.ReasonName(r)
}

func claimOf(v abi.Verdict) string {
	switch p := v.Payload.(type) {
	case abi.WitnessPayload:
		return p.Claim
	}
	if v.Meta != nil {
		return v.Meta["claim"]
	}
	return ""
}

func claimSuffix(claim string) string {
	if claim == "" {
		return ""
	}
	return " (claim: " + claim + ")"
}

func or(s, fallback string) string {
	if s == "" {
		return fallback
	}
	return s
}

// changedKeys returns the sorted set of top-level keys whose JSON value differs
// between the original and transformed args — the redaction/rewrite diff a
// TRANSFORM verdict applied. Best-effort: if either side is not a JSON object it
// returns nil (the transform is still reported, just without the key list).
func changedKeys(orig, next []byte) []string {
	var a, b map[string]any
	if json.Unmarshal(orig, &a) != nil || json.Unmarshal(next, &b) != nil {
		return nil
	}
	seen := map[string]bool{}
	for k := range a {
		seen[k] = true
	}
	for k := range b {
		seen[k] = true
	}
	var out []string
	for k := range seen {
		if !sameJSON(a[k], b[k]) {
			out = append(out, k)
		}
	}
	sort.Strings(out)
	return out
}

func sameJSON(x, y any) bool {
	bx, _ := json.Marshal(x)
	by, _ := json.Marshal(y)
	return string(bx) == string(by)
}

// refBytesK resolves a Ref to its bytes via the active resolver (off the hot
// path; the explain trace only). Named with a K suffix to avoid colliding with
// any future kernel-local helper.
func refBytesK(ctx context.Context, r abi.Ref) []byte {
	if r.Kind == abi.RefInline {
		return r.Inline
	}
	res := abi.ActiveResolver()
	if res == nil {
		return nil
	}
	b, err := res.Resolve(ctx, r)
	if err != nil {
		return nil
	}
	return b
}
