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
	Index    int    `json:"index"`            // position in the folded chain (rank order)
	Rung     string `json:"rung"`             // concrete adjudicator type, e.g. "adjudicator.Adjudicator"
	By       string `json:"by,omitempty"`     // verdict.By: who claims the decision ("" on a bare defer)
	Kind     string `json:"kind"`             // ALLOW/DENY/TRANSFORM/QUARANTINE/WITNESS/DEFER/...
	Reason   string `json:"reason,omitempty"` // reason name, omitted when NONE
	Claim    string `json:"claim,omitempty"`  // bounded-disclosure witness, if this rung disclosed one
	Rank     int    `json:"rank"`             // FoldRank(kind): restrictiveness-lattice position
	Deferred bool   `json:"deferred"`         // Kind==DEFER: this rung abstained, did not participate
	Winner   bool   `json:"winner"`           // this rung's verdict was the folded result
	Elided   bool   `json:"elided,omitempty"` // this rung did NOT run: a max-rank verdict short-circuited the fold before it
}

// Decision is the full, explainable trace of one adjudication fold. It is the
// structured dual of the one-line verdict: every rung consulted, what each
// returned, which won, the bounded-disclosure witness, the loopback
// disposition, and a one-line human explanation. Built only off the hot path.
type Decision struct {
	Tool        string `json:"tool"`
	ArgsDigest  string `json:"args_digest,omitempty"` // sha256[:12] of the args bytes — never the raw args
	ArgsBytes   int    `json:"args_bytes"`            // size of the args payload
	Consistency string `json:"consistency"`           // the call's declared consistency level (#1317): STRICT (the default) / BOUNDED_STALE / BEST_EFFORT / SPECULATIVE — recorded verbatim so the relaxation contract is an audit field, not a hidden mode
	Verdict     string `json:"verdict"`               // final verdict kind name
	Reason      string `json:"reason,omitempty"`      // final reason name (omitted when NONE)
	By          string `json:"by,omitempty"`          // winning rung's By (or synthesized: empty-policy/all-defer)
	Claim       string `json:"claim,omitempty"`       // final bounded-disclosure witness
	Disposition string `json:"disposition,omitempty"` // deny loopback: RETRYABLE/WAIT/ESCALATE/TERMINAL
	// DenyRule is the CLOSED-vocabulary id of the policy RUNG that refused
	// (abi.DenyRuleID). Reason names the refusal's CLASS — POLICY_BLOCK covers the
	// recursive-delete rung, the out-of-tree-write rung and seven gitgate laws at
	// once — so a reader given only the class cannot tell WHICH rule matched, and
	// an operator auditing a denial cannot check whether the advice they were given
	// is even about the same rule (#5213). Re-validated through abi.DenyRuleID, so
	// a non-member value is dropped WHOLE and no input byte can appear here.
	DenyRule string `json:"deny_rule,omitempty"`
	// Remedy is the sanctioned alternative declared BY THE REFUSING RUNG ITSELF
	// (verdict Meta "fix" / "dry_run_hint" — the same one seam the wire renders).
	// It is empty when the matched rule declares none, and an empty Remedy MUST be
	// reported as "no sanctioned alternative is known" rather than back-filled from
	// the reason CLASS: a class-keyed lookup is what advised a Slack send and a
	// core-lock maintenance witness for an ordinary refused file edit (#5213).
	// Same disclosure budget as Claim — static operator-authored text, never an arg
	// value — so it stays safe to log.
	Remedy      string        `json:"remedy,omitempty"`
	Posture     string        `json:"posture,omitempty"`    // verdict Meta: e.g. admit_and_log
	WouldDeny   string        `json:"would_deny,omitempty"` // verdict Meta: the reason a posture downgrade suppressed
	Redacted    []string      `json:"redacted,omitempty"`   // TRANSFORM: arg keys whose value the rung rewrote
	Rungs       []RungVerdict `json:"rungs"`                // every rung consulted, in fold order
	Explanation string        `json:"explanation"`          // one-line human summary
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
	// verdict is selected identically: the highest-FoldRank conclusive verdict,
	// ties won by the FIRST rung to reach that rank (strict-greater update).
	var v abi.Verdict
	switch {
	case len(chain) == 0:
		v = abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonDefaultDeny, By: "empty-policy"}
	default:
		best := abi.Verdict{Kind: abi.VerdictDefer, By: "no-link"}
		bestRank, bestIdx := -1, -1
		indeterminateIdx := -1
		sawConclusive := false
		sawIndeterminate := false
		brokeAt := -1 // index of the max-rank verdict that short-circuited the fold, or -1
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
			switch rv.Kind {
			case abi.VerdictDefer:
				continue
			case abi.VerdictIndeterminate:
				sawIndeterminate = true
				if indeterminateIdx < 0 {
					indeterminateIdx = len(d.Rungs) - 1
				}
				continue
			}
			if rank > bestRank {
				bestRank, best, bestIdx = rank, rv, i
				sawConclusive = true
				if isMaxFoldRank(rank) {
					brokeAt = i
					break
				}
			}
		}
		// A max-rank conclusive verdict short-circuits the fold (Fold does the same): the
		// rungs after it did NOT run — they were ELIDED by the decision. Record them
		// (un-evaluated, so no By/Kind) instead of dropping them, so the trace answers
		// "which rung DECIDED and which were skipped" — the chain-level dual of a
		// RungProfile eliding sub-rungs inside a single adjudicator.
		for i := brokeAt + 1; brokeAt >= 0 && i < len(chain); i++ {
			d.Rungs = append(d.Rungs, RungVerdict{Index: i, Rung: rungType(chain[i]), Elided: true})
		}
		switch {
		case sawConclusive:
			v = best
			d.Rungs[bestIdx].Winner = true
		default:
			if pc, ok := abi.PolicyFromContext(ctx); ok && pc.Posture == abi.PostureDefaultOpen {
				v = abi.Verdict{Kind: abi.VerdictAllow, By: "all-defer(default-open)"}
			} else if sawIndeterminate {
				rv := d.Rungs[indeterminateIdx]
				v = abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonDefaultDeny, By: rv.By,
					Meta: map[string]string{"fold": "indeterminate"}}
				d.Rungs[indeterminateIdx].Winner = true
			} else {
				v = abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonDefaultDeny, By: "all-defer"}
			}
		}
	}

	d.populate(ctx, c, v)
	return v, d
}

// populate fills the final-verdict summary fields and the human explanation from
// the resolved verdict v.
func (d *Decision) populate(ctx context.Context, c *abi.ToolCall, v abi.Verdict) {
	// Record the call's declared consistency level (#1317) verbatim. ConsistencyOf
	// applies the STRICT default for an unset/unknown value, so the journal always
	// carries a concrete level — the relaxation contract is an audit field, never
	// inferred. This is forensic surplus only: it never feeds the verdict v.
	d.Consistency = abi.ConsistencyOf(c).String()
	d.Verdict = kindName(v.Kind)
	d.Reason = reasonOrEmpty(v.Reason)
	d.By = v.By
	d.Claim = claimOf(v)
	if v.Kind == abi.VerdictDeny {
		d.Disposition = VerdictDisposition(v)
	}
	// Bind the refusal to the rule that actually matched, and to that rule's OWN
	// declared alternative. Both are read off the winning verdict only — never
	// synthesized from the reason class — so a consumer can always tell "this
	// remedy came from the rung that refused" from "this rung offered none".
	if isRefusal(v.Kind) {
		if id, ok := abi.DenyRuleID(v.Meta[abi.MetaDenyRule]); ok {
			d.DenyRule = id
		}
		d.Remedy = remedyOf(v)
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
		if d.DenyRule != "" {
			b += " matched rule: " + d.DenyRule + "."
		}
		if d.Claim != "" {
			b += " offending set: " + d.Claim + "."
		}
		if d.By == "empty-policy" || d.By == "all-defer" {
			b += " No rung affirmatively allowed it — fail-closed default deny."
		}
		return b + remedySentence(d.Remedy)
	case "TRANSFORM":
		if len(d.Redacted) > 0 {
			return d.Tool + " transformed by " + or(d.By, "a rung") + ": rewrote " + strings.Join(d.Redacted, ", ") + " before dispatch (e.g. secret redaction)."
		}
		return d.Tool + " transformed by " + or(d.By, "a rung") + " before dispatch."
	case "WITNESS":
		return d.Tool + " held pending an independent witness read-back" + claimSuffix(d.Claim) + "." + remedySentence(d.Remedy)
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
	if d.DenyRule != "" {
		fmt.Fprintf(&b, "rule: %s\n", d.DenyRule)
	}
	if d.Claim != "" {
		fmt.Fprintf(&b, "witness: %s\n", d.Claim)
	}
	// Print the remedy line for every refusal, including the empty case: an
	// operator reading a denial trace must be able to tell "this rule offers no
	// alternative" from "the trace forgot to print one".
	if isRefusalName(d.Verdict) {
		fmt.Fprintf(&b, "remedy: %s\n", or(d.Remedy, "(none — no sanctioned alternative is known for this call)"))
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
		if r.Elided {
			fmt.Fprintf(&b, "   [%d] %-26s %-9s   (elided: a max-rank verdict decided before this rung)\n", r.Index, r.Rung, "ELIDED")
			continue
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
	case abi.VerdictIndeterminate:
		return "INDETERMINATE"
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

// isRefusal reports whether a verdict kind withheld the call. Allow / Transform /
// Defer proceed (a Transform proceeds with rewritten args); Quarantine acts on the
// RESULT, not the call, so it is not a refused attempt either. Everything else —
// Deny, RequireWitness, and any registered restrictive kind the fold held — is a
// refusal whose matched rule and alternative the caller is owed.
func isRefusal(k abi.VerdictKind) bool {
	switch k {
	case abi.VerdictAllow, abi.VerdictTransform, abi.VerdictDefer, abi.VerdictQuarantine:
		return false
	}
	return true
}

// isRefusalName is isRefusal over a rendered verdict NAME, for the Decision's own
// already-stringified Verdict field. Kept in lockstep with isRefusal above.
func isRefusalName(name string) bool {
	switch name {
	case "ALLOW", "TRANSFORM", "DEFER", "QUARANTINE":
		return false
	}
	return true
}

// remedyOf returns the sanctioned alternative the REFUSING RUNG declared, through
// the same one seam the wire renders: the arg-predicate rung stamps Meta["fix"]
// (its manifest arg_rules[].fix), the reversibility rung stamps
// Meta["dry_run_hint"] (its preview affordance). Empty means the matched rule
// declared no alternative — reported as such, never back-filled from the class.
func remedyOf(v abi.Verdict) string {
	if fix := v.Meta["fix"]; fix != "" {
		return fix
	}
	return v.Meta["dry_run_hint"]
}

// remedySentence renders the actionable half of a refusal. An absent remedy is
// stated OUT LOUD — a refusal that silently omits the alternative reads as "there
// must be one, go find it", which is how a governed agent talks itself onto a more
// privileged path than the one it was refused (#5213).
func remedySentence(remedy string) string {
	if remedy == "" {
		return " No sanctioned alternative is known for this call."
	}
	return " Sanctioned alternative: " + remedy + "."
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
