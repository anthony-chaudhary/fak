package guardrsi

// Refusal-recovery telemetry (#2143): for every closed-vocabulary refusal reason
// the guard journals, measure how often the refused call actually RECOVERED —
// the same call (tool + args_digest) later adjudicated ALLOW — versus LOOPED
// (re-denied, never cleared) versus never exactly re-attempted. The fold answers
// "which refusal token's fix hint is not working", worst token first, from the
// same hash-chained decision journals the verdict loop folds.
//
// IDENTITY, STATED. A "call" is (tool, args_digest) WITHIN one journal file —
// journals are per-session, so cross-session digest collisions never correlate.
// The journal carries only the args digest, so a call the agent REPAIRED (changed
// args per the cited law) is a different identity and lands in no_retry alongside
// a genuinely abandoned task; those two are statically indistinguishable here and
// the report never claims otherwise. Only exact re-attempts are decided.
//
// EPISODES. Within one identity, each refusal opens an episode: every further
// refusal of the same identity is a wasted row (a turn burned re-hitting the same
// wall), and the episode closes cleared when the identity is next adjudicated
// through (ALLOW, or TRANSFORM — a rewritten call that proceeded). A refusal
// after a clear opens a NEW episode, so deny→allow→deny counts one recovery and
// one fresh refusal, not a muddle.

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
)

// RecoverySchema versions the RecoveryReport JSON emitted by the CLI.
const RecoverySchema = "guard-verdict-rsi.recovery/1"

// ReasonRecovery aggregates every refusal episode attributed to one reason token
// (the token of the episode's FIRST refusal — the fix hint the agent acted on).
// RefusedCalls partitions exactly into Cleared + Looped + NoRetry.
type ReasonRecovery struct {
	Reason       string `json:"reason"`
	RefusedCalls int    `json:"refused_calls"` // episodes opened under this token
	Cleared      int    `json:"cleared"`       // the same call later went through
	Looped       int    `json:"looped"`        // re-denied at least once, never cleared
	NoRetry      int    `json:"no_retry"`      // never exactly re-attempted (repaired-with-changed-args OR abandoned)
	WastedRows   int    `json:"wasted_rows"`   // re-denied rows after each episode's first refusal — pure burned turns
}

// Retried is the number of episodes whose outcome the journal actually decided.
func (r ReasonRecovery) Retried() int { return r.Cleared + r.Looped }

// ClearRate is Cleared over the decided episodes, in [0,1]; 0 when nothing was
// retried (an undecided token, not a failing one — check NoRetry).
func (r ReasonRecovery) ClearRate() float64 {
	if d := r.Retried(); d > 0 {
		return float64(r.Cleared) / float64(d)
	}
	return 0
}

// RecoveryReport is the whole-fleet fold, worst token first.
type RecoveryReport struct {
	Schema       string           `json:"schema"`
	JournalPaths []string         `json:"journal_paths"`
	Rows         int              `json:"rows"`             // adjudicated rows parsed
	RefusedCalls int              `json:"refused_calls"`    // episodes across all tokens
	Unkeyed      int              `json:"unkeyed_refusals"` // refusal rows with no args_digest — uncorrelatable, excluded from fates
	ByReason     []ReasonRecovery `json:"by_reason"`
}

// auditRow is the typed projection of one journal line the recovery fold reads.
type auditRow struct {
	Tool    string `json:"tool"`
	Digest  string `json:"args_digest"`
	Verdict string `json:"verdict"`
	Kind    string `json:"kind"`
	Reason  string `json:"reason"`
}

// normalizeVerdict folds a row's explicit verdict with the kind fallback older
// journal rows rely on — the one verdict vocabulary FoldRows and the recovery
// fold share, so the two lenses can never disagree on what a row decided.
func normalizeVerdict(verdict, kind string) string {
	if v := strings.ToUpper(strings.TrimSpace(verdict)); v != "" {
		return v
	}
	switch k := strings.ToUpper(strings.TrimSpace(kind)); k {
	case "DENY", "RESULT_DENY":
		return "DENY"
	case "QUARANTINE":
		return "QUARANTINE"
	case "DECIDE", "VDSO_HIT":
		return "ALLOW"
	default:
		return k
	}
}

func isRefusalVerdict(v string) bool { return v == "DENY" || v == "QUARANTINE" }

// isClearVerdict reports whether a verdict let the call proceed: ALLOW, or
// TRANSFORM (the kernel rewrote the call and let the rewrite run). DEFER /
// WITNESS / INDETERMINATE neither refuse nor clear, so they leave an episode open.
func isClearVerdict(v string) bool { return v == "ALLOW" || v == "TRANSFORM" }

// BuildRecovery discovers the journals the way every other guardrsi lens does
// (JournalPaths: the fleet dir, plus the per-user journal, or one explicit file)
// and folds them into the worst-first recovery report.
func BuildRecovery(root, auditPath string) RecoveryReport {
	return FoldRecovery(JournalPaths(root, auditPath))
}

// FoldRecovery folds explicit journal files into a RecoveryReport. Pure given
// the file bytes: same journals, same report.
func FoldRecovery(paths []string) RecoveryReport {
	rep := RecoveryReport{Schema: RecoverySchema, JournalPaths: paths}
	byReason := map[string]*ReasonRecovery{}
	for _, path := range paths {
		foldJournalRecovery(path, &rep, byReason)
	}
	rep.ByReason = make([]ReasonRecovery, 0, len(byReason))
	for _, r := range byReason {
		rep.ByReason = append(rep.ByReason, *r)
	}
	// Worst first: most wasted rows, then most loops, then most refusals; name
	// breaks ties so the order is deterministic.
	sort.Slice(rep.ByReason, func(i, j int) bool {
		a, b := rep.ByReason[i], rep.ByReason[j]
		if a.WastedRows != b.WastedRows {
			return a.WastedRows > b.WastedRows
		}
		if a.Looped != b.Looped {
			return a.Looped > b.Looped
		}
		if a.RefusedCalls != b.RefusedCalls {
			return a.RefusedCalls > b.RefusedCalls
		}
		return a.Reason < b.Reason
	})
	return rep
}

// episode is one open refusal→(clear|loop|silence) span for a single identity.
type episode struct {
	reason string
	wasted int
}

// foldJournalRecovery walks ONE journal in row order, correlating episodes per
// (tool, digest) identity. Correlation never crosses journal files.
func foldJournalRecovery(path string, rep *RecoveryReport, byReason map[string]*ReasonRecovery) {
	b, err := os.ReadFile(path)
	if err != nil {
		return
	}
	open := map[string]*episode{} // identity key -> the open episode
	closeEpisode := func(ep *episode, cleared bool) {
		r := byReason[ep.reason]
		if r == nil {
			r = &ReasonRecovery{Reason: ep.reason}
			byReason[ep.reason] = r
		}
		r.RefusedCalls++
		r.WastedRows += ep.wasted
		rep.RefusedCalls++
		switch {
		case cleared:
			r.Cleared++
		case ep.wasted > 0:
			r.Looped++
		default:
			r.NoRetry++
		}
	}
	for _, line := range strings.Split(string(b), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var row auditRow
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			continue
		}
		v := normalizeVerdict(row.Verdict, row.Kind)
		if v == "" {
			continue
		}
		rep.Rows++
		refusal, clear := isRefusalVerdict(v), isClearVerdict(v)
		if !refusal && !clear {
			continue // DEFER/WITNESS/... — neither refuses nor clears
		}
		if row.Digest == "" {
			if refusal {
				rep.Unkeyed++
			}
			continue // no identity to correlate
		}
		key := row.Tool + "\x00" + row.Digest
		ep := open[key]
		switch {
		case refusal && ep == nil:
			reason := strings.TrimSpace(row.Reason)
			if reason == "" {
				reason = "(blank)"
			}
			open[key] = &episode{reason: reason}
		case refusal:
			ep.wasted++
		case clear && ep != nil:
			closeEpisode(ep, true)
			delete(open, key)
		}
	}
	// Episodes still open at end-of-journal never cleared: looped or no-retry.
	// Deterministic order does not matter here — closing is order-independent.
	for _, ep := range open {
		closeEpisode(ep, false)
	}
}

// RenderRecovery renders the report worst token first, with the identity limit
// stated so a reader never over-trusts no_retry.
func RenderRecovery(rep RecoveryReport) string {
	lines := []string{
		"guard-verdict-rsi recovery: which refusal token's fix hint clears, from the real journal",
	}
	if rep.RefusedCalls == 0 {
		lines = append(lines, "  no refused calls with a correlatable identity in the journal(s) — nothing to grade")
		if rep.Unkeyed > 0 {
			lines = append(lines, renderUnkeyed(rep.Unkeyed))
		}
		return strings.Join(lines, "\n")
	}
	lines = append(lines, renderRecoverySummary(rep))
	for _, r := range rep.ByReason {
		lines = append(lines, renderReasonLine(r))
	}
	if rep.Unkeyed > 0 {
		lines = append(lines, renderUnkeyed(rep.Unkeyed))
	}
	worst := rep.ByReason[0]
	if worst.WastedRows > 0 {
		lines = append(lines, "  worst: "+worst.Reason+" — its refusals are re-attempted verbatim and re-denied; the cited fix is not landing (improve the law text / the fix hint for this token)")
	}
	lines = append(lines, "  limit: identity is (tool, args-digest) within one session journal; a repaired call (changed args) and an abandoned task both land in no-retry — only exact re-attempts are decided")
	return strings.Join(lines, "\n")
}

func renderRecoverySummary(rep RecoveryReport) string {
	return fmt.Sprintf("  %d refused call(s) across %d journal(s), %d adjudicated row(s)",
		rep.RefusedCalls, len(rep.JournalPaths), rep.Rows)
}

func renderReasonLine(r ReasonRecovery) string {
	line := fmt.Sprintf("  %s: refused %d  cleared %d  looped %d  no-retry %d  wasted-rows %d",
		r.Reason, r.RefusedCalls, r.Cleared, r.Looped, r.NoRetry, r.WastedRows)
	if r.Retried() > 0 {
		line += fmt.Sprintf("  clear-rate %.0f%%", r.ClearRate()*100)
	}
	return line
}

func renderUnkeyed(n int) string {
	return fmt.Sprintf("  (%d refusal row(s) carried no args-digest and could not be correlated)", n)
}
