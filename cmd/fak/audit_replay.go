package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/journal"
)

// cmdAuditReplay is the REPLAY side of the durable decision journal (#2905, part of the
// Hermes-inspiration epic #2871). Hermes generates trajectories as a research asset but a
// live product run is not itself replayable/witnessed; fak already persists a per-decision
// session ledger (the hash-chained journal `fak guard`/FAK_AUDIT_JOURNAL write), so a guarded
// session can be re-driven from that ledger to make "did this actually happen" a witness, not
// a claim.
//
// It re-drives the RECORDED tool-call sequence against the floor's own recorded verdicts and
// asserts DETERMINISM: the capability floor is a pure function of a tool call, so every
// identical recorded call — the SAME tool over the SAME args-content-digest — must carry the
// SAME (verdict, reason). A call identity that the ledger shows landing on two or more distinct
// verdicts is not reproducible: the floor answered the same question two different ways within
// one session (a hot policy swap mid-run, or genuine non-determinism). Each such divergence is
// emitted as a structured non-determinism finding; a clean replay (every identical call → one
// verdict) is the reproducible-trajectory witness the issue's first slice asks for.
//
// WHY `fak audit replay` AND NOT `fak replay <session>`. The bare `fak replay` verb is already
// bound to the trace-replay path (cmd/fak/main.go: `fak run --trace` re-drives a bench trace
// through the kernel) — a different kind of replay. The recorded SESSION here IS the durable
// decision journal, whose home is `fak audit` (verify / diagnose / export). This is the replay
// of that record, so it lives next to its siblings rather than colliding with the trace alias.
//
// HONEST LIMIT (the record it can and cannot re-drive). The journal records ArgsDigest (a
// content hash), never the raw args — a deliberate secret-safety choice (see internal/journal).
// So replay asserts determinism over the equivalence class "same tool + same args digest" from
// the durable ledger; it does not reconstruct the raw args to re-execute the live adjudicator
// (that would need a richer record than fak persists today). Tamper-evidence — that the ledger
// itself was not edited since written — is the complementary witness `fak audit verify` /
// `fak audit diagnose` already provide; replay surfaces the linear-chain status as context and
// points there, but does not fail on a benign concurrent-writer interleave.
func cmdAuditReplay(args []string) {
	fs := flag.NewFlagSet("audit replay", flag.ExitOnError)
	asJSON := fs.Bool("json", false, "emit the replay report (and any non-determinism findings) as JSON")
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "usage: fak audit replay [--json] <journal.jsonl>")
		fmt.Fprintln(os.Stderr, "  (re-drive a recorded guard session's decisions; assert every identical call replayed to one verdict)")
	}
	_ = fs.Parse(args)
	if fs.NArg() != 1 {
		fs.Usage()
		os.Exit(2)
	}
	os.Exit(runAuditReplay(os.Stdout, os.Stderr, fs.Arg(0), *asJSON))
}

// replayOutcome is one (verdict, reason) the floor recorded for a call identity, with how many
// times it occurred and the seq bounds of those rows — the evidence a divergence really happened.
type replayOutcome struct {
	Verdict  string `json:"verdict"`
	Reason   string `json:"reason,omitempty"`
	Count    int    `json:"count"`
	FirstSeq uint64 `json:"first_seq,omitempty"`
	LastSeq  uint64 `json:"last_seq,omitempty"`
}

// replayFinding is one structured non-determinism finding: a single recorded call identity
// (tool + args digest) that the floor mapped to more than one verdict across the session.
type replayFinding struct {
	Tool        string          `json:"tool"`
	ArgsDigest  string          `json:"args_digest"`
	ArgsLabel   string          `json:"args_label,omitempty"`
	Occurrences int             `json:"occurrences"`
	Outcomes    []replayOutcome `json:"outcomes"` // the >1 distinct verdicts, most-frequent first
}

// replayReport is the structured result of a journal replay — the JSON payload and the source
// the human render reads from, so the two never disagree (the audit_diagnose pattern).
type replayReport struct {
	Path string `json:"path"`
	Rows int    `json:"rows"`

	// ChainLinearOK is whether the file verifies as ONE linear chain (what `fak audit verify`
	// checks). It is NOT exit-driving: a shared default journal legitimately interleaves
	// concurrent writers. ChainNote explains a non-linear read and points at `fak audit diagnose`.
	ChainLinearOK bool   `json:"chain_linear_ok"`
	ChainNote     string `json:"chain_note,omitempty"`

	// DecisionRows counts call-adjudication rows (DECIDE/DENY) carrying an args digest — the
	// rows replay can key. DistinctCalls is the number of distinct (tool, args_digest) identities.
	// UnkeyableRows counts adjudication rows with no args digest (blob-backed args, no inline
	// hash): honestly un-assertable, surfaced rather than silently dropped.
	DecisionRows  int `json:"decision_rows"`
	DistinctCalls int `json:"distinct_calls"`
	UnkeyableRows int `json:"unkeyable_rows,omitempty"`

	// Deterministic is the assertion: every recorded call identity replayed to exactly one
	// verdict. Findings holds every violation, worst (most distinct verdicts) first.
	Deterministic bool            `json:"deterministic"`
	Findings      []replayFinding `json:"findings,omitempty"`
}

// runAuditReplay reads a decision journal, folds its call-adjudication rows by call identity,
// and asserts that each identity replayed to a single verdict. Exit: 0 deterministic (the
// reproducible-trajectory witness), 1 non-determinism findings, 2 read/setup error.
func runAuditReplay(stdout, stderr io.Writer, path string, asJSON bool) int {
	// Segment-aware (#6488): a determinism claim over a rotated journal's live
	// segment would silently exclude every call adjudicated before the cut.
	rows, err := journal.ReadAllSegments(path)
	if err != nil {
		fmt.Fprintf(stderr, "fak audit replay: %v\n", err)
		return 2
	}
	rows = journal.WithoutCutAnchors(rows)
	rep := replayRows(path, rows)

	if asJSON {
		if code := encodeJSONOrFail(stdout, stderr, rep, "fak audit replay"); code != 0 {
			return code
		}
	} else {
		fmt.Fprint(stdout, renderReplayReport(rep))
	}
	if !rep.Deterministic {
		return 1
	}
	return 0
}

// replayCallKey is a recorded call's identity: the tool and its args-content digest. Two rows
// with the same key are the SAME call replayed — they MUST carry the same verdict.
type replayCallKey struct {
	tool   string
	digest string
}

type replayOutcomeKey struct {
	verdict string
	reason  string
}

type replayOutcomeAcc struct {
	count    int
	firstSeq uint64
	lastSeq  uint64
}

type replayCallAcc struct {
	label    string
	total    int
	outcomes map[replayOutcomeKey]*replayOutcomeAcc
	order    []replayOutcomeKey // first-seen order, so a tie in count renders stably
}

// replayRows is the pure fold: it takes journal rows as data and decides deterministic vs
// non-deterministic. Kept side-effect-free so the determinism logic is unit-tested without a
// file (the audit_diagnose diagnoseRows pattern).
func replayRows(path string, rows []journal.Row) replayReport {
	rep := replayReport{Path: path, Rows: len(rows), ChainLinearOK: true, Deterministic: true}

	// Chain status is context only. A clean linear verify is the single-writer happy path; a
	// break is usually a benign concurrent-writer interleave (adjudicated by `fak audit diagnose`),
	// so replay notes it and moves on rather than false-failing a fleet's shared journal.
	if len(rows) > 0 {
		if _, verr := journal.VerifyRows(rows); verr != nil {
			rep.ChainLinearOK = false
			rep.ChainNote = "chain not linearly verifiable (likely concurrent-writer interleave; run `fak audit diagnose` to tell interleave from tampering): " + verr.Error()
		}
	}

	calls := map[replayCallKey]*replayCallAcc{}
	var keyOrder []replayCallKey
	for _, r := range rows {
		if !replayIsAdjudication(r) {
			continue
		}
		rep.DecisionRows++
		digest := strings.TrimSpace(r.ArgsDigest)
		if digest == "" {
			rep.UnkeyableRows++ // an adjudicated call whose args were blob-backed with no inline digest
			continue
		}
		key := replayCallKey{tool: strings.TrimSpace(r.Tool), digest: digest}
		acc := calls[key]
		if acc == nil {
			acc = &replayCallAcc{outcomes: map[replayOutcomeKey]*replayOutcomeAcc{}}
			calls[key] = acc
			keyOrder = append(keyOrder, key)
		}
		if acc.label == "" {
			acc.label = strings.TrimSpace(r.ArgsLabel)
		}
		acc.total++
		ok := replayOutcomeKey{verdict: replayVerdictOf(r), reason: strings.TrimSpace(r.Reason)}
		oa := acc.outcomes[ok]
		if oa == nil {
			oa = &replayOutcomeAcc{}
			acc.outcomes[ok] = oa
			acc.order = append(acc.order, ok)
		}
		oa.count++
		if oa.firstSeq == 0 || r.Seq < oa.firstSeq {
			oa.firstSeq = r.Seq
		}
		if r.Seq > oa.lastSeq {
			oa.lastSeq = r.Seq
		}
	}
	rep.DistinctCalls = len(keyOrder)

	for _, key := range keyOrder {
		acc := calls[key]
		if len(acc.outcomes) < 2 {
			continue // one verdict for this call identity: reproducible
		}
		rep.Findings = append(rep.Findings, replayFinding{
			Tool:        key.tool,
			ArgsDigest:  key.digest,
			ArgsLabel:   acc.label,
			Occurrences: acc.total,
			Outcomes:    sortedReplayOutcomes(acc),
		})
	}
	rep.Deterministic = len(rep.Findings) == 0
	sortReplayFindings(rep.Findings)
	return rep
}

// replayIsAdjudication reports whether a row is the floor's decision on a proposed tool call —
// the DECIDE (allow/transform) and DENY rows keyed by the call's args. QUARANTINE / RESULT_DENY
// are result-side re-decisions on a returned payload (a different axis, keyed by result digest);
// VDSO_HIT / CAP_* are lifecycle, not call verdicts. Restricting to call adjudications keeps the
// determinism assertion about "did the floor allow/deny the same call the same way".
func replayIsAdjudication(r journal.Row) bool {
	switch strings.ToUpper(strings.TrimSpace(r.Kind)) {
	case "DECIDE", "DENY":
		return strings.TrimSpace(r.Verdict) != ""
	default:
		return false
	}
}

// replayVerdictOf is the row's verdict, falling back to its Kind if the Verdict field is blank
// (older rows) so a DENY row without an explicit verdict still keys distinctly from an ALLOW.
func replayVerdictOf(r journal.Row) string {
	if v := strings.ToUpper(strings.TrimSpace(r.Verdict)); v != "" {
		return v
	}
	return strings.ToUpper(strings.TrimSpace(r.Kind))
}

// sortedReplayOutcomes renders a call's distinct outcomes most-frequent first, breaking ties by
// first-seen order so the report is deterministic regardless of Go map iteration order.
func sortedReplayOutcomes(acc *replayCallAcc) []replayOutcome {
	seen := make(map[replayOutcomeKey]int, len(acc.order))
	for i, k := range acc.order {
		seen[k] = i
	}
	out := make([]replayOutcome, 0, len(acc.outcomes))
	for k, oa := range acc.outcomes {
		out = append(out, replayOutcome{
			Verdict:  k.verdict,
			Reason:   k.reason,
			Count:    oa.count,
			FirstSeq: oa.firstSeq,
			LastSeq:  oa.lastSeq,
		})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Count != out[j].Count {
			return out[i].Count > out[j].Count
		}
		ki := replayOutcomeKey{verdict: out[i].Verdict, reason: out[i].Reason}
		kj := replayOutcomeKey{verdict: out[j].Verdict, reason: out[j].Reason}
		return seen[ki] < seen[kj]
	})
	return out
}

// sortReplayFindings orders findings worst-first: most distinct verdicts, then most occurrences,
// then tool/digest for a stable render.
func sortReplayFindings(f []replayFinding) {
	sort.Slice(f, func(i, j int) bool {
		if len(f[i].Outcomes) != len(f[j].Outcomes) {
			return len(f[i].Outcomes) > len(f[j].Outcomes)
		}
		if f[i].Occurrences != f[j].Occurrences {
			return f[i].Occurrences > f[j].Occurrences
		}
		if f[i].Tool != f[j].Tool {
			return f[i].Tool < f[j].Tool
		}
		return f[i].ArgsDigest < f[j].ArgsDigest
	})
}

// renderReplayReport is the human report: chain status, what was replayed, and the determinism
// verdict with each non-determinism finding's diverging outcomes.
func renderReplayReport(rep replayReport) string {
	var b strings.Builder
	out := func(format string, a ...any) { fmt.Fprintf(&b, format, a...) }

	out("fak audit replay: %s\n", rep.Path)
	out("  rows           : %d\n", rep.Rows)
	if rep.ChainLinearOK {
		out("  chain (linear) : OK (verifies as one hash chain)\n")
	} else {
		out("  chain (linear) : NOT LINEAR — %s\n", rep.ChainNote)
	}
	out("  replayed       : %d adjudication row(s) over %d distinct call(s)", rep.DecisionRows, rep.DistinctCalls)
	if rep.UnkeyableRows > 0 {
		out(" (%d row(s) un-keyable: blob-backed args, no digest)", rep.UnkeyableRows)
	}
	out("\n")

	if rep.Deterministic {
		out("  determinism    : DETERMINISTIC — every recorded call replayed to one floor verdict (reproducible witness)\n")
		return b.String()
	}
	out("  determinism    : NON-DETERMINISTIC — %d call identity(s) replayed to >1 verdict\n", len(rep.Findings))
	for _, f := range rep.Findings {
		label := ""
		if f.ArgsLabel != "" {
			label = " call_label=" + f.ArgsLabel
		}
		out("    diverged: %-20s args_digest=%s%s occurrences=%d\n", f.Tool, f.ArgsDigest, label, f.Occurrences)
		for _, o := range f.Outcomes {
			reason := o.Reason
			if reason == "" {
				reason = "NONE"
			}
			out("        %-10s %-16s x%d seq=%d..%d\n", o.Verdict, reason, o.Count, o.FirstSeq, o.LastSeq)
		}
	}
	return b.String()
}
