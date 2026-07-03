// relaunch.go — the pure decision core of the RELAUNCH-OUTCOME audit (the Go port of
// tools/resume_relaunch_audit.py's verdict/fold half; #1343/#1346 scope bullet
// "fak resume audit").
//
// # The question it answers
//
// The sweep answers "what failure is this crashed session in?" (crash state). This
// answers the downstream question the sweep cannot: of the sessions a relaunch was
// ATTEMPTED on (the ledger records the attempt), which actually TOOK — advanced past
// their error — and which are still STRANDED on it? The ledger is a self-report of the
// attempt; it does not prove the outcome. This verifies the outcome from the transcript,
// never the ledger's word — the same distrust discipline as dos_verify.
//
// # The verdict rule (inherited verbatim from the Python)
//
// A session is RELAUNCHED_OK iff its superset copy's last real (non-error, non-banner)
// assistant turn is NEWER than its last error record — record order + ISO-8601 timestamp
// string comparison, exactly the Python's keying. Else it is STRANDED on that error,
// classified by the shared sessionsignals.TerminalFailure taxonomy (AUTH/LIMIT/API_ERR,
// "OTHER" when the text matches none). NEVER_WORKED if it produced no timestamped real
// turn at all; NO_TRANSCRIPT (shell-assigned) if no copy is on disk.
//
// Pure by construction: the I/O shell reads the ledger and the transcript copies; this
// leaf only folds. Same records in, same verdict out — no clock, no filesystem.
package sweep

import (
	"bufio"
	"encoding/json"
	"io"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/sessionsignals"
)

// The closed relaunch-verdict vocabulary, ordered by operator urgency (see RelaunchOrder).
const (
	// VerdictRelaunchedOK: the transcript advanced past its last error — the relaunch took.
	VerdictRelaunchedOK = "RELAUNCHED_OK"
	// VerdictStranded: the last error record is still the newest thing — the session is
	// stuck on it, classified by Kind (AUTH / LIMIT / API_ERR / OTHER).
	VerdictStranded = "STRANDED"
	// VerdictNeverWorked: no timestamped real assistant turn at all — it never produced work.
	VerdictNeverWorked = "NEVER_WORKED"
	// VerdictNoTranscript: the ledger names a sid with no on-disk copy (shell-assigned:
	// this leaf never sees the filesystem, so it can only be handed the fact).
	VerdictNoTranscript = "NO_TRANSCRIPT"
)

// relaunchBannerMark is the synthetic usage-limit banner text: a record carrying it is
// an error for verdict purposes even when the error channel bits are unset, because a
// re-capped resume writes the banner as an ordinary assistant turn.
const relaunchBannerMark = "hit your session limit"

// RelaunchResult is the transcript-derived outcome of one session's relaunch attempts —
// the same fields the Python emitted, so a consumer of the machine record sees an
// unchanged shape.
type RelaunchResult struct {
	Verdict string `json:"verdict"`
	// Kind classifies a STRANDED verdict's error (AUTH / LIMIT / API_ERR / OTHER); empty
	// for every other verdict.
	Kind       string `json:"kind"`
	LastRealTS string `json:"last_real_ts"`
	LastErrTS  string `json:"last_err_ts"`
	// Evidence is the clipped error text that drove a STRANDED / NEVER_WORKED verdict.
	Evidence string `json:"evidence"`
}

// RelaunchVerdict folds one transcript's ordered records into the relaunch outcome: did
// this session advance PAST its last error? Keyed off record order + timestamps — the
// last error/banner record vs the last real (non-error, non-banner) assistant turn.
// ISO-8601 UTC timestamps compare correctly as strings, the same order the Python
// relied on.
func RelaunchVerdict(recs []Record) RelaunchResult {
	var lastErr, lastErrTS, lastRealTS string
	for _, r := range recs {
		if r.IsError || strings.Contains(strings.ToLower(r.Text), relaunchBannerMark) {
			lastErr, lastErrTS = r.Text, r.Timestamp
		} else if r.Role == "assistant" && strings.TrimSpace(r.Text) != "" {
			lastRealTS = r.Timestamp
		}
	}
	if lastRealTS == "" {
		return RelaunchResult{Verdict: VerdictNeverWorked, LastErrTS: lastErrTS,
			Evidence: clip(lastErr, 90)}
	}
	if lastErrTS == "" || lastRealTS > lastErrTS {
		return RelaunchResult{Verdict: VerdictRelaunchedOK, LastRealTS: lastRealTS,
			LastErrTS: lastErrTS}
	}
	kind, _ := sessionsignals.TerminalFailure(lastErr)
	if kind == "" {
		kind = "OTHER"
	}
	return RelaunchResult{Verdict: VerdictStranded, Kind: kind, LastRealTS: lastRealTS,
		LastErrTS: lastErrTS, Evidence: clip(lastErr, 90)}
}

// RelaunchOrder is the operator-urgency sort rank of a verdict: still-broken first
// (STRANDED, then the never-produced and missing-transcript oddities), healthy last.
// Unknown verdicts sort last.
func RelaunchOrder(verdict string) int {
	switch verdict {
	case VerdictStranded:
		return 0
	case VerdictNeverWorked:
		return 1
	case VerdictNoTranscript:
		return 2
	case VerdictRelaunchedOK:
		return 3
	}
	return 9
}

// SupersetIndex resolves which copy is the SUPERSET: latest last-ts, then most records —
// NOT file mtime, because a re-capped resume rewrites only the banner and bumps mtime on
// a stale PREFIX copy (the same rule Classify applies). Returns -1 for no copies.
func SupersetIndex(copies []Copy) int {
	if len(copies) == 0 {
		return -1
	}
	best := 0
	for i := 1; i < len(copies); i++ {
		bTS, iTS := lastTS(copies[best].Records), lastTS(copies[i].Records)
		if iTS > bTS || (iTS == bTS && len(copies[i].Records) > len(copies[best].Records)) {
			best = i
		}
	}
	return best
}

// LedgerActions folds the resume ledger into sid -> the sorted distinct actions recorded
// for it — the audit's roster of "sessions a relaunch was attempted on". A row without an
// action still counts as an attempt ("?", the Python's placeholder); malformed lines are
// skipped, matching the ledger tolerance RecentlyResumed applies.
func LedgerActions(r io.Reader) map[string][]string {
	acts := map[string]map[string]bool{}
	if r == nil {
		return map[string][]string{}
	}
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var rec struct {
			Session string `json:"session"`
			Action  string `json:"action"`
		}
		if json.Unmarshal([]byte(line), &rec) != nil || rec.Session == "" {
			continue
		}
		action := rec.Action
		if action == "" {
			action = "?"
		}
		if acts[rec.Session] == nil {
			acts[rec.Session] = map[string]bool{}
		}
		acts[rec.Session][action] = true
	}
	out := make(map[string][]string, len(acts))
	for sid, set := range acts {
		list := make([]string, 0, len(set))
		for a := range set {
			list = append(list, a)
		}
		sort.Strings(list)
		out[sid] = list
	}
	return out
}
