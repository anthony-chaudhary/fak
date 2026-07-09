package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

// guard_stophook_score.go — issue #2539, spine step 6 of the trajectory-control
// epic (#2533): wire the pure turn-end / compaction folds (internal/trajctl's
// Sample and CompactionBoundary) into the guard hook processes so the curve gains a
// point every turn end and a boundary row at every context reset — live during a
// session instead of only when a scorer is run by hand.
//
// Both entry points are STRICTLY BOUNDED (a wall-clock deadline caps the whole
// load->sample->append) and FAIL-OPEN (any error, timeout, or scorer panic costs at
// most its own row — never the hook's exit code or the session). They are gated on
// an explicitly-configured ledger path (guardTrajctlEnvLedger), which the guard
// installer injects for a real session; absent it (a bare `fak guard-stophook`
// invocation, or any unrelated hook test) they are a total no-op and touch no
// ledger. Judge scorers are excluded here by construction — only the cheap set runs
// on the hook path.

const (
	// guardTrajctlEnvLedger is the absolute trajctl ledger path the guard installer
	// injects into the hook child. Empty (unset) disables turn-end/compaction
	// sampling entirely, so the folds only run for a session the guard wired.
	guardTrajctlEnvLedger = "FAK_GUARD_TRAJCTL_LEDGER"
	// guardTrajctlEnvMode opts the sampling out (value "off") even when a ledger is
	// configured — the escape hatch for a lean harness that wants no curve writes.
	guardTrajctlEnvMode = "FAK_GUARD_TRAJCTL_MODE"
	// guardTrajctlDeadline bounds the whole sampling pass so hook latency stays
	// bounded no matter how large the ledger or how slow the disk. On a timeout the
	// pass is abandoned fail-open; a torn append is skipped by the ledger parser.
	guardTrajctlDeadline = 250 * time.Millisecond
)

// guardTrajctlLedgerConfigured returns the wired ledger path, or "" when turn-end
// sampling is disabled (no ledger injected, or mode=off).
func guardTrajctlLedgerConfigured() string {
	if strings.EqualFold(strings.TrimSpace(os.Getenv(guardTrajctlEnvMode)), "off") {
		return ""
	}
	return strings.TrimSpace(os.Getenv(guardTrajctlEnvLedger))
}

// guardTrajctlLedgerDefault is the absolute default ledger path the installer
// injects: <repo root>/docs/nightrun/trajctl.jsonl. Empty when the root cannot be
// resolved, in which case the installer injects nothing and the hook stays inert.
func guardTrajctlLedgerDefault() string {
	root := repoRoot()
	if strings.TrimSpace(root) == "" {
		return ""
	}
	return filepath.Join(root, filepath.FromSlash(trajctl.DefaultLedgerRel))
}

// cheapTurnEndScorers is the cheap scorer set run at turn cadence: the W3
// witnessed-commit progress scorer and the W2 activity/progress divergence (stall)
// scorer. For an objective with a declared plan the W3 scorer yields a row every
// pass (its progress fraction, even 0), so a planned objective accumulates at least
// one point per turn end. Judge scorers are deliberately excluded (cost; opt-in
// only, a separate child).
func cheapTurnEndScorers() []trajctl.Scorer {
	return []trajctl.Scorer{trajctl.CommitProgressScorer{}, trajctl.ActivityDivergenceScorer{}}
}

// scoreTurnEndFailOpen runs the cheap scorers over every OPEN objective at a turn
// end and appends the produced rows to the wired ledger. Called from the Stop hook.
// Bounded + fail-open; a no-op when no ledger is configured.
func scoreTurnEndFailOpen(stderr io.Writer, stamp trajctl.Stamp, nowMillis int64) {
	ledger := guardTrajctlLedgerConfigured()
	if ledger == "" {
		return
	}
	guardTrajctlSampleBounded(stderr, "turn-end", ledger, nowMillis, func(st trajctl.State, win trajctl.EvidenceWindow) trajctl.TurnSample {
		return trajctl.Sample(st.Objectives, cheapTurnEndScorers(), win, stamp)
	})
}

// appendCompactionBoundaryFailOpen appends the PreCompact twin: one boundary-marker
// row per OPEN objective so a curve reader can see the context reset. Called from
// the PreCompact hook. Bounded + fail-open; a no-op when no ledger is configured.
func appendCompactionBoundaryFailOpen(stderr io.Writer, stamp trajctl.Stamp, nowMillis int64) {
	ledger := guardTrajctlLedgerConfigured()
	if ledger == "" {
		return
	}
	guardTrajctlSampleBounded(stderr, "compaction", ledger, nowMillis, func(st trajctl.State, win trajctl.EvidenceWindow) trajctl.TurnSample {
		return trajctl.CompactionBoundary(st.Objectives, win, stamp)
	})
}

// guardTrajctlSampleResult carries a bounded sampling pass's outcome back over the
// done channel.
type guardTrajctlSampleResult struct {
	rows     int
	failures []trajctl.SampleFailure
	err      error
}

// guardTrajctlSampleBounded folds the ledger, builds the evidence window, produces a
// TurnSample via build, and appends its rows — all inside a goroutine capped by a
// wall-clock deadline. It NEVER returns an error to the caller: on a deadline it
// abandons the in-flight pass (the goroutine's late append, if any, lands a torn
// line the parser skips), and it recovers any panic so a bug in a fold costs at most
// its own row rather than the hook. The single side effect is the ledger append; the
// stderr line is advisory (observability), not a decision.
func guardTrajctlSampleBounded(stderr io.Writer, label, ledger string, nowMillis int64, build func(trajctl.State, trajctl.EvidenceWindow) trajctl.TurnSample) {
	done := make(chan guardTrajctlSampleResult, 1)
	go func() {
		defer func() {
			if r := recover(); r != nil {
				done <- guardTrajctlSampleResult{err: fmt.Errorf("panic: %v", r)}
			}
		}()
		st := trajctl.Fold(trajctl.ReadLedgerFile(ledger))
		win := trajctl.EvidenceWindow{PriorScores: st.Scores, UnixMillis: nowMillis}
		sample := build(st, win)
		n, err := trajctl.AppendSample(ledger, sample)
		done <- guardTrajctlSampleResult{rows: n, failures: sample.Failures, err: err}
	}()
	select {
	case <-time.After(guardTrajctlDeadline):
		fmt.Fprintf(stderr, "fak guard: trajctl %s scoring timed out (>%s); continuing fail-open\n", label, guardTrajctlDeadline)
	case res := <-done:
		switch {
		case res.err != nil:
			fmt.Fprintf(stderr, "fak guard: trajctl %s scoring skipped (fail-open): %v\n", label, res.err)
		case res.rows > 0:
			fmt.Fprintf(stderr, "fak guard: trajctl %s scored %d row(s) into %s", label, res.rows, ledger)
			if len(res.failures) > 0 {
				fmt.Fprintf(stderr, " (%d scorer failure(s) swallowed)", len(res.failures))
			}
			fmt.Fprintln(stderr)
		}
	}
}

// readHookStdin drains a hook's stdin payload once (bounded at 1 MiB) so every
// field consumer parses the SAME bytes — a hook's stdin is not rewindable. A nil
// reader or read error yields nil; every parse below tolerates that.
func readHookStdin(stdin io.Reader) []byte {
	if stdin == nil {
		return nil
	}
	b, err := io.ReadAll(io.LimitReader(stdin, 1<<20))
	if err != nil {
		return nil
	}
	return b
}

// parseHookSessionID best-effort parses the session_id a Claude Code hook payload
// carries on stdin (Stop and PreCompact both include it). An empty body or parse
// miss returns "" — the stamp is advisory attribution, never a gate.
func parseHookSessionID(payload []byte) string {
	if len(payload) == 0 {
		return ""
	}
	var p struct {
		SessionID string `json:"session_id"`
	}
	if json.Unmarshal(payload, &p) != nil {
		return ""
	}
	return p.SessionID
}
