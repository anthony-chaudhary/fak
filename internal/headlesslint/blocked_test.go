package headlesslint

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestScanNextAction exercises both arms of the next-action fold: a terminal-block
// summary with no named next action is refused, while a blocked summary that names one
// (or cites a ticket) and a summary with no block grammar at all pass clean. The cases
// double as the fold's re-derivable corpus (blocked cases are stop-shape-specific, so
// they live here rather than in fixture.go).
func TestScanNextAction(t *testing.T) {
	cases := []struct {
		name    string
		summary string
		want    string
	}{
		// (a) A bare stop: the block is stated, nothing is to be executed.
		{"bare-blocked-terminal", "Blocked on the gateway handshake. Nothing further to try here.", NextActionTerminalBlock},
		{"cannot-proceed-terminal", "I can't proceed without the staging credentials, so the run stops here.", NextActionTerminalBlock},
		{"stuck-on-terminal", "Stuck on the admission fence and out of ranked candidates.", NextActionTerminalBlock},
		{"no-path-forward-terminal", "No path forward on the vendor quota; an operator has to unblock this.", NextActionTerminalBlock},
		{"waiting-on-blocker-terminal", "Waiting on a blocker that never clears, with no disposition named.", NextActionTerminalBlock},
		{"externally-blocked-terminal", "Externally blocked by the upstream outage.", NextActionTerminalBlock},
		// (b) Block named, disposition named: a described next step, not a stop.
		{"blocked-with-next-action", "Blocked by a live peer lease on item 3; next action: spawn a disjoint subtask on item 4.", NextActionClean},
		{"cannot-proceed-stated-intent", "Cannot proceed on the lease fabric until the peer drains; will retry once the peer reports done.", NextActionClean},
		{"externally-blocked-routes-around", "Externally blocked by the vendor rate limit; routing around by deferring that lane.", NextActionClean},
		{"bare-blocked-with-ticket", "Blocked on the peer's merge; the unblock recipe is tracked in #4821.", NextActionClean},
		{"blocked-then-handed-off", "Stuck on the reserved seat, so I handed off the billing lane to the queue owner.", NextActionClean},
		// (c) A block whose next action is a cited ticket.
		{"cannot-proceed-tracked-in-ticket", "Cannot proceed on the lease fabric until the peer drains; tracked in #1234.", NextActionClean},
		// (d) No block language at all.
		{"plain-completion-clean", "Shipped the retry fix and pushed abc123. go test ./internal/retry is green.", NextActionClean},
		{"bulleted-next-step-clean", "- Head 4 entered, pushed def567\n- Next: land the docs (#4900)", NextActionClean},
		{"empty-clean", "", NextActionClean},
		{"blank-only-clean", "\n\n   \n", NextActionClean},
		// The next action may sit anywhere in the summary — the fold reads the whole turn.
		{"block-line-one-next-action-line-three", "Blocked on the live peer lease.\nTests for the bypass path are green.\nNext action: queue item 4 behind it.", NextActionClean},
		{"multi-line-terminal-block", "Blocked on the shared lease.\nStuck on the stale reader.\nNo path forward from here.", NextActionTerminalBlock},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rep := ScanNextAction(tc.summary)
			if rep.Verdict != tc.want {
				t.Fatalf("ScanNextAction(%q) verdict = %q, want %q (blocked=%d, next=%q, hits=%+v)",
					tc.name, rep.Verdict, tc.want, rep.Blocked, rep.Next, rep.Hits)
			}
			if rep.Schema != NextActionSchema {
				t.Errorf("Schema=%q, want %q", rep.Schema, NextActionSchema)
			}
			if rep.Doctrine != NextActionDoctrine {
				t.Errorf("Doctrine=%q, want %q — the report must carry the rule it enforces", rep.Doctrine, NextActionDoctrine)
			}
			if rep.Refused() != (tc.want == NextActionTerminalBlock) {
				t.Errorf("Refused()=%v does not agree with verdict %q", rep.Refused(), rep.Verdict)
			}
			if rep.Blocked != len(rep.Hits) {
				t.Errorf("Blocked=%d does not match %d recorded hits", rep.Blocked, len(rep.Hits))
			}
			if tc.want == NextActionClean && rep.Blocked == 0 && rep.Next != "" {
				t.Errorf("no block grammar was found, so Next must be empty, got %q", rep.Next)
			}
			if rep.Verdict == NextActionTerminalBlock {
				if rep.Resolve == "" {
					t.Error("a refused report must carry a resolve line")
				}
				if rep.Next != "" {
					t.Errorf("a refused report cannot name a next action, got %q", rep.Next)
				}
			}
		})
	}
}

// TestScanNextActionCleanWhenBlockedRecordsWhy: a clean verdict on a blocked summary is
// only honest if it says WHICH phrase counted as the next action — otherwise "clean"
// reads as "no block was seen". The report is the audit trail for the whole arm.
func TestScanNextActionCleanWhenBlockedRecordsWhy(t *testing.T) {
	rep := ScanNextAction("Blocked by the reserved seat; queueing item 5 behind it.")
	if rep.Blocked != 1 {
		t.Fatalf("expected one block hit, got %d", rep.Blocked)
	}
	if rep.Verdict != NextActionClean {
		t.Fatalf("verdict = %q, want %q — a named disposition discharges the block", rep.Verdict, NextActionClean)
	}
	if rep.Next == "" {
		t.Fatal("a clean-but-blocked report must record the next action that discharged it")
	}
	if rep.Resolve != "" {
		t.Errorf("a clean report carries no resolve line, got %q", rep.Resolve)
	}
}

// TestScanNextActionDoesNotCountTicketWords: the next-action test is a REACHABLE action —
// a bare "#1234" citation qualifies because the work is tracked somewhere, but the word
// "filed" alone is not a next action. Keeping hasTicketRef's wider trigger out of this
// fold keeps it fail-closed.
func TestScanNextActionDoesNotCountTicketWords(t *testing.T) {
	rep := ScanNextAction("Blocked on the admission fence; a note was filed about it earlier.")
	if rep.Verdict != NextActionTerminalBlock {
		t.Fatalf("verdict = %q, want %q — the word \"filed\" is not a named next action", rep.Verdict, NextActionTerminalBlock)
	}
}

// TestScanNextActionIsPureAndDeterministic: the fold takes no clock, no I/O and no
// caller state, so the same summary must serialize identically every time — including the
// hit ORDER, which a caller routing on line numbers depends on.
func TestScanNextActionIsPureAndDeterministic(t *testing.T) {
	const summary = "Blocked on the live peer lease.\nTests are green.\nStuck on nothing else."
	first, err := json.Marshal(ScanNextAction(summary))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	for i := 0; i < 4; i++ {
		b, err := json.Marshal(ScanNextAction(summary))
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		if string(b) != string(first) {
			t.Fatalf("run %d differs:\n got %s\nwant %s", i, b, first)
		}
	}
}

// TestScanNextActionHitLinesAreFaithful: the recorded hit must point at the line that
// actually carries the block grammar, so a Stop hook can echo the offending line.
func TestScanNextActionHitLinesAreFaithful(t *testing.T) {
	rep := ScanNextAction("Shipped the parser, pushed abc123.\n\nBlocked on the stale reader; no disposition named.")
	if rep.Blocked != 1 {
		t.Fatalf("expected one block hit, got %d (%+v)", rep.Blocked, rep.Hits)
	}
	h := rep.Hits[0]
	if h.Line != 3 {
		t.Errorf("hit line = %d, want 3", h.Line)
	}
	if !strings.Contains(h.Match, "Blocked") {
		t.Errorf("hit match = %q, want the original-case phrase", h.Match)
	}
	if !strings.Contains(h.Excerpt, "stale reader") {
		t.Errorf("hit excerpt = %q, want the offending line", h.Excerpt)
	}
}

// TestNextActionDoctrineBindsAgentsMd couples code to doctrine: the fold quotes the
// AGENTS.md next-action rule verbatim, and this asserts AGENTS.md still carries that exact
// line. If the doctrine text moves, this reds — forcing the constant and the rule to stay
// in lockstep rather than drifting silently. Mirrors TestLeftoversDoctrineBindsAgentsMd and
// TestClosingDoctrineBindsAgentsMd.
func TestNextActionDoctrineBindsAgentsMd(t *testing.T) {
	path := filepath.Join("..", "..", "AGENTS.md")
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	if !strings.Contains(string(b), NextActionDoctrine) {
		t.Fatalf("AGENTS.md must carry the doctrine line %q that ScanNextAction binds to (code↔doctrine coupling broke)", NextActionDoctrine)
	}
}
