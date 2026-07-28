package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/loopfleet"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/relay"
	"github.com/anthony-chaudhary/fak/internal/superloop"
)

// TestMemberProgressSpinningEndToEnd drives the #4956 shell seam end to end over a
// hermetic intent ledger: a ticking throughput loop whose bound ledger records no
// advanced verified step reads SPINNING with the closed relay reason and carries the
// one-unit progress debt term in loopDebt — where the pre-#4956 loopDebt read it
// clean (0). A re-verifiable step flips it to advancing/clean, and the live default
// (no bound ledger anchor) stays unmeasured and surface-only: never a fabricated
// zero, never debt.
func TestMemberProgressSpinningEndToEnd(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, ".dos", "runs"), 0o755); err != nil {
		t.Fatal(err)
	}
	ledger := filepath.Join(root, ".dos", "runs", "dispatch.jsonl")

	old := progressLedgerRef
	t.Cleanup(func() { progressLedgerRef = old })
	progressLedgerRef = func(_, kind string) string {
		if kind == "dispatch" {
			return ".dos/runs/dispatch.jsonl"
		}
		return ""
	}

	c := &superloopCollector{root: root}
	m := superloop.Member{Kind: superloop.KindLoop, Ref: "dispatch"}
	live := loopfleet.LoopHealth{Kind: "dispatch", State: loopmgr.HealthLive}

	// Ticking + a real ledger with NO re-verifiable step ref: verified-empty is an
	// honest zero, and zero advance on a throughput loop is SPINNING.
	if err := os.WriteFile(ledger, []byte("{\"note\":\"tick, nothing witnessed\"}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	prog, reason := c.memberProgress(m, live)
	if prog != superloop.ProgressSpinning || reason != relay.ReasonNoProgress {
		t.Fatalf("ticking + zero verified steps = (%q, %q), want (spinning, %s)", prog, reason, relay.ReasonNoProgress)
	}
	if got := loopDebt(live, prog, ""); got != 1 {
		t.Fatalf("loopDebt(live, spinning) = %d, want 1 (the #4956 progress term; pre-#4956 this read clean)", got)
	}

	// A re-verifiable step lands in the ledger: the high-water rises, the loop is
	// advancing, and the progress term vanishes.
	if err := os.WriteFile(ledger, []byte("{\"ref\":\"#123\",\"note\":\"closed #123\"}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	prog, reason = c.memberProgress(m, live)
	if prog != superloop.ProgressAdvancing || reason != "" {
		t.Fatalf("ticking + verified step = (%q, %q), want (advancing, \"\")", prog, reason)
	}
	if got := loopDebt(live, prog, ""); got != 0 {
		t.Fatalf("loopDebt(live, advancing) = %d, want 0", got)
	}

	// A stale loop keeps its one-unit liveness debt AND stacks the progress term.
	stale := loopfleet.LoopHealth{Kind: "dispatch", State: loopmgr.HealthStale}
	if err := os.WriteFile(ledger, []byte("{\"note\":\"tick, nothing witnessed\"}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	prog, _ = c.memberProgress(m, stale)
	if prog != superloop.ProgressSpinning {
		t.Fatalf("stale ticking loop progress = %q, want spinning (stale still ticks)", prog)
	}
	if got := loopDebt(stale, prog, ""); got != 2 {
		t.Fatalf("loopDebt(stale, spinning) = %d, want 2 (liveness unit + progress term)", got)
	}

	// The live default: no bound intent-ledger anchor. ReadVerifiedProgress fails
	// closed to ProgressUnknown and the fold surfaces unmeasured — shown, no reason
	// token, no debt (surface-only; never a fabricated zero).
	progressLedgerRef = func(_, _ string) string { return "" }
	prog, reason = c.memberProgress(m, live)
	if prog != superloop.ProgressUnmeasured || reason != "" {
		t.Fatalf("unbound anchor = (%q, %q), want (unmeasured, \"\")", prog, reason)
	}
	if got := loopDebt(live, prog, ""); got != 0 {
		t.Fatalf("loopDebt(live, unmeasured) = %d, want 0 (surface-only)", got)
	}

	// A dark loop reads no progress axis at all — its urgency is the liveness
	// verdict, never double-counted.
	dark := loopfleet.LoopHealth{Kind: "dispatch", State: loopmgr.HealthDark, Dark: true}
	if prog, reason = c.memberProgress(m, dark); prog != "" || reason != "" {
		t.Fatalf("dark loop progress axis = (%q, %q), want unread (\"\", \"\")", prog, reason)
	}
}
