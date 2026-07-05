package main

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/superloop"
)

func TestSuperloopWalkAcceptsFlagsAfterName(t *testing.T) {
	root := t.TempDir()
	var out, errb bytes.Buffer
	// The walk of an empty temp dir is legitimately UNSATISFIED (no baseline, no
	// ledgers), so an honest exit is 1 — not a flag-parse error (2). This test pins
	// that flags AFTER the positional name are still parsed: assert it is not a usage
	// error (2) and that the JSON report was emitted (proof the flags took effect).
	code := runSuperloop(&out, &errb, []string{"walk", "manage-benchmarks", "--workspace", root, "--json"})
	if code == 2 {
		t.Fatalf("flags after name were not parsed (usage error): stderr=%s", errb.String())
	}
	var rep superloop.WalkReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("walk json: %v\n%s", err, out.String())
	}
	if rep.Schema != superloop.WalkSchema || rep.Name != "manage-benchmarks" {
		t.Fatalf("unexpected report: schema=%q name=%q", rep.Schema, rep.Name)
	}

	out.Reset()
	errb.Reset()
	code = runSuperloop(&out, &errb, []string{"walk", "manage-benchmarks", "--workspace=" + filepath.ToSlash(root), "--json"})
	if code == 2 {
		t.Fatalf("walk with --workspace= was a usage error: stderr=%s", errb.String())
	}
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("walk with --workspace= json: %v\n%s", err, out.String())
	}
}

// TestSuperloopWalkTendDescends exercises the live recursion: walking the root
// "tend" intent in an empty workspace DESCENDS all three sub-super-loops inline —
// each arrives as a MEASURED member (walked, not a container pointer), each is
// honestly unsatisfied (no baseline, no ledgers) and so carries at least one unit
// of folded debt, and the root walk exits 1.
func TestSuperloopWalkTendDescends(t *testing.T) {
	root := t.TempDir()
	var out, errb bytes.Buffer
	code := runSuperloop(&out, &errb, []string{"walk", "tend", "--workspace", root, "--json"})
	if code != 1 {
		t.Fatalf("tend walk of an empty workspace must be honestly unsatisfied (exit 1), got %d: %s", code, errb.String())
	}
	var rep superloop.WalkReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("walk json: %v\n%s", err, out.String())
	}
	if rep.Name != "tend" {
		t.Fatalf("report name = %q", rep.Name)
	}
	if rep.Walked != rep.Members || rep.Unmeasured != 0 {
		t.Fatalf("every sub-super-loop must be DESCENDED as measured: walked=%d members=%d unmeasured=%d",
			rep.Walked, rep.Members, rep.Unmeasured)
	}
	for _, st := range rep.Statuses {
		if st.Member.Kind != superloop.KindSuperloop {
			t.Errorf("tend member %q should be a sub-super-loop, got %q", st.Member.Ref, st.Member.Kind)
			continue
		}
		if !st.Measured || st.Container {
			t.Errorf("sub %q must arrive measured (descended), got measured=%v container=%v", st.Member.Ref, st.Measured, st.Container)
		}
		if !strings.Contains(st.Detail, "descended:") {
			t.Errorf("sub %q detail should carry the sub-walk fold, got %q", st.Member.Ref, st.Detail)
		}
		if st.Debt < 1 {
			t.Errorf("sub %q read clean in an EMPTY workspace (debt %d) — an unsatisfied sub must carry debt", st.Member.Ref, st.Debt)
		}
	}
	if len(rep.Worklist) == 0 || !strings.Contains(rep.Worklist[0].Action, "fak superloop walk") {
		t.Errorf("worst-first action should point at descending a sub-intent, got %+v", rep.Worklist)
	}
}

// TestSuperloopWalkJSONCarriesBudget is the end-to-end budget witness: `walk --json`
// emits the intent's declared generation budget as four dimension rows (matching the
// registry's declared caps), divides each budgeted cap down across the worklist
// members, and annotates every worklist member with its share. The unbudgeted review
// dimension arrives as a HOLD (a reason, not a blank), and the held dimension shows
// up in each member's Allocation.Held. manage-benchmarks is chosen because it carries
// both budgeted dimensions and a held one.
func TestSuperloopWalkJSONCarriesBudget(t *testing.T) {
	root := t.TempDir()
	var out, errb bytes.Buffer
	code := runSuperloop(&out, &errb, []string{"walk", "manage-benchmarks", "--workspace", root, "--json"})
	if code == 2 {
		t.Fatalf("walk was a usage error: %s", errb.String())
	}
	var rep superloop.WalkReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("walk json: %v\n%s", err, out.String())
	}

	// The declared caps in the report must be exactly what the registry declares.
	s, ok := superloop.Lookup("manage-benchmarks")
	if !ok {
		t.Fatal("manage-benchmarks not registered")
	}
	wantDims := []string{superloop.BudgetTime, superloop.BudgetTokens, superloop.BudgetWorkers, superloop.BudgetReview}
	wantTotal := map[string]int{
		superloop.BudgetTime:    s.Budget.MaxMinutes,
		superloop.BudgetTokens:  s.Budget.TokenCeiling,
		superloop.BudgetWorkers: s.Budget.MaxWorkers,
		superloop.BudgetReview:  s.Budget.ReviewSlots,
	}
	if len(rep.Budget) != len(wantDims) {
		t.Fatalf("budget rows = %d, want %d", len(rep.Budget), len(wantDims))
	}
	rowByDim := map[string]superloop.BudgetRow{}
	for i, row := range rep.Budget {
		if row.Dimension != wantDims[i] {
			t.Errorf("budget row[%d] = %q, want %q (dimension order)", i, row.Dimension, wantDims[i])
		}
		if row.Total != wantTotal[row.Dimension] {
			t.Errorf("row %q declared total = %d, want registry cap %d", row.Dimension, row.Total, wantTotal[row.Dimension])
		}
		rowByDim[row.Dimension] = row
	}

	// review is declared 0 in the registry -> it must arrive as a HOLD, never budgeted.
	if review := rowByDim[superloop.BudgetReview]; review.Budgeted || review.Hold == "" {
		t.Errorf("review must be an unbudgeted HOLD, got budgeted=%v hold=%q", review.Budgeted, review.Hold)
	}

	if len(rep.Worklist) == 0 {
		t.Fatal("an empty-workspace walk of manage-benchmarks must have a worklist to allocate across")
	}
	for _, it := range rep.Worklist {
		a := it.Allocation
		// Each budgeted dimension's share equals the row's per-member value.
		if a.MaxMinutes != rowByDim[superloop.BudgetTime].PerMember {
			t.Errorf("member %q minutes share = %d, want row per-member %d", it.Member.Ref, a.MaxMinutes, rowByDim[superloop.BudgetTime].PerMember)
		}
		if a.TokenCeiling != rowByDim[superloop.BudgetTokens].PerMember {
			t.Errorf("member %q tokens share = %d, want %d", it.Member.Ref, a.TokenCeiling, rowByDim[superloop.BudgetTokens].PerMember)
		}
		// The held review dimension must be listed under Held and allocate nothing.
		held := false
		for _, h := range a.Held {
			if h == superloop.BudgetReview {
				held = true
			}
		}
		if !held {
			t.Errorf("member %q must list the unbudgeted review dimension under Held, got %v", it.Member.Ref, a.Held)
		}
		if a.ReviewSlots != 0 {
			t.Errorf("member %q allocates a held dimension: review=%d", it.Member.Ref, a.ReviewSlots)
		}
	}

	// Floor honesty carries through the shell: no budgeted dimension over-allocates.
	n := len(rep.Worklist)
	for _, row := range rep.Budget {
		if row.Budgeted && row.PerMember*n > row.Total {
			t.Errorf("dimension %q over-allocates through the shell: %d*%d > %d", row.Dimension, row.PerMember, n, row.Total)
		}
	}
}

// TestRenderSuperloopWalkShowsBudget pins the human surface: the non-JSON walk prints
// the declared budget block with the per-dimension caps and marks the held dimension
// HELD, so an operator reading the terminal sees the reservation without --json.
func TestRenderSuperloopWalkShowsBudget(t *testing.T) {
	rep := superloop.WalkReport{
		Name:       "manage-benchmarks",
		Verdict:    "ACTION",
		Finding:    "superloop_debt",
		Reason:     "test",
		NextAction: "descend",
		Budget: []superloop.BudgetRow{
			{Dimension: superloop.BudgetTime, Unit: "minutes", Stream: "gen/next", Budgeted: true, Total: 20, Members: 2, PerMember: 10},
			{Dimension: superloop.BudgetReview, Unit: "slots", Stream: "gen/next", Budgeted: false, Hold: "unbudgeted — held for later-horizon work"},
		},
	}
	var out bytes.Buffer
	renderSuperloopWalk(&out, rep)
	got := out.String()
	for _, want := range []string{"budget gen/next", "DIMENSION", "time", "20 minutes", "HELD"} {
		if !strings.Contains(got, want) {
			t.Errorf("human render missing %q:\n%s", want, got)
		}
	}
}

func TestRenderSuperloopWalkMarksSurfaceAsDescend(t *testing.T) {
	rep := superloop.WalkReport{
		Name:       "manage-benchmarks",
		Verdict:    "ACTION",
		Finding:    "superloop_debt",
		Reason:     "test",
		NextAction: "descend",
		Worklist: []superloop.WorkItem{{
			Rank: 1,
			Member: superloop.Member{
				Kind: superloop.KindSurface,
				Ref:  "fak bench-loop status",
			},
			Container: true,
			Action:    "enter `fak bench-loop status`",
			Detail:    "DESCEND - domain fold",
		}},
	}
	var out bytes.Buffer
	renderSuperloopWalk(&out, rep)
	if !strings.Contains(out.String(), "surface fak bench-loop status") || !strings.Contains(out.String(), "→") {
		t.Fatalf("surface member should render as a descend pointer:\n%s", out.String())
	}
}
