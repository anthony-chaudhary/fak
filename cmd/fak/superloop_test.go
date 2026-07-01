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
