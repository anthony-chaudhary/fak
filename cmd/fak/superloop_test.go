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
			Action: "enter `fak bench-loop status`",
			Detail: "DESCEND - domain fold",
		}},
	}
	var out bytes.Buffer
	renderSuperloopWalk(&out, rep)
	if !strings.Contains(out.String(), "surface fak bench-loop status") || !strings.Contains(out.String(), "→") {
		t.Fatalf("surface member should render as a descend pointer:\n%s", out.String())
	}
}
