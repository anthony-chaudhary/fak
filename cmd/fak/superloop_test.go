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
	code := runSuperloop(&out, &errb, []string{"walk", "manage-benchmarks", "--workspace", root, "--json"})
	if code != 0 {
		t.Fatalf("walk code=%d stderr=%s", code, errb.String())
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
	if code != 0 {
		t.Fatalf("walk with --workspace= code=%d stderr=%s", code, errb.String())
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
