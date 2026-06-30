package propagationscore

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/issuecontract"
)

func probe(verb string, exists bool, adopted map[string]bool) Probe {
	return Probe{
		Member:  Member{Verb: verb, CmdFile: "cmd/fak/" + verb + ".go", PkgDir: "internal/" + verb},
		Exists:  exists,
		Adopted: adopted,
	}
}

// A declared convention makes EVERY laggard HARD debt regardless of adoption count -- even one
// adopter out of three still reds the two stragglers, because the standard is explicit.
func TestDeclaredConventionIsHardBelowQuorum(t *testing.T) {
	c := Convention{Key: "kernel", Short: "the kernel", Label: "ride the kernel", Declared: true, Source: "doc"}
	probes := []Probe{
		probe("a", true, map[string]bool{"kernel": true}),
		probe("b", true, map[string]bool{"kernel": false}),
		probe("c", true, map[string]bool{"kernel": false}),
	}
	k, gaps := kpiForConvention(c, probes)
	if len(k.Defects) != 2 {
		t.Fatalf("declared convention: want 2 HARD defects, got %d (%v)", len(k.Defects), k.Defects)
	}
	if len(k.Soft) != 0 {
		t.Fatalf("declared convention must not emit SOFT, got %v", k.Soft)
	}
	if len(gaps) != 2 {
		t.Fatalf("want 2 fan-out gaps, got %d", len(gaps))
	}
}

// A non-declared convention is HARD only once it clears the quorum; below it the stragglers are
// an emerging SOFT signal (never debt, never a gap to fan out).
func TestEmergingConventionIsSoftNotHard(t *testing.T) {
	c := Convention{Key: "compare", Short: "--compare", Label: "expose --compare"}
	belowQuorum := []Probe{ // 1/3 adoption < 0.5
		probe("a", true, map[string]bool{"compare": true}),
		probe("b", true, map[string]bool{"compare": false}),
		probe("c", true, map[string]bool{"compare": false}),
	}
	k, gaps := kpiForConvention(c, belowQuorum)
	if len(k.Defects) != 0 || len(gaps) != 0 {
		t.Fatalf("below quorum: want 0 HARD/0 gaps, got %d/%d", len(k.Defects), len(gaps))
	}
	if len(k.Soft) != 2 {
		t.Fatalf("below quorum: want 2 SOFT, got %d (%v)", len(k.Soft), k.Soft)
	}
}

// Once a non-declared convention is past the quorum, the remaining laggards become HARD + gaps.
func TestQuorumConventionIsHard(t *testing.T) {
	c := Convention{Key: "compare", Short: "--compare", Label: "expose --compare"}
	pastQuorum := []Probe{ // 2/3 adoption >= 0.5
		probe("a", true, map[string]bool{"compare": true}),
		probe("b", true, map[string]bool{"compare": true}),
		probe("c", true, map[string]bool{"compare": false}),
	}
	k, gaps := kpiForConvention(c, pastQuorum)
	if len(k.Defects) != 1 || len(gaps) != 1 {
		t.Fatalf("past quorum: want 1 HARD/1 gap, got %d/%d", len(k.Defects), len(gaps))
	}
}

// Full adoption is clean: no defects, no soft, no gaps, 100 score.
func TestFullAdoptionIsClean(t *testing.T) {
	c := Convention{Key: "compare", Short: "--compare", Label: "expose --compare"}
	all := []Probe{
		probe("a", true, map[string]bool{"compare": true}),
		probe("b", true, map[string]bool{"compare": true}),
	}
	k, gaps := kpiForConvention(c, all)
	if len(k.Defects) != 0 || len(k.Soft) != 0 || len(gaps) != 0 {
		t.Fatalf("full adoption: want clean, got def=%d soft=%d gaps=%d", len(k.Defects), len(k.Soft), len(gaps))
	}
	if k.Score != 100 {
		t.Fatalf("full adoption score = %v, want 100", k.Score)
	}
}

// Non-existent members are excluded from the adoption denominator entirely.
func TestMissingMemberExcludedFromDenominator(t *testing.T) {
	c := Convention{Key: "compare", Short: "--compare", Label: "expose --compare"}
	probes := []Probe{
		probe("a", true, map[string]bool{"compare": true}),
		probe("ghost", false, map[string]bool{"compare": false}), // not on the tree
	}
	k, _ := kpiForConvention(c, probes)
	if k.Score != 100 {
		t.Fatalf("a missing member must not count against adoption; score=%v want 100", k.Score)
	}
}

// ProbeMembers reads the SOURCE: the kernel import, the scorecardCmdSetup helper (json+markdown
// but not compare), an inline compare flag, a package test, and control-pane registration.
func TestProbeMembersReadsSource(t *testing.T) {
	root := t.TempDir()
	mustWrite(t, root, "tools/scorecard_control_pane.py", `SCORECARDS = [{"cmd": "go run ./cmd/fak helper-card --json"}]`)

	// Member 1: rides the kernel, inline --compare, has a test.
	mustWrite(t, root, "cmd/fak/kernelcard.go", `package main
func cmdKernelCard(a []string){ fs.String("compare","",""); fs.Bool("markdown",false,""); fs.Bool("json",false,"") }`)
	mustWrite(t, root, "internal/kernelcard/kernelcard.go", `package kernelcard
import _ "github.com/anthony-chaudhary/fak/pkg/scorecard"`)
	mustWrite(t, root, "internal/kernelcard/kernelcard_test.go", `package kernelcard`)

	// Member 2: uses the shared setup helper (json+markdown, NO compare), no kernel, no test.
	mustWrite(t, root, "cmd/fak/helpercard.go", `package main
func cmdHelperCard(a []string){ scorecardCmdSetup("fak helper-card", a, collect) }`)
	mustWrite(t, root, "internal/helpercard/helpercard.go", `package helpercard`)

	family := []Member{
		{Verb: "kernel-card", CmdFile: "cmd/fak/kernelcard.go", PkgDir: "internal/kernelcard", DebtKey: "kernel_debt"},
		{Verb: "helper-card", CmdFile: "cmd/fak/helpercard.go", PkgDir: "internal/helpercard", DebtKey: "helper_debt"},
	}
	probes := ProbeMembers(root, family)

	k := probes[0].Adopted
	if !k["kernel"] || !k["compare"] || !k["markdown"] || !k["json"] || !k["test"] {
		t.Fatalf("kernel-card probe wrong: %+v", k)
	}
	if !probes[0].Exists {
		t.Fatalf("kernel-card should exist on the synthetic tree")
	}

	h := probes[1].Adopted
	if h["kernel"] {
		t.Fatalf("helper-card must not import the kernel")
	}
	if !h["json"] || !h["markdown"] {
		t.Fatalf("scorecardCmdSetup should provide json+markdown: %+v", h)
	}
	if h["compare"] {
		t.Fatalf("scorecardCmdSetup does NOT provide --compare; helper-card must read as a compare laggard")
	}
	if h["test"] {
		t.Fatalf("helper-card has no _test.go")
	}
	if !h["controlpane"] {
		t.Fatalf("helper-card is registered in the control pane")
	}
}

// Every fanned-out gap must produce a dispatchable issuecontract candidate (the dispatcher only
// syncs OK candidates, so a non-dispatchable mapping would silently drop the issue).
func TestGapActionItemsAreDispatchable(t *testing.T) {
	gaps := []Gap{
		{Member: Family[1], Convention: Conventions[0], Adopters: 3, Total: 8}, // kernel gap
		{Member: Family[4], Convention: Conventions[2], Adopters: 7, Total: 8}, // compare gap
	}
	for _, g := range gaps {
		item := g.ToActionItem("fak propagation-scorecard --json")
		c := issuecontract.Candidate{
			Schema: issuecontract.Schema, Key: item.Key, Title: item.Title,
			ParentRef: item.ParentRef, CurrentState: item.CurrentState, WhyNow: item.WhyNow,
			WorkingSpine: item.WorkingSpine, InScope: item.InScope, OutOfScope: item.OutOfScope,
			DoneCondition: item.DoneCondition, Witness: item.Witness, AcceptanceGate: item.AcceptanceGate,
			Lane: item.Lane, Paths: item.Paths, Labels: item.Labels, BoundaryNotes: item.BoundaryNotes,
			ClosureBinding: item.ClosureBinding,
		}
		rv := issuecontract.ReviewCandidate(c, issuecontract.Options{Live: true, DedupeChecked: true, DedupeCap: 10})
		if !rv.OK {
			t.Fatalf("gap %s -> non-dispatchable issue: verdict=%s reasons=%v missing=%v", g.Key(), rv.Verdict, rv.Reasons, rv.MissingFields)
		}
	}
}

// --- live smoke over the real tree: deterministic, roster fresh, detects real kernel drift -----

func TestLiveTreeIsDeterministicAndDetectsDrift(t *testing.T) {
	root := repoRootForTest(t)
	a := Build(root)
	b := Build(root)
	if a.Corpus[DebtKey] != b.Corpus[DebtKey] {
		t.Fatalf("non-deterministic debt: %v != %v", a.Corpus[DebtKey], b.Corpus[DebtKey])
	}
	if a.Schema != Schema {
		t.Fatalf("schema = %q, want %q", a.Schema, Schema)
	}
	// member_integrity is the roster-freshness floor: every rostered member must resolve on the
	// real tree, so this KPI carries zero defects (a non-zero here means the roster drifted).
	for _, k := range a.KPIs {
		if k.Key == "member_integrity" && len(k.Defects) != 0 {
			t.Fatalf("roster drifted from the tree: %v", k.Defects)
		}
	}
	// The card must be measuring REAL drift: the kernel convention has at least one adopter and
	// at least one laggard on the live tree (it is neither fully propagated nor absent).
	var kernel struct{ adopters, laggards int }
	probes := ProbeMembers(root, Family)
	for _, p := range probes {
		if !p.Exists {
			continue
		}
		if p.Adopted["kernel"] {
			kernel.adopters++
		} else {
			kernel.laggards++
		}
	}
	if kernel.adopters == 0 || kernel.laggards == 0 {
		t.Fatalf("kernel adoption is %d adopters / %d laggards -- expected real drift (both > 0)", kernel.adopters, kernel.laggards)
	}
	if len(Gaps(root)) == 0 {
		t.Fatalf("expected at least one HARD propagation gap on the live tree")
	}
}

func mustWrite(t *testing.T, root, rel, content string) {
	t.Helper()
	p := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func repoRootForTest(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Skip("go.mod not found; skipping live smoke")
		}
		dir = parent
	}
}
