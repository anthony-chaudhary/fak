package devindex

// Proof for #5648. The audit's whole claim is that it measures reachability rather
// than mentions, so the fixtures below are a SYNTHETIC MODULE in which every state the
// issue names exists on purpose and is separable:
//
//	cmd/integrated   test + Makefile build target            -> ok
//	cmd/dispatched   test + Go registry string literal       -> ok   (dispatcher edge)
//	cmd/untested     no test + shell script that runs it     -> untested
//	cmd/installed    no test + `go install` line in docs     -> untested (installer edge)
//	cmd/lonely       test, nothing outside invokes it        -> unreachable
//	cmd/orphan       no test; only its OWN comment and a
//	                 markdown inventory row name it          -> orphan
//	cmd/pinned       no test, no edge, admitted by a pin     -> pinned
//	tools/nested/deep/cmd/runner
//	                 test + fenced doc command, nested two
//	                 levels below a non-cmd directory        -> ok   (recursive discovery)
//
// cmd/orphan is the load-bearing one: it is named by its own source and by a catalog
// table, which is exactly the evidence a mention-counting audit would accept. It must
// still come out orphan, or the instrument measures nothing.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// writeSynthFile writes one fixture file, creating parents.
func writeSynthFile(t *testing.T, root, rel, body string) {
	t.Helper()
	p := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", rel, err)
	}
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatalf("write %s: %v", rel, err)
	}
}

const synthMain = "package main\n\nfunc main() {}\n"
const synthTest = "package main\n\nimport \"testing\"\n\nfunc TestBuilds(t *testing.T) { _ = t }\n"

// newSynthModule materializes the fixture module described above and returns its root.
func newSynthModule(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	writeSynthFile(t, root, "go.mod", "module example.com/synth\n\ngo 1.22\n")

	for _, p := range []string{"integrated", "dispatched", "untested", "installed", "lonely", "pinned"} {
		writeSynthFile(t, root, "cmd/"+p+"/main.go", synthMain)
	}
	for _, p := range []string{"integrated", "dispatched", "lonely"} {
		writeSynthFile(t, root, "cmd/"+p+"/main_test.go", synthTest)
	}
	// The nested executable: two levels below a directory that is not "cmd", so a
	// `cmd/*` glob would miss it entirely.
	writeSynthFile(t, root, "tools/nested/deep/cmd/runner/main.go", synthMain)
	writeSynthFile(t, root, "tools/nested/deep/cmd/runner/main_test.go", synthTest)

	// cmd/orphan names ITSELF in its own source, in both a comment and a string
	// literal. Neither may count: self-reference is not reachability.
	writeSynthFile(t, root, "cmd/orphan/main.go",
		"package main\n\n// cmd/orphan is the orphan fixture; run it with go run ./cmd/orphan.\n\n"+
			"const self = \"cmd/orphan\"\n\nfunc main() { _ = self }\n")

	// Evidence carriers, one per class.
	writeSynthFile(t, root, "Makefile", "build:\n\tgo build -o bin/ ./cmd/integrated\n")
	writeSynthFile(t, root, "scripts/run.sh", "#!/bin/sh\nset -e\ngo run ./cmd/untested \"$@\"\n")
	writeSynthFile(t, root, "internal/dispatch/reg.go",
		"package dispatch\n\n// The spawn registry: a main package cannot be imported, so the edge is a path.\n"+
			"var Registry = []string{\"cmd/dispatched\"}\n")
	writeSynthFile(t, root, "docs/GUIDE.md", strings.Join([]string{
		"# Guide",
		"",
		"Install the publisher:",
		"",
		"```sh",
		"go install example.com/synth/cmd/installed@latest",
		"```",
		"",
		"Run the nested runner:",
		"",
		"```sh",
		"go run ./tools/nested/deep/cmd/runner",
		"```",
		"",
		"## Command inventory",
		"",
		"| command | purpose |",
		"| --- | --- |",
		"| cmd/orphan | the orphan fixture |",
		"| cmd/lonely | the lonely fixture |",
		"| cmd/pinned | the pinned fixture |",
		"",
	}, "\n"))
	return root
}

// auditSynth runs the audit over the fixture module and indexes rows by import path.
func auditSynth(t *testing.T, root string, pins []ExecPin, now time.Time) (*ExecAuditResult, map[string]ExecPackage) {
	t.Helper()
	res, err := AuditExecutables(ExecAuditOptions{Root: root, Pins: pins, Now: now})
	if err != nil {
		t.Fatalf("AuditExecutables: %v", err)
	}
	byPath := map[string]ExecPackage{}
	for _, p := range res.Packages {
		byPath[strings.TrimPrefix(p.ImportPath, "example.com/synth/")] = p
	}
	return res, byPath
}

var synthNow = time.Date(2026, 8, 6, 0, 0, 0, 0, time.UTC)

// TestExecAuditClassifiesEveryState is the issue's four required states plus the two
// the two-axis split adds (tested-but-unreachable, and the pinned admission).
func TestExecAuditClassifiesEveryState(t *testing.T) {
	root := newSynthModule(t)
	pins := []ExecPin{{
		Package: "example.com/synth/cmd/pinned",
		Reason:  "fixture: an admitted exception with a live justification",
		Until:   "2099-01-01",
	}}
	res, by := auditSynth(t, root, pins, synthNow)

	if res.Domain != 8 {
		t.Fatalf("domain = %d, want 8 executable packages; got %v", res.Domain, importPaths(res))
	}
	want := map[string]ExecStatus{
		"cmd/integrated":               ExecStatusOK,
		"cmd/dispatched":               ExecStatusOK,
		"tools/nested/deep/cmd/runner": ExecStatusOK,
		"cmd/untested":                 ExecStatusUntested,
		"cmd/installed":                ExecStatusUntested,
		"cmd/lonely":                   ExecStatusUnreachable,
		"cmd/orphan":                   ExecStatusOrphan,
		"cmd/pinned":                   ExecStatusPinned,
	}
	for name, wantStatus := range want {
		p, ok := by[name]
		if !ok {
			t.Errorf("%s: not discovered", name)
			continue
		}
		if p.Status != wantStatus {
			t.Errorf("%s: status = %q, want %q (has_test=%v evidence=%v)",
				name, p.Status, wantStatus, p.HasTest, p.Evidence)
		}
	}
	// The two axes must stay independent, not collapsed into the status label.
	if p := by["cmd/lonely"]; !p.HasTest || p.Reachable() {
		t.Errorf("cmd/lonely: want tested+unreachable, got has_test=%v reachable=%v", p.HasTest, p.Reachable())
	}
	if p := by["cmd/untested"]; p.HasTest || !p.Reachable() {
		t.Errorf("cmd/untested: want untested+reachable, got has_test=%v reachable=%v", p.HasTest, p.Reachable())
	}
	// A pinned row keeps its reason so the exception is readable, not silent.
	if p := by["cmd/pinned"]; !strings.Contains(p.PinReason, "admitted exception") {
		t.Errorf("cmd/pinned: pin reason not carried through: %q", p.PinReason)
	}
	// Everything failing an axis and not pinned is named, and the audit reds.
	if res.Status != "fail" {
		t.Errorf("status = %q, want fail (failures=%v)", res.Status, res.Failures)
	}
	wantFailures := []string{
		"example.com/synth/cmd/installed",
		"example.com/synth/cmd/lonely",
		"example.com/synth/cmd/orphan",
		"example.com/synth/cmd/untested",
	}
	if got := strings.Join(res.Failures, ","); got != strings.Join(wantFailures, ",") {
		t.Errorf("failures = %v, want %v", res.Failures, wantFailures)
	}
	if res.Tested != 4 || res.Reached != 5 {
		t.Errorf("tested/reached = %d/%d, want 4/5", res.Tested, res.Reached)
	}
}

// TestExecAuditRejectsSelfReferentialEvidence is the anti-vacuity arm: cmd/orphan is
// named by its own source (comment AND string literal) and by a markdown inventory
// row, and must still be unreachable. cmd/lonely and cmd/pinned appear in the same
// table, so the table cannot be quietly wiring anything either.
func TestExecAuditRejectsSelfReferentialEvidence(t *testing.T) {
	root := newSynthModule(t)
	_, by := auditSynth(t, root, nil, synthNow)
	for _, name := range []string{"cmd/orphan", "cmd/lonely", "cmd/pinned"} {
		p, ok := by[name]
		if !ok {
			t.Fatalf("%s: not discovered", name)
		}
		if p.Reachable() {
			t.Errorf("%s: self-reference/inventory prose was accepted as reachability: %+v", name, p.Evidence)
		}
	}
	// Sanity: the fixture really does name them, so the test above is not vacuous.
	guide, err := os.ReadFile(filepath.Join(root, "docs", "GUIDE.md"))
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"cmd/orphan", "cmd/lonely", "cmd/pinned"} {
		if !strings.Contains(string(guide), name) {
			t.Fatalf("fixture broken: docs/GUIDE.md no longer names %s", name)
		}
	}
	own, err := os.ReadFile(filepath.Join(root, "cmd", "orphan", "main.go"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(own), "go run ./cmd/orphan") {
		t.Fatal("fixture broken: cmd/orphan no longer names itself in an invocation form")
	}
}

// TestExecAuditDistinguishesEvidenceClasses: the classes the issue lists are separate
// answers, not one bool. Which edge holds is what tells a fixer what to add.
func TestExecAuditDistinguishesEvidenceClasses(t *testing.T) {
	root := newSynthModule(t)
	_, by := auditSynth(t, root, nil, synthNow)
	want := map[string]ExecEvidenceClass{
		"cmd/integrated":               ExecEvidenceBuildTarget,
		"cmd/dispatched":               ExecEvidenceDispatch,
		"cmd/untested":                 ExecEvidenceScript,
		"cmd/installed":                ExecEvidenceInstaller,
		"tools/nested/deep/cmd/runner": ExecEvidenceDocExample,
	}
	for name, class := range want {
		p := by[name]
		var got []ExecEvidenceClass
		hit := false
		for _, e := range p.Evidence {
			got = append(got, e.Class)
			if e.Class == class {
				hit = true
				if e.File == "" || e.Line <= 0 || e.Text == "" {
					t.Errorf("%s: %s evidence has no checkable locator: %+v", name, class, e)
				}
			}
		}
		if !hit {
			t.Errorf("%s: want evidence class %q, got %v", name, class, got)
		}
	}
}

// TestExecAuditDiscoversNestedExecutables: discovery is recursive and toolchain-derived,
// so an executable buried below a non-cmd directory is in the denominator.
func TestExecAuditDiscoversNestedExecutables(t *testing.T) {
	root := newSynthModule(t)
	res, by := auditSynth(t, root, nil, synthNow)
	p, ok := by["tools/nested/deep/cmd/runner"]
	if !ok {
		t.Fatalf("nested executable missing from a domain of %d: %v", res.Domain, importPaths(res))
	}
	if p.Dir != "tools/nested/deep/cmd/runner" {
		t.Errorf("nested dir = %q, want the slash-separated repo-relative path", p.Dir)
	}
	if p.Binary != "runner" {
		t.Errorf("nested binary = %q, want runner", p.Binary)
	}
}

// TestExecAuditPinStaleness: a pin is only worth more than an ignore list because it
// FAILS once the condition it names stops holding. All four staleness modes red.
func TestExecAuditPinStaleness(t *testing.T) {
	root := newSynthModule(t)
	cases := []struct {
		name    string
		pin     ExecPin
		wantWhy string
	}{
		{"expired", ExecPin{Package: "example.com/synth/cmd/pinned", Reason: "r", Until: "2026-01-01"}, "expired"},
		{"unknown-package", ExecPin{Package: "example.com/synth/cmd/gone", Reason: "r"}, "no longer in the executable domain"},
		{"no-longer-needed", ExecPin{Package: "example.com/synth/cmd/integrated", Reason: "r"}, "no longer doing work"},
		{"unreasoned", ExecPin{Package: "example.com/synth/cmd/pinned", Reason: "  "}, "no reason"},
		{"bad-date", ExecPin{Package: "example.com/synth/cmd/pinned", Reason: "r", Until: "soon"}, "not a YYYY-MM-DD"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			res, _ := auditSynth(t, root, []ExecPin{tc.pin}, synthNow)
			if len(res.StalePins) != 1 || res.StalePins[0] != tc.pin.Package {
				t.Fatalf("stale pins = %v, want [%s]", res.StalePins, tc.pin.Package)
			}
			if res.Status != "fail" {
				t.Errorf("status = %q, want fail on a stale pin", res.Status)
			}
			if len(res.Exceptions) != 1 || !strings.Contains(res.Exceptions[0].Why, tc.wantWhy) {
				t.Errorf("why = %q, want it to mention %q", res.Exceptions[0].Why, tc.wantWhy)
			}
		})
	}
	// A live pin does the opposite: it admits its package and does not red.
	res, _ := auditSynth(t, root, []ExecPin{
		{Package: "example.com/synth/cmd/pinned", Reason: "still justified", Until: "2099-01-01"},
	}, synthNow)
	if len(res.StalePins) != 0 {
		t.Errorf("a live pin went stale: %v", res.Exceptions)
	}
}

// TestExecAuditFailsClosedWithoutSourceMetadata: the one bug that would quietly retire
// the instrument is a green audit over an empty domain. A directory the toolchain
// cannot resolve must report could-not-establish-domain, not ok.
func TestExecAuditFailsClosedWithoutSourceMetadata(t *testing.T) {
	// A directory with no module at all: `go list ./...` cannot resolve a domain.
	bare := t.TempDir()
	writeSynthFile(t, bare, "notes.txt", "no go module here\n")
	res, err := AuditExecutables(ExecAuditOptions{Root: bare, Now: synthNow})
	if err == nil {
		t.Fatal("want an error when the executable domain cannot be established")
	}
	if res.Established {
		t.Error("Established = true for a tree with no resolvable module")
	}
	if res.Status != ExecDomainNotEstablished {
		t.Errorf("status = %q, want %q", res.Status, ExecDomainNotEstablished)
	}
	if res.Reason == "" {
		t.Error("fail-closed result carries no reason")
	}

	// A module that resolves cleanly but contains NO executable is the subtler half:
	// every fold over it finds zero problems, so a naive audit reports success.
	empty := t.TempDir()
	writeSynthFile(t, empty, "go.mod", "module example.com/empty\n\ngo 1.22\n")
	writeSynthFile(t, empty, "lib/lib.go", "package lib\n\nfunc F() {}\n")
	res, err = AuditExecutables(ExecAuditOptions{Root: empty, Now: synthNow})
	if err == nil {
		t.Fatal("want an error for a module with an empty executable domain")
	}
	if res.Status != ExecDomainNotEstablished || res.Established {
		t.Errorf("empty domain: status = %q established = %v, want %q/false",
			res.Status, res.Established, ExecDomainNotEstablished)
	}
	if res.Domain != 0 {
		t.Errorf("empty domain: Domain = %d, want 0", res.Domain)
	}
}

// TestExecAuditJSONCarriesTheDenominator: the witness has to answer "out of how many",
// name the exceptions, and stay host-independent (no absolute paths).
func TestExecAuditJSONCarriesTheDenominator(t *testing.T) {
	root := newSynthModule(t)
	res, _ := auditSynth(t, root, []ExecPin{
		{Package: "example.com/synth/cmd/pinned", Reason: "fixture pin", Until: "2099-01-01"},
	}, synthNow)
	b, err := json.Marshal(res)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var round ExecAuditResult
	if err := json.Unmarshal(b, &round); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if round.Domain != res.Domain || len(round.Packages) != res.Domain {
		t.Errorf("denominator lost in JSON: domain=%d packages=%d want %d",
			round.Domain, len(round.Packages), res.Domain)
	}
	if round.Schema != ExecAuditSchema {
		t.Errorf("schema = %q, want %q", round.Schema, ExecAuditSchema)
	}
	if len(round.Exceptions) != 1 || round.Exceptions[0].Reason != "fixture pin" {
		t.Errorf("exceptions did not survive JSON: %+v", round.Exceptions)
	}
	if strings.Contains(string(b), filepath.ToSlash(root)) {
		t.Error("witness embeds the absolute host root; it must be repo-relative")
	}
}

// TestExecAuditOverCurrentTree runs the audit on fak itself. It asserts only what is
// stable — the domain establishes, rows are well-formed, discovery reaches beyond
// cmd/* — because the fleet adds executables daily and a pinned count here would red
// the trunk for reasons that have nothing to do with this gate. The graded numbers
// live in the committed witness instead.
func TestExecAuditOverCurrentTree(t *testing.T) {
	if testing.Short() {
		t.Skip("walks the whole tree")
	}
	root := FindRoot(".")
	res, err := AuditExecutables(ExecAuditOptions{Root: root, Pins: ExecAuditPins, Now: time.Now()})
	if err != nil {
		t.Skipf("executable domain unavailable in this environment: %v", err)
	}
	if !res.Established || res.Domain == 0 {
		t.Fatalf("domain not established over the live tree: %s", res.Reason)
	}
	nested := 0
	for _, p := range res.Packages {
		if p.ImportPath == "" || p.Dir == "" || p.Binary == "" {
			t.Errorf("malformed row: %+v", p)
		}
		if filepath.IsAbs(p.Dir) {
			t.Errorf("row %s carries an absolute dir %q", p.ImportPath, p.Dir)
		}
		if p.Status == "" {
			t.Errorf("row %s has no status", p.ImportPath)
		}
		if !strings.HasPrefix(p.Dir, "cmd/") {
			nested++
		}
	}
	if nested == 0 {
		t.Error("discovery found no executable outside cmd/* — a glob would have sufficed, so recursion is untested here")
	}
	// Every committed pin must still name a package that exists; a pin for a deleted
	// command is exactly the stale state the pin mechanism promises to catch.
	inDomain := map[string]bool{}
	for _, p := range res.Packages {
		inDomain[p.ImportPath] = true
	}
	for _, pin := range ExecAuditPins {
		if !inDomain[pin.Package] {
			t.Errorf("stale pin: %s is no longer an executable package — remove it", pin.Package)
		}
		if strings.TrimSpace(pin.Reason) == "" {
			t.Errorf("pin %s carries no reason", pin.Package)
		}
	}
	t.Logf("live tree: domain=%d tested=%d reached=%d failures=%d pins=%d",
		res.Domain, res.Tested, res.Reached, len(res.Failures), len(res.Exceptions))
}

// TestRegenExecAuditWitness rewrites the committed witness
// (docs/exec-audit.witness.json) from a live audit of this tree. It is a NO-OP unless
// FAK_REGEN_EXEC_WITNESS is set, so CI never writes; an operator refreshes it when the
// executable domain moves.
//
// The issue's proof requirement is to preserve the complete denominator and the
// admitted exceptions as a machine-readable artifact. Generating that artifact from
// the SAME code path the gate runs is what stops it from decaying into a
// hand-maintained claim about the tree — the failure mode a witness exists to prevent.
func TestRegenExecAuditWitness(t *testing.T) {
	if os.Getenv("FAK_REGEN_EXEC_WITNESS") == "" {
		t.Skip("set FAK_REGEN_EXEC_WITNESS=1 to regenerate docs/exec-audit.witness.json")
	}
	root := FindRoot(".")
	res, err := AuditExecutables(ExecAuditOptions{Root: root, Pins: ExecAuditPins, Now: time.Now()})
	if err != nil {
		t.Fatalf("AuditExecutables over %s: %v", root, err)
	}
	b, err := json.MarshalIndent(res, "", "  ")
	if err != nil {
		t.Fatalf("marshal witness: %v", err)
	}
	out := filepath.Join(root, "docs", "exec-audit.witness.json")
	if err := os.WriteFile(out, append(b, '\n'), 0o644); err != nil {
		t.Fatalf("write %s: %v", out, err)
	}
	t.Logf("wrote %s: domain=%d tested=%d reached=%d failures=%d exceptions=%d",
		out, res.Domain, res.Tested, res.Reached, len(res.Failures), len(res.Exceptions))
}

func importPaths(res *ExecAuditResult) []string {
	out := make([]string, 0, len(res.Packages))
	for _, p := range res.Packages {
		out = append(out, p.ImportPath)
	}
	return out
}
