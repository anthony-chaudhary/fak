package witness

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// candRedRig fakes the candidate and parent scratch trees and the go test runner.
// The parent scratch is "parcand" while it holds the candidate test overlay and
// "parown" while it holds the parent's own test file. Verdicts are keyed
// "<where>|<Name>", consumed in order (the last entry repeats). A name with no
// entry, or the verdict "none", starts without a final event; "absent" emits no run
// event; "infra" makes the whole invocation return an error.
type candRedRig struct {
	cand, par string
	candSrc   string
	// diags overrides the failure message printed for "<where>|<Name>".
	diags    map[string]string
	verdicts map[string][]string
	next     map[string]int
	log      []candRedCall
}

type candRedCall struct {
	where string
	names []string
}

var candRedSelectorName = regexp.MustCompile(`\^([A-Za-z_][A-Za-z0-9_]*)\$`)

const candRedRel = "pkg/fix_test.go"

const candRedFixOnly = "package pkg\n\nimport \"testing\"\n\nfunc TestFix(t *testing.T) {}\n"

func newCandRedRig(t *testing.T, verdicts map[string][]string) *candRedRig {
	t.Helper()
	return &candRedRig{cand: t.TempDir(), par: t.TempDir(), verdicts: verdicts, next: map[string]int{}}
}

func (f *candRedRig) verdict(key string) string {
	seq := f.verdicts[key]
	if len(seq) == 0 {
		return "none"
	}
	i := f.next[key]
	f.next[key]++
	if i >= len(seq) {
		i = len(seq) - 1
	}
	return seq[i]
}

// exec answers each selected go test run from the verdict script. The parent scratch
// is told apart by its file content: the candidate overlay or the parent's own file.
func (f *candRedRig) exec(_ context.Context, dir string, argv ...string) (string, int, error) {
	where := "other"
	switch dir {
	case f.cand:
		where = "cand"
	case f.par:
		where = "parown"
		if body, err := os.ReadFile(filepath.Join(f.par, filepath.FromSlash(candRedRel))); err == nil && string(body) == f.candSrc {
			where = "parcand"
		}
	}
	run := ""
	for i, a := range argv {
		if a == "-run" && i+1 < len(argv) {
			run = argv[i+1]
		}
	}
	if run == "" {
		return "", 0, nil
	}
	var names []string
	for _, m := range candRedSelectorName.FindAllStringSubmatch(run, -1) {
		names = append(names, m[1])
	}
	f.log = append(f.log, candRedCall{where: where, names: names})
	var b strings.Builder
	code := 0
	for _, name := range names {
		v := f.verdict(where + "|" + name)
		if v == "absent" {
			code = 1
			continue
		}
		if v == "infra" {
			return "", 0, errors.New("runner exploded")
		}
		b.WriteString(`{"Action":"run","Package":"m/pkg","Test":"` + name + `"}` + "\n")
		switch v {
		case "pass":
			b.WriteString(`{"Action":"pass","Package":"m/pkg","Test":"` + name + `"}` + "\n")
		case "skip":
			b.WriteString(`{"Action":"skip","Package":"m/pkg","Test":"` + name + `"}` + "\n")
		case "fail":
			msg := "    fix_test.go:7: " + name + " is red"
			if override, ok := f.diags[where+"|"+name]; ok {
				msg = override
			}
			b.WriteString(`{"Action":"output","Package":"m/pkg","Test":"` + name + `","Output":"` + msg + `\n"}` + "\n")
			b.WriteString(`{"Action":"fail","Package":"m/pkg","Test":"` + name + `"}` + "\n")
			code = 1
		case "passexit1":
			b.WriteString(`{"Action":"pass","Package":"m/pkg","Test":"` + name + `"}` + "\n")
			code = 1
		default:
			code = 1
		}
	}
	return b.String(), code, nil
}

// count returns how many runs in where selected exactly names (order-insensitive).
func (f *candRedRig) count(where string, names ...string) int {
	want := append([]string(nil), names...)
	sort.Strings(want)
	n := 0
	for _, c := range f.log {
		got := append([]string(nil), c.names...)
		sort.Strings(got)
		if c.where == where && strings.Join(got, ",") == strings.Join(want, ",") {
			n++
		}
	}
	return n
}

// touched returns how many runs in where selected name.
func (f *candRedRig) touched(where, name string) int {
	n := 0
	for _, c := range f.log {
		for _, got := range c.names {
			if c.where == where && got == name {
				n++
			}
		}
	}
	return n
}

// candRedResolve runs the witness over a one-file change: candSrc is the candidate
// test file and parSrc the parent's ("" when the parent has no such file).
func candRedResolve(t *testing.T, rig *candRedRig, candSrc, parSrc string) (abi.WitnessOutcome, string) {
	t.Helper()
	rig.candSrc = candSrc
	git := func(_ context.Context, _ string, args ...string) (string, int, error) {
		switch {
		case len(args) == 3 && args[0] == "rev-parse" && args[2] == "HEAD^{commit}":
			return "cand\n", 0, nil
		case len(args) == 3 && args[0] == "rev-parse" && args[2] == "cand^":
			return "par\n", 0, nil
		case len(args) == 2 && args[0] == "show" && args[1] == "cand:"+candRedRel:
			return candSrc, 0, nil
		case len(args) == 2 && args[0] == "show" && args[1] == "par:"+candRedRel && parSrc != "":
			return parSrc, 0, nil
		}
		return "", 128, nil
	}
	ctx := WithExecutionMaterializer(context.Background(), func(_ context.Context, _ string, ref string) (string, func(), error) {
		switch ref {
		case "cand":
			return rig.cand, func() {}, nil
		case "par":
			return rig.par, func() {}, nil
		}
		return "", func() {}, errors.New("unexpected scratch ref " + ref)
	})
	r := NewWithRunners(git, rig.exec, t.TempDir())
	return r.resolveSymptomExec(ctx, "HEAD", []string{candRedRel})
}

// fak-test:runtime fast est=10ms
func TestSymptomCandidateRedToleratesPreexistingRedSibling(t *testing.T) {
	// TestStale is modified by the change, so discovery selects it, and it is red at the
	// candidate, at the parent's own body, and with the candidate body at the parent.
	const cand = candRedFixOnly + "\nfunc TestStale(t *testing.T) { t.Log(\"changed\") }\n"
	const par = "package pkg\n\nimport \"testing\"\n\nfunc TestStale(t *testing.T) {}\n"
	rig := newCandRedRig(t, map[string][]string{
		"cand|TestFix":      {"pass"},
		"cand|TestStale":    {"fail", "fail"},
		"parown|TestStale":  {"fail"},
		"parcand|TestStale": {"fail"},
		"parcand|TestFix":   {"fail"},
	})
	got, detail := candRedResolve(t, rig, cand, par)
	want := "selected symptom test failed at parent and passed at candidate (tolerated at candidate: pre-existing red at parent TestStale)"
	if got != abi.WitnessConfirmed || detail != want {
		t.Fatalf("outcome=%v detail=%q, want CONFIRMED %q", got, detail, want)
	}
	if n := rig.count("cand", "TestStale"); n != 1 {
		t.Fatalf("candidate rerun of TestStale ran %d time(s), want exactly 1", n)
	}
	if n := rig.count("parown", "TestStale"); n != 1 {
		t.Fatalf("parent own-version run of TestStale ran %d time(s), want 1", n)
	}
	if n := rig.count("parcand", "TestStale"); n != 1 {
		t.Fatalf("candidate-overlay run of TestStale at the parent ran %d time(s), want 1", n)
	}
	if n := rig.count("parcand", "TestFix"); n != 1 || rig.touched("parcand", "TestStale") != 1 {
		t.Fatalf("parent proof must run only TestFix: TestFix runs=%d, TestStale runs=%d", n, rig.touched("parcand", "TestStale"))
	}
}

// fak-test:runtime fast est=10ms
func TestSymptomCandidateRedFlakyOnRerunRefutes(t *testing.T) {
	// A red that passes on the rerun is intermittent: the fix may have introduced it, so it
	// is never tolerated.
	const cand = candRedFixOnly + "\nfunc TestParallel(t *testing.T) {}\n"
	rig := newCandRedRig(t, map[string][]string{
		"cand|TestFix":      {"pass"},
		"cand|TestParallel": {"fail", "pass"},
		"parcand|TestFix":   {"fail"},
	})
	got, detail := candRedResolve(t, rig, cand, "")
	want := "candidate selected symptom test failed"
	if got != abi.WitnessRefuted || detail != want {
		t.Fatalf("outcome=%v detail=%q, want REFUTED %q", got, detail, want)
	}
	if n := rig.touched("parcand", "TestParallel") + rig.touched("parown", "TestParallel"); n != 0 {
		t.Fatalf("a flaky-on-rerun name ran at the parent %d time(s), want 0", n)
	}
}

// fak-test:runtime fast est=10ms
func TestSymptomCandidateRedDifferentDiagnosticRefutes(t *testing.T) {
	// Both sides are red, but the candidate body fails at the parent with a different
	// message: the red is not the same pre-existing failure, so it refutes as a regression.
	const cand = candRedFixOnly + "\nfunc TestStale(t *testing.T) { t.Log(\"changed\") }\n"
	const par = "package pkg\n\nimport \"testing\"\n\nfunc TestStale(t *testing.T) {}\n"
	rig := newCandRedRig(t, map[string][]string{
		"cand|TestFix":      {"pass"},
		"cand|TestStale":    {"fail", "fail"},
		"parown|TestStale":  {"fail"},
		"parcand|TestStale": {"fail"},
		"parcand|TestFix":   {"fail"},
	})
	rig.diags = map[string]string{"parcand|TestStale": "    fix_test.go:12: a different message"}
	got, detail := candRedResolve(t, rig, cand, par)
	want := "candidate selected symptom test regressed: TestStale failed at candidate on rerun and at parent with a different or missing diagnostic"
	if got != abi.WitnessRefuted || detail != want {
		t.Fatalf("outcome=%v detail=%q, want REFUTED %q", got, detail, want)
	}
}

// fak-test:runtime fast est=10ms
func TestSymptomCandidateRedAllSkipFallsThroughToBase(t *testing.T) {
	// No PASS and no explicit fail with a zero exit: base falls through and proves the
	// selected name at the parent, so a parent red still confirms.
	const cand = "package pkg\n\nimport \"testing\"\n\nfunc TestFix(t *testing.T) {}\n"
	rig := newCandRedRig(t, map[string][]string{
		"cand|TestFix":    {"skip"},
		"parcand|TestFix": {"fail"},
	})
	got, detail := candRedResolve(t, rig, cand, "")
	want := "selected symptom test failed at parent and passed at candidate"
	if got != abi.WitnessConfirmed || detail != want {
		t.Fatalf("outcome=%v detail=%q, want CONFIRMED %q (base behavior)", got, detail, want)
	}
}

// fak-test:runtime fast est=10ms
func TestSymptomCandidateRedRegressionPassedAtParentRefutes(t *testing.T) {
	const cand = candRedFixOnly + "\nfunc TestOld(t *testing.T) { t.Log(\"changed\") }\n"
	const par = "package pkg\n\nimport \"testing\"\n\nfunc TestOld(t *testing.T) {}\n"
	rig := newCandRedRig(t, map[string][]string{
		"cand|TestFix":    {"pass"},
		"cand|TestOld":    {"fail", "fail"},
		"parown|TestOld":  {"pass"},
		"parcand|TestOld": {"fail"},
		"parcand|TestFix": {"fail"},
	})
	got, detail := candRedResolve(t, rig, cand, par)
	want := "candidate selected symptom test regressed: TestOld failed at candidate on rerun and passed at parent"
	if got != abi.WitnessRefuted || detail != want {
		t.Fatalf("outcome=%v detail=%q, want REFUTED %q", got, detail, want)
	}
}

// fak-test:runtime fast est=10ms
func TestSymptomCandidateRedUnrelatedRedOldBodyRefutes(t *testing.T) {
	// The parent's old TestOld body is red for an unrelated reason, but the candidate body
	// passes at the parent: the candidate's change is a real regression, not pre-existing.
	const cand = candRedFixOnly + "\nfunc TestOld(t *testing.T) { t.Log(\"changed\") }\n"
	const par = "package pkg\n\nimport \"testing\"\n\nfunc TestOld(t *testing.T) { t.Fatal(\"old body is red\") }\n"
	rig := newCandRedRig(t, map[string][]string{
		"cand|TestFix":    {"pass"},
		"cand|TestOld":    {"fail", "fail"},
		"parown|TestOld":  {"fail"},
		"parcand|TestOld": {"pass"},
		"parcand|TestFix": {"fail"},
	})
	got, detail := candRedResolve(t, rig, cand, par)
	want := "candidate selected symptom test regressed: TestOld failed at candidate on rerun and passed at parent"
	if got != abi.WitnessRefuted || detail != want {
		t.Fatalf("outcome=%v detail=%q, want REFUTED %q (an unrelated red old body must not tolerate a regression)", got, detail, want)
	}
}

// fak-test:runtime fast est=10ms
func TestSymptomCandidateRedInfraAfterExplicitFailRefutes(t *testing.T) {
	const cand = candRedFixOnly + "\nfunc TestStale(t *testing.T) { t.Log(\"changed\") }\n"
	const par = "package pkg\n\nimport \"testing\"\n\nfunc TestStale(t *testing.T) {}\n"
	for _, tc := range []struct {
		name     string
		verdicts map[string][]string
		want     string
	}{
		{
			name: "rerun runner error",
			verdicts: map[string][]string{
				"cand|TestFix":   {"pass"},
				"cand|TestStale": {"fail", "infra"},
			},
			want: "candidate selected symptom test failed: TestStale gave no explicit verdict on rerun",
		},
		{
			name: "parent-own runner error",
			verdicts: map[string][]string{
				"cand|TestFix":      {"pass"},
				"cand|TestStale":    {"fail", "fail"},
				"parown|TestStale":  {"infra"},
				"parcand|TestStale": {"fail"},
				"parcand|TestFix":   {"fail"},
			},
			want: "candidate selected symptom test failed: TestStale gave no explicit verdict at parent",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rig := newCandRedRig(t, tc.verdicts)
			got, detail := candRedResolve(t, rig, cand, par)
			if got != abi.WitnessRefuted || detail != tc.want {
				t.Fatalf("outcome=%v detail=%q, want REFUTED %q (an infrastructure error after an explicit fail must refute, not abstain)", got, detail, tc.want)
			}
		})
	}
}

// fak-test:runtime fast est=10ms
func TestSymptomCandidateRedTeardownMaskedRefutes(t *testing.T) {
	// TestFix passes its first run but the candidate-PASS rerun exits non-zero with pass
	// events: a teardown failure masked as a pass must not confirm the tolerated red.
	const cand = candRedFixOnly + "\nfunc TestStale(t *testing.T) { t.Log(\"changed\") }\n"
	const par = "package pkg\n\nimport \"testing\"\n\nfunc TestStale(t *testing.T) {}\n"
	rig := newCandRedRig(t, map[string][]string{
		"cand|TestFix":      {"pass", "passexit1"},
		"cand|TestStale":    {"fail", "fail"},
		"parown|TestStale":  {"fail"},
		"parcand|TestStale": {"fail"},
		"parcand|TestFix":   {"fail"},
	})
	got, detail := candRedResolve(t, rig, cand, par)
	want := "candidate selected symptom test failed"
	if got != abi.WitnessRefuted || detail != want {
		t.Fatalf("outcome=%v detail=%q, want REFUTED %q", got, detail, want)
	}
}

// fak-test:runtime fast est=10ms
func TestSymptomCandidateRedNewTestRegressionRefutes(t *testing.T) {
	const cand = candRedFixOnly + "\nfunc TestNew(t *testing.T) {}\n"
	rig := newCandRedRig(t, map[string][]string{
		"cand|TestFix": {"pass"},
		"cand|TestNew": {"fail", "fail"},
	})
	// The parent has no such file, so TestNew is absent there.
	got, detail := candRedResolve(t, rig, cand, "")
	want := "candidate selected symptom test regressed: TestNew failed at candidate on rerun and is absent at parent"
	if got != abi.WitnessRefuted || detail != want {
		t.Fatalf("outcome=%v detail=%q, want REFUTED %q", got, detail, want)
	}
	for _, c := range rig.log {
		if strings.HasPrefix(c.where, "par") {
			t.Fatalf("a new test regression reached the parent with runs %v; it must refute before parent preparation", c.names)
		}
	}
}

// fak-test:runtime fast est=10ms
func TestSymptomCandidateRedNoPassStillRefutesWithoutRerun(t *testing.T) {
	const cand = "package pkg\n\nimport \"testing\"\n\nfunc TestFix(t *testing.T) {}\n"
	rig := newCandRedRig(t, map[string][]string{"cand|TestFix": {"fail"}})
	got, detail := candRedResolve(t, rig, cand, "")
	if got != abi.WitnessRefuted || detail != "candidate selected symptom test failed" {
		t.Fatalf("outcome=%v detail=%q, want REFUTED \"candidate selected symptom test failed\"", got, detail)
	}
	if len(rig.log) != 1 {
		t.Fatalf("candidate with no PASS ran %d go test invocations, want 1 (no rerun)", len(rig.log))
	}
}

// fak-test:runtime fast est=10ms
func TestSymptomCandidateRedTestMainFailureIsUnknown(t *testing.T) {
	const cand = "package pkg\n\nimport \"testing\"\n\nfunc TestFix(t *testing.T) {}\n"
	rig := newCandRedRig(t, map[string][]string{"cand|TestFix": {"passexit1"}})
	got, detail := candRedResolve(t, rig, cand, "")
	if got != abi.WitnessRefuted || detail != "candidate selected symptom test failed" {
		t.Fatalf("outcome=%v detail=%q, want REFUTED \"candidate selected symptom test failed\"", got, detail)
	}
	if len(rig.log) != 1 {
		t.Fatalf("a pass under a non-zero exit ran %d invocations, want 1 (no rerun, no tolerance)", len(rig.log))
	}
}

// fak-test:runtime fast est=10ms
func TestSymptomCandidateRedPartialMatchNonZeroRefutes(t *testing.T) {
	// TestB never starts (no run event) while the run exits non-zero: the base refusal
	// stands, with no rerun.
	const cand = candRedFixOnly + "\nfunc TestB(t *testing.T) {}\n"
	rig := newCandRedRig(t, map[string][]string{
		"cand|TestFix": {"pass"},
		"cand|TestB":   {"absent"},
	})
	got, detail := candRedResolve(t, rig, cand, "")
	if got != abi.WitnessRefuted || detail != "candidate symptom selector matched no executed test" {
		t.Fatalf("outcome=%v detail=%q, want REFUTED \"candidate symptom selector matched no executed test\"", got, detail)
	}
	if len(rig.log) != 1 {
		t.Fatalf("partial match ran %d go test invocations, want 1 (no rerun)", len(rig.log))
	}
}

// fak-test:runtime fast est=10ms
func TestSymptomCandidateRedSiblingPanicUnknownRerun(t *testing.T) {
	// TestB is red everywhere (pre-existing). TestC starts without a verdict on the first
	// run and passes on the rerun, so it refutes. TestFix is the genuine candidate PASS.
	const cand = candRedFixOnly + "\nfunc TestB(t *testing.T) { t.Log(\"changed\") }\n\nfunc TestC(t *testing.T) { t.Log(\"changed\") }\n"
	const par = "package pkg\n\nimport \"testing\"\n\nfunc TestB(t *testing.T) {}\n\nfunc TestC(t *testing.T) {}\n"
	rig := newCandRedRig(t, map[string][]string{
		"cand|TestFix":    {"pass"},
		"cand|TestB":      {"fail", "fail"},
		"cand|TestC":      {"none", "pass"},
		"parown|TestB":    {"fail"},
		"parcand|TestB":   {"fail"},
		"parcand|TestFix": {"fail"},
	})
	got, detail := candRedResolve(t, rig, cand, par)
	want := "candidate selected symptom test failed"
	if got != abi.WitnessRefuted || detail != want {
		t.Fatalf("outcome=%v detail=%q, want REFUTED %q", got, detail, want)
	}
}

// fak-test:runtime fast est=10ms
func TestSymptomCandidateRedUnresolvedRerunRefutes(t *testing.T) {
	const cand = candRedFixOnly + "\nfunc TestB(t *testing.T) {}\n"
	rig := newCandRedRig(t, map[string][]string{
		"cand|TestFix": {"pass"},
		"cand|TestB":   {"fail", "none"},
	})
	got, detail := candRedResolve(t, rig, cand, "")
	want := "candidate selected symptom test failed: TestB gave no explicit verdict on rerun"
	if got != abi.WitnessRefuted || detail != want {
		t.Fatalf("outcome=%v detail=%q, want REFUTED %q", got, detail, want)
	}
}

// fak-test:runtime fast est=10ms
func TestSymptomCandidateRedParentOwnUnresolvedRefutes(t *testing.T) {
	const cand = candRedFixOnly + "\nfunc TestStale(t *testing.T) { t.Log(\"changed\") }\n"
	const par = "package pkg\n\nimport \"testing\"\n\nfunc TestStale(t *testing.T) {}\n"
	rig := newCandRedRig(t, map[string][]string{
		"cand|TestFix":      {"pass"},
		"cand|TestStale":    {"fail", "fail"},
		"parown|TestStale":  {"none"},
		"parcand|TestStale": {"fail"},
		"parcand|TestFix":   {"fail"},
	})
	got, detail := candRedResolve(t, rig, cand, par)
	want := "candidate selected symptom test failed: TestStale gave no explicit verdict at parent"
	if got != abi.WitnessRefuted || detail != want {
		t.Fatalf("outcome=%v detail=%q, want REFUTED %q (a parent without a verdict must not abstain)", got, detail, want)
	}
}

// candRedTwoSelections resolves a change selecting two packages. Selections run in
// package-name order, so "alpha" is the first selection and "beta" the second.
func candRedTwoSelections(t *testing.T, alphaOut string, alphaCode int, betaOut string, betaCode int) (abi.WitnessOutcome, string) {
	t.Helper()
	const alphaRel = "alpha/red_test.go"
	const betaRel = "beta/slow_test.go"
	const alphaSrc = "package alpha\n\nimport \"testing\"\n\nfunc TestRed(t *testing.T) {}\n"
	const betaSrc = "package beta\n\nimport \"testing\"\n\nfunc TestSlow(t *testing.T) {}\n"
	git := func(_ context.Context, _ string, args ...string) (string, int, error) {
		switch {
		case len(args) == 3 && args[0] == "rev-parse" && args[2] == "HEAD^{commit}":
			return "cand\n", 0, nil
		case len(args) == 3 && args[0] == "rev-parse" && args[2] == "cand^":
			return "par\n", 0, nil
		case len(args) == 2 && args[0] == "show" && args[1] == "cand:"+alphaRel:
			return alphaSrc, 0, nil
		case len(args) == 2 && args[0] == "show" && args[1] == "cand:"+betaRel:
			return betaSrc, 0, nil
		}
		return "", 128, nil
	}
	exec := func(_ context.Context, _ string, argv ...string) (string, int, error) {
		run := ""
		for i, a := range argv {
			if a == "-run" && i+1 < len(argv) {
				run = argv[i+1]
			}
		}
		switch {
		case strings.Contains(run, "TestRed"):
			return alphaOut, alphaCode, nil
		case strings.Contains(run, "TestSlow"):
			return betaOut, betaCode, nil
		}
		return "", 0, nil
	}
	cand, par := t.TempDir(), t.TempDir()
	ctx := WithExecutionMaterializer(context.Background(), func(_ context.Context, _ string, ref string) (string, func(), error) {
		if ref == "cand" {
			return cand, func() {}, nil
		}
		return par, func() {}, nil
	})
	r := NewWithRunners(git, exec, t.TempDir())
	return r.resolveSymptomExec(ctx, "HEAD", []string{alphaRel, betaRel})
}

// fak-test:runtime fast est=10ms
func TestSymptomCandidateRedLaterTimeoutAfterRedRefutes(t *testing.T) {
	// The first selection is red; base refutes there, so a later timeout must not abstain.
	const redOut = "{\"Action\":\"run\",\"Test\":\"TestRed\"}\n{\"Action\":\"fail\",\"Test\":\"TestRed\"}\n"
	got, detail := candRedTwoSelections(t, redOut, 1, "panic: test timed out after 10m0s\n", 1)
	want := "candidate selected symptom test failed"
	if got != abi.WitnessRefuted || detail != want {
		t.Fatalf("outcome=%v detail=%q, want REFUTED %q", got, detail, want)
	}
}

// fak-test:runtime fast est=10ms
func TestSymptomCandidateRedLaterBuildFailureAfterRedRefutes(t *testing.T) {
	const redOut = "{\"Action\":\"run\",\"Test\":\"TestRed\"}\n{\"Action\":\"fail\",\"Test\":\"TestRed\"}\n"
	got, detail := candRedTwoSelections(t, redOut, 1, "# m/beta\nbeta/slow_test.go:3:2: undefined: Missing\n", 1)
	want := "candidate selected symptom test failed"
	if got != abi.WitnessRefuted || detail != want {
		t.Fatalf("outcome=%v detail=%q, want REFUTED %q", got, detail, want)
	}
}
