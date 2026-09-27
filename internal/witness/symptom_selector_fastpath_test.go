package witness

import (
	"context"
	"fmt"
	"reflect"
	"strings"
	"testing"
)

func TestDiscoverSymptomSelectionsExactChangedSelectorUsesBatchedGrep(t *testing.T) {
	const candidate = `package p
import "testing"
func TestSharedWitness(t *testing.T) {}
`
	const parent = `package p
import "testing"
func TestSharedWitness(t *testing.T) { t.Log("parent") }
`
	changed := []string{"pkg/alpha/changed_test.go", "pkg/beta/changed_test.go"}
	var showCalls, grepCalls, treeCalls int
	run := func(_ context.Context, _ string, args ...string) (string, int, error) {
		switch args[0] {
		case "show":
			showCalls++
			spec := args[1]
			if strings.HasPrefix(spec, "candidate:") {
				return candidate, 0, nil
			}
			if strings.HasPrefix(spec, "parent:") {
				return parent, 0, nil
			}
			t.Fatalf("unexpected show spec %q", spec)
		case "grep":
			grepCalls++
			want := []string{"grep", "-l", "-F", "-e", "TestSharedWitness", "candidate", "--", "./pkg/alpha", "./pkg/beta"}
			if !reflect.DeepEqual(args, want) {
				t.Fatalf("grep args = %v, want %v", args, want)
			}
			return "candidate:pkg/alpha/changed_test.go\ncandidate:pkg/beta/changed_test.go\n", 0, nil
		case "ls-tree":
			treeCalls++
			paths := make([]string, 1964)
			for i := range paths {
				paths[i] = fmt.Sprintf("pkg/alpha/decoy_%04d_test.go", i)
			}
			return strings.Join(paths, "\n"), 0, nil
		}
		return "", 128, fmt.Errorf("unexpected git args %v", args)
	}

	got, detail := discoverSymptomSelections(context.Background(), run, "repo", "candidate", "parent", changed, []string{"^TestSharedWitness$"})
	want := []symptomSelection{
		{Package: "./pkg/alpha", Tests: []string{"TestSharedWitness"}},
		{Package: "./pkg/beta", Tests: []string{"TestSharedWitness"}},
	}
	if !reflect.DeepEqual(got, want) || detail != "" {
		t.Fatalf("selections=%v detail=%q, want %v with no detail", got, detail, want)
	}
	if grepCalls != 1 || treeCalls != 0 {
		t.Fatalf("grep calls=%d ls-tree calls=%d, want one batched grep and no tree scan", grepCalls, treeCalls)
	}
	if showCalls != 6 {
		t.Fatalf("git show calls=%d, want 6 (changed candidate+parent and two grep hits), not 1,964 serial fallback reads", showCalls)
	}
}

func TestDiscoverSymptomSelectionsBroadSelectorPreservesFullScan(t *testing.T) {
	changed := []string{"pkg/alpha/changed_test.go", "pkg/beta/changed_test.go"}
	sources := map[string]string{
		"pkg/alpha/changed_test.go": goTestSource("TestSharedWitness"),
		"pkg/beta/changed_test.go":  goTestSource("TestSharedWitness"),
		"pkg/alpha/legacy_test.go":  goTestSource("TestSharedLegacy"),
		"pkg/beta/legacy_test.go":   goTestSource("TestSharedLegacy"),
	}
	var grepCalls, treeCalls int
	run := func(_ context.Context, _ string, args ...string) (string, int, error) {
		switch args[0] {
		case "show":
			spec := args[1]
			if strings.HasPrefix(spec, "parent:") {
				return "", 128, nil
			}
			body, ok := sources[strings.TrimPrefix(spec, "candidate:")]
			if !ok {
				return "", 128, nil
			}
			return body, 0, nil
		case "grep":
			grepCalls++
			return "", 0, nil
		case "ls-tree":
			treeCalls++
			return strings.Join([]string{
				"pkg/alpha/changed_test.go", "pkg/alpha/legacy_test.go",
				"pkg/beta/changed_test.go", "pkg/beta/legacy_test.go",
			}, "\n"), 0, nil
		}
		return "", 128, fmt.Errorf("unexpected git args %v", args)
	}

	got, detail := discoverSymptomSelections(context.Background(), run, "repo", "candidate", "parent", changed, []string{"^TestShared.*$"})
	wantTests := []string{"TestSharedLegacy", "TestSharedWitness"}
	want := []symptomSelection{
		{Package: "./pkg/alpha", Tests: wantTests},
		{Package: "./pkg/beta", Tests: wantTests},
	}
	if !reflect.DeepEqual(got, want) || detail != "" {
		t.Fatalf("selections=%v detail=%q, want %v with no detail", got, detail, want)
	}
	if grepCalls != 0 || treeCalls != 1 {
		t.Fatalf("grep calls=%d ls-tree calls=%d, want broad selector to use one full scan", grepCalls, treeCalls)
	}
}

func TestDiscoverSymptomSelectionsUnmatchedExactSelectorPreservesFullScan(t *testing.T) {
	changed := []string{"pkg/alpha/changed_test.go"}
	var grepCalls, treeCalls int
	run := func(_ context.Context, _ string, args ...string) (string, int, error) {
		switch args[0] {
		case "show":
			if strings.HasPrefix(args[1], "parent:") {
				return "", 128, nil
			}
			return goTestSource("TestExisting"), 0, nil
		case "grep":
			grepCalls++
			return "", 1, nil
		case "ls-tree":
			treeCalls++
			return "pkg/alpha/changed_test.go\n", 0, nil
		}
		return "", 128, fmt.Errorf("unexpected git args %v", args)
	}

	got, detail := discoverSymptomSelections(context.Background(), run, "repo", "candidate", "parent", changed, []string{"^TestMissing$"})
	if got != nil || detail != "explicit symptom selector matched no changed top-level Test/Example" {
		t.Fatalf("selections=%v detail=%q, want unmatched-selector abstention detail", got, detail)
	}
	if grepCalls != 0 || treeCalls != 1 {
		t.Fatalf("grep calls=%d ls-tree calls=%d, want unmatched exact selector to use one full scan", grepCalls, treeCalls)
	}
}

func goTestSource(name string) string {
	return "package p\nimport \"testing\"\nfunc " + name + "(t *testing.T) {}\n"
}
