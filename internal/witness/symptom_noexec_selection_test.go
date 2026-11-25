package witness

import (
	"context"
	"reflect"
	"runtime"
	"sort"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// fak-test:runtime fast est=10ms
func TestTopLevelGoTestsNoExecEntriesExcluded(t *testing.T) {
	const source = `package p

import (
	"fmt"
	"os"
	"testing"
)

func TestMain(m *testing.M) { os.Exit(m.Run()) }

func TestReal(t *testing.T) {}

func TestWrongSignature(b *testing.B) {}

func TestTwoParams(t *testing.T, n int) {}

func TestReturns(t *testing.T) error { return nil }

func ExampleQuiet() { fmt.Println("x") }

func ExampleLoud() {
	fmt.Println("x")
	// Output: x
}

func ExampleUnordered() {
	fmt.Println("a")
	// Unordered output: a
}
`
	funcs, err := topLevelGoTests(source)
	if err != nil {
		t.Fatal(err)
	}
	var got []string
	for name := range funcs {
		got = append(got, name)
	}
	sort.Strings(got)
	want := []string{"ExampleLoud", "ExampleUnordered", "TestReal"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("topLevelGoTests names = %v, want %v", got, want)
	}
}

// fak-test:runtime fast est=10ms
func TestMergeTagsNoExecDropsToolchainOwnedTags(t *testing.T) {
	got := mergeTags([]string{"linux", "!linux", "unix", "cgo", "go1.21"}, []string{"gc", "gccgo", "ignore", "vulkan"})
	if !reflect.DeepEqual(got, []string{"vulkan"}) {
		t.Fatalf("mergeTags = %v, want [vulkan]", got)
	}
}

// fak-test:runtime fast est=50ms
func TestSymptomNoExecForeignGOOSTestAbstainsOnSelector(t *testing.T) {
	foreign := "linux"
	if runtime.GOOS == "linux" {
		foreign = "windows"
	}
	const rel = "pkg/runner_pgroup_test.go"
	source := "//go:build " + foreign + "\n\npackage pkg\n\nimport \"testing\"\n\nfunc TestProcessGroupKill(t *testing.T) {}\n"
	git := func(_ context.Context, _ string, args ...string) (string, int, error) {
		switch {
		case len(args) == 3 && args[0] == "rev-parse" && args[2] == "HEAD^{commit}":
			return "cand\n", 0, nil
		case len(args) == 3 && args[0] == "rev-parse" && args[2] == "cand^":
			return "par\n", 0, nil
		case len(args) == 2 && args[0] == "show" && args[1] == "cand:"+rel:
			return source, 0, nil
		}
		return "", 128, nil
	}
	execCalls := 0
	exec := func(_ context.Context, _ string, _ ...string) (string, int, error) {
		execCalls++
		return "{\"Action\":\"pass\",\"Package\":\"m/pkg\"}\n", 0, nil
	}
	scratch := t.TempDir()
	ctx := WithExecutionMaterializer(context.Background(), func(context.Context, string, string) (string, func(), error) {
		return scratch, func() {}, nil
	})
	r := NewWithRunners(git, exec, t.TempDir())
	got, detail := r.resolveSymptomExec(ctx, "HEAD", []string{rel})
	if got != abi.WitnessAbstain || !strings.Contains(detail, "selector") {
		t.Fatalf("foreign-GOOS changed test = %v (%s), want abstain with a selector detail", got, detail)
	}
	for _, refuting := range []string{"matched no", "did not build", "failed", "passed at parent", "timed out"} {
		if strings.Contains(detail, refuting) {
			t.Fatalf("detail %q contains %q, which maps to a refusing token", detail, refuting)
		}
	}
	if execCalls != 0 {
		t.Fatalf("host-excluded selection ran %d test command(s), want 0", execCalls)
	}
}

// fak-test:runtime fast est=10ms
func TestTopLevelGoTestsNoExecKeepsDotImportedT(t *testing.T) {
	funcs, err := topLevelGoTests("package p\n\nimport . \"testing\"\n\nfunc TestDot(t *T) {}\n\nfunc TestMain(m *M) {}\n")
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := funcs["TestDot"]; !ok || len(funcs) != 1 {
		t.Fatalf("topLevelGoTests = %v, want only TestDot", funcs)
	}
}

// fak-test:runtime fast est=50ms
func TestSymptomNoExecKeepsToolchainDependentConstraint(t *testing.T) {
	const rel = "pkg/fallback_test.go"
	source := "//go:build !cgo\n\npackage pkg\n\nimport \"testing\"\n\nfunc TestFallback(t *testing.T) {}\n"
	git := func(_ context.Context, _ string, args ...string) (string, int, error) {
		switch {
		case len(args) == 3 && args[0] == "rev-parse" && args[2] == "HEAD^{commit}":
			return "cand\n", 0, nil
		case len(args) == 3 && args[0] == "rev-parse" && args[2] == "cand^":
			return "par\n", 0, nil
		case len(args) == 2 && args[0] == "show" && args[1] == "cand:"+rel:
			return source, 0, nil
		}
		return "", 128, nil
	}
	execCalls := 0
	exec := func(_ context.Context, _ string, _ ...string) (string, int, error) {
		execCalls++
		return "{\"Action\":\"pass\",\"Package\":\"m/pkg\"}\n", 0, nil
	}
	scratch := t.TempDir()
	ctx := WithExecutionMaterializer(context.Background(), func(context.Context, string, string) (string, func(), error) {
		return scratch, func() {}, nil
	})
	r := NewWithRunners(git, exec, t.TempDir())
	if _, detail := r.resolveSymptomExec(ctx, "HEAD", []string{rel}); execCalls == 0 {
		t.Fatalf("a cgo-dependent changed test was dropped without running (%s)", detail)
	}
}
