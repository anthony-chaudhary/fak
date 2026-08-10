package main

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/buildoverlay"
)

// withGoShimSeams swaps the shim's impure seams (the SAME untracked/modified-dirs listers
// buildcheck uses, plus the go runner) for the duration of the test, so runGoShim is
// exercised hermetically without git or spawning go.
func withGoShimSeams(t *testing.T, untracked []string, modifiedDirs map[string]bool, runFn func(root string, args []string, stdout, stderr io.Writer) (int, error)) {
	t.Helper()
	origU, origM, origL, origR := buildCheckUntracked, buildCheckModifiedDirs, buildCheckLoadBearing, goShimRun
	t.Cleanup(func() {
		buildCheckUntracked, buildCheckModifiedDirs, buildCheckLoadBearing, goShimRun = origU, origM, origL, origR
	})
	buildCheckUntracked = func(string) ([]string, error) { return untracked, nil }
	buildCheckModifiedDirs = func(string) (map[string]bool, error) { return modifiedDirs, nil }
	buildCheckLoadBearing = func(string, []string) ([]string, error) { return nil, nil }
	goShimRun = runFn
}

func TestExtractMineFlags(t *testing.T) {
	cases := []struct {
		name     string
		argv     []string
		wantMine []string
		wantRest []string
	}{
		{"none", []string{"build", "./..."}, nil, []string{"build", "./..."}},
		{"space form", []string{"--mine", "a/b.go", "build", "./..."}, []string{"a/b.go"}, []string{"build", "./..."}},
		{"equals form", []string{"build", "--mine=a/b.go", "./..."}, []string{"a/b.go"}, []string{"build", "./..."}},
		{"single dash", []string{"-mine", "a/b.go", "vet"}, []string{"a/b.go"}, []string{"vet"}},
		{"repeated", []string{"--mine", "a.go", "test", "--mine=b.go", "./x"}, []string{"a.go", "b.go"}, []string{"test", "./x"}},
		{"trailing dangling mine dropped", []string{"build", "--mine"}, nil, []string{"build"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			mine, rest := extractMineFlags(tc.argv)
			if !reflect.DeepEqual(mine, tc.wantMine) {
				t.Errorf("mine = %v, want %v", mine, tc.wantMine)
			}
			if !reflect.DeepEqual(rest, tc.wantRest) {
				t.Errorf("rest = %v, want %v", rest, tc.wantRest)
			}
		})
	}
}

func TestGoShimArgsInjectsOverlayAfterSubcommand(t *testing.T) {
	got := goShimArgs("/tmp/ov.json", []string{"test", "-run", "X", "./..."})
	want := []string{"test", "-overlay", "/tmp/ov.json", "-run", "X", "./..."}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("goShimArgs = %v, want %v", got, want)
	}
}

func TestGoShimArgsNoOverlayIsVerbatim(t *testing.T) {
	rest := []string{"build", "./..."}
	got := goShimArgs("", rest)
	if !reflect.DeepEqual(got, rest) {
		t.Errorf("goShimArgs with no overlay = %v, want verbatim %v", got, rest)
	}
}

// TestGoShimOverlayEqualsBuildcheckSelection is the DoD anti-fork assertion: for the same
// tree state the shim masks the IDENTICAL file set as fak buildcheck, because both call the
// shared selectMaskedFiles fold — and the overlay bytes it writes equal buildOverlay's.
func TestGoShimOverlayKeepsLoadBearingUntrackedPackage(t *testing.T) {
	untracked := []string{"internal/loadbearing/load.go", "internal/orphan/orphan.go"}
	withGoShimSeams(t, untracked, nil, nil)
	buildCheckLoadBearing = func(string, []string) ([]string, error) {
		return []string{"internal/loadbearing/load.go"}, nil
	}
	var errb bytes.Buffer
	masked, _, code := goShimOverlay("/repo", nil, t.TempDir(), &errb)
	if code != 0 {
		t.Fatalf("goShimOverlay code = %d, stderr=%s", code, errb.String())
	}
	want := []string{"internal/orphan/orphan.go"}
	if !reflect.DeepEqual(masked, want) {
		t.Fatalf("masked = %v, want only orphan %v", masked, want)
	}
}

func TestGoShimOverlayEqualsBuildcheckSelection(t *testing.T) {
	untracked := []string{"cmd/fak/peer_wip.go", "internal/x/new.go", "cmd/fak/mine.go"}
	modifiedDirs := map[string]bool{"internal/x": true} // internal/x has an in-flight edit -> its untracked sibling is KEPT
	mine := []string{"cmd/fak/mine.go"}

	// buildcheck's canonical selection for this tree state:
	wantMasked, _, _ := buildoverlay.SelectMaskedFiles(untracked, mine, modifiedDirs)

	withGoShimSeams(t, untracked, modifiedDirs, nil)
	scratch := t.TempDir()
	var errb bytes.Buffer
	gotMasked, overlayPath, code := goShimOverlay("/repo", mine, scratch, &errb)
	if code != 0 {
		t.Fatalf("goShimOverlay code = %d, stderr=%s", code, errb.String())
	}
	if !reflect.DeepEqual(gotMasked, wantMasked) {
		t.Fatalf("shim masked %v, buildcheck masks %v — masking logic has forked", gotMasked, wantMasked)
	}
	// The overlay file's bytes must equal buildOverlay(root, masked) — same hiding.
	raw, err := os.ReadFile(overlayPath)
	if err != nil {
		t.Fatalf("read overlay: %v", err)
	}
	var got buildoverlay.Overlay
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("overlay json: %v", err)
	}
	want := buildoverlay.Build("/repo", wantMasked)
	if !reflect.DeepEqual(got.Replace, want.Replace) {
		t.Errorf("overlay Replace = %v, want %v", got.Replace, want.Replace)
	}
	// The KEPT untracked sibling in the edited dir must NOT be masked.
	if _, ok := want.Replace[filepath.Clean(filepath.Join("/repo", filepath.FromSlash("internal/x/new.go")))]; ok {
		t.Error("matched untracked sibling in an edited dir was masked; should be kept")
	}
}

func TestRunGoShimInjectsOverlayForBuild(t *testing.T) {
	var gotArgs []string
	withGoShimSeams(t, []string{"cmd/fak/peer_wip.go"}, nil, func(_ string, args []string, _, _ io.Writer) (int, error) {
		gotArgs = args
		return 0, nil
	})
	var out, errb bytes.Buffer
	if rc := runGoShim(&out, &errb, []string{"build", "./..."}); rc != 0 {
		t.Fatalf("rc = %d, want 0; stderr=%s", rc, errb.String())
	}
	if len(gotArgs) < 3 || gotArgs[0] != "build" || gotArgs[1] != "-overlay" {
		t.Errorf("args %v: want overlay injected right after `build`", gotArgs)
	}
	if gotArgs[len(gotArgs)-1] != "./..." {
		t.Errorf("last arg = %q, want ./... (user args forwarded)", gotArgs[len(gotArgs)-1])
	}
}

func TestRunGoShimForwardsExitCode(t *testing.T) {
	withGoShimSeams(t, nil, nil, func(_ string, args []string, _, _ io.Writer) (int, error) {
		if strings.Contains(strings.Join(args, " "), "-overlay") {
			t.Errorf("no untracked files -> no overlay expected; got %v", args)
		}
		return 7, nil
	})
	var out, errb bytes.Buffer
	if rc := runGoShim(&out, &errb, []string{"test", "./..."}); rc != 7 {
		t.Fatalf("rc = %d, want 7 (exit code forwarded verbatim)", rc)
	}
}

func TestRunGoShimPassthroughForNonBuildSubcommand(t *testing.T) {
	var gotArgs []string
	withGoShimSeams(t, []string{"cmd/fak/peer_wip.go"}, nil, func(_ string, args []string, _, _ io.Writer) (int, error) {
		gotArgs = args
		return 0, nil
	})
	var out, errb bytes.Buffer
	if rc := runGoShim(&out, &errb, []string{"env", "GOFLAGS"}); rc != 0 {
		t.Fatalf("rc = %d, want 0", rc)
	}
	if !reflect.DeepEqual(gotArgs, []string{"env", "GOFLAGS"}) {
		t.Errorf("env passthrough args = %v, want verbatim; -overlay must not be injected for `go env`", gotArgs)
	}
}

func TestRunGoShimLiveCrossCheckSuppressesFalseRed(t *testing.T) {
	calls := 0
	withGoShimSeams(t, []string{"cmd/fak/peer_wip.go"}, nil, func(_ string, args []string, _, _ io.Writer) (int, error) {
		calls++
		hasOverlay := strings.Contains(strings.Join(args, " "), "-overlay")
		if calls == 1 {
			if !hasOverlay {
				t.Errorf("first (masked) run missing -overlay: %v", args)
			}
			return 2, nil // masked build reds...
		}
		// ...second run is the live cross-check (no overlay) and it compiles.
		if hasOverlay {
			t.Errorf("cross-check run should have NO overlay: %v", args)
		}
		return 0, nil
	})
	var out, errb bytes.Buffer
	if rc := runGoShim(&out, &errb, []string{"build", "./..."}); rc != 0 {
		t.Fatalf("rc = %d, want 0 (mask-induced false red suppressed by live cross-check)", rc)
	}
	if calls != 2 {
		t.Fatalf("go runner called %d times, want 2 (masked + live cross-check)", calls)
	}
	if !strings.Contains(errb.String(), "live tree compiles") {
		t.Errorf("stderr should explain the false red; got %q", errb.String())
	}
}

func TestRunGoShimRealRedSurvivesCrossCheck(t *testing.T) {
	withGoShimSeams(t, []string{"cmd/fak/peer_wip.go"}, nil, func(_ string, args []string, stdout, _ io.Writer) (int, error) {
		io.WriteString(stdout, "undefined: RealBreak\n")
		return 2, nil // both masked and live runs red -> a genuine break
	})
	var out, errb bytes.Buffer
	if rc := runGoShim(&out, &errb, []string{"build", "./..."}); rc != 2 {
		t.Fatalf("rc = %d, want 2 (a real red must survive the cross-check)", rc)
	}
	if !strings.Contains(errb.String(), "undefined: RealBreak") {
		t.Errorf("captured real-red output should be flushed; got %q", errb.String())
	}
}
