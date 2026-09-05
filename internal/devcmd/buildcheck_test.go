package devcmd

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/buildoverlay"
)

// argsHave reports whether flag and its value appear adjacently in args.
func argsHave(args []string, flag, val string) bool {
	for i := 0; i+1 < len(args); i++ {
		if args[i] == flag && args[i+1] == val {
			return true
		}
	}
	return false
}

func TestSelectMaskedFiles(t *testing.T) {
	untracked := []string{
		"cmd/fak/peer_wip.go",
		"cmd/fak/mine_new.go",
		"docs/notes.md",
		"cmd/fak/data.txt",
		"internal/foo/peer2.go",
	}
	masked, kept, stale := buildoverlay.SelectMaskedFiles(untracked, []string{"cmd/fak/mine_new.go"}, nil)
	wantMasked := []string{"cmd/fak/peer_wip.go", "internal/foo/peer2.go"}
	if !reflect.DeepEqual(masked, wantMasked) {
		t.Errorf("masked = %v, want %v (only untracked .go not declared --mine)", masked, wantMasked)
	}
	if len(kept) != 0 {
		t.Errorf("kept = %v, want none (no modified dirs -> nothing kept)", kept)
	}
	if len(stale) != 0 {
		t.Errorf("staleMine = %v, want none (mine_new.go IS untracked)", stale)
	}
}

func TestSelectMaskedFilesStaleAndBackslashMine(t *testing.T) {
	untracked := []string{"cmd/fak/peer.go", "cmd/fak/keep.go"}
	// --mine given with a backslash separator (Windows paste) and a duplicate, plus one
	// path that is not actually untracked -> normalized, deduped, and reported as stale.
	masked, _, stale := buildoverlay.SelectMaskedFiles(untracked, []string{`cmd\fak\keep.go`, "cmd/fak/keep.go", "cmd/fak/tracked.go"}, nil)
	if !reflect.DeepEqual(masked, []string{"cmd/fak/peer.go"}) {
		t.Errorf("masked = %v, want [cmd/fak/peer.go] (keep.go protected via slash-normalized --mine)", masked)
	}
	if !reflect.DeepEqual(stale, []string{"cmd/fak/tracked.go"}) {
		t.Errorf("staleMine = %v, want [cmd/fak/tracked.go] (declared --mine that is not untracked)", stale)
	}
}

func TestSelectMaskedFilesEmpty(t *testing.T) {
	masked, kept, stale := buildoverlay.SelectMaskedFiles(nil, nil, nil)
	if len(masked) != 0 || len(kept) != 0 || len(stale) != 0 {
		t.Errorf("empty inputs -> masked=%v kept=%v stale=%v, want all empty", masked, kept, stale)
	}
}

func TestSelectMaskedFilesKeepsInModifiedDir(t *testing.T) {
	untracked := []string{
		"internal/slackoutbox/compact.go", // matched new file for an edited drain.go
		"internal/peer/newthing.go",       // independent peer WIP, its dir has no edits
	}
	modified := map[string]bool{"internal/slackoutbox": true}
	masked, kept, stale := buildoverlay.SelectMaskedFiles(untracked, nil, modified)
	if !reflect.DeepEqual(kept, []string{"internal/slackoutbox/compact.go"}) {
		t.Errorf("kept = %v, want [internal/slackoutbox/compact.go] (untracked .go in an edited pkg is kept, not masked)", kept)
	}
	if !reflect.DeepEqual(masked, []string{"internal/peer/newthing.go"}) {
		t.Errorf("masked = %v, want [internal/peer/newthing.go] (untracked .go in an un-edited pkg is still masked)", masked)
	}
	if len(stale) != 0 {
		t.Errorf("staleMine = %v, want none", stale)
	}
}

func TestBuildOverlayHidesEachFile(t *testing.T) {
	root := filepath.FromSlash("/repo/root")
	ov := buildoverlay.Build(root, []string{"cmd/fak/peer.go", "internal/foo/bar.go"})
	if len(ov.Replace) != 2 {
		t.Fatalf("Replace has %d entries, want 2", len(ov.Replace))
	}
	wantKey := filepath.Clean(filepath.Join(root, filepath.FromSlash("cmd/fak/peer.go")))
	backing, ok := ov.Replace[wantKey]
	if !ok {
		t.Fatalf("overlay missing key %q; keys=%v", wantKey, ov.Replace)
	}
	if backing != "" {
		t.Errorf("backing for masked file = %q, want \"\" (empty = treated as absent by go)", backing)
	}
}

func TestBuildCheckArgs(t *testing.T) {
	cases := []struct {
		name                             string
		mode, overlay, outTarget, wipTag string
		pkgs, want                       []string
	}{
		{"build discards to null", "build", "", "NUL", "", []string{"./..."},
			[]string{"build", "-trimpath", "-buildvcs=false", "-o", "NUL", "./..."}},
		{"build with overlay and out dir", "build", "ov.json", "out", "", []string{"./cmd/fak"},
			[]string{"build", "-trimpath", "-buildvcs=false", "-overlay", "ov.json", "-o", "out", "./cmd/fak"}},
		{"vet never takes -o", "vet", "ov.json", "out", "", []string{"./..."},
			[]string{"vet", "-trimpath", "-overlay", "ov.json", "./..."}},
		{"build with wip tag", "build", "", "NUL", "wip_feat", []string{"./..."},
			[]string{"build", "-trimpath", "-buildvcs=false", "-tags", "wip_feat", "-o", "NUL", "./..."}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := buildCheckArgs(tc.mode, tc.overlay, tc.outTarget, tc.wipTag, tc.pkgs)
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("buildCheckArgs = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestBuildCheckArgsCarryOneTrimpathFlag(t *testing.T) {
	for _, mode := range []string{"build", "vet"} {
		got := buildCheckArgs(mode, "", "NUL", "", []string{"./p"})
		if len(got) < 2 || got[1] != "-trimpath" {
			t.Fatalf("%s args = %v; want -trimpath as the first build flag", mode, got)
		}
		count := 0
		for _, arg := range got {
			if arg == "-trimpath" {
				count++
			}
		}
		if count != 1 {
			t.Fatalf("%s args = %v; want exactly one -trimpath", mode, got)
		}
		hasBuildVCS := false
		for _, arg := range got {
			if arg == "-buildvcs=false" {
				hasBuildVCS = true
			}
		}
		if mode == "build" && !hasBuildVCS {
			t.Fatalf("build args = %v; want -buildvcs=false present for build mode", got)
		}
		if mode == "vet" && hasBuildVCS {
			t.Fatalf("vet args = %v; want -buildvcs=false omitted for vet mode", got)
		}
	}
}

func TestBuildCheckTail(t *testing.T) {
	in := "a\n\nb\n  \nc\nd\n"
	if got := buildCheckTail(in, 2); got != "c\nd" {
		t.Errorf("buildCheckTail(_,2) = %q, want \"c\\nd\"", got)
	}
	if got := buildCheckTail("only\n", 5); got != "only" {
		t.Errorf("buildCheckTail fewer-than-n = %q, want \"only\"", got)
	}
}

func TestJoinReason(t *testing.T) {
	if got := joinReason("", "b"); got != "b" {
		t.Errorf("joinReason empty-a = %q", got)
	}
	if got := joinReason("a", ""); got != "a" {
		t.Errorf("joinReason empty-b = %q", got)
	}
	if got := joinReason("a", "b"); got != "a: b" {
		t.Errorf("joinReason both = %q, want \"a: b\"", got)
	}
}

// withBuildCheckSeams swaps the impure seams (untracked lister, modified-dirs lister, go
// runner, clock) for the duration of fn, restoring them after -- so the shell is exercised
// hermetically without touching git or spawning go. modifiedDirs is the set of packages
// with in-flight tracked edits (nil = none, i.e. mask every untracked sibling).
func withBuildCheckSeams(t *testing.T, untracked []string, modifiedDirs map[string]bool, runFn func(root string, args []string, stdout, stderr io.Writer) (int, error)) {
	t.Helper()
	origU, origM, origR, origN, origS := buildCheckUntracked, buildCheckModifiedDirs, buildCheckRun, buildCheckNow, buildCheckAcquireSlot
	t.Cleanup(func() {
		buildCheckUntracked, buildCheckModifiedDirs, buildCheckRun, buildCheckNow, buildCheckAcquireSlot = origU, origM, origR, origN, origS
	})
	buildCheckUntracked = func(string) ([]string, error) { return untracked, nil }
	buildCheckModifiedDirs = func(string) (map[string]bool, error) { return modifiedDirs, nil }
	buildCheckRun = runFn
	buildCheckNow = func() time.Time { return time.Unix(0, 0) }
	buildCheckAcquireSlot = func(context.Context, time.Duration) (func(), error) { return func() {}, nil }
}

func TestRunBuildCheckIsolatesSiblings(t *testing.T) {
	var gotArgs []string
	withBuildCheckSeams(t, []string{"cmd/fak/peer_wip.go"}, nil, func(_ string, args []string, _, _ io.Writer) (int, error) {
		gotArgs = args
		return 0, nil
	})
	var out, errb bytes.Buffer
	if rc := RunBuildCheck(&out, &errb, []string{"./cmd/fak"}); rc != 0 {
		t.Fatalf("rc = %d, want 0; stderr=%s", rc, errb.String())
	}
	joined := strings.Join(gotArgs, " ")
	if !strings.Contains(joined, "-overlay") {
		t.Errorf("args %v missing -overlay (sibling isolation not applied)", gotArgs)
	}
	if !argsHave(gotArgs, "-o", os.DevNull) {
		t.Errorf("args %v missing `-o %s` (a compile check must discard, never drop a binary in the tree)", gotArgs, os.DevNull)
	}
	if gotArgs[len(gotArgs)-1] != "./cmd/fak" {
		t.Errorf("last arg = %q, want ./cmd/fak", gotArgs[len(gotArgs)-1])
	}
	if !strings.Contains(errb.String(), "peer_wip.go") {
		t.Errorf("stderr does not name the masked file; got %q", errb.String())
	}
}

func TestRunBuildCheckNoIsolate(t *testing.T) {
	called := false
	withBuildCheckSeams(t, []string{"cmd/fak/peer_wip.go"}, nil, func(_ string, args []string, _, _ io.Writer) (int, error) {
		called = true
		if strings.Contains(strings.Join(args, " "), "-overlay") {
			t.Errorf("--isolate=false still passed -overlay: %v", args)
		}
		return 0, nil
	})
	var out, errb bytes.Buffer
	if rc := RunBuildCheck(&out, &errb, []string{"--isolate=false", "./..."}); rc != 0 {
		t.Fatalf("rc = %d, want 0; stderr=%s", rc, errb.String())
	}
	if !called {
		t.Fatal("go runner never invoked")
	}
}

func TestRunBuildCheckJSONReport(t *testing.T) {
	withBuildCheckSeams(t, []string{"cmd/fak/peer_wip.go"}, nil, func(_ string, _ []string, _, _ io.Writer) (int, error) {
		return 0, nil
	})
	var out, errb bytes.Buffer
	if rc := RunBuildCheck(&out, &errb, []string{"--json", "./..."}); rc != 0 {
		t.Fatalf("rc = %d, want 0", rc)
	}
	var rep buildCheckReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("stdout is not valid JSON report: %v\n%s", err, out.String())
	}
	if rep.Schema != "fak.buildcheck.v1" || rep.Verdict != "OK" || rep.Mode != "build" {
		t.Errorf("report = %+v, want schema/verdict OK/build", rep)
	}
	if rep.MaskedCount != 1 || !rep.Isolate {
		t.Errorf("report masked/isolate = %d/%v, want 1/true", rep.MaskedCount, rep.Isolate)
	}
	if rep.Output != os.DevNull {
		t.Errorf("report.Output = %q, want the null device %q (default discard)", rep.Output, os.DevNull)
	}
	if len(rep.Command) < 2 || rep.Command[0] != "go" || rep.Command[1] != "build" {
		t.Errorf("report.Command = %v, want it to start with [go build ...]", rep.Command)
	}
	if rep.Delivery == nil || rep.Delivery.Receipt == nil || rep.Delivery.Receipt.Transition.Axis != "verification" || rep.Delivery.Receipt.Transition.To != "passed" {
		t.Errorf("delivery = %+v, want verification-only passed receipt", rep.Delivery)
	}
	if rep.Delivery.Receipt.Transition.Axis == "integration" || rep.Delivery.Receipt.Transition.Axis == "release" {
		t.Fatal("buildcheck inferred a downstream delivery axis")
	}
}

func TestRunBuildCheckIsolateWIPTrunk(t *testing.T) {
	var gotArgs []string
	withBuildCheckSeams(t, []string{"cmd/fak/peer_wip.go"}, nil, func(_ string, args []string, _, _ io.Writer) (int, error) {
		gotArgs = args
		return 0, nil
	})
	var out, errb bytes.Buffer
	if rc := RunBuildCheck(&out, &errb, []string{"--isolate-wip", "./cmd/fak"}); rc != 0 {
		t.Fatalf("rc = %d, want 0; stderr=%s", rc, errb.String())
	}
	for i, arg := range gotArgs {
		if arg == "-tags" {
			t.Errorf("expected no -tags for trunk isolate-wip, got %v", gotArgs[i:i+2])
		}
	}
}

func TestRunBuildCheckIsolateWIPTag(t *testing.T) {
	var gotArgs []string
	withBuildCheckSeams(t, []string{"cmd/fak/peer_wip.go"}, nil, func(_ string, args []string, _, _ io.Writer) (int, error) {
		gotArgs = args
		return 0, nil
	})
	var out, errb bytes.Buffer
	if rc := RunBuildCheck(&out, &errb, []string{"--isolate-wip=myfeat", "--json", "./cmd/fak"}); rc != 0 {
		t.Fatalf("rc = %d, want 0; stderr=%s", rc, errb.String())
	}
	hasTag := false
	for i, arg := range gotArgs {
		if arg == "-tags" && i+1 < len(gotArgs) && gotArgs[i+1] == "wip_myfeat" {
			hasTag = true
			break
		}
	}
	if !hasTag {
		t.Errorf("args %v missing -tags wip_myfeat", gotArgs)
	}
	var rep buildCheckReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("stdout is not valid JSON: %v", err)
	}
	if rep.IsolateWIP != "wip_myfeat" {
		t.Errorf("rep.IsolateWIP = %q, want wip_myfeat", rep.IsolateWIP)
	}
}

func TestLabelUntrackedBuildFailures(t *testing.T) {
	var output bytes.Buffer
	output.WriteString("internal/broken/broken.go:3:27: undefined: missing\n")
	labelUntrackedBuildFailures(&output, []string{"internal/broken/broken.go", "internal/other/clean.go"})

	got := output.String()
	for _, want := range []string{
		"internal/broken/broken.go:3:27: undefined: missing",
		"internal/broken/broken.go is UNTRACKED -- this red is not from the committed base",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("output missing %q:\n%s", want, got)
		}
	}
	if strings.Contains(got, "internal/other/clean.go is UNTRACKED") {
		t.Fatalf("unrelated untracked path was labelled:\n%s", got)
	}
}

func TestRunBuildCheckLabelsKeptUntrackedFailure(t *testing.T) {
	withBuildCheckSeams(t, []string{"cmd/fak/peer.go"}, map[string]bool{"cmd/fak": true}, func(_ string, _ []string, _, stderr io.Writer) (int, error) {
		stderr.Write([]byte("cmd/fak/peer.go:9:2: undefined: Missing\n"))
		return 1, nil
	})
	var stdout, stderr bytes.Buffer
	if got := RunBuildCheck(&stdout, &stderr, []string{"./cmd/fak"}); got != 1 {
		t.Fatalf("RunBuildCheck()=%d, want 1", got)
	}
	for _, want := range []string{
		"cmd/fak/peer.go:9:2: undefined: Missing",
		"cmd/fak/peer.go is UNTRACKED -- this red is not from the committed base",
	} {
		if !strings.Contains(stderr.String(), want) {
			t.Fatalf("stderr missing %q:\n%s", want, stderr.String())
		}
	}
}

func TestRunBuildCheckBuildFailedExit(t *testing.T) {
	withBuildCheckSeams(t, nil, nil, func(_ string, _ []string, _, stderr io.Writer) (int, error) {
		stderr.Write([]byte("cmd/fak/x.go:9:2: undefined: Foo\n"))
		return 2, nil
	})
	var out, errb bytes.Buffer
	rc := RunBuildCheck(&out, &errb, []string{"--json", "./..."})
	if rc != 2 {
		t.Fatalf("rc = %d, want 2 (mirror go's exit code)", rc)
	}
	var rep buildCheckReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("bad JSON: %v", err)
	}
	if rep.Verdict != "BUILD_FAILED" {
		t.Errorf("verdict = %q, want BUILD_FAILED", rep.Verdict)
	}
	if !strings.Contains(rep.Reason, "undefined: Foo") {
		t.Errorf("reason %q should carry the captured go failure tail", rep.Reason)
	}
}

// TestRunBuildCheckKeepsMatchedSibling is the failing-before/passing-after proof of the
// dir-scoped masking fix: an untracked .go whose package has an in-flight tracked edit is
// KEPT (no overlay generated), so the edit that references it still compiles -- instead of
// being masked into a false "undefined" red.
func TestRunBuildCheckKeepsMatchedSibling(t *testing.T) {
	var gotArgs []string
	withBuildCheckSeams(t, []string{"cmd/fak/info_color.go"}, map[string]bool{"cmd/fak": true},
		func(_ string, args []string, _, _ io.Writer) (int, error) {
			gotArgs = args
			return 0, nil
		})
	var out, errb bytes.Buffer
	if rc := RunBuildCheck(&out, &errb, []string{"--json", "./cmd/fak"}); rc != 0 {
		t.Fatalf("rc = %d, want 0; stderr=%s", rc, errb.String())
	}
	if strings.Contains(strings.Join(gotArgs, " "), "-overlay") {
		t.Errorf("args %v carry -overlay; the sole untracked .go was in an edited pkg and must be kept, not masked", gotArgs)
	}
	var rep buildCheckReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("bad JSON: %v\n%s", err, out.String())
	}
	if rep.KeptCount != 1 || !reflect.DeepEqual(rep.KeptFiles, []string{"cmd/fak/info_color.go"}) {
		t.Errorf("kept = %d/%v, want 1/[cmd/fak/info_color.go]", rep.KeptCount, rep.KeptFiles)
	}
	if rep.MaskedCount != 0 {
		t.Errorf("masked = %d/%v, want 0 (the only untracked .go was kept)", rep.MaskedCount, rep.MaskedFiles)
	}
	if rep.Verdict != "OK" {
		t.Errorf("verdict = %q, want OK", rep.Verdict)
	}
}

// TestRunBuildCheckLiveCrossCheckFailsClosed covers the cross-package case: a masked
// untracked file (a brand-new package in an un-edited dir) is imported by kept/tracked
// code, so the isolate build reds -- but the LIVE tree compiles. Buildcheck remains
// strictly fail-closed: it records live_cross_checked = true, but does NOT override
// the non-zero exit code or BUILD_FAILED verdict (#11457).
func TestRunBuildCheckLiveCrossCheckFailsClosed(t *testing.T) {
	withBuildCheckSeams(t, []string{"internal/conformance/conformance.go"}, nil,
		func(_ string, args []string, _, stderr io.Writer) (int, error) {
			if strings.Contains(strings.Join(args, " "), "-overlay") {
				// masked build: the hidden new package cannot be resolved.
				stderr.Write([]byte("cmd/fak/conformance.go:9:2: no required module provides package .../internal/conformance\n"))
				return 1, nil
			}
			return 0, nil // live tree (no overlay) compiles
		})
	var out, errb bytes.Buffer
	rc := RunBuildCheck(&out, &errb, []string{"--json", "./..."})
	if rc != 1 {
		t.Fatalf("rc = %d, want 1 (strictly fail-closed when isolated overlay compilation fails)", rc)
	}
	var rep buildCheckReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("bad JSON: %v\n%s", err, out.String())
	}
	if rep.Verdict != "BUILD_FAILED" || rep.ExitCode != 1 {
		t.Errorf("verdict/exit = %q/%d, want BUILD_FAILED/1", rep.Verdict, rep.ExitCode)
	}
	if !rep.LiveCrossChecked {
		t.Error("live_cross_checked = false, want true (live cross check succeeded)")
	}
	if !strings.Contains(rep.Reason, "conformance") {
		t.Errorf("reason %q should carry the failure reason", rep.Reason)
	}
}

// TestRunBuildCheckRefusesMaskedOverlayCompilationFailure verifies that when an isolated
// overlay build fails, buildcheck strictly fails closed with non-zero exit code, preserves
// the failure verdict and reason, and does NOT report OK -- even if the ambient live tree compiles (#11457).
func TestRunBuildCheckRefusesMaskedOverlayCompilationFailure(t *testing.T) {
	withBuildCheckSeams(t, []string{"internal/conformance/conformance.go"}, nil,
		func(_ string, args []string, _, stderr io.Writer) (int, error) {
			if strings.Contains(strings.Join(args, " "), "-overlay") {
				stderr.Write([]byte("cmd/fak/conformance.go:9:2: no required module provides package .../internal/conformance\n"))
				return 1, nil
			}
			return 0, nil // live tree (no overlay) compiles
		})
	// 1. JSON report must fail-closed: non-zero exit, BUILD_FAILED, LiveCrossChecked=true, error retained.
	var out, errb bytes.Buffer
	rc := RunBuildCheck(&out, &errb, []string{"--json", "./..."})
	if rc != 1 {
		t.Fatalf("rc = %d, want 1 (strictly fail-closed on isolated overlay build failure)", rc)
	}
	var rep buildCheckReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("bad JSON: %v\n%s", err, out.String())
	}
	if rep.Verdict != "BUILD_FAILED" {
		t.Errorf("rep.Verdict = %q, want BUILD_FAILED (must not report OK)", rep.Verdict)
	}
	if rep.ExitCode != 1 {
		t.Errorf("rep.ExitCode = %d, want 1", rep.ExitCode)
	}
	if !rep.LiveCrossChecked {
		t.Error("rep.LiveCrossChecked = false, want true")
	}
	if !strings.Contains(rep.Reason, "conformance") {
		t.Errorf("rep.Reason = %q, want captured failure output", rep.Reason)
	}

	// 2. Non-JSON output must retain compiler error and advise adding untracked files with git add.
	var textOut, textErr bytes.Buffer
	textRc := RunBuildCheck(&textOut, &textErr, []string{"./..."})
	if textRc != 1 {
		t.Fatalf("textRc = %d, want 1", textRc)
	}
	textErrStr := textErr.String()
	if strings.Contains(textErrStr, "reporting OK") {
		t.Errorf("textErr should not report OK: %s", textErrStr)
	}
	if !strings.Contains(textErrStr, "git add") {
		t.Errorf("textErr should advise `git add`: %s", textErrStr)
	}
	if !strings.Contains(textErrStr, "conformance.go:9:2") {
		t.Errorf("textErr should retain compilation error: %s", textErrStr)
	}
}

// TestRunBuildCheckLiveCrossCheckAlsoFails: when the live tree ALSO reds, the breakage is in
// tracked/kept code, not mask-induced -- the fallback must NOT rescue it.
func TestRunBuildCheckLiveCrossCheckAlsoFails(t *testing.T) {
	withBuildCheckSeams(t, []string{"internal/peer/wip.go"}, nil,
		func(_ string, _ []string, _, stderr io.Writer) (int, error) {
			stderr.Write([]byte("internal/x/y.go:1:1: undefined: Real\n"))
			return 2, nil // both masked and live builds fail
		})
	var out, errb bytes.Buffer
	rc := RunBuildCheck(&out, &errb, []string{"--json", "./..."})
	if rc != 2 {
		t.Fatalf("rc = %d, want 2 (live tree also red -> real breakage, not rescued)", rc)
	}
	var rep buildCheckReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("bad JSON: %v\n%s", err, out.String())
	}
	if rep.Verdict != "BUILD_FAILED" {
		t.Errorf("verdict = %q, want BUILD_FAILED", rep.Verdict)
	}
	if rep.LiveCrossChecked {
		t.Error("live_cross_checked = true, want false (live pass also failed)")
	}
	if !strings.Contains(rep.Reason, "undefined: Real") {
		t.Errorf("reason %q should carry the real go failure tail", rep.Reason)
	}
}

func TestLoadBearingUntrackedFilesKeepsImportedPackageClosure(t *testing.T) {
	root := t.TempDir()
	write := func(name, body string) {
		t.Helper()
		path := filepath.Join(root, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("go.mod", "module example.test/repo\n\ngo 1.26\n")
	write("cmd/app/main.go", "package main\nimport _ \"example.test/repo/internal/newpkg\"\nfunc main() {}\n")
	runGit := func(args ...string) {
		t.Helper()
		cmd := exec.Command("git", args...)
		cmd.Dir = root
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v: %s", args, err, out)
		}
	}
	runGit("init", "-q")
	runGit("add", "go.mod", "cmd/app/main.go")
	write("internal/newpkg/new.go", "package newpkg\nimport _ \"example.test/repo/internal/nested\"\n")
	write("internal/newpkg/sibling.go", "package newpkg\n")
	write("internal/nested/nested.go", "package nested\n")
	write("internal/orphan/orphan.go", "package orphan\n")
	untracked := []string{"internal/newpkg/new.go", "internal/newpkg/sibling.go", "internal/nested/nested.go", "internal/orphan/orphan.go"}
	got, err := buildoverlay.LoadBearingUntrackedFiles(root, untracked)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"internal/nested/nested.go", "internal/newpkg/new.go", "internal/newpkg/sibling.go"}
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("load-bearing files = %#v, want %#v", got, want)
	}
}

func TestRunBuildCheckCompileManifestFailsClosed(t *testing.T) {
	root := t.TempDir()
	writeBuildcheckFile(t, root, "go.mod", "module example.com/buildcheck\n\ngo 1.26\n")
	writeBuildcheckFile(t, root, "main.go", "package main\nfunc main() {}\n")
	manifest := filepath.Join(root, "unit.json")
	writeBuildcheckFile(t, root, "unit.json", `{"schema":"fak.work-delivery/v1","id":"unit","axes":{"authoring":"recorded","compile_admission":"undeclared","verification":"unverified","integration":"unintegrated","release":"not_ready"},"artifacts":[{"path":"peer.go","kind":"go-source"}]}`)

	oldWD, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(root); err != nil {
		t.Fatal(err)
	}
	defer os.Chdir(oldWD)
	var out, errOut bytes.Buffer
	code := RunBuildCheck(&out, &errOut, []string{"--json", "--compile-manifest", manifest, "."})
	if code == 0 {
		t.Fatalf("code = 0, out=%s", out.String())
	}
	if !strings.Contains(out.String(), "COMPILE_ADMISSION_BLOCKED") || !strings.Contains(out.String(), "MISSING_DECLARATION") {
		t.Fatalf("out = %s", out.String())
	}
}

func TestRunBuildCheckCompileManifestExcludesRecordedSource(t *testing.T) {
	root := t.TempDir()
	writeBuildcheckFile(t, root, "go.mod", "module example.com/buildcheck\n\ngo 1.26\n")
	writeBuildcheckFile(t, root, "main.go", "package main\nfunc main() {}\n")
	writeBuildcheckFile(t, root, "recorded.go", "package main\nfunc broken( {\n")
	manifest := filepath.Join(root, "unit.json")
	writeBuildcheckFile(t, root, "unit.json", `{"schema":"fak.work-delivery/v1","id":"unit","axes":{"authoring":"recorded","compile_admission":"excluded","verification":"unverified","integration":"unintegrated","release":"not_ready"},"artifacts":[{"path":"recorded.go","kind":"go-source"}]}`)

	oldWD, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(root); err != nil {
		t.Fatal(err)
	}
	defer os.Chdir(oldWD)
	var out, errOut bytes.Buffer
	code := RunBuildCheck(&out, &errOut, []string{"--json", "--compile-manifest", manifest, "."})
	if code != 0 {
		t.Fatalf("code = %d, out=%s err=%s", code, out.String(), errOut.String())
	}
	if !strings.Contains(out.String(), `"excluded_files"`) || !strings.Contains(out.String(), "recorded.go") {
		t.Fatalf("out = %s", out.String())
	}
}

func writeBuildcheckFile(t *testing.T, root, name, contents string) {
	t.Helper()
	path := filepath.Join(root, name)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(contents), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestRunBuildCheckCompileManifestIncludesAdmittedSource(t *testing.T) {
	root := t.TempDir()
	writeBuildcheckFile(t, root, "go.mod", "module example.com/buildcheck\n\ngo 1.26\n")
	writeBuildcheckFile(t, root, "main.go", "package main\nfunc main() {}\n")
	writeBuildcheckFile(t, root, "admitted.go", "package main\nfunc broken( {\n")
	manifest := filepath.Join(root, "unit.json")
	writeBuildcheckFile(t, root, "unit.json", `{"schema":"fak.work-delivery/v1","id":"unit","axes":{"authoring":"recorded","compile_admission":"admitted","verification":"unverified","integration":"unintegrated","release":"not_ready"},"artifacts":[{"path":"admitted.go","kind":"go-source"}]}`)

	oldWD, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(root); err != nil {
		t.Fatal(err)
	}
	defer os.Chdir(oldWD)
	var out, errOut bytes.Buffer
	code := RunBuildCheck(&out, &errOut, []string{"--json", "--compile-manifest", manifest, "."})
	if code == 0 {
		t.Fatalf("admitted broken source was not compiled: out=%s err=%s", out.String(), errOut.String())
	}
	if !strings.Contains(out.String(), `"admitted_files"`) || !strings.Contains(out.String(), "admitted.go") {
		t.Fatalf("out = %s", out.String())
	}
}

func TestRunBuildCheckAcquiresSlot(t *testing.T) {
	acquired := false
	released := false
	withBuildCheckSeams(t, nil, nil, func(_ string, _ []string, _, _ io.Writer) (int, error) {
		if !acquired {
			t.Error("expected slot to be acquired before buildCheckRun executes")
		}
		if released {
			t.Error("expected slot to remain held while buildCheckRun is executing")
		}
		return 0, nil
	})
	buildCheckAcquireSlot = func(ctx context.Context, timeout time.Duration) (func(), error) {
		acquired = true
		return func() {
			released = true
		}, nil
	}

	var out, errb bytes.Buffer
	rc := RunBuildCheck(&out, &errb, []string{"./cmd/fak"})
	if rc != 0 {
		t.Fatalf("rc = %d, want 0", rc)
	}
	if !acquired {
		t.Error("buildCheckAcquireSlot was not invoked")
	}
	if !released {
		t.Error("build slot was not released after RunBuildCheck returned")
	}
}

func TestRunBuildCheckSlotUnavailable(t *testing.T) {
	withBuildCheckSeams(t, nil, nil, func(_ string, _ []string, _, _ io.Writer) (int, error) {
		t.Fatal("buildCheckRun should not execute when slot acquisition fails")
		return 0, nil
	})
	buildCheckAcquireSlot = func(ctx context.Context, timeout time.Duration) (func(), error) {
		return nil, ErrBuildSlotTimeout
	}

	var out, errb bytes.Buffer
	rc := RunBuildCheck(&out, &errb, []string{"--json", "./cmd/fak"})
	if rc != 1 {
		t.Fatalf("rc = %d, want 1", rc)
	}
	var rep buildCheckReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("bad json: %v", err)
	}
	if rep.Verdict != "BUILD_SLOT_UNAVAILABLE" {
		t.Fatalf("verdict = %q, want BUILD_SLOT_UNAVAILABLE", rep.Verdict)
	}
	if !strings.Contains(rep.Reason, "timed out") {
		t.Fatalf("rep.Reason = %q, want timeout explanation", rep.Reason)
	}
}
