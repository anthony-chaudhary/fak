package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/safesync"
)

type syncPhaseTestReport struct {
	Schema          string `json:"schema"`
	Route           string `json:"route"`
	Error           string `json:"error"`
	Phase           string `json:"phase"`
	OK              bool   `json:"ok"`
	AssessmentKnown bool   `json:"assessment_known"`
	PhaseEvidence   *struct {
		Schema             string `json:"schema"`
		ElapsedNS          int64  `json:"elapsed_ns"`
		FetchCallAttempted *bool  `json:"fetch_call_attempted"`
		Phases             []struct {
			Phase     string `json:"phase"`
			Status    string `json:"status"`
			ElapsedNS int64  `json:"elapsed_ns"`
		} `json:"phases"`
	} `json:"phase_evidence"`
}

func requireSyncPhases(t *testing.T, got syncPhaseTestReport, want ...string) {
	t.Helper()
	e := got.PhaseEvidence
	if e == nil || e.Schema != "fak.sync.reconcile.phases.v1" || e.ElapsedNS <= 0 {
		t.Fatalf("missing observed monotonic phase evidence: %+v", e)
	}
	if len(e.Phases) != len(want) || len(e.Phases) > 6 {
		t.Fatalf("phase count=%d, want %d (bounded by six): %+v", len(e.Phases), len(want), e.Phases)
	}
	var sum int64
	fetchEntered := false
	for i, phase := range e.Phases {
		fetchEntered = fetchEntered || phase.Phase == "fetch"
		if phase.Phase+":"+phase.Status != want[i] || phase.ElapsedNS < 0 {
			t.Fatalf("phase[%d]=%+v, want %s with nonnegative elapsed", i, phase, want[i])
		}
		sum += phase.ElapsedNS
	}
	if sum > e.ElapsedNS {
		t.Fatalf("overlapping/fabricated accounting: phases=%d total=%d", sum, e.ElapsedNS)
	}
	if fetchEntered != (e.FetchCallAttempted != nil) {
		t.Fatalf("fetch attempt evidence does not match entered work: %+v", e)
	}
}

// Exercise runSync and the real router/Git inspection, not a fabricated success
// report. Only named failure sites use the existing Runner seam. Timings describe
// observed boundaries, never a performance threshold or network-only attribution.
// fak-test:runtime medium est=2s lane=default
func TestRunSyncReconcilePhaseEvidence(t *testing.T) {
	repo := syncCLIFixture(t) // Behind origin/work by the fixture's second commit.
	head, target := syncRev(t, repo, "HEAD"), syncRev(t, repo, "origin/work")
	syncWriteFile(t, filepath.Join(repo, "keep.txt"), "peer staged\n")
	syncGit(t, repo, "add", "keep.txt")
	syncWriteFile(t, filepath.Join(repo, "keep.txt"), "peer unstaged\n")
	syncWriteFile(t, filepath.Join(repo, "untracked.txt"), "unrelated\n")
	peerIndex := detachedSyncGitOutput(t, repo, "ls-files", "--stage", "-z", "--", "keep.txt")
	refs := detachedSyncGitOutput(t, repo, "show-ref")
	old := syncRouteReconciliation
	t.Cleanup(func() { syncRouteReconciliation = old })
	const sensitive = "credential-canary /private/repo git-secret-command"

	run := func(t *testing.T, extra []string, fail string) (syncPhaseTestReport, int, string, [][]string) {
		t.Helper()
		var calls [][]string
		syncRouteReconciliation = func(ctx context.Context, opts safesync.ReconcileOptions) (safesync.ReconcileAssessment, error) {
			opts.Runner = func(ctx context.Context, dir string, args ...string) safesync.RunResult {
				calls = append(calls, append([]string(nil), args...))
				if (fail == "fetch" && len(args) > 0 && args[0] == "fetch") ||
					(fail == "packet" && len(args) > 0 && args[0] == "log") ||
					(fail == "inspect" && strings.Join(args, " ") == "rev-parse --verify HEAD") {
					return safesync.RunResult{Code: -1, Err: errors.New(sensitive)}
				}
				return safesync.RealRunner(ctx, dir, args...)
			}
			return safesync.RouteReconciliation(ctx, opts)
		}
		args := []string{"reconcile", "--repo", repo, "--remote", "origin", "--branch", "work", "--goal", "integrate", "--json"}
		args = append(args, extra...)
		var out, errb bytes.Buffer
		code := runSync(&out, &errb, args)
		var got syncPhaseTestReport
		if err := json.Unmarshal(out.Bytes(), &got); err != nil {
			t.Fatalf("phase JSON missing: exit=%d err=%v stdout=%s stderr=%s", code, err, out.String(), errb.String())
		}
		if code == syncExitInternal {
			if got.Schema != "fak.sync.reconcile.error.v1" || got.Error != "RECONCILE_FAILED" || got.OK {
				t.Fatalf("failure became an untyped/success report: %+v", got)
			}
			for _, forbidden := range []string{sensitive, repo, "credential-canary", "git-secret-command"} {
				if strings.Contains(out.String(), forbidden) {
					t.Fatalf("new JSON telemetry leaked %q: %s", forbidden, out.String())
				}
			}
		}
		return got, code, errb.String(), calls
	}
	assertPeers := func(t *testing.T, wantHead string) {
		t.Helper()
		if syncRev(t, repo, "HEAD") != wantHead || syncRev(t, repo, "origin/work") != target {
			t.Fatal("unexpected HEAD or remote-tracking mutation")
		}
		if got := detachedSyncGitOutput(t, repo, "ls-files", "--stage", "-z", "--", "keep.txt"); !bytes.Equal(got, peerIndex) {
			t.Fatal("peer stage changed")
		}
		if syncReadFile(t, filepath.Join(repo, "keep.txt")) != "peer unstaged\n" || syncReadFile(t, filepath.Join(repo, "untracked.txt")) != "unrelated\n" {
			t.Fatal("peer working bytes changed")
		}
		if wantHead == head && !bytes.Equal(detachedSyncGitOutput(t, repo, "show-ref"), refs) {
			t.Fatal("read-only/refused path changed a named ref")
		}
	}
	assertNoMutation := func(t *testing.T, calls [][]string, allowFetch bool) {
		t.Helper()
		for _, args := range calls {
			if len(args) == 0 {
				continue
			}
			switch args[0] {
			case "fetch":
				if !allowFetch {
					t.Fatal("unrequested fetch")
				}
			case "merge", "push", "commit-tree", "update-ref", "read-tree", "checkout", "reset", "add":
				t.Fatalf("unexpected mutating Git call: %v", args)
			}
		}
	}

	t.Run("plan omits unentered fetch and primitive", func(t *testing.T) {
		got, code, stderr, calls := run(t, nil, "")
		if code != syncExitOK || got.Route != safesync.RouteApply || !got.OK || stderr != "" {
			t.Fatalf("plan semantics changed: exit=%d report=%+v stderr=%s", code, got, stderr)
		}
		requireSyncPhases(t, got, "prepare:completed", "inspect:completed")
		assertNoMutation(t, calls, false)
		assertPeers(t, head)
	})
	t.Run("fetch has its own whole phase", func(t *testing.T) {
		got, code, stderr, calls := run(t, []string{"--fetch"}, "")
		if code != syncExitOK || got.Route != safesync.RouteApply || stderr != "" {
			t.Fatalf("fetch plan changed: exit=%d report=%+v stderr=%s", code, got, stderr)
		}
		requireSyncPhases(t, got, "prepare:completed", "fetch:completed", "inspect:completed")
		if !*got.PhaseEvidence.FetchCallAttempted {
			t.Fatal("actual fetch call was not recorded")
		}
		fetches := 0
		for _, args := range calls {
			if strings.Join(args, " ") == "fetch origin work" {
				fetches++
			}
		}
		if fetches != 1 {
			t.Fatalf("fetch count=%d, want exactly one existing call", fetches)
		}
		assertNoMutation(t, calls, true)
		assertPeers(t, head)
	})
	t.Run("bounded prefetch guard does not invent transport", func(t *testing.T) {
		got, code, stderr, calls := run(t, []string{"--fetch", "--fetch-depth", "2", "--fetch-base", strings.Repeat("1", 40), "--fetch-target", target}, "")
		if code != syncExitInternal || got.Phase != "fetch" || !strings.Contains(stderr, "bounded fetch") {
			t.Fatalf("prefetch refusal changed: exit=%d report=%+v stderr=%s", code, got, stderr)
		}
		requireSyncPhases(t, got, "prepare:completed", "fetch:failed")
		if *got.PhaseEvidence.FetchCallAttempted {
			t.Fatal("guard refusal falsely claims a fetch attempt")
		}
		assertNoMutation(t, calls, false)
		assertPeers(t, head)
	})
	t.Run("guard refusal is not a successful primitive", func(t *testing.T) {
		syncWriteFile(t, filepath.Join(repo, "a.txt"), "colliding working bytes\n")
		defer syncWriteFile(t, filepath.Join(repo, "a.txt"), "v1\n")
		got, code, _, calls := run(t, []string{"--apply"}, "")
		if code != syncExitRefused || got.Route != safesync.RouteHoldDirtyCollision || got.OK {
			t.Fatalf("guard outcome changed: exit=%d report=%+v", code, got)
		}
		requireSyncPhases(t, got, "prepare:completed", "inspect:refused")
		if syncReadFile(t, filepath.Join(repo, "a.txt")) != "colliding working bytes\n" {
			t.Fatal("refusal cleared collision")
		}
		assertNoMutation(t, calls, false)
		assertPeers(t, head)
	})
	for _, phase := range []string{"fetch", "inspect"} {
		t.Run(phase+" error retains entered evidence only", func(t *testing.T) {
			var extra []string
			if phase == "fetch" {
				extra = []string{"--fetch"}
			}
			got, code, stderr, calls := run(t, extra, phase)
			if code != syncExitInternal || got.Phase != phase || got.AssessmentKnown || !strings.Contains(stderr, sensitive) {
				t.Fatalf("error contract changed: exit=%d report=%+v stderr=%s", code, got, stderr)
			}
			requireSyncPhases(t, got, "prepare:completed", phase+":failed")
			if phase == "fetch" && !*got.PhaseEvidence.FetchCallAttempted {
				t.Fatal("failed fetch Runner attempt disappeared")
			}
			if len(calls) != 1 {
				t.Fatalf("error path made %d Git calls, want one existing failed call: %v", len(calls), calls)
			}
			assertPeers(t, head)
		})
	}
	t.Run("preparation error emits no guessed work", func(t *testing.T) {
		got, code, _, calls := run(t, []string{"--goal", "invalid-credential-canary"}, "")
		if code != syncExitInternal || got.Phase != "prepare" || got.AssessmentKnown || len(calls) != 0 {
			t.Fatalf("preparation error reached later work: exit=%d report=%+v calls=%v", code, got, calls)
		}
		requireSyncPhases(t, got, "prepare:failed")
		assertPeers(t, head)
	})
	t.Run("optional packet error keeps existing exit policy", func(t *testing.T) {
		got, code, stderr, calls := run(t, []string{"--emit-packet"}, "packet")
		if code != syncExitOK || got.Route != safesync.RouteApply || !got.OK || stderr != "" {
			t.Fatalf("optional packet failure changed existing plan policy: exit=%d report=%+v stderr=%s", code, got, stderr)
		}
		requireSyncPhases(t, got, "prepare:completed", "inspect:completed", "packet_build:failed")
		logs := 0
		for _, args := range calls {
			if len(args) > 0 && args[0] == "log" {
				logs++
			}
		}
		if logs != 1 {
			t.Fatalf("expected one actual failed packet read, got %d", logs)
		}
		assertNoMutation(t, calls, false)
		assertPeers(t, head)
	})
	t.Run("actual primitive retains peer staging", func(t *testing.T) {
		got, code, stderr, _ := run(t, []string{"--apply"}, "")
		if code != syncExitOK || got.Route != safesync.RouteApply || stderr != "" {
			t.Fatalf("actual apply changed: exit=%d report=%+v stderr=%s", code, got, stderr)
		}
		requireSyncPhases(t, got, "prepare:completed", "inspect:completed", "primitive:completed")
		assertPeers(t, target)
		if syncReadFile(t, filepath.Join(repo, "a.txt")) != "v2\n" || syncReadFile(t, filepath.Join(repo, "new.txt")) != "n1\n" {
			t.Fatal("reported primitive did not materialize incoming bytes")
		}
	})
}

// fak-test:runtime fast est=1ms lane=default
func TestRunSyncReconcilePhaseUnknownAndTextFailure(t *testing.T) {
	old := syncRouteReconciliation
	t.Cleanup(func() { syncRouteReconciliation = old })
	const sensitive = "private credential-canary /private/path"
	for _, known := range []bool{false, true} {
		syncRouteReconciliation = func(context.Context, safesync.ReconcileOptions) (safesync.ReconcileAssessment, error) {
			var a safesync.ReconcileAssessment
			if known {
				a.Schema = safesync.ReconcileSchema
			}
			return a, errors.New(sensitive)
		}
		for _, asJSON := range []bool{false, true} {
			args := []string{"reconcile"}
			if asJSON {
				args = append(args, "--json")
			}
			var out, errb bytes.Buffer
			code := runSync(&out, &errb, args)
			if code != syncExitInternal || errb.String() != "fak sync reconcile: "+sensitive+"\n" {
				t.Fatalf("existing exit/stderr changed: %d %q", code, errb.String())
			}
			if !asJSON {
				if out.Len() != 0 {
					t.Fatalf("text mode gained output: %s", out.String())
				}
				continue
			}
			var got syncPhaseTestReport
			if err := json.Unmarshal(out.Bytes(), &got); err != nil {
				t.Fatal(err)
			}
			if got.Schema != "fak.sync.reconcile.error.v1" || got.Error != "RECONCILE_FAILED" || got.AssessmentKnown != known || got.Phase != "unknown" || got.PhaseEvidence != nil || got.OK {
				t.Fatalf("unknown evidence was fabricated: %+v", got)
			}
			if strings.Contains(out.String(), sensitive) || strings.Contains(out.String(), "elapsed_ns") {
				t.Fatalf("unknown error leaked text or fabricated timing: %s", out.String())
			}
		}
	}
}
