package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchauto"
	"github.com/anthony-chaudhary/fak/internal/dispatchcache"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

func TestIssue8480DispatchAutoSlowBuildReturnsTypedPartialReceipt(t *testing.T) {
	finished := make(chan struct{})
	backlogCalls := 0
	stubDispatchAutoIssue8480Probes(t,
		func(ctx context.Context, _ string) (dispatchAutoBuildResult, error) {
			<-ctx.Done()
			close(finished)
			return dispatchAutoBuildResult{}, ctx.Err()
		},
		func(context.Context, string) (dispatchAutoBacklogResult, error) {
			backlogCalls++
			return dispatchAutoBacklogResult{}, nil
		},
		nil,
		nil,
	)

	receipt, code, elapsed := runDispatchAutoIssue8480(t,
		"--build-timeout=15ms", "--backlog-timeout=100ms", "--ranking-timeout=100ms",
		"--pricing-timeout=100ms", "--render-timeout=100ms", "--timeout=500ms",
	)
	if code != 1 || elapsed >= time.Second {
		t.Fatalf("code=%d elapsed=%s, want bounded failure", code, elapsed)
	}
	assertDispatchAutoIssue8480Error(t, receipt, dispatchAutoPhaseBuild, dispatchAutoPhaseTimeoutCode)
	assertDispatchAutoIssue8480Phase(t, receipt, dispatchAutoPhaseBuild, "timeout")
	assertDispatchAutoIssue8480Phase(t, receipt, dispatchAutoPhaseBacklogFetch, "skipped")
	if backlogCalls != 0 {
		t.Fatalf("backlog calls=%d, want 0 after build timeout", backlogCalls)
	}
	select {
	case <-finished:
	case <-time.After(time.Second):
		t.Fatal("slow build fixture did not stop with its phase context")
	}
}

func TestIssue8480DispatchAutoWholeRunBudgetHasDistinctTimeoutCode(t *testing.T) {
	finished := make(chan struct{})
	stubDispatchAutoIssue8480Probes(t,
		func(ctx context.Context, _ string) (dispatchAutoBuildResult, error) {
			<-ctx.Done()
			close(finished)
			return dispatchAutoBuildResult{}, ctx.Err()
		},
		func(context.Context, string) (dispatchAutoBacklogResult, error) {
			return dispatchAutoBacklogResult{}, errors.New("unexpected backlog phase")
		},
		nil,
		nil,
	)

	receipt, code, elapsed := runDispatchAutoIssue8480(t,
		"--build-timeout=200ms", "--backlog-timeout=200ms", "--ranking-timeout=200ms",
		"--pricing-timeout=200ms", "--render-timeout=200ms", "--timeout=15ms",
	)
	if code != 1 || elapsed >= time.Second {
		t.Fatalf("code=%d elapsed=%s, want bounded whole-run failure", code, elapsed)
	}
	assertDispatchAutoIssue8480Error(t, receipt, dispatchAutoPhaseBuild, dispatchAutoTotalTimeoutCode)
	select {
	case <-finished:
	case <-time.After(time.Second):
		t.Fatal("whole-run timeout did not cancel the active build phase")
	}
}

func TestIssue8480DispatchAutoSlowGitHubReturnsBacklogPhase(t *testing.T) {
	finished := make(chan struct{})
	stubDispatchAutoIssue8480Probes(t,
		func(context.Context, string) (dispatchAutoBuildResult, error) {
			return dispatchAutoBuildResult{Evidence: dispatchTreeBuildEvidence{Source: "running_binary", Reused: true}}, nil
		},
		func(ctx context.Context, _ string) (dispatchAutoBacklogResult, error) {
			<-ctx.Done()
			close(finished)
			return dispatchAutoBacklogResult{}, ctx.Err()
		},
		nil,
		nil,
	)

	receipt, code, elapsed := runDispatchAutoIssue8480(t,
		"--build-timeout=100ms", "--backlog-timeout=15ms", "--ranking-timeout=100ms",
		"--pricing-timeout=100ms", "--render-timeout=100ms", "--timeout=500ms",
	)
	if code != 1 || elapsed >= time.Second {
		t.Fatalf("code=%d elapsed=%s, want bounded failure", code, elapsed)
	}
	assertDispatchAutoIssue8480Phase(t, receipt, dispatchAutoPhaseBuild, "ok")
	assertDispatchAutoIssue8480Error(t, receipt, dispatchAutoPhaseBacklogFetch, dispatchAutoPhaseTimeoutCode)
	assertDispatchAutoIssue8480Phase(t, receipt, dispatchAutoPhaseBacklogFetch, "timeout")
	assertDispatchAutoIssue8480Phase(t, receipt, dispatchAutoPhaseRanking, "skipped")
	select {
	case <-finished:
	case <-time.After(time.Second):
		t.Fatal("slow GitHub fixture did not stop with its phase context")
	}
}

func TestIssue8480DispatchAutoSuccessfulDryRunNamesEveryPhase(t *testing.T) {
	stubDispatchAutoIssue8480Probes(t,
		func(context.Context, string) (dispatchAutoBuildResult, error) {
			return dispatchAutoBuildResult{Evidence: dispatchTreeBuildEvidence{Source: "running_binary", Reused: true}}, nil
		},
		func(context.Context, string) (dispatchAutoBacklogResult, error) {
			return dispatchAutoBacklogResult{Fetched: dispatchFetchedBacklog{
				IssueLimit: 100000,
				Issues:     []dispatchtick.Issue{{Number: 8480, Title: "bounded auto dispatch"}},
			}}, nil
		},
		func(context.Context, string, dispatchAutoBacklogResult) (dispatchtick.RouterPayload, error) {
			return dispatchtick.RouterPayload{
				OK: true, Verdict: "ACTION",
				Issues: []dispatchtick.IssueRoute{{Number: 8480, Lane: "cmd"}},
			}, nil
		},
		func(context.Context, string, io.Writer, int, string, string, string, []string, int, int, dispatchtick.TreeCheck, dispatchtick.RouterPayload) (dispatchAutoPricingResult, error) {
			input := dispatchauto.Input{EffectiveCap: 4, DistinctPools: 3, ReadyWork: 1}
			return dispatchAutoPricingResult{Input: input, Plan: dispatchauto.PlanAuto(input), Preflight: map[string]any{"verdict": "GO"}}, nil
		},
	)

	receipt, code, _ := runDispatchAutoIssue8480(t,
		"--build-timeout=100ms", "--backlog-timeout=100ms", "--ranking-timeout=100ms",
		"--pricing-timeout=100ms", "--render-timeout=100ms", "--timeout=500ms",
	)
	if code != 0 || receipt["ok"] != true || receipt["partial"] != false {
		t.Fatalf("receipt=%+v code=%d, want complete successful dry-run", receipt, code)
	}
	for _, phase := range dispatchAutoPhaseOrder {
		assertDispatchAutoIssue8480Phase(t, receipt, phase, "ok")
	}
	build, _ := receipt["build"].(map[string]any)
	if build["source"] != "running_binary" || build["reused"] != true {
		t.Fatalf("build evidence=%+v, want provenance-valid binary reuse", build)
	}
}

func TestIssue8480ConcurrentIdenticalBacklogFetchesReuseCompletedSnapshot(t *testing.T) {
	root := t.TempDir()
	path := dispatchBacklogSnapshotPath(root)
	key := dispatchcache.Key(root, "", 1000)
	watermark := time.Now().Add(-time.Hour).UTC()
	if err := dispatchcache.WriteBacklog(path, key, watermark, dispatchIssueRows([]dispatchtick.Issue{{Number: 1, Title: "old"}})); err != nil {
		t.Fatal(err)
	}
	writeDispatchBacklogFetchReceipt(path, key, watermark)

	oldWait := dispatchBacklogFetchLockWait
	contended := make(chan struct{}, 1)
	dispatchBacklogFetchLockWait = func(ctx context.Context) error {
		select {
		case contended <- struct{}{}:
		default:
		}
		timer := time.NewTimer(time.Millisecond)
		defer timer.Stop()
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-timer.C:
			return nil
		}
	}
	t.Cleanup(func() { dispatchBacklogFetchLockWait = oldWait })

	started, release := make(chan struct{}), make(chan struct{})
	var deltaCalls atomic.Int32
	fetchDelta := func(ctx context.Context, _ string, _ time.Time, _ int) (dispatchBacklogDelta, error) {
		if deltaCalls.Add(1) != 1 {
			return dispatchBacklogDelta{}, errors.New("duplicate GitHub delta fetch")
		}
		close(started)
		select {
		case <-ctx.Done():
			return dispatchBacklogDelta{}, ctx.Err()
		case <-release:
			return dispatchBacklogDelta{
				Issues:    []dispatchtick.Issue{{Number: 1, Title: "updated"}},
				Watermark: time.Now().UTC(),
			}, nil
		}
	}
	fetchFull := func(context.Context, string, int) ([]dispatchtick.Issue, error) {
		return nil, errors.New("unexpected full GitHub fetch")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	type result struct {
		issues []dispatchtick.Issue
		err    error
	}
	results := make(chan result, 2)
	run := func() {
		issues, err := dispatchFetchBacklogIncrementalWith(ctx, root, 1000, time.Now(), fetchDelta, fetchFull)
		results <- result{issues: issues, err: err}
	}
	go run()
	select {
	case <-started:
	case <-ctx.Done():
		t.Fatal("first backlog fetch did not start")
	}
	go run()
	select {
	case <-contended:
	case <-ctx.Done():
		t.Fatal("second backlog fetch did not contend on the shared lock")
	}
	close(release)
	for range 2 {
		got := <-results
		if got.err != nil || len(got.issues) != 1 || got.issues[0].Title != "updated" {
			t.Fatalf("fetch result=%+v err=%v", got.issues, got.err)
		}
	}
	if got := deltaCalls.Load(); got != 1 {
		t.Fatalf("GitHub delta calls=%d, want exactly 1", got)
	}
}

func stubDispatchAutoIssue8480Probes(
	t *testing.T,
	build func(context.Context, string) (dispatchAutoBuildResult, error),
	backlog func(context.Context, string) (dispatchAutoBacklogResult, error),
	ranking func(context.Context, string, dispatchAutoBacklogResult) (dispatchtick.RouterPayload, error),
	pricing func(context.Context, string, io.Writer, int, string, string, string, []string, int, int, dispatchtick.TreeCheck, dispatchtick.RouterPayload) (dispatchAutoPricingResult, error),
) {
	t.Helper()
	oldBuild, oldBacklog := dispatchAutoBuildProbe, dispatchAutoBacklogProbe
	oldRanking, oldPricing, oldRender := dispatchAutoRankingProbe, dispatchAutoPricingProbe, dispatchAutoRenderProbe
	dispatchAutoBuildProbe = build
	dispatchAutoBacklogProbe = backlog
	if ranking == nil {
		ranking = func(context.Context, string, dispatchAutoBacklogResult) (dispatchtick.RouterPayload, error) {
			return dispatchtick.RouterPayload{}, errors.New("unexpected ranking phase")
		}
	}
	dispatchAutoRankingProbe = ranking
	if pricing == nil {
		pricing = func(context.Context, string, io.Writer, int, string, string, string, []string, int, int, dispatchtick.TreeCheck, dispatchtick.RouterPayload) (dispatchAutoPricingResult, error) {
			return dispatchAutoPricingResult{}, errors.New("unexpected pricing phase")
		}
	}
	dispatchAutoPricingProbe = pricing
	dispatchAutoRenderProbe = func(ctx context.Context, _ map[string]any, _ dispatchauto.Plan, _ bool) error { return ctx.Err() }
	t.Cleanup(func() {
		dispatchAutoBuildProbe, dispatchAutoBacklogProbe = oldBuild, oldBacklog
		dispatchAutoRankingProbe, dispatchAutoPricingProbe, dispatchAutoRenderProbe = oldRanking, oldPricing, oldRender
	})
}

func runDispatchAutoIssue8480(t *testing.T, budgetArgs ...string) (map[string]any, int, time.Duration) {
	t.Helper()
	args := append([]string{"--workspace", t.TempDir(), "--json"}, budgetArgs...)
	var stdout, stderr bytes.Buffer
	started := time.Now()
	code := runDispatchAuto(&stdout, &stderr, args)
	elapsed := time.Since(started)
	var receipt map[string]any
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
		t.Fatalf("decode receipt: %v\nstdout=%s\nstderr=%s", err, stdout.String(), stderr.String())
	}
	return receipt, code, elapsed
}

func assertDispatchAutoIssue8480Error(t *testing.T, receipt map[string]any, phase, code string) {
	t.Helper()
	got, _ := receipt["error"].(map[string]any)
	if got["phase"] != phase || got["code"] != code || got["timeout"] != true || got["partial"] != true {
		t.Fatalf("typed error=%+v, want phase=%s code=%s timeout partial", got, phase, code)
	}
}

func assertDispatchAutoIssue8480Phase(t *testing.T, receipt map[string]any, phase, status string) {
	t.Helper()
	rows, _ := receipt["phase_timings"].([]any)
	for _, raw := range rows {
		row, _ := raw.(map[string]any)
		if row["phase"] == phase {
			if row["status"] != status {
				t.Fatalf("phase %s status=%v, want %s", phase, row["status"], status)
			}
			if row["budget_ms"].(float64) <= 0 {
				t.Fatalf("phase %s missing positive budget: %+v", phase, row)
			}
			return
		}
	}
	t.Fatalf("phase %s missing from timings: %+v", phase, rows)
}
