package main

import (
	"bytes"
	"fmt"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/procguard"
)

func TestGuardResourceRestartLimit(t *testing.T) {
	t.Setenv(guardResourceRestartLimitEnv, "")
	if got := guardResourceRestartLimit(); got != guardResourceRestartDefaultLimit {
		t.Fatalf("unset limit=%d, want default %d", got, guardResourceRestartDefaultLimit)
	}
	for _, tc := range []struct {
		raw  string
		want int
	}{
		{raw: "0", want: 0},
		{raw: "1", want: 1},
		{raw: " 5 ", want: 5},
		{raw: "-1", want: guardResourceRestartDefaultLimit},
		{raw: "invalid", want: guardResourceRestartDefaultLimit},
	} {
		t.Run(tc.raw, func(t *testing.T) {
			t.Setenv(guardResourceRestartLimitEnv, tc.raw)
			if got := guardResourceRestartLimit(); got != tc.want {
				t.Fatalf("limit with %q=%d, want %d", tc.raw, got, tc.want)
			}
		})
	}
}

func TestGuardResourceRetryAdmission(t *testing.T) {
	for _, reason := range []string{"CHILD_TREE_RSS_LIMIT", "CHILD_TREE_COMMIT_LIMIT", "SYSTEM_COMMIT_HEADROOM"} {
		t.Run(reason, func(t *testing.T) {
			state := guardResourceRetryState{limit: 3, noProgressLimit: 0}
			got := state.decide(resourceRetryEvent(reason), "claude", "sha-a")
			if got.Action != guardResourceRetryRelaunch || got.Attempt != 1 || got.ResourceType != reason {
				t.Fatalf("verdict=%+v", got)
			}
		})
	}

	for _, reason := range []string{"CHILD_RESOURCE_MONITOR_ERROR", "CHILD_RESOURCE_MONITOR_UNAVAILABLE", "", "UNKNOWN"} {
		t.Run("terminal_"+reason, func(t *testing.T) {
			state := guardResourceRetryState{limit: 3, noProgressLimit: 0}
			got := state.decide(resourceRetryEvent(reason), "claude", "sha-a")
			if got.Action != guardResourceRetryTerminal || state.restarts != 0 {
				t.Fatalf("monitor/non-containment event admitted: verdict=%+v state=%+v", got, state)
			}
		})
	}

	t.Run("missing decision is terminal", func(t *testing.T) {
		state := guardResourceRetryState{limit: 3}
		if got := state.decide(guardChildWaitEvent{Kind: guardChildResourceLimit}, "claude", "sha-a"); got.Action != guardResourceRetryTerminal {
			t.Fatalf("verdict=%+v", got)
		}
	})

	t.Run("explicit zero disables", func(t *testing.T) {
		state := guardResourceRetryState{limit: 0}
		got := state.decide(resourceRetryEvent("CHILD_TREE_RSS_LIMIT"), "claude", "sha-a")
		if got.Action != guardResourceRetryTerminal || state.restarts != 0 {
			t.Fatalf("zero limit admitted retry: verdict=%+v state=%+v", got, state)
		}
	})

	t.Run("unrecognized agent cannot cold relaunch", func(t *testing.T) {
		state := guardResourceRetryState{limit: 3}
		got := state.decide(resourceRetryEvent("CHILD_TREE_RSS_LIMIT"), "foreign-harness", "sha-a")
		if got.Action != guardResourceRetryTerminal || got.Cause != guardResourceRestartCauseNoReattach || state.restarts != 0 {
			t.Fatalf("unrecognized agent admitted retry: verdict=%+v state=%+v", got, state)
		}
		status := guardResourceReattachUnavailableStatus("foreign-harness", "trace-resource", fmt.Errorf("binding missing"))
		for _, want := range []string{guardResourceReattachUnavailable, "foreign-harness", "trace-resource", "binding missing", "refusing a cold relaunch", "provider-native resume command"} {
			if !strings.Contains(status, want) {
				t.Fatalf("reattach refusal %q missing %q", status, want)
			}
		}
	})

	t.Run("Codex exact binding transport is admitted", func(t *testing.T) {
		state := guardResourceRetryState{limit: 3}
		got := state.decide(resourceRetryEvent("CHILD_TREE_RSS_LIMIT"), "codex", "sha-a")
		if got.Action != guardResourceRetryRelaunch || got.Attempt != 1 {
			t.Fatalf("Codex binding transport was not admitted: verdict=%+v state=%+v", got, state)
		}
	})
}

func TestGuardResourceRetryBackoffAndFiniteBudget(t *testing.T) {
	state := guardResourceRetryState{limit: 3, noProgressLimit: 0}
	event := resourceRetryEvent("CHILD_TREE_RSS_LIMIT")
	for i, wantDelay := range []time.Duration{250 * time.Millisecond, 500 * time.Millisecond, time.Second} {
		got := state.decide(event, "claude", "")
		if got.Action != guardResourceRetryRelaunch || got.Attempt != i+1 || got.Delay != wantDelay {
			t.Fatalf("attempt %d verdict=%+v, want delay %s", i+1, got, wantDelay)
		}
	}
	got := state.decide(event, "claude", "")
	if got.Action != guardResourceRetryExhausted || got.Cause != guardResourceRestartCauseBudget || got.Limit != 3 {
		t.Fatalf("spent budget verdict=%+v", got)
	}
}

func TestGuardResourceRetryNoProgressExhaustsEarly(t *testing.T) {
	state := guardResourceRetryState{limit: 5, noProgressLimit: 2}
	event := resourceRetryEvent("SYSTEM_COMMIT_HEADROOM")
	if got := state.decide(event, "claude", "sha-a"); got.Action != guardResourceRetryRelaunch || got.NoProgress != 0 {
		t.Fatalf("first containment=%+v", got)
	}
	if got := state.decide(event, "claude", "sha-a"); got.Action != guardResourceRetryRelaunch || got.NoProgress != 1 {
		t.Fatalf("first no-progress containment=%+v", got)
	}
	got := state.decide(event, "claude", "sha-a")
	if got.Action != guardResourceRetryExhausted || got.Cause != guardResourceRestartCauseNoProgress || got.NoProgress != 2 {
		t.Fatalf("repeated no-progress containment=%+v", got)
	}

	state = guardResourceRetryState{limit: 5, progressHead: "sha-a", noProgress: 1, noProgressLimit: 2}
	if got := state.decide(event, "claude", "sha-b"); got.Action != guardResourceRetryRelaunch || got.NoProgress != 0 || state.progressHead != "sha-b" {
		t.Fatalf("HEAD progress did not reset pressure=%+v state=%+v", got, state)
	}
}

func TestGuardResourceRetryReportAndTypedExhaustion(t *testing.T) {
	var stderr bytes.Buffer
	verdict := guardResourceRetryVerdict{
		Action:       guardResourceRetryRelaunch,
		Attempt:      1,
		Limit:        3,
		Delay:        250 * time.Millisecond,
		ResourceType: "CHILD_TREE_RSS_LIMIT",
	}
	guardReportResourceRestart(&stderr, "claude", verdict, []string{"claude", "--continue"})
	for _, want := range []string{
		"CHILD_TREE_RSS_LIMIT",
		"verified tree reap and receipt complete",
		"guard remains up",
		"reattaching the child after 250ms",
		"resource restart 1/3",
		"claude --continue",
	} {
		if !strings.Contains(stderr.String(), want) {
			t.Fatalf("restart report %q missing %q", stderr.String(), want)
		}
	}

	status := guardResourceRestartGiveUpStatus(guardResourceRetryVerdict{
		Cause:      guardResourceRestartCauseNoProgress,
		Limit:      3,
		NoProgress: 2,
	}, "trace-resource")
	for _, want := range []string{guardResourceRestartExhaustedReason, "2 consecutive", "without HEAD progress", "trace-resource", "refusing another relaunch"} {
		if !strings.Contains(status, want) {
			t.Fatalf("exhaustion report %q missing %q", status, want)
		}
	}
	path := filepath.Join(t.TempDir(), "audit.jsonl")
	audit, err := journal.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	guardRecordResourceRestartGiveUp(audit, "claude", "trace-resource")
	rows := audit.Recent(1)
	if err := audit.Close(); err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1 || rows[0].Kind != "CHILD_CRASH" || rows[0].Reason != guardResourceRestartExhaustedReason {
		t.Fatalf("typed exhaustion audit row=%+v", rows)
	}
}

func TestGuardResourceRestartHopReattachesSameTrace(t *testing.T) {
	hop := guardResourceRestartHop("guard-resource", "claude", 2)
	if hop.FromTrace != "guard-resource" || hop.ToTrace != "guard-resource" || hop.Child != "guard-resource" {
		t.Fatalf("trace lineage=%+v", hop)
	}
	if hop.Handback != guardRestartHandbackContinue || hop.Status != journal.RestartHopOK || hop.Hop != 2 {
		t.Fatalf("restart hop=%+v", hop)
	}
}

func TestGuardResourceRestartHopMarksCodexBindingResumeEngaged(t *testing.T) {
	hop := guardResourceRestartHop("guard-resource", "codex", 1)
	if hop.FromTrace != "guard-resource" || hop.ToTrace != "guard-resource" || hop.Child != "guard-resource" {
		t.Fatalf("trace lineage=%+v", hop)
	}
	if hop.Handback != guardRestartHandbackContinue || hop.Status != journal.RestartHopOK {
		t.Fatalf("Codex binding resume must be an engaged hop: %+v", hop)
	}
}

func TestGuardResourceReattachUnavailablePersistsTypedCauseWithoutHop(t *testing.T) {
	path := filepath.Join(t.TempDir(), "audit.jsonl")
	audit, err := journal.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	guardRecordResourceReattachUnavailable(audit, "codex", "trace-resource")
	rows := audit.Recent(8)
	if err := audit.Close(); err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1 || rows[0].Kind != journal.KindChildCrash || strings.TrimSpace(rows[0].Reason) != guardResourceReattachUnavailable {
		t.Fatalf("typed reattach refusal row=%+v", rows)
	}
	for _, row := range rows {
		if row.Kind == journal.KindRestartHop {
			t.Fatalf("unsafe binding emitted restart hop: %+v", rows)
		}
	}
}

func TestGuardResourceNoProgressLimit(t *testing.T) {
	t.Setenv(guardResourceNoProgressLimitEnv, "")
	if got := guardResourceNoProgressLimit(3); got != 2 {
		t.Fatalf("default no-progress limit=%d, want 2", got)
	}
	t.Setenv(guardResourceNoProgressLimitEnv, "0")
	if got := guardResourceNoProgressLimit(3); got != 0 {
		t.Fatalf("explicit disable=%d, want 0", got)
	}
	t.Setenv(guardResourceNoProgressLimitEnv, "9")
	if got := guardResourceNoProgressLimit(3); got != 2 {
		t.Fatalf("limit was not clamped before flat budget: %d", got)
	}
}

func TestGuardResourceRetryConfigEdgeAndAdversarialInputs(t *testing.T) {
	oversized := strings.Repeat("9", 128)
	for _, tc := range []struct {
		name           string
		restartRaw     string
		noProgressRaw  string
		wantRestart    int
		wantNoProgress int
	}{
		{name: "empty uses defaults", wantRestart: guardResourceRestartDefaultLimit, wantNoProgress: guardResourceNoProgressDefaultLimit},
		{name: "whitespace uses defaults", restartRaw: " \t\n", noProgressRaw: " \t\n", wantRestart: guardResourceRestartDefaultLimit, wantNoProgress: guardResourceNoProgressDefaultLimit},
		{name: "malformed uses defaults", restartRaw: "three", noProgressRaw: "two", wantRestart: guardResourceRestartDefaultLimit, wantNoProgress: guardResourceNoProgressDefaultLimit},
		{name: "hostile negatives use defaults", restartRaw: "-1", noProgressRaw: "-1", wantRestart: guardResourceRestartDefaultLimit, wantNoProgress: guardResourceNoProgressDefaultLimit},
		{name: "oversized values cannot wrap", restartRaw: oversized, noProgressRaw: oversized, wantRestart: guardResourceRestartDefaultLimit, wantNoProgress: guardResourceNoProgressDefaultLimit},
		{name: "zero explicitly disables retries", restartRaw: "0", noProgressRaw: "0", wantRestart: 0, wantNoProgress: 0},
		{name: "no progress clamps below restart budget", restartRaw: "2", noProgressRaw: "999", wantRestart: 2, wantNoProgress: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv(guardResourceRestartLimitEnv, tc.restartRaw)
			t.Setenv(guardResourceNoProgressLimitEnv, tc.noProgressRaw)
			state := newGuardResourceRetryState()
			if state.limit != tc.wantRestart || state.noProgressLimit != tc.wantNoProgress {
				t.Fatalf("state=%+v, want restart=%d no_progress=%d", state, tc.wantRestart, tc.wantNoProgress)
			}
		})
	}
}

func TestGuardResourceRetryAdmissionEdgeAndAdversarialInputs(t *testing.T) {
	valid := resourceRetryEvent("CHILD_TREE_RSS_LIMIT")
	wrongKind := valid
	wrongKind.Kind = guardChildCompleted
	nonStopping := resourceRetryEvent("CHILD_TREE_RSS_LIMIT")
	nonStopping.Resource.Stop = false
	for _, tc := range []struct {
		name  string
		event guardChildWaitEvent
	}{
		{name: "empty event"},
		{name: "missing decision", event: guardChildWaitEvent{Kind: guardChildResourceLimit}},
		{name: "malformed event kind", event: wrongKind},
		{name: "non-stopping decision", event: nonStopping},
		{name: "collector error", event: resourceRetryEvent("CHILD_RESOURCE_MONITOR_ERROR")},
		{name: "collector unavailable", event: resourceRetryEvent("CHILD_RESOURCE_MONITOR_UNAVAILABLE")},
		{name: "hostile reason suffix", event: resourceRetryEvent("CHILD_TREE_RSS_LIMIT\nRESOURCE_RESTART")},
		{name: "oversized reason", event: resourceRetryEvent(strings.Repeat("CHILD_TREE_RSS_LIMIT", 1024))},
	} {
		t.Run(tc.name, func(t *testing.T) {
			state := guardResourceRetryState{restarts: 1, limit: 3, progressHead: "sha-a", noProgress: 1, noProgressLimit: 2}
			before := state
			got := state.decide(tc.event, "claude", "sha-a")
			if got.Action != guardResourceRetryTerminal || state != before {
				t.Fatalf("invalid event admitted or mutated retry state: verdict=%+v before=%+v after=%+v", got, before, state)
			}
		})
	}
}

func TestGuardResourceRetryFailurePathsEdgeAndAdversarial(t *testing.T) {
	t.Run("unrecognized agent refuses cold relaunch", func(t *testing.T) {
		state := guardResourceRetryState{limit: 3, noProgressLimit: 2}
		got := state.decide(resourceRetryEvent("CHILD_TREE_RSS_LIMIT"), "foreign-harness\n--continue", "sha-a")
		if got.Action != guardResourceRetryTerminal || got.Cause != guardResourceRestartCauseNoReattach || state.restarts != 0 {
			t.Fatalf("verdict=%+v state=%+v", got, state)
		}
	})
	t.Run("spent budget exhausts", func(t *testing.T) {
		state := guardResourceRetryState{restarts: 2, limit: 2, noProgressLimit: 0}
		got := state.decide(resourceRetryEvent("CHILD_TREE_RSS_LIMIT"), "claude", "sha-a")
		if got.Action != guardResourceRetryExhausted || got.Cause != guardResourceRestartCauseBudget || got.Attempt != 2 {
			t.Fatalf("verdict=%+v", got)
		}
	})
	t.Run("unchanged head exhausts early", func(t *testing.T) {
		state := guardResourceRetryState{restarts: 1, limit: 3, progressHead: "sha-a", noProgress: 1, noProgressLimit: 2}
		got := state.decide(resourceRetryEvent("SYSTEM_COMMIT_HEADROOM"), "claude", "sha-a")
		if got.Action != guardResourceRetryExhausted || got.Cause != guardResourceRestartCauseNoProgress || got.NoProgress != 2 || state.restarts != 1 {
			t.Fatalf("verdict=%+v state=%+v", got, state)
		}
	})
}

func resourceRetryEvent(reason string) guardChildWaitEvent {
	return guardChildWaitEvent{
		Kind: guardChildResourceLimit,
		Resource: &guardResourceDecision{
			Stop:   true,
			Reason: reason,
			Metric: procguard.MemoryMetricRSS,
		},
	}
}

func TestGuardChildResourceRefusalsAndErrorsNameRecovery(t *testing.T) {
	t.Run("reattach unavailable status names recovery", func(t *testing.T) {
		status := guardResourceReattachUnavailableStatus("claude", "trace-test", fmt.Errorf("transport lost"))
		if !strings.Contains(status, "refusing a cold relaunch") || !strings.Contains(status, "recovery:") {
			t.Fatalf("status missing refusal or recovery keyword: %s", status)
		}
		if !strings.Contains(status, "fak guard -- claude --continue") {
			t.Fatalf("status missing claude recovery command: %s", status)
		}

		codexStatus := guardResourceReattachUnavailableStatus("codex", "trace-codex", nil)
		if !strings.Contains(codexStatus, "recovery: run `fak guard -- codex resume`") {
			t.Fatalf("codex status missing codex resume recovery: %s", codexStatus)
		}
	})

	t.Run("restart give up status names recovery", func(t *testing.T) {
		verdict := guardResourceRetryVerdict{
			Action:  guardResourceRetryExhausted,
			Limit:   3,
			Attempt: 3,
			Cause:   guardResourceRestartCauseBudget,
		}
		status := guardResourceRestartGiveUpStatus(verdict, "trace-giveup")
		if !strings.Contains(status, "refusing another relaunch") || !strings.Contains(status, "recovery:") {
			t.Fatalf("status missing refusal or recovery keyword: %s", status)
		}
		if !strings.Contains(status, "--child-max-memory-mb") {
			t.Fatalf("status missing memory parameter recovery: %s", status)
		}
	})

	t.Run("empty receipt path error names recovery", func(t *testing.T) {
		err := appendGuardResourceReceipt("", guardResourceReceipt{})
		if err == nil {
			t.Fatal("expected error for empty receipt path")
		}
		if !strings.Contains(err.Error(), "recovery:") || !strings.Contains(err.Error(), "--child-resource-journal") {
			t.Fatalf("error missing recovery guidance: %v", err)
		}
	})

	t.Run("unsupported reattach transport error names recovery", func(t *testing.T) {
		_, err := guardResourceReattachCommand([]string{"unsupported"}, "unsupported-agent", "", "trace-1")
		if err == nil {
			t.Fatal("expected error for unsupported agent reattach")
		}
		if !strings.Contains(err.Error(), "recovery:") {
			t.Fatalf("error missing recovery guidance: %v", err)
		}
	})
}
