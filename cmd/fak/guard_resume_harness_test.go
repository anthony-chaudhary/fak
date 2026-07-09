package main

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/preflight"
	"github.com/anthony-chaudhary/fak/internal/resume"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// TestGuardResumeHarness* / TestResumeHarness* witness `fak guard --resume <id>` (#1206, epic
// #1193 C10): the cache-safe resume PLANNER that composes the C1 durable session registry with
// the C9 posture rung. The witness the issue names is `go test ./cmd/fak -run
// 'GuardResume|ResumeHarness'`.

func joinArgv(argv []string) string { return strings.Join(argv, " ") }

// A WARM posture (idle within the 5m tier's effective reuse window, small session) keeps the
// byte-identical prefix: WARM path, no --compact-history-budget splice, and the recorded agent
// relaunched with its own --continue resume flag.
func TestGuardResumeHarnessWarmPrefixKeepsPrefixByteIdentical(t *testing.T) {
	desc := session.Descriptor{ID: "guard-abc", Trace: "guard-abc", Argv: []string{"claude"}}
	plan := planGuardResume("guard-abc", desc, guardResumeInput{
		IdleSeconds:    60, // < 900s effective 5m reuse window -> warm
		ResidentTokens: 0,  // small session
		TTL:            resume.TTL5m,
	})
	if plan.Path != preflight.ResumePathWarm {
		t.Fatalf("warm small session: path = %q, want WARM", plan.Path)
	}
	if plan.CompactHistoryBudget != 0 {
		t.Errorf("WARM must preserve the prefix (no byte-splice); got compact budget %d", plan.CompactHistoryBudget)
	}
	if got := joinArgv(plan.RelaunchArgv); got != "fak guard -- claude --continue" {
		t.Errorf("relaunch argv = %q, want the recorded agent reattached with --continue and no compact flag", got)
	}
	if plan.ContinueFlag != "--continue" {
		t.Errorf("continue flag = %q, want --continue for a recognized Claude child", plan.ContinueFlag)
	}
}

// A COLD posture (idle past the effective reuse window) over a LARGE resident context selects
// CUT and splices the #745 --compact-history-budget cache-safe lever at the rung's priced
// prefill target — so resume defaults to the cheapest safe path, not a naive cold re-prefill.
func TestGuardResumeHarnessColdLargeSessionCutsWithCompactBudget(t *testing.T) {
	desc := session.Descriptor{ID: "guard-big", Trace: "guard-big", Argv: []string{"claude", "--model", "opus"}}
	plan := planGuardResume("guard-big", desc, guardResumeInput{
		IdleSeconds:    1200,   // > 900s effective 5m reuse window -> cold
		ResidentTokens: 200000, // large: > the 48000 shed budget, so CUT differs from resume_full
		TTL:            resume.TTL5m,
	})
	if plan.Path != preflight.ResumePathCut {
		t.Fatalf("cold large session: path = %q, want CUT", plan.Path)
	}
	if plan.CompactHistoryBudget != plan.ProjectedPrefillTokens {
		t.Errorf("CUT must splice --compact-history-budget at the priced prefill target: budget %d != prefill %d",
			plan.CompactHistoryBudget, plan.ProjectedPrefillTokens)
	}
	if plan.CompactHistoryBudget <= 0 {
		t.Fatalf("CUT must carry a positive compact budget, got %d", plan.CompactHistoryBudget)
	}
	want := "fak guard --compact-history-budget " + itoa(plan.CompactHistoryBudget) + " -- claude --model opus --continue"
	if got := joinArgv(plan.RelaunchArgv); got != want {
		t.Errorf("relaunch argv = %q, want %q", got, want)
	}
}

// When the host asserts an in-process warm-KV splice is available (internal/session/warmsplice),
// the plan selects WARM-SPLICE: zero re-prefill, prefix preserved, no compact splice.
func TestGuardResumeHarnessWarmSpliceSelectsNoReprefill(t *testing.T) {
	desc := session.Descriptor{ID: "guard-ws", Trace: "guard-ws", Argv: []string{"claude"}}
	plan := planGuardResume("guard-ws", desc, guardResumeInput{
		IdleSeconds:         1200, // even a would-be-cold idle: the live splice wins
		ResidentTokens:      200000,
		WarmSpliceAvailable: true,
		TTL:                 resume.TTL5m,
	})
	if plan.Path != preflight.ResumePathWarmSplice {
		t.Fatalf("warm-splice available: path = %q, want WARM-SPLICE", plan.Path)
	}
	if plan.ProjectedPrefillTokens != 0 {
		t.Errorf("WARM-SPLICE reattaches in-process KV with no re-prefill; got %d prefill tokens", plan.ProjectedPrefillTokens)
	}
	if plan.CompactHistoryBudget != 0 {
		t.Errorf("WARM-SPLICE must not byte-splice; got compact budget %d", plan.CompactHistoryBudget)
	}
}

// An unrecognized wrapped binary gets NO continue flag — fak never guesses a foreign tool's
// resume syntax (the same fence guardContinueFlagForAgent enforces on the restart path).
func TestResumeHarnessUnknownAgentGetsNoContinueFlag(t *testing.T) {
	desc := session.Descriptor{ID: "guard-x", Trace: "guard-x", Argv: []string{"opencode", "run"}}
	plan := planGuardResume("guard-x", desc, guardResumeInput{IdleSeconds: 60, TTL: resume.TTL5m})
	if plan.ContinueFlag != "" {
		t.Errorf("unrecognized agent must get no continue flag, got %q", plan.ContinueFlag)
	}
	if got := joinArgv(plan.RelaunchArgv); got != "fak guard -- opencode run" {
		t.Errorf("relaunch argv = %q, want the recorded command with no injected resume flag", got)
	}
}

// Resuming a BRANCH id (C4/#1200) brings the branch up from its OWN descriptor — its own Argv
// and trace — and the parent descriptor sharing the registry is never the one resolved, so the
// parent is left untouched by construction.
func TestResumeHarnessBranchResumeUsesBranchDescriptorParentUntouched(t *testing.T) {
	descs := []session.Descriptor{
		{ID: "guard-parent", Trace: "guard-parent", Argv: []string{"claude", "--model", "sonnet"}},
		{ID: "guard-parent-branch", Trace: "guard-parent-branch", ParentID: "guard-parent", Argv: []string{"claude", "--model", "opus"}},
	}
	got, matched, _ := resolveGuardResumeDescriptor(descs, "guard-parent-branch")
	if matched != 1 {
		t.Fatalf("branch id must resolve to exactly one descriptor, matched=%d", matched)
	}
	if got.ID != "guard-parent-branch" {
		t.Fatalf("resolved the wrong descriptor: %q (parent must stay untouched)", got.ID)
	}
	plan := planGuardResume("guard-parent-branch", got, guardResumeInput{IdleSeconds: 60, TTL: resume.TTL5m})
	if !plan.Branch || plan.ParentID != "guard-parent" {
		t.Errorf("branch plan must carry the parent link: branch=%v parent=%q", plan.Branch, plan.ParentID)
	}
	if !strings.Contains(joinArgv(plan.RelaunchArgv), "--model opus") {
		t.Errorf("branch relaunch must use the BRANCH's recorded argv (opus), got %q", joinArgv(plan.RelaunchArgv))
	}
}

// An exact id/trace equality wins over a longer session it is a prefix of.
func TestResumeHarnessResolveExactBeatsPrefix(t *testing.T) {
	descs := []session.Descriptor{
		{ID: "guard-1", Trace: "guard-1"},
		{ID: "guard-12", Trace: "guard-12"},
	}
	got, matched, _ := resolveGuardResumeDescriptor(descs, "guard-1")
	if matched != 1 || got.ID != "guard-1" {
		t.Fatalf("exact id must win: matched=%d id=%q", matched, got.ID)
	}
}

// idle derivation: an explicit override wins; otherwise idle comes from the descriptor's
// LastSeen; an unstamped descriptor yields "unknown" (-1) which the C9 rung projects to the
// conservative RESUME_FULL.
func TestGuardResumeHarnessIdleFromLastSeen(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	desc := session.Descriptor{LastSeen: now.Add(-10 * time.Minute)}
	if got := guardResumeIdleSeconds(desc, -1, now); got != 600 {
		t.Errorf("idle from LastSeen = %d, want 600", got)
	}
	if got := guardResumeIdleSeconds(desc, 42, now); got != 42 {
		t.Errorf("explicit override must win: got %d, want 42", got)
	}
	if got := guardResumeIdleSeconds(session.Descriptor{}, -1, now); got != -1 {
		t.Errorf("unstamped descriptor must be unknown (-1), got %d", got)
	}
}

// The CLI resolves an id against a durable registry file and prints/JSON-emits the plan; it
// exits 1 on no match, 3 on an ambiguous prefix, 2 on a missing id, and 0 on a unique resolve.
func TestGuardResumeCLIResolvesAgainstRegistry(t *testing.T) {
	regPath := filepath.Join(t.TempDir(), "session-registry.json")
	reg := session.NewRegistry(session.NewFileStore(regPath))
	now := time.Now()
	seed := func(id string, argv []string) {
		if _, err := reg.RegisterWithMeta(id, "host", session.State{TraceID: id, Run: session.Running},
			session.DefaultDescriptorTTL, now, session.DescriptorMeta{Argv: argv}); err != nil {
			t.Fatalf("seed %s: %v", id, err)
		}
	}
	seed("guard-alpha", []string{"claude"})
	seed("guard-alto", []string{"claude", "--model", "opus"})

	// usage: no id
	var out, errb bytes.Buffer
	if rc := runGuardResume(&out, &errb, []string{"--registry", regPath}); rc != 2 {
		t.Fatalf("no id: rc=%d want 2 (%s)", rc, errb.String())
	}

	// no match
	out.Reset()
	errb.Reset()
	if rc := runGuardResume(&out, &errb, []string{"--registry", regPath, "nope"}); rc != 1 {
		t.Fatalf("no match: rc=%d want 1", rc)
	}

	// ambiguous prefix ("guard-al" matches both)
	out.Reset()
	errb.Reset()
	if rc := runGuardResume(&out, &errb, []string{"--registry", regPath, "guard-al"}); rc != 3 {
		t.Fatalf("ambiguous: rc=%d want 3 (%s)", rc, errb.String())
	}

	// unique resolve -> JSON plan
	out.Reset()
	errb.Reset()
	rc := runGuardResume(&out, &errb, []string{"--registry", regPath, "--json", "--idle-seconds", "60", "guard-alto"})
	if rc != 0 {
		t.Fatalf("unique resolve: rc=%d want 0 (%s)", rc, errb.String())
	}
	var plan guardResumePlan
	if err := json.Unmarshal(out.Bytes(), &plan); err != nil {
		t.Fatalf("plan JSON: %v\n%s", err, out.String())
	}
	if plan.Schema != guardResumePlanSchema || plan.ID != "guard-alto" {
		t.Fatalf("plan mismatch: schema=%q id=%q", plan.Schema, plan.ID)
	}
	if got := joinArgv(plan.RelaunchArgv); got != "fak guard -- claude --model opus --continue" {
		t.Errorf("CLI relaunch argv = %q", got)
	}
}
