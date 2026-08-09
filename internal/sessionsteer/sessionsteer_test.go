package sessionsteer

import (
	"strings"
	"testing"
)

// TestSteerGolden is the spine witness (#3512): the pure decision core over a golden table of
// the load-bearing session snapshots. Every row asserts the full typed directive, so a change
// to the admission ladder, the persist ladder, or the advice->directive mapping is caught.
func TestSteerGolden(t *testing.T) {
	cases := []struct {
		name          string
		in            SteerInput
		wantAdmit     AdmitClass
		wantAdmitRsn  AdmitReason
		wantPersist   PersistDecision
		wantPersRsn   PersistReason
		wantDirective bool // whether ContextDirective is non-empty
	}{
		{
			// The headline case: an over-budget headless worker with a standing, unmet goal.
			// Admitted MANAGED; the Stop hook is told to BLOCK; the checkpoint advice yields a directive.
			name: "over-budget headless worker, goal unmet -> managed + block + checkpoint",
			in: SteerInput{
				Advice: AdviceCheckpoint, Headless: true, GoalActive: true, GoalMet: false,
				PendingWork: true, DurableStore: true, Phase: "crowding",
			},
			wantAdmit: AdmitManaged, wantAdmitRsn: AdmitReasonHeadlessGoal,
			wantPersist: PersistBlockStop, wantPersRsn: PersistReasonGoalUnmet,
			wantDirective: true,
		},
		{
			// Clean completion: goal met, no work left -> allow the stop.
			name: "headless worker, goal met, no work -> managed + allow",
			in: SteerInput{
				Advice: AdviceAny, Headless: true, GoalActive: true, GoalMet: true,
				PendingWork: false, DurableStore: true,
			},
			wantAdmit: AdmitManaged, wantAdmitRsn: AdmitReasonHeadlessGoal,
			wantPersist: PersistAllowStop, wantPersRsn: PersistReasonGoalMet,
			wantDirective: false,
		},
		{
			// No durable store: cannot checkpoint/relay -> LEGACY, but with a structured reason (never silent).
			name: "no durable store -> legacy with structured reason",
			in: SteerInput{
				Advice: AdviceCheckpoint, Headless: true, GoalActive: true, GoalMet: false,
				PendingWork: true, DurableStore: false,
			},
			wantAdmit: AdmitLegacy, wantAdmitRsn: AdmitReasonNoDurableStore,
			// persist ladder still reports the truth about work state, independent of admission.
			wantPersist: PersistBlockStop, wantPersRsn: PersistReasonGoalUnmet,
			wantDirective: true,
		},
		{
			// Attended human-driven session, no goal -> LEGACY (no heavy posture imposed).
			name: "attended, no goal -> legacy attended",
			in: SteerInput{
				Advice: AdviceBounded, Headless: false, GoalActive: false,
				PendingWork: true, DurableStore: true,
			},
			wantAdmit: AdmitLegacy, wantAdmitRsn: AdmitReasonAttendedNoGoal,
			wantPersist: PersistBlockStop, wantPersRsn: PersistReasonWorkRemains,
			wantDirective: true,
		},
		{
			// Attended but the operator set a standing goal -> MANAGED.
			name: "attended with goal -> managed attended",
			in: SteerInput{
				Advice: AdviceRebuild, Headless: false, GoalActive: true, GoalMet: false,
				DurableStore: true,
			},
			wantAdmit: AdmitManaged, wantAdmitRsn: AdmitReasonAttendedGoal,
			wantPersist: PersistBlockStop, wantPersRsn: PersistReasonGoalUnmet,
			wantDirective: true,
		},
		{
			// Headless, no goal, but the task-handoff artifact is not yet written -> block on handoff.
			name: "headless, handoff required -> managed + block on handoff",
			in: SteerInput{
				Advice: AdviceAny, Headless: true, GoalActive: false,
				HandoffRequired: true, DurableStore: true,
			},
			wantAdmit: AdmitManaged, wantAdmitRsn: AdmitReasonHeadless,
			wantPersist: PersistBlockStop, wantPersRsn: PersistReasonHandoff,
			wantDirective: false,
		},
		{
			// Headless, nothing pending -> allow with the no-work reason.
			name: "headless, nothing pending -> managed + allow no-work",
			in: SteerInput{
				Advice: AdviceUnknown, Headless: true, DurableStore: true,
			},
			wantAdmit: AdmitManaged, wantAdmitRsn: AdmitReasonHeadless,
			wantPersist: PersistAllowStop, wantPersRsn: PersistReasonNoWork,
			wantDirective: false,
		},
		{
			// Floor reconciliation: the same headline over-budget/goal-unmet case, but the capability
			// floor denies the agent's durable-persist path. A BLOCK_STOP here would demand an
			// impossible commit and wedge the session, so it is downgraded to ALLOW_STOP with the
			// floor-denied reason (persistence must be operator-mediated). Regression guard for the
			// persist-hook vs write-floor deadlock that blocked the spine's own commit.
			name: "headless, goal unmet, but persist floor-denied -> allow + floor-denied",
			in: SteerInput{
				Advice: AdviceCheckpoint, Headless: true, GoalActive: true, GoalMet: false,
				PendingWork: true, DurableStore: true, PersistFloorDenied: true,
			},
			wantAdmit: AdmitManaged, wantAdmitRsn: AdmitReasonHeadlessGoal,
			wantPersist: PersistAllowStop, wantPersRsn: PersistReasonFloorDenied,
			wantDirective: true,
		},
		{
			// Floor-denied but the work state was already ALLOW_STOP (goal met): the downgrade only
			// fires on a would-be block, so this stays the clean goal-met stop, not relabeled.
			name: "goal met + persist floor-denied -> allow stays goal-met (no spurious relabel)",
			in: SteerInput{
				Advice: AdviceAny, Headless: true, GoalActive: true, GoalMet: true,
				DurableStore: true, PersistFloorDenied: true,
			},
			wantAdmit: AdmitManaged, wantAdmitRsn: AdmitReasonHeadlessGoal,
			wantPersist: PersistAllowStop, wantPersRsn: PersistReasonGoalMet,
			wantDirective: false,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := Steer(tc.in)
			if got.Admit != tc.wantAdmit || got.AdmitReason != tc.wantAdmitRsn {
				t.Errorf("admit = %s/%s, want %s/%s", got.Admit, got.AdmitReason, tc.wantAdmit, tc.wantAdmitRsn)
			}
			if got.Persist != tc.wantPersist || got.PersistReason != tc.wantPersRsn {
				t.Errorf("persist = %s/%s, want %s/%s", got.Persist, got.PersistReason, tc.wantPersist, tc.wantPersRsn)
			}
			if (got.ContextDirective != "") != tc.wantDirective {
				t.Errorf("directive present = %v, want %v (%q)", got.ContextDirective != "", tc.wantDirective, got.ContextDirective)
			}
			// The SessionStart rule is injected iff the session is MANAGED.
			rule := SessionStartRule(got)
			if (rule != "") != got.Managed() {
				t.Errorf("SessionStartRule present = %v, want managed=%v", rule != "", got.Managed())
			}
		})
	}
}

// TestNormalizeAdvice locks the fail-closed mapping: any unrecognized advice string becomes
// AdviceUnknown, never a headroom class.
func TestNormalizeAdvice(t *testing.T) {
	for in, want := range map[string]Advice{
		"any": AdviceAny, "BOUNDED": AdviceBounded, " checkpoint ": AdviceCheckpoint,
		"Rebuild": AdviceRebuild, "unknown": AdviceUnknown, "garbage": AdviceUnknown, "": AdviceUnknown,
	} {
		if got := NormalizeAdvice(in); got != want {
			t.Errorf("NormalizeAdvice(%q) = %s, want %s", in, got, want)
		}
	}
}

func TestStepAdviceAffordance(t *testing.T) {
	cases := []struct{ token, want string }{
		{"any", "Keep going with full headroom (8.0k tokens available)."},
		{"bounded", "Wrap the current sub-task, land its durable state, then continue with one bounded step."},
		{"checkpoint", "Checkpoint durable state now, then proceed from that checkpoint."},
		{"rebuild", "Rebuild context from the checkpoint, then take the next step."},
		{"", "Keep going with full headroom (8.0k tokens available)."},
		{"future-token", "Keep going with full headroom (8.0k tokens available)."},
	}
	for _, tc := range cases {
		got := StepAdviceAffordance(tc.token, 8000)
		if got != tc.want {
			t.Errorf("%q => %q, want %q", tc.token, got, tc.want)
		}
		for _, forbidden := range []string{"don't", "can't", "avoid"} {
			if strings.Contains(strings.ToLower(got), forbidden) {
				t.Errorf("%q contains %q: %q", tc.token, forbidden, got)
			}
		}
	}
}

func TestManagedRuleAffordanceFirst(t *testing.T) {
	rule := SessionStartRule(Steer(SteerInput{Headless: true, DurableStore: true}))
	if !strings.HasPrefix(rule, "Keep working while checkable work remains; managed context is ON.") {
		t.Fatalf("rule does not lead with the allowed action: %q", rule)
	}
	for _, forbidden := range []string{"do not stop", "only end", "not memory"} {
		if strings.Contains(strings.ToLower(rule), forbidden) {
			t.Fatalf("rule retains negation-first framing %q: %q", forbidden, rule)
		}
	}
	for _, token := range []string{"managed context is ON", "CHECKPOINT", "REBUILD", "mcp__fak__fak_context_value", "mcp__fak__fak_context_restore"} {
		if !strings.Contains(rule, token) {
			t.Fatalf("rule dropped must-keep token %q: %q", token, rule)
		}
	}
}

// TestManagedRuleNamesTheTools guards the rule's load-bearing content: a managed rule must tell
// the agent to keep going AND name the context-state tools, or the injection is inert.
func TestManagedRuleNamesTheTools(t *testing.T) {
	rule := SessionStartRule(Steer(SteerInput{Headless: true, DurableStore: true}))
	for _, want := range []string{"managed context is ON", "Keep working", "CHECKPOINT", "REBUILD", "fak_context_value"} {
		if !contains(rule, want) {
			t.Errorf("managed rule missing %q", want)
		}
	}
}

// TestManagedRuleBindsClosingShape pins the proactive TEACH half of the closing-shape rung: the
// managed posture must carry the scannable-close clause (verdict first, bulleted body, next step
// as the final bullet) so the guard's output-style Stop-hook enforce rung rarely has to fire.
// This clause targets the same headless/long-horizon population the gate caps to.
func TestManagedRuleBindsClosingShape(t *testing.T) {
	rule := SessionStartRule(Steer(SteerInput{Headless: true, DurableStore: true}))
	for _, want := range []string{"lead with the verdict", "scannable", "next checkable step"} {
		if !contains(rule, want) {
			t.Errorf("managed rule missing closing-shape clause %q", want)
		}
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

func TestIndependentToolHintIsAdvisoryAndConditional(t *testing.T) {
	rule := IndependentToolHint(true)
	for _, want := range []string{"shadow-advisory", "independent", "dependent calls sequential", "never permission"} {
		if !strings.Contains(rule, want) {
			t.Fatalf("rule %q missing %q", rule, want)
		}
	}
}
