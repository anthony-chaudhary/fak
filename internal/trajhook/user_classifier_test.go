package trajhook

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/trajctl"
	"github.com/anthony-chaudhary/fak/internal/trajctlhook"
)

func TestClassifierRegistrationAndRetrieval(t *testing.T) {
	ResetStepClassifiersForTest()
	defer ResetStepClassifiersForTest()

	if len(GetStepClassifiers()) != 0 {
		t.Fatalf("expected empty classifiers at start, got %d", len(GetStepClassifiers()))
	}

	c1 := StepClassifierFunc(func(_ context.Context, _, _, _ string) (StepClassification, error) {
		return StepClassification{Action: ActionSteer, Reason: "c1"}, nil
	})
	c2 := StepClassifierFunc(func(_ context.Context, _, _, _ string) (StepClassification, error) {
		return StepClassification{Action: ActionPause, Reason: "c2"}, nil
	})

	RegisterStepClassifier("first", c1)
	RegisterStepClassifier("second", c2)

	classifiers := GetStepClassifiers()
	if len(classifiers) != 2 {
		t.Fatalf("expected 2 classifiers, got %d", len(classifiers))
	}

	// Overwrite "first" and verify order is preserved
	c1Updated := StepClassifierFunc(func(_ context.Context, _, _, _ string) (StepClassification, error) {
		return StepClassification{Action: ActionRetry, Reason: "c1_updated"}, nil
	})
	RegisterStepClassifier("first", c1Updated)

	classifiers = GetStepClassifiers()
	if len(classifiers) != 2 {
		t.Fatalf("expected 2 classifiers after update, got %d", len(classifiers))
	}

	res, err := classifiers[0].ClassifyPostResult(context.Background(), "", "", "")
	if err != nil || res.Reason != "c1_updated" {
		t.Fatalf("expected c1_updated, got res=%v, err=%v", res, err)
	}
	resPre, err := classifiers[0].ClassifyPreCall(context.Background(), "", "")
	if err != nil || resPre.Reason != "c1_updated" {
		t.Fatalf("expected c1_updated for pre-call, got res=%v, err=%v", resPre, err)
	}
}

func TestEvaluateTurn_PredicateTriggering(t *testing.T) {
	ResetStepClassifiersForTest()
	defer ResetStepClassifiersForTest()

	RegisterStepClassifier("danger_detector", StepClassifierFunc(func(_ context.Context, tool, args, _ string) (StepClassification, error) {
		if tool == "bash" && strings.Contains(args, "rm -rf") {
			return StepClassification{
				Action:              ActionRollback,
				Confidence:          0.99,
				Reason:              "destructive filesystem removal detected",
				Guidance:            "Use safe file tools or narrow paths",
				NegativeConstraints: []string{"never use rm -rf on repo root"},
			}, nil
		}
		if tool == "bash" && strings.Contains(args, "hang") {
			return StepClassification{
				Action:     ActionPause,
				Confidence: 0.95,
				Reason:     "hang detected",
			}, nil
		}
		return StepClassification{}, nil
	}))

	ctx := context.Background()

	// 1. Trigger rollback
	sc, tripped := EvaluateTurn(ctx, "bash", `{"cmd": "rm -rf /"}`, "")
	if !tripped {
		t.Fatalf("expected EvaluateTurn to trip on dangerous command")
	}
	if sc.Action != ActionRollback {
		t.Fatalf("expected ActionRollback, got %s", sc.Action)
	}
	if sc.Confidence != 0.99 {
		t.Fatalf("expected confidence 0.99, got %f", sc.Confidence)
	}
	if sc.Reason != "destructive filesystem removal detected" {
		t.Fatalf("unexpected reason: %s", sc.Reason)
	}

	// 2. Trigger pause
	sc, tripped = EvaluateTurn(ctx, "bash", `{"cmd": "hang indefinitely"}`, "")
	if !tripped || sc.Action != ActionPause {
		t.Fatalf("expected ActionPause, got tripped=%v, action=%s", tripped, sc.Action)
	}

	// 3. Benign call
	sc, tripped = EvaluateTurn(ctx, "read", `{"path": "foo.go"}`, "content")
	if tripped {
		t.Fatalf("expected benign call not to trip, got tripped=true, action=%s", sc.Action)
	}
}

func TestEvaluateTurn_ShortCircuiting(t *testing.T) {
	ResetStepClassifiersForTest()
	defer ResetStepClassifiersForTest()

	firstEvaluated := false
	secondEvaluated := false
	thirdEvaluated := false

	RegisterStepClassifier("c1_err", StepClassifierFunc(func(_ context.Context, _, _, _ string) (StepClassification, error) {
		firstEvaluated = true
		return StepClassification{}, errors.New("temporary error")
	}))

	RegisterStepClassifier("c2_trips", StepClassifierFunc(func(_ context.Context, tool, _, _ string) (StepClassification, error) {
		secondEvaluated = true
		if tool == "target" {
			return StepClassification{
				Action:   ActionSteer,
				Reason:   "tripped on c2",
				Guidance: "steer away from target",
			}, nil
		}
		return StepClassification{}, nil
	}))

	RegisterStepClassifier("c3_unreachable", StepClassifierFunc(func(_ context.Context, _, _, _ string) (StepClassification, error) {
		thirdEvaluated = true
		return StepClassification{Action: ActionPause, Reason: "c3"}, nil
	}))

	sc, tripped := EvaluateTurn(context.Background(), "target", "", "")
	if !tripped {
		t.Fatalf("expected EvaluateTurn to trip")
	}
	if sc.Action != ActionSteer || sc.Reason != "tripped on c2" {
		t.Fatalf("unexpected classification: %+v", sc)
	}

	if !firstEvaluated {
		t.Errorf("expected first classifier to be evaluated")
	}
	if !secondEvaluated {
		t.Errorf("expected second classifier to be evaluated")
	}
	if thirdEvaluated {
		t.Errorf("expected third classifier NOT to be evaluated due to short-circuiting")
	}
}

func TestStepClassifierSemanticScreen_Integration(t *testing.T) {
	ResetStepClassifiersForTest()
	defer ResetStepClassifiersForTest()

	var _ abi.SemanticScreen = StepClassifierSemanticScreen{}
	var _ abi.SemanticScreen = (*StepClassifierSemanticScreen)(nil)

	screen := NewStepClassifierSemanticScreen()

	// 1. When no classifiers are registered, advice is ScreenAllow
	advice := screen.ScreenResult(context.Background(), &abi.ToolCall{Tool: "test"}, []byte("ok"))
	if advice.Disposition != abi.ScreenAllow {
		t.Fatalf("expected ScreenAllow with no classifiers, got %v", advice.Disposition)
	}

	// 2. Test ActionPause -> ScreenQuarantine
	RegisterStepClassifier("pause_rule", StepClassifierFunc(func(_ context.Context, tool, _, _ string) (StepClassification, error) {
		if tool == "pause_me" {
			return StepClassification{Action: ActionPause, Reason: "suspended by policy"}, nil
		}
		return StepClassification{}, nil
	}))

	advice = screen.ScreenResult(context.Background(), &abi.ToolCall{Tool: "pause_me"}, []byte("body"))
	if advice.Disposition != abi.ScreenQuarantine {
		t.Fatalf("expected ScreenQuarantine for ActionPause, got %v", advice.Disposition)
	}
	if advice.Reason != abi.ReasonTrustViolation {
		t.Fatalf("expected ReasonTrustViolation, got %v", advice.Reason)
	}
	if advice.Digest != "suspended by policy" {
		t.Fatalf("unexpected digest: %s", advice.Digest)
	}

	// 3. Test ActionSteer -> ScreenDigest
	ResetStepClassifiersForTest()
	RegisterStepClassifier("steer_rule", StepClassifierFunc(func(_ context.Context, tool, _, _ string) (StepClassification, error) {
		if tool == "steer_me" {
			return StepClassification{
				Action:   ActionSteer,
				Reason:   "drift detected",
				Guidance: "focus on core issue",
			}, nil
		}
		return StepClassification{}, nil
	}))

	advice = screen.ScreenResult(context.Background(), &abi.ToolCall{Tool: "steer_me"}, []byte("body"))
	if advice.Disposition != abi.ScreenDigest {
		t.Fatalf("expected ScreenDigest for ActionSteer, got %v", advice.Disposition)
	}
	if advice.Digest != "focus on core issue" {
		t.Fatalf("expected guidance as digest, got %q", advice.Digest)
	}

	// 4. Test ActionRollback -> ScreenQuarantine
	ResetStepClassifiersForTest()
	RegisterStepClassifier("rollback_rule", StepClassifierFunc(func(_ context.Context, tool, _, _ string) (StepClassification, error) {
		if tool == "rollback_me" {
			return StepClassification{Action: ActionRollback, Reason: "corrupted state"}, nil
		}
		return StepClassification{}, nil
	}))

	advice = screen.ScreenResult(context.Background(), &abi.ToolCall{Tool: "rollback_me"}, []byte("body"))
	if advice.Disposition != abi.ScreenQuarantine {
		t.Fatalf("expected ScreenQuarantine for ActionRollback, got %v", advice.Disposition)
	}

	// 5. Test ActionRetry -> ScreenQuarantine
	ResetStepClassifiersForTest()
	RegisterStepClassifier("retry_rule", StepClassifierFunc(func(_ context.Context, tool, _, _ string) (StepClassification, error) {
		if tool == "retry_me" {
			return StepClassification{Action: ActionRetry, Reason: "retry needed"}, nil
		}
		return StepClassification{}, nil
	}))

	advice = screen.ScreenResult(context.Background(), &abi.ToolCall{Tool: "retry_me"}, []byte("body"))
	if advice.Disposition != abi.ScreenQuarantine {
		t.Fatalf("expected ScreenQuarantine for ActionRetry, got %v", advice.Disposition)
	}

	// 6. Test isolated custom registry
	customReg := NewClassifierRegistry()
	customReg.Register("custom", StepClassifierFunc(func(_ context.Context, _, _, _ string) (StepClassification, error) {
		return StepClassification{Action: ActionSteer, Guidance: "custom guidance"}, nil
	}))
	customScreen := NewStepClassifierSemanticScreen(customReg)
	adv := customScreen.ScreenResult(context.Background(), nil, []byte("x"))
	if adv.Disposition != abi.ScreenDigest || adv.Digest != "custom guidance" {
		t.Fatalf("unexpected custom screen advice: %+v", adv)
	}
}

func TestTypedActionOutput(t *testing.T) {
	orig := StepClassification{
		Action:              ActionRetry,
		Confidence:          0.88,
		Reason:              "model repeated hallucinated import",
		Guidance:            "Use internal/abi directly",
		NegativeConstraints: []string{"do not import external packages", "no github.com/foo"},
	}

	data, err := json.Marshal(orig)
	if err != nil {
		t.Fatalf("marshal error: %v", err)
	}

	var parsed StepClassification
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	if parsed.Action != ActionRetry {
		t.Errorf("action mismatch: got %s, want %s", parsed.Action, ActionRetry)
	}
	if parsed.Confidence != 0.88 {
		t.Errorf("confidence mismatch: got %f, want 0.88", parsed.Confidence)
	}
	if parsed.Reason != orig.Reason {
		t.Errorf("reason mismatch: got %s, want %s", parsed.Reason, orig.Reason)
	}
	if parsed.Guidance != orig.Guidance {
		t.Errorf("guidance mismatch: got %s, want %s", parsed.Guidance, orig.Guidance)
	}
	if len(parsed.NegativeConstraints) != 2 || parsed.NegativeConstraints[0] != orig.NegativeConstraints[0] {
		t.Errorf("negative constraints mismatch: got %v, want %v", parsed.NegativeConstraints, orig.NegativeConstraints)
	}
}

func TestSkillClassifierDecl_ValidationAndMount(t *testing.T) {
	ResetStepClassifiersForTest()
	defer ResetStepClassifiersForTest()

	// 1. Missing skill and name
	err := DeclareSkillClassifier(SkillClassifierDecl{
		Action: ActionSteer,
		Reason: "missing name",
	})
	if err == nil {
		t.Errorf("expected error when skill and name are both empty")
	}

	// 2. Invalid Action
	err = DeclareSkillClassifier(SkillClassifierDecl{
		Skill:  "my-skill",
		Action: Action("INVALID_ACTION"),
		Reason: "bad action",
	})
	if err == nil {
		t.Errorf("expected error for invalid action")
	}

	// 3. FullName canonical formatting
	testCases := []struct {
		skill    string
		name     string
		expected string
	}{
		{"goal", "drift", "skill:goal:drift"},
		{"goal", "", "skill:goal"},
		{"goal", "skill:goal:already_prefixed", "skill:goal:already_prefixed"},
		{"", "standalone_classifier", "standalone_classifier"},
	}
	for _, tc := range testCases {
		d := SkillClassifierDecl{Skill: tc.skill, Name: tc.name}
		if got := d.FullName(); got != tc.expected {
			t.Errorf("FullName() = %q, want %q", got, tc.expected)
		}
	}

	// 4. Clean mounting into custom registry
	reg := NewClassifierRegistry()
	decl1 := SkillClassifierDecl{
		Skill:    "goal",
		Name:     "drift_detector",
		Action:   ActionSteer,
		Reason:   "goal drift detected",
		Guidance: "re-align with core objective",
	}
	if err := MountSkillClassifier(reg, decl1); err != nil {
		t.Fatalf("MountSkillClassifier error: %v", err)
	}

	if reg.Len() != 1 {
		t.Errorf("expected reg.Len() == 1, got %d", reg.Len())
	}
	if !reg.Has("skill:goal:drift_detector") {
		t.Errorf("expected registry to have skill:goal:drift_detector")
	}

	// 5. Clean mounting into default registry
	decl2 := SkillClassifierDecl{
		Skill:  "safety",
		Name:   "pause_guard",
		Action: ActionPause,
		Reason: "suspend on high risk",
	}
	if err := DeclareSkillClassifier(decl2); err != nil {
		t.Fatalf("DeclareSkillClassifier error: %v", err)
	}
	if !defaultClassifierRegistry.Has("skill:safety:pause_guard") {
		t.Errorf("expected default registry to have skill:safety:pause_guard")
	}
}

func TestSkillClassifier_AdditiveEvaluation_AllClosedActions(t *testing.T) {
	reg := NewClassifierRegistry()
	ctx := context.Background()

	// Declare 4 distinct skills covering each closed trajctl action
	skillSteer := SkillClassifierDecl{
		Skill:     "skill-steer",
		Name:      "drift_detector",
		Tools:     []string{"bash"},
		MatchArgs: []string{"drift_query"},
		Action:    ActionSteer,
		Reason:    "detected off-task query pattern",
		Guidance:  "focus on Issue #11410 implementation",
	}

	skillPause := SkillClassifierDecl{
		Skill:     "skill-pause",
		Name:      "infinite_loop_detector",
		Tools:     []string{"bash"},
		MatchArgs: []string{"loop_recurse"},
		Action:    ActionPause,
		Reason:    "unbounded recursive tool call detected",
	}

	skillRollback := SkillClassifierDecl{
		Skill:       "skill-rollback",
		Name:        "corrupted_write_guard",
		Tools:       []string{"write_file"},
		MatchResult: []string{"ERR_CORRUPT_BLOCK"},
		Action:      ActionRollback,
		Reason:      "step produced corrupted write output",
	}

	skillRetry := SkillClassifierDecl{
		Skill:               "skill-retry",
		Name:                "syntax_deprecation_guard",
		Tools:               []string{"eval_tool"},
		MatchResult:         []string{"DEPRECATED_SYNTAX_ERROR"},
		Action:              ActionRetry,
		Reason:              "model used deprecated syntax in eval_tool",
		Guidance:            "re-run step using modern v2 syntax",
		NegativeConstraints: []string{"do not use v1 syntax", "do not call legacy APIs"},
	}

	// Mount all 4 additively into registry
	if err := MountSkillClassifiers(reg, skillSteer, skillPause, skillRollback, skillRetry); err != nil {
		t.Fatalf("MountSkillClassifiers failed: %v", err)
	}

	if reg.Len() != 4 {
		t.Fatalf("expected 4 mounted classifiers, got %d", reg.Len())
	}
	expectedNames := []string{
		"skill:skill-steer:drift_detector",
		"skill:skill-pause:infinite_loop_detector",
		"skill:skill-rollback:corrupted_write_guard",
		"skill:skill-retry:syntax_deprecation_guard",
	}
	names := reg.GetNames()
	for i, exp := range expectedNames {
		if names[i] != exp {
			t.Errorf("name[%d] = %q, want %q", i, names[i], exp)
		}
	}

	// Set up InterventionExecutor and trajctl State for end-to-end verification
	state := &trajctl.State{
		Objectives: map[string]trajctl.Objective{
			"obj-11410": {
				ID:        "obj-11410",
				Statement: "Issue #11410 implementation",
				Status:    trajctl.StatusActive,
			},
		},
	}
	deliveredSteer := ""
	mockSteer := func(sessionID, text string) error {
		deliveredSteer = text
		return nil
	}

	exec := &trajctlhook.InterventionExecutor{
		SessionID:    "session-user-classifier",
		ObjectiveID:  "obj-11410",
		RunID:        "run-1",
		State:        state,
		SteerDeliver: mockSteer,
	}

	// Turn 1: Triggers ActionSteer
	sc1, tripped := reg.EvaluateTurn(ctx, "bash", "run --drift_query arg", "output")
	if !tripped {
		t.Fatalf("Turn 1 expected to trip ActionSteer")
	}
	if sc1.Action != ActionSteer || sc1.Guidance != "focus on Issue #11410 implementation" {
		t.Errorf("unexpected sc1: %+v", sc1)
	}
	res1, err := exec.Execute(ctx, sc1)
	if err != nil {
		t.Fatalf("exec.Execute(ActionSteer) error: %v", err)
	}
	if res1.Action != ActionSteer || res1.State != "steered" || res1.Guidance != sc1.Guidance {
		t.Errorf("unexpected res1: %+v", res1)
	}
	if deliveredSteer != sc1.Guidance {
		t.Errorf("deliveredSteer = %q, want %q", deliveredSteer, sc1.Guidance)
	}
	if len(state.Steers) != 1 || state.Steers[0].Packet != sc1.Guidance {
		t.Errorf("steer record not appended in state: %+v", state.Steers)
	}

	// Turn 2: Triggers ActionPause
	sc2, tripped := reg.EvaluateTurn(ctx, "bash", "loop_recurse --depth=99", "")
	if !tripped {
		t.Fatalf("Turn 2 expected to trip ActionPause")
	}
	if sc2.Action != ActionPause || sc2.Reason != "unbounded recursive tool call detected" {
		t.Errorf("unexpected sc2: %+v", sc2)
	}
	res2, err := exec.Execute(ctx, sc2)
	if err != nil {
		t.Fatalf("exec.Execute(ActionPause) error: %v", err)
	}
	if res2.Action != ActionPause || res2.State != "paused" {
		t.Errorf("unexpected res2: %+v", res2)
	}
	if state.Objectives["obj-11410"].Status != trajctl.StatusPaused {
		t.Errorf("expected objective to transition to StatusPaused, got: %s", state.Objectives["obj-11410"].Status)
	}

	// Turn 3: Triggers ActionRollback
	tempDir := t.TempDir()
	scratchDir := filepath.Join(tempDir, "_scratch")
	_ = os.MkdirAll(scratchDir, 0o755)
	scratchFile := filepath.Join(scratchDir, "uncommitted_block.tmp")
	_ = os.WriteFile(scratchFile, []byte("broken provisional"), 0o644)
	committedFile := filepath.Join(tempDir, "preserved.go")
	_ = os.WriteFile(committedFile, []byte("package main\n"), 0o644)

	execRollback := trajctlhook.NewInterventionExecutor(tempDir, "_scratch")

	sc3, tripped := reg.EvaluateTurn(ctx, "write_file", "path/data.bin", "ERR_CORRUPT_BLOCK_ENCOUNTERED")
	if !tripped {
		t.Fatalf("Turn 3 expected to trip ActionRollback")
	}
	if sc3.Action != ActionRollback {
		t.Errorf("unexpected sc3: %+v", sc3)
	}
	res3, err := execRollback.Execute(ctx, sc3)
	if err != nil {
		t.Fatalf("execRollback.Execute error: %v", err)
	}
	if res3.Action != ActionRollback || res3.State != "rolled_back" {
		t.Errorf("unexpected res3: %+v", res3)
	}
	if _, err := os.Stat(scratchFile); !os.IsNotExist(err) {
		t.Errorf("expected provisional scratch file to be deleted by ActionRollback")
	}
	if data, err := os.ReadFile(committedFile); err != nil || string(data) != "package main\n" {
		t.Errorf("committed file should be preserved: err=%v", err)
	}

	// Turn 4: Triggers ActionRetry
	deliveredRetry := ""
	execRetry := &trajctlhook.InterventionExecutor{
		SessionID:   "session-retry",
		ObjectiveID: "obj-11410",
		SteerDeliver: func(_, text string) error {
			deliveredRetry = text
			return nil
		},
	}
	sc4, tripped := reg.EvaluateTurn(ctx, "eval_tool", "run()", "DEPRECATED_SYNTAX_ERROR: at line 1")
	if !tripped {
		t.Fatalf("Turn 4 expected to trip ActionRetry")
	}
	if sc4.Action != ActionRetry {
		t.Errorf("unexpected sc4: %+v", sc4)
	}
	if len(sc4.NegativeConstraints) != 2 {
		t.Errorf("expected 2 negative constraints in sc4, got %d", len(sc4.NegativeConstraints))
	}
	res4, err := execRetry.Execute(ctx, sc4)
	if err != nil {
		t.Fatalf("execRetry.Execute error: %v", err)
	}
	if res4.Action != ActionRetry || res4.State != "retrying" {
		t.Errorf("unexpected res4: %+v", res4)
	}
	if len(res4.Constraints) != 2 || res4.Constraints[0] != "do not use v1 syntax" {
		t.Errorf("unexpected constraints in res4: %v", res4.Constraints)
	}
	if !strings.Contains(deliveredRetry, "do not use v1 syntax") {
		t.Errorf("expected delivered text to contain negative constraints, got: %s", deliveredRetry)
	}

	// Turn 5: Benign tool call, does not trip
	sc5, tripped := reg.EvaluateTurn(ctx, "read_file", "foo.go", "package foo")
	if tripped {
		t.Errorf("expected benign call not to trip, got tripped=true, action=%s", sc5.Action)
	}
}

func TestSkillClassifier_AdditiveOrderAndShortCircuit(t *testing.T) {
	reg := NewClassifierRegistry()
	ctx := context.Background()

	skillA := SkillClassifierDecl{
		Skill:     "skill-a",
		Name:      "first_rule",
		Tools:     []string{"bash"},
		MatchArgs: []string{"target_command"},
		Action:    ActionSteer,
		Reason:    "rule A tripped first",
	}

	skillB := SkillClassifierDecl{
		Skill:     "skill-b",
		Name:      "second_rule",
		Tools:     []string{"bash"},
		MatchArgs: []string{"target_command"},
		Action:    ActionPause,
		Reason:    "rule B tripped second",
	}

	if err := MountSkillClassifiers(reg, skillA, skillB); err != nil {
		t.Fatalf("MountSkillClassifiers error: %v", err)
	}

	// When both match, skillA was registered first and must short-circuit
	sc, tripped := reg.EvaluateTurn(ctx, "bash", "target_command", "")
	if !tripped {
		t.Fatalf("expected evaluate to trip")
	}
	if sc.Action != ActionSteer || sc.Reason != "rule A tripped first" {
		t.Errorf("expected Skill A to short-circuit, got %+v", sc)
	}
}

func TestSkillClassifier_DiskDiscovery_AgentsSkills(t *testing.T) {
	tempDir := t.TempDir()

	// 1. Create .agents/skills/git-guard with classifiers.json (ActionRollback)
	gitGuardDir := filepath.Join(tempDir, ".agents", "skills", "git-guard")
	_ = os.MkdirAll(gitGuardDir, 0o755)
	gitGuardJSON := `[
		{
			"name": "prevent_destructive_rm",
			"tools": ["bash"],
			"match_args": ["rm -rf"],
			"action": "ROLLBACK",
			"reason": "destructive rm detected"
		}
	]`
	_ = os.WriteFile(filepath.Join(gitGuardDir, "classifiers.json"), []byte(gitGuardJSON), 0o644)

	// 2. Create .agents/skills/prompt-steer with SKILL.md frontmatter (ActionSteer)
	steerDir := filepath.Join(tempDir, ".agents", "skills", "prompt-steer")
	_ = os.MkdirAll(steerDir, 0o755)
	steerMD := `---
name: prompt-steer
description: Prompts steering
trajectory_action: STEER
trajectory_tool: bash
trajectory_match: drift_signal
trajectory_reason: drift detected in prompt
trajectory_guidance: refocus on plan
---
# Prompt Steer Body
`
	_ = os.WriteFile(filepath.Join(steerDir, "SKILL.md"), []byte(steerMD), 0o644)

	// 3. Create .claude/skills/watchdog with step_classifiers.json (ActionPause)
	watchdogDir := filepath.Join(tempDir, ".claude", "skills", "watchdog")
	_ = os.MkdirAll(watchdogDir, 0o755)
	watchdogJSON := `{
		"name": "hang_detector",
		"tools": ["bash"],
		"match_args": ["hang_proc"],
		"action": "PAUSE",
		"reason": "hang detected"
	}`
	_ = os.WriteFile(filepath.Join(watchdogDir, "step_classifiers.json"), []byte(watchdogJSON), 0o644)

	// 4. Create .agents/skills/eval-guard with classifiers.json (ActionRetry)
	evalDir := filepath.Join(tempDir, ".agents", "skills", "eval-guard")
	_ = os.MkdirAll(evalDir, 0o755)
	evalJSON := `[
		{
			"name": "retry_on_syntax",
			"tools": ["eval"],
			"match_result": ["SYNTAX_ERR"],
			"action": "RETRY",
			"reason": "syntax error",
			"guidance": "fix syntax",
			"negative_constraints": ["no invalid keywords"]
		}
	]`
	_ = os.WriteFile(filepath.Join(evalDir, "classifiers.json"), []byte(evalJSON), 0o644)

	// Discover skill classifiers across workspace
	decls, err := DiscoverSkillClassifiers(tempDir)
	if err != nil {
		t.Fatalf("DiscoverSkillClassifiers failed: %v", err)
	}
	if len(decls) != 4 {
		t.Fatalf("expected 4 discovered skill classifiers, got %d", len(decls))
	}

	// Mount into clean registry
	reg := NewClassifierRegistry()
	count, err := MountDiscoveredSkillClassifiers(reg, tempDir)
	if err != nil {
		t.Fatalf("MountDiscoveredSkillClassifiers failed: %v", err)
	}
	if count != 4 {
		t.Errorf("expected count 4, got %d", count)
	}

	ctx := context.Background()

	// Verify Rollback
	scRollback, tripped := reg.EvaluateTurn(ctx, "bash", "rm -rf /tmp/data", "")
	if !tripped || scRollback.Action != ActionRollback {
		t.Errorf("expected ActionRollback for git-guard, got tripped=%v, sc=%+v", tripped, scRollback)
	}

	// Verify Steer
	scSteer, tripped := reg.EvaluateTurn(ctx, "bash", "echo drift_signal", "")
	if !tripped || scSteer.Action != ActionSteer || scSteer.Guidance != "refocus on plan" {
		t.Errorf("expected ActionSteer for prompt-steer, got tripped=%v, sc=%+v", tripped, scSteer)
	}

	// Verify Pause
	scPause, tripped := reg.EvaluateTurn(ctx, "bash", "run hang_proc", "")
	if !tripped || scPause.Action != ActionPause {
		t.Errorf("expected ActionPause for watchdog, got tripped=%v, sc=%+v", tripped, scPause)
	}

	// Verify Retry
	scRetry, tripped := reg.EvaluateTurn(ctx, "eval", "code", "SYNTAX_ERR on line 1")
	if !tripped || scRetry.Action != ActionRetry || len(scRetry.NegativeConstraints) != 1 {
		t.Errorf("expected ActionRetry for eval-guard, got tripped=%v, sc=%+v", tripped, scRetry)
	}
}

func TestSkillClassifier_ProgrammaticAndPredicateHooks(t *testing.T) {
	reg := NewClassifierRegistry()
	ctx := context.Background()

	// 1. Declarative with custom MatchPredicate and MatchAnyArgs
	declPredicate := SkillClassifierDecl{
		Skill:        "math-skill",
		Name:         "overflow_guard",
		Tools:        []string{"calc"},
		MatchAnyArgs: []string{"pow", "exp", "fact"},
		MatchPredicate: func(_ context.Context, _, args, _ string) bool {
			return strings.Contains(args, "999999")
		},
		Action: ActionPause,
		Reason: "potential integer overflow calculation",
	}

	if err := MountSkillClassifier(reg, declPredicate); err != nil {
		t.Fatalf("MountSkillClassifier error: %v", err)
	}

	// Should not trip: math tool, has "pow", but doesn't have "999999"
	_, tripped := reg.EvaluateTurn(ctx, "calc", "pow(2, 8)", "")
	if tripped {
		t.Errorf("expected no trip for pow(2, 8)")
	}

	// Should trip: math tool, has "pow", has "999999"
	sc, tripped := reg.EvaluateTurn(ctx, "calc", "pow(2, 999999)", "")
	if !tripped || sc.Action != ActionPause {
		t.Errorf("expected ActionPause for pow(2, 999999), got tripped=%v, action=%s", tripped, sc.Action)
	}

	// 2. Programmatic StepClassifier hook with tool filter
	progCalled := false
	declProgrammatic := SkillClassifierDecl{
		Skill: "prog-skill",
		Name:  "custom_classifier",
		Tools: []string{"special_tool"},
		Classifier: StepClassifierFunc(func(_ context.Context, tool, args, result string) (StepClassification, error) {
			progCalled = true
			if result == "special_error" {
				return StepClassification{Action: ActionRetry, Reason: "special retry"}, nil
			}
			return StepClassification{}, nil
		}),
	}
	if err := MountSkillClassifier(reg, declProgrammatic); err != nil {
		t.Fatalf("MountSkillClassifier programmatic error: %v", err)
	}

	// Call with different tool: programmatic classifier should NOT be called due to Tools filter
	progCalled = false
	_, _ = reg.EvaluateTurn(ctx, "other_tool", "", "special_error")
	if progCalled {
		t.Errorf("expected programmatic classifier not to be called when tool does not match")
	}

	// Call with matching tool: programmatic classifier called and trips
	progCalled = false
	scProg, tripped := reg.EvaluateTurn(ctx, "special_tool", "", "special_error")
	if !progCalled {
		t.Errorf("expected programmatic classifier to be called")
	}
	if !tripped || scProg.Action != ActionRetry {
		t.Errorf("expected ActionRetry, got tripped=%v, action=%s", tripped, scProg.Action)
	}
}

type customSplitClassifier struct {
	preCalled  bool
	postCalled bool
}

func (c *customSplitClassifier) ClassifyPreCall(_ context.Context, tool, args string) (StepClassification, error) {
	c.preCalled = true
	if tool == "dangerous_tool" {
		return StepClassification{
			Action: ActionRollback,
			Reason: "prohibited tool invocation",
		}, nil
	}
	return StepClassification{}, nil
}

func (c *customSplitClassifier) ClassifyPostResult(_ context.Context, _, _, result string) (StepClassification, error) {
	c.postCalled = true
	if strings.Contains(result, "CORRUPTED_PAYLOAD") {
		return StepClassification{
			Action: ActionPause,
			Reason: "corrupted payload encountered",
		}, nil
	}
	return StepClassification{}, nil
}

func TestStepClassifier_PreCallAndPostResultLifecycle(t *testing.T) {
	ResetStepClassifiersForTest()
	defer ResetStepClassifiersForTest()

	split := &customSplitClassifier{}
	var _ StepClassifier = split

	reg := NewClassifierRegistry()
	reg.Register("split_classifier", split)

	ctx := context.Background()

	// 1. EvaluatePreCall trips on dangerous_tool
	sc, tripped := reg.EvaluatePreCall(ctx, "dangerous_tool", "arg")
	if !tripped || sc.Action != ActionRollback || sc.Reason != "prohibited tool invocation" {
		t.Fatalf("expected ActionRollback from pre-call, got tripped=%v, sc=%+v", tripped, sc)
	}
	if !split.preCalled {
		t.Errorf("expected ClassifyPreCall to have been invoked")
	}

	// 2. EvaluatePreCall on safe tool does not trip
	sc, tripped = reg.EvaluatePreCall(ctx, "safe_tool", "arg")
	if tripped {
		t.Errorf("expected safe_tool not to trip pre-call")
	}

	// 3. EvaluatePostResult trips on corrupted payload
	sc, tripped = reg.EvaluatePostResult(ctx, "safe_tool", "arg", "output: CORRUPTED_PAYLOAD")
	if !tripped || sc.Action != ActionPause || sc.Reason != "corrupted payload encountered" {
		t.Fatalf("expected ActionPause from post-result, got tripped=%v, sc=%+v", tripped, sc)
	}
	if !split.postCalled {
		t.Errorf("expected ClassifyPostResult to have been invoked")
	}

	// 4. Test PreCallClassifierFunc and PostResultClassifierFunc adapters
	preAdapter := PreCallClassifierFunc(func(_ context.Context, tool, _ string) (StepClassification, error) {
		if tool == "pre_only" {
			return StepClassification{Action: ActionSteer, Guidance: "steer early"}, nil
		}
		return StepClassification{}, nil
	})
	postAdapter := PostResultClassifierFunc(func(_ context.Context, _, _, result string) (StepClassification, error) {
		if result == "retry_me" {
			return StepClassification{Action: ActionRetry, Guidance: "try again"}, nil
		}
		return StepClassification{}, nil
	})

	var _ StepClassifier = preAdapter
	var _ StepClassifier = postAdapter

	regAdapters := NewClassifierRegistry()
	regAdapters.Register("pre", preAdapter)
	regAdapters.Register("post", postAdapter)

	sc, tripped = regAdapters.EvaluatePreCall(ctx, "pre_only", "")
	if !tripped || sc.Action != ActionSteer || sc.Guidance != "steer early" {
		t.Errorf("preAdapter failed to trip in EvaluatePreCall: %+v", sc)
	}
	// preAdapter should not trip in post-result
	sc, tripped = regAdapters.EvaluatePostResult(ctx, "pre_only", "", "ok")
	if tripped {
		t.Errorf("preAdapter should not trip in EvaluatePostResult")
	}

	// postAdapter trips in post-result
	sc, tripped = regAdapters.EvaluatePostResult(ctx, "any", "", "retry_me")
	if !tripped || sc.Action != ActionRetry || sc.Guidance != "try again" {
		t.Errorf("postAdapter failed to trip in EvaluatePostResult: %+v", sc)
	}
	// postAdapter should not trip in pre-call
	sc, tripped = regAdapters.EvaluatePreCall(ctx, "any", "retry_me")
	if tripped {
		t.Errorf("postAdapter should not trip in EvaluatePreCall")
	}

	// 5. Semantic screen ScreenToolCall check
	screen := NewStepClassifierSemanticScreen(regAdapters)
	callAdvice := screen.ScreenToolCall(ctx, &abi.ToolCall{Tool: "pre_only"})
	if callAdvice.Disposition != abi.ScreenDigest || callAdvice.Digest != "steer early" {
		t.Errorf("ScreenToolCall advice mismatch: %+v", callAdvice)
	}
}

func TestClassifierRegistry_ConcurrencySafety(t *testing.T) {
	reg := NewClassifierRegistry()
	ctx := context.Background()
	var wg sync.WaitGroup

	// Concurrently register
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			name := fmt.Sprintf("classifier_%d", id)
			reg.Register(name, StepClassifierFunc(func(_ context.Context, _, _, _ string) (StepClassification, error) {
				if id%2 == 0 {
					return StepClassification{Action: ActionSteer, Reason: "steer"}, nil
				}
				return StepClassification{}, nil
			}))
		}(i)
	}

	// Concurrently evaluate and query
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _ = reg.EvaluateTurn(ctx, "tool", "args", "result")
			_ = reg.GetClassifiers()
			_ = reg.GetNames()
			_ = reg.Len()
			_ = reg.Has("classifier_1")
		}()
	}

	wg.Wait()
	if reg.Len() != 50 {
		t.Fatalf("expected 50 registered classifiers, got %d", reg.Len())
	}
}
