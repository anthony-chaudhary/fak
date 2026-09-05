package guardrsi

import "testing"

func TestArgsDigestCanonicalizesJSONObjectWhitespaceAndOrder(t *testing.T) {
	a := ArgsDigest(`{"a":1,"b":[true,"x"]}`)
	b := ArgsDigest("{\n  \"b\": [true, \"x\"],\n  \"a\": 1\n}")
	if a != b {
		t.Fatalf("digest changed across semantic JSON reorder/whitespace:\n  %s\n  %s", a, b)
	}
}

func TestLivelockDetectorFiresOnThirdIdenticalFailure(t *testing.T) {
	d := NewLivelockDetector(3)
	obs := LivelockObservation{
		TraceID:     "trace-1",
		Tool:        "Bash",
		ArgsDigest:  "sha256:abc",
		Verdict:     "DENY",
		Reason:      "POLICY_BLOCK",
		Disposition: "TERMINAL",
	}
	for i := 1; i <= 2; i++ {
		if env, ok := d.ObserveFailure(obs); ok {
			t.Fatalf("turn %d fired early: %+v", i, env)
		}
	}
	env, ok := d.ObserveFailure(obs)
	if !ok {
		t.Fatal("third identical failure did not fire")
	}
	if env.Event != LivelockEvent || env.RepeatCount != 3 || env.Tool != "Bash" || env.ArgsDigest != "sha256:abc" {
		t.Fatalf("envelope = %+v, want LIVELOCK_DETECTED repeat=3 for Bash@sha256:abc", env)
	}
	if env.SuggestedChange != "change_approach_fetch_merge_escalate_or_not_yet_with_witness" {
		t.Fatalf("suggested change = %q", env.SuggestedChange)
	}
}

func TestLivelockDetectorCheckpointsVariedSelfModifyByFourthRefusal(t *testing.T) {
	d := NewLivelockDetector(DefaultLivelockThreshold)
	digests := []string{
		ArgsDigest(`{"command":"python bridge_helper.py --publish"}`),
		ArgsDigest(`{"command":"pwsh -File bridge-helper.ps1 -Publish"}`),
		ArgsDigest(`{"command":"go run ./scratch/bridge_helper.go publish"}`),
		ArgsDigest(`{"command":"cmd /c bridge-helper.cmd publish"}`),
	}
	var last LivelockEnvelope
	for i, digest := range digests {
		env, ok := d.ObserveFailure(LivelockObservation{
			TraceID:     "session-4986",
			Tool:        "shell_command",
			ArgsDigest:  digest,
			Verdict:     "DENY",
			Reason:      "SELF_MODIFY",
			Disposition: "ESCALATE",
		})
		switch i {
		case 0, 1:
			if ok {
				t.Fatalf("refusal %d checkpointed early: %+v", i+1, env)
			}
		case 2:
			if !ok || env.Checkpoint != nil {
				t.Fatalf("third refusal = %+v ok=%v, want advisory without checkpoint", env, ok)
			}
		case 3:
			if !ok {
				t.Fatal("fourth semantically equivalent SELF_MODIFY refusal did not produce a recoverable checkpoint")
			}
			last = env
		}
	}
	if last.RepeatCount != 4 || last.Verdict != "DENY" || last.Reason != "SELF_MODIFY" {
		t.Fatalf("fourth refusal envelope = %+v, want repeat=4 while preserving DENY/SELF_MODIFY", last)
	}
	if last.IdentityKind != "semantic_refusal" || last.SemanticIdentity == "" {
		t.Fatalf("fourth refusal identity = %q/%q, want a content-free semantic identity", last.IdentityKind, last.SemanticIdentity)
	}
	checkpoint := last.Checkpoint
	if checkpoint == nil {
		t.Fatal("fourth refusal did not carry the typed checkpoint")
	}
	if checkpoint.Schema != RefusalCheckpointSchema || checkpoint.Event != RefusalCheckpointEvent ||
		checkpoint.Outcome != RefusalPausedRecoverable || checkpoint.Reason != "SELF_MODIFY" ||
		checkpoint.SemanticIdentity != last.SemanticIdentity || checkpoint.RepeatCount != 4 ||
		checkpoint.Confirmable || checkpoint.GoalAction != "preserve_active" ||
		checkpoint.NextAction != "select a sanctioned actuator, or escalate with a witness" {
		t.Fatalf("checkpoint = %+v, want a non-confirmable paused_recoverable contract preserving the active goal", checkpoint)
	}
	if last.Escalate {
		t.Fatal("fourth refusal must pause recoverably, not arm terminal escalation")
	}
	if last.SuggestedChange != string(RefusalPausedRecoverable) {
		t.Fatalf("fourth refusal approach = %q, want a typed recoverable checkpoint", last.SuggestedChange)
	}
}

func TestLivelockDetectorKeepsDeclaredSemanticRefusalRoutesDistinct(t *testing.T) {
	d := NewLivelockDetector(DefaultLivelockThreshold)
	routeA := failureHash("target-a/effect-publish")
	routeB := failureHash("target-b/effect-configure")
	identities := []string{routeA, routeA, routeB, routeB}
	for i, identity := range identities {
		env, _ := d.ObserveFailure(LivelockObservation{
			TraceID:          "session-distinct-routes",
			Tool:             "shell_command",
			ArgsDigest:       ArgsDigest(identity),
			Verdict:          "DENY",
			Reason:           "SELF_MODIFY",
			Disposition:      "ESCALATE",
			SemanticIdentity: identity,
		})
		if env.Checkpoint != nil {
			t.Fatalf("observation %d conflated distinct target/effect routes: %+v", i+1, env)
		}
	}
}

func TestLivelockDetectorFiresOnThirdIdenticalAllowedCall(t *testing.T) {
	d := NewLivelockDetector(3)
	obs := LivelockObservation{
		TraceID:    "trace-allow",
		Tool:       "shell_command",
		ArgsDigest: "sha256:abc",
	}
	for i := 1; i <= 2; i++ {
		if env, ok := d.ObserveAllowed(obs); ok {
			t.Fatalf("turn %d fired early: %+v", i, env)
		}
	}
	env, ok := d.ObserveAllowed(obs)
	if !ok {
		t.Fatal("third identical allowed call did not fire")
	}
	if env.Event != LivelockEvent || env.RepeatCount != 3 || env.Tool != "shell_command" || env.Verdict != "ALLOW" {
		t.Fatalf("envelope = %+v, want LIVELOCK_DETECTED repeat=3 ALLOW shell_command@sha256:abc", env)
	}
	if env.SuggestedChange != "change_approach_stop_repeating_successful_call_or_summarize_result" {
		t.Fatalf("suggested change = %q", env.SuggestedChange)
	}
}

func TestLivelockDetectorFiresOnThirdIdenticalTransform(t *testing.T) {
	d := NewLivelockDetector(3)
	obs := LivelockObservation{
		TraceID:    "trace-transform",
		Tool:       "Bash",
		ArgsDigest: "sha256:abc",
		Verdict:    "TRANSFORM",
		Reason:     "REPAIR",
	}
	for i := 1; i <= 2; i++ {
		if env, ok := d.ObserveAdmitted(obs); ok {
			t.Fatalf("turn %d fired early: %+v", i, env)
		}
	}
	env, ok := d.ObserveAdmitted(obs)
	if !ok {
		t.Fatal("third identical transform did not fire")
	}
	if env.Event != LivelockEvent || env.RepeatCount != 3 || env.Tool != "Bash" || env.Verdict != "TRANSFORM" || env.Reason != "REPAIR" {
		t.Fatalf("envelope = %+v, want LIVELOCK_DETECTED repeat=3 TRANSFORM Bash@sha256:abc", env)
	}
	if env.SuggestedChange != "change_approach_stop_repeating_successful_call_or_summarize_result" {
		t.Fatalf("suggested change = %q", env.SuggestedChange)
	}
}

func TestLivelockDetectorArmsFuseAfterAdvisory(t *testing.T) {
	// Advisory fires at 3; the default fuse factor is 2, so the fuse arms at 6.
	d := NewLivelockDetector(3)
	obs := LivelockObservation{
		TraceID:    "trace-fuse",
		Tool:       "Bash",
		ArgsDigest: "sha256:abc",
		Verdict:    "ALLOW",
	}
	// Repeats 1..2: nothing fires.
	for i := 1; i <= 2; i++ {
		if env, ok := d.ObserveAdmitted(obs); ok {
			t.Fatalf("repeat %d fired early: %+v", i, env)
		}
	}
	// Repeats 3..5: advisory fires but the fuse is NOT yet armed.
	for i := 3; i <= 5; i++ {
		env, ok := d.ObserveAdmitted(obs)
		if !ok {
			t.Fatalf("repeat %d did not fire the advisory", i)
		}
		if env.RepeatCount != i {
			t.Fatalf("repeat %d: envelope repeat=%d", i, env.RepeatCount)
		}
		if env.Fuse {
			t.Fatalf("repeat %d armed the fuse before the fuse count (6): %+v", i, env)
		}
	}
	// Repeat 6: the fuse arms.
	env, ok := d.ObserveAdmitted(obs)
	if !ok {
		t.Fatal("repeat 6 did not fire")
	}
	if env.RepeatCount != 6 || !env.Fuse {
		t.Fatalf("repeat 6 envelope = %+v, want repeat=6 fuse=true", env)
	}
	// Beyond the fuse count it stays armed.
	env, ok = d.ObserveAdmitted(obs)
	if !ok || !env.Fuse || env.RepeatCount != 7 {
		t.Fatalf("repeat 7 envelope = %+v, want repeat=7 fuse=true", env)
	}
}

func TestLivelockDetectorArmsTerminalEscalateAboveFuse(t *testing.T) {
	// Advisory at 3, fuse at 6 (factor 2), abort/Escalate at 9 (factor 3).
	d := NewLivelockDetector(3)
	obs := LivelockObservation{
		TraceID:     "trace-escalate",
		Tool:        "Bash",
		ArgsDigest:  "sha256:abc",
		Verdict:     "ALLOW",
		Disposition: "RETRYABLE",
	}
	// Repeats 1..8: Escalate must stay false (fuse arms at 6, but the terminal rung
	// must not fire until 9 so the retryable fuse always gets its turns first).
	for i := 1; i <= 8; i++ {
		env, _ := d.ObserveAdmitted(obs)
		if env.Escalate {
			t.Fatalf("repeat %d armed Escalate before the abort count (9): %+v", i, env)
		}
	}
	// Repeat 9: Escalate arms, and it implies Fuse.
	env, ok := d.ObserveAdmitted(obs)
	if !ok || env.RepeatCount != 9 || !env.Escalate || !env.Fuse {
		t.Fatalf("repeat 9 envelope = %+v, want repeat=9 escalate=true fuse=true", env)
	}
	// Beyond the abort count it stays escalated.
	env, ok = d.ObserveAdmitted(obs)
	if !ok || !env.Escalate || env.RepeatCount != 10 {
		t.Fatalf("repeat 10 envelope = %+v, want repeat=10 escalate=true", env)
	}
}

func TestLivelockDetectorFuseOptOutAlsoDisablesEscalate(t *testing.T) {
	// A detector with the fuse opted out (fuse<=0) must also never arm the terminal
	// Escalate rung — an advisory-only detector stays advisory-only.
	off := NewLivelockDetectorWithFuse(3, 0)
	obs := LivelockObservation{TraceID: "t", Tool: "Bash", ArgsDigest: "sha256:x", Verdict: "DENY", Reason: "POLICY_BLOCK"}
	for i := 1; i <= 12; i++ {
		env, _ := off.ObserveFailure(obs)
		if env.Escalate || env.Fuse {
			t.Fatalf("opt-out detector armed fuse/escalate at repeat %d: %+v", i, env)
		}
	}
}

func TestLivelockDetectorExplicitFuseKeepsEscalateAbove(t *testing.T) {
	// An explicit fuse at 4 (advisory 3) must push the terminal rung strictly above 4
	// so the retryable fuse fires before the terminal stop.
	d := NewLivelockDetectorWithFuse(3, 4)
	obs := LivelockObservation{TraceID: "t", Tool: "Bash", ArgsDigest: "sha256:x", Verdict: "DENY", Reason: "POLICY_BLOCK", Disposition: "RETRYABLE"}
	var last LivelockEnvelope
	for i := 1; i <= 4; i++ {
		last, _ = d.ObserveFailure(obs)
	}
	if !last.Fuse || last.Escalate {
		t.Fatalf("repeat 4 (explicit fuse) = %+v, want fuse=true escalate=false", last)
	}
}

func TestLivelockDetectorWithFuseHonorsExplicitFuseAndOptOut(t *testing.T) {
	// Explicit fuse at 4 with advisory at 3.
	d := NewLivelockDetectorWithFuse(3, 4)
	obs := LivelockObservation{TraceID: "t", Tool: "Bash", ArgsDigest: "sha256:x", Verdict: "DENY", Reason: "POLICY_BLOCK"}
	var last LivelockEnvelope
	for i := 1; i <= 4; i++ {
		last, _ = d.ObserveFailure(obs)
	}
	if !last.Fuse || last.RepeatCount != 4 {
		t.Fatalf("explicit fuse=4 envelope = %+v, want repeat=4 fuse=true", last)
	}

	// A fuse below the advisory threshold clamps up to the threshold (can't precede
	// the first advisory).
	clamped := NewLivelockDetectorWithFuse(3, 1)
	for i := 1; i <= 2; i++ {
		if env, ok := clamped.ObserveFailure(obs); ok {
			t.Fatalf("clamped fuse fired at repeat %d: %+v", i, env)
		}
	}
	env, ok := clamped.ObserveFailure(obs)
	if !ok || !env.Fuse {
		t.Fatalf("clamped fuse envelope = %+v, want first advisory to also arm the fuse", env)
	}

	// Fuse opt-out: advisory still fires, fuse never arms.
	off := NewLivelockDetectorWithFuse(3, 0)
	for i := 1; i <= 10; i++ {
		env, _ := off.ObserveFailure(obs)
		if env.Fuse {
			t.Fatalf("opt-out detector armed the fuse at repeat %d: %+v", i, env)
		}
	}
}

func TestLivelockDetectorResetsOnDifferentFailureAndClear(t *testing.T) {
	d := NewLivelockDetector(3)
	obs := LivelockObservation{TraceID: "trace-1", Tool: "Bash", ArgsDigest: "sha256:one", Verdict: "DENY", Reason: "POLICY_BLOCK"}
	if _, ok := d.ObserveFailure(obs); ok {
		t.Fatal("first failure fired")
	}
	if _, ok := d.ObserveFailure(obs); ok {
		t.Fatal("second failure fired")
	}
	changed := obs
	changed.ArgsDigest = "sha256:two"
	if _, ok := d.ObserveFailure(changed); ok {
		t.Fatal("changed call must reset the consecutive run")
	}
	if _, ok := d.ObserveFailure(obs); ok {
		t.Fatal("returning to the first call starts a fresh run, not repeat=3")
	}
	d.Clear("trace-1")
	if _, ok := d.ObserveFailure(obs); ok {
		t.Fatal("clear must reset the run")
	}
}

func TestCargoCultPrefixDetection(t *testing.T) {
	d := NewLivelockDetector(3)
	trace := "session-prefix-1"

	// 1. Prove compound shell commands with ';' and varying suffixes trigger prefix livelock detection on the 3rd repetition.
	commands := []string{
		"dos man wedge TRUST_VIOLATION --explain; fak worktree worker list",
		"dos man wedge TRUST_VIOLATION --explain; fak orchestration plan --help",
		"dos man wedge TRUST_VIOLATION --explain; go test ./...",
	}

	for i, cmd := range commands[:2] {
		obs := LivelockObservation{
			TraceID: trace,
			Tool:    "Bash",
			Command: cmd,
			Verdict: "ALLOW",
		}
		if env, ok := d.ObserveAdmitted(obs); ok {
			t.Fatalf("turn %d fired early: %+v", i+1, env)
		}
	}

	// 3rd repetition with a different suffix: exact-match livelock does not fire,
	// but cargo-cult prefix detection fires!
	obs3 := LivelockObservation{
		TraceID: trace,
		Tool:    "Bash",
		Command: commands[2],
		Verdict: "ALLOW",
	}
	env, ok := d.ObserveAdmitted(obs3)
	if !ok {
		t.Fatal("third repeat with varying suffix failed to trigger cargo-cult prefix detection")
	}
	if env.Event != CargoCultPrefixEvent {
		t.Fatalf("event = %q, want %q", env.Event, CargoCultPrefixEvent)
	}
	if env.RepeatCount != 3 {
		t.Fatalf("repeat_count = %d, want 3", env.RepeatCount)
	}
	expectedPrefix := "dos man wedge TRUST_VIOLATION --explain"
	if env.Prefix != expectedPrefix || env.CommandPrefix != expectedPrefix {
		t.Fatalf("prefix = %q / command_prefix = %q, want %q", env.Prefix, env.CommandPrefix, expectedPrefix)
	}
	if env.CargoCultPrefix == nil {
		t.Fatal("CargoCultPrefix record was nil")
	}
	if env.CargoCultPrefix.Event != CargoCultPrefixEvent || env.CargoCultPrefix.Prefix != expectedPrefix || env.CargoCultPrefix.RepeatCount != 3 {
		t.Fatalf("CargoCultPrefix = %+v, want prefix=%q repeat=3", env.CargoCultPrefix, expectedPrefix)
	}
	if env.IdentityKind != "cargo_cult_prefix" {
		t.Fatalf("identity_kind = %q, want cargo_cult_prefix", env.IdentityKind)
	}
	if env.SuggestedChange != SuggestedChangeCargoCultPrefix {
		t.Fatalf("suggested_change = %q, want %q", env.SuggestedChange, SuggestedChangeCargoCultPrefix)
	}

	// 2. 4th turn with same prefix continues to fire with incremented repeat count
	obs4 := LivelockObservation{
		TraceID: trace,
		Tool:    "Bash",
		Command: "dos man wedge TRUST_VIOLATION --explain; git status",
		Verdict: "ALLOW",
	}
	env4, ok := d.ObserveAdmitted(obs4)
	if !ok || env4.RepeatCount != 4 || env4.Event != CargoCultPrefixEvent {
		t.Fatalf("turn 4 = %+v ok=%v, want repeat=4 event=%s", env4, ok, CargoCultPrefixEvent)
	}

	// 3. Reset: substantive command without the prefix breaks consecutive run
	obs5 := LivelockObservation{
		TraceID: trace,
		Tool:    "Bash",
		Command: "git status",
		Verdict: "ALLOW",
	}
	if env5, ok := d.ObserveAdmitted(obs5); ok {
		t.Fatalf("substantive command without prefix fired: %+v", env5)
	}

	// Returning to prefix starts fresh at 1 (does not fire)
	obs6 := LivelockObservation{
		TraceID: trace,
		Tool:    "Bash",
		Command: "dos man wedge TRUST_VIOLATION --explain; git diff",
		Verdict: "ALLOW",
	}
	if env6, ok := d.ObserveAdmitted(obs6); ok {
		t.Fatalf("first repetition after reset fired early: %+v", env6)
	}

	// 4. Test '&&' compound commands with 'fak recover' prefix
	d.Clear(trace)
	recoverCommands := []string{
		"fak recover TRUST_VIOLATION && fak worktree worker list",
		"fak recover TRUST_VIOLATION && fak orchestration plan --help",
		"fak recover TRUST_VIOLATION && fak buildcheck",
	}
	for i, cmd := range recoverCommands[:2] {
		if env, ok := d.ObserveAdmitted(LivelockObservation{
			TraceID: trace,
			Tool:    "Bash",
			Command: cmd,
			Verdict: "ALLOW",
		}); ok {
			t.Fatalf("recover turn %d fired early: %+v", i+1, env)
		}
	}
	envRec, ok := d.ObserveAdmitted(LivelockObservation{
		TraceID: trace,
		Tool:    "Bash",
		Command: recoverCommands[2],
		Verdict: "ALLOW",
	})
	if !ok {
		t.Fatal("fak recover compound commands failed to trigger on third repeat")
	}
	if envRec.Event != CargoCultPrefixEvent || envRec.Prefix != "fak recover TRUST_VIOLATION" {
		t.Fatalf("recover envelope = %+v, want event=%s prefix='fak recover TRUST_VIOLATION'", envRec, CargoCultPrefixEvent)
	}

	// 5. Test JSON RawArgs extraction
	d.Clear(trace)
	rawArgsCalls := []string{
		`{"command":"dos man wedge TRUST_VIOLATION --explain; echo 1"}`,
		`{"command":"dos man wedge TRUST_VIOLATION --explain; echo 2"}`,
		`{"command":"dos man wedge TRUST_VIOLATION --explain; echo 3"}`,
	}
	for i, raw := range rawArgsCalls[:2] {
		if env, ok := d.ObserveAdmitted(LivelockObservation{
			TraceID: trace,
			Tool:    "Bash",
			RawArgs: raw,
			Verdict: "ALLOW",
		}); ok {
			t.Fatalf("raw args turn %d fired early: %+v", i+1, env)
		}
	}
	envRaw, ok := d.ObserveAdmitted(LivelockObservation{
		TraceID: trace,
		Tool:    "Bash",
		RawArgs: rawArgsCalls[2],
		Verdict: "ALLOW",
	})
	if !ok || envRaw.Event != CargoCultPrefixEvent || envRaw.RepeatCount != 3 {
		t.Fatalf("raw args turn 3 = %+v ok=%v, want repeat=3 event=%s", envRaw, ok, CargoCultPrefixEvent)
	}

	// 6. Benign compound commands (e.g. cd /dir && ...) do NOT trigger cargo-cult prefixing
	d.Clear(trace)
	benignCommands := []string{
		"cd /some/dir && make test",
		"cd /some/dir && make build",
		"cd /some/dir && git status",
	}
	for i, cmd := range benignCommands {
		if env, ok := d.ObserveAdmitted(LivelockObservation{
			TraceID: trace,
			Tool:    "Bash",
			Command: cmd,
			Verdict: "ALLOW",
		}); ok {
			t.Fatalf("benign compound command turn %d falsely triggered prefix livelock: %+v", i+1, env)
		}
	}
}

func TestDecomposeCompoundCommand(t *testing.T) {
	tests := []struct {
		input      string
		wantPrefix string
		wantSuffix string
		wantComp   bool
	}{
		{"dos man wedge TRUST_VIOLATION --explain; fak worktree worker list", "dos man wedge TRUST_VIOLATION --explain", "fak worktree worker list", true},
		{"fak recover TRUST_VIOLATION && fak orchestration plan", "fak recover TRUST_VIOLATION", "fak orchestration plan", true},
		{"echo 'foo; bar' && git status", "echo 'foo; bar'", "git status", true},
		{"echo \"foo && bar\"; git status", "echo \"foo && bar\"", "git status", true},
		{"git status", "", "", false},
		{"", "", "", false},
		{"; git status", "", "", false},
		{"git status;", "", "", false},
	}
	for _, tt := range tests {
		p, s, comp := DecomposeCompoundCommand(tt.input)
		if comp != tt.wantComp || p != tt.wantPrefix || s != tt.wantSuffix {
			t.Errorf("DecomposeCompoundCommand(%q) = (%q, %q, %v), want (%q, %q, %v)",
				tt.input, p, s, comp, tt.wantPrefix, tt.wantSuffix, tt.wantComp)
		}
	}
}

func TestLivelockDetectorTargetAwarenessDistinguishesFiles(t *testing.T) {
	d := NewLivelockDetector(DefaultLivelockThreshold)
	trace := "session-target-aware"

	obsFileA := LivelockObservation{
		TraceID:          trace,
		Tool:             "file_editor",
		Verdict:          "DENY",
		Reason:           "POLICY_BLOCK",
		Disposition:      "RETRYABLE",
		SemanticIdentity: "internal/guardrsi/livelock.go",
	}
	obsFileB := LivelockObservation{
		TraceID:          trace,
		Tool:             "file_editor",
		Verdict:          "DENY",
		Reason:           "POLICY_BLOCK",
		Disposition:      "RETRYABLE",
		SemanticIdentity: "internal/guardrsi/recovery.go",
	}

	// 1. Two failures on file A (repeat 1 and 2, below threshold 3).
	for i := 1; i <= 2; i++ {
		if env, ok := d.ObserveFailure(obsFileA); ok {
			t.Fatalf("file A failure %d fired early: %+v", i, env)
		}
	}

	// 2. Failures on file B must not conflate with file A's failure streak.
	// If files were conflated, the first failure on file B would count as failure 3
	// and fire livelock.
	for i := 1; i <= 2; i++ {
		if env, ok := d.ObserveFailure(obsFileB); ok {
			t.Fatalf("file B failure %d fired early: %+v", i, env)
		}
	}

	// 3. A 3rd consecutive failure on file B reaches threshold and fires livelock for file B.
	envB, ok := d.ObserveFailure(obsFileB)
	if !ok {
		t.Fatal("3rd consecutive failure on file B did not fire livelock")
	}
	if envB.RepeatCount != 3 {
		t.Fatalf("file B repeat count = %d, want 3", envB.RepeatCount)
	}
	if envB.Event != LivelockEvent {
		t.Fatalf("file B event = %q, want %q", envB.Event, LivelockEvent)
	}
	if envB.IdentityKind != "semantic_refusal" || envB.SemanticIdentity == "" {
		t.Fatalf("file B identity = %q/%q, want semantic_refusal", envB.IdentityKind, envB.SemanticIdentity)
	}

	// 4. Switching back to file A starts a distinct run for file A (repeat count = 1, does not fire).
	envA, ok := d.ObserveFailure(obsFileA)
	if ok {
		t.Fatalf("file A after file B fired early: %+v", envA)
	}

	// 5. Subsequent consecutive failures on file A reach threshold independently.
	for i := 2; i <= 3; i++ {
		env, ok := d.ObserveFailure(obsFileA)
		if i < 3 && ok {
			t.Fatalf("file A repetition %d fired early: %+v", i, env)
		}
		if i == 3 {
			if !ok || env.RepeatCount != 3 {
				t.Fatalf("file A 3rd repeat = %+v, ok=%v, want repeat=3", env, ok)
			}
			if env.SemanticIdentity == envB.SemanticIdentity {
				t.Fatalf("file A and file B must have distinct semantic identities: got %q", env.SemanticIdentity)
			}
		}
	}
}

func TestLivelockDetectorAdmittedCallResetsRefusalStreak(t *testing.T) {
	d := NewLivelockDetector(DefaultLivelockThreshold)
	trace := "session-admitted-reset"

	refusal := LivelockObservation{
		TraceID:          trace,
		Tool:             "file_editor",
		Verdict:          "DENY",
		Reason:           "POLICY_BLOCK",
		Disposition:      "RETRYABLE",
		SemanticIdentity: "internal/guardrsi/livelock.go",
	}

	// 1. Two consecutive refusals (streak = 2, threshold = 3).
	for i := 1; i <= 2; i++ {
		if env, ok := d.ObserveFailure(refusal); ok {
			t.Fatalf("refusal %d fired early: %+v", i, env)
		}
	}

	// 2. An admitted call representing a forward-progressing recovery action.
	admitted := LivelockObservation{
		TraceID:          trace,
		Tool:             "file_editor",
		Verdict:          "ALLOW",
		SemanticIdentity: "internal/guardrsi/livelock.go",
	}
	if env, ok := d.ObserveAdmitted(admitted); ok {
		t.Fatalf("admitted recovery action fired unexpectedly: %+v", env)
	}

	// 3. Next refusal must start a fresh refusal streak (count=1), not fire as repeat 3.
	if env, ok := d.ObserveFailure(refusal); ok {
		t.Fatalf("refusal after admitted action fired livelock (streak was not reset): %+v", env)
	}

	// 4. Second refusal after recovery (count=2, does not fire).
	if env, ok := d.ObserveFailure(refusal); ok {
		t.Fatalf("second refusal after recovery fired early: %+v", env)
	}

	// 5. Third consecutive refusal after recovery reaches threshold=3 and fires livelock.
	env, ok := d.ObserveFailure(refusal)
	if !ok {
		t.Fatal("third refusal after recovery did not fire livelock")
	}
	if env.RepeatCount != 3 {
		t.Fatalf("repeat count = %d, want 3", env.RepeatCount)
	}
}
