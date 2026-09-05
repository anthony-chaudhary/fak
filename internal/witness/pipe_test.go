package witness

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestRunPipeWitnessRealCommand(t *testing.T) {
	ctx := context.Background()
	receipt, stdoutBytes, stderrBytes, err := RunPipeWitness(ctx, "", "go", "version")
	if err != nil {
		t.Fatalf("RunPipeWitness failed: %v", err)
	}

	if receipt.Schema != PipeReceiptSchema {
		t.Errorf("Schema = %q, want %q", receipt.Schema, PipeReceiptSchema)
	}
	if receipt.ExitCode != 0 {
		t.Errorf("ExitCode = %d, want 0", receipt.ExitCode)
	}
	if receipt.StdoutBytes <= 0 {
		t.Errorf("StdoutBytes = %d, want > 0", receipt.StdoutBytes)
	}
	if int64(len(stdoutBytes)) != receipt.StdoutBytes {
		t.Errorf("len(stdoutBytes) = %d != StdoutBytes = %d", len(stdoutBytes), receipt.StdoutBytes)
	}
	wantStdoutSHA := sha256Hex(stdoutBytes)
	if receipt.StdoutSHA256 != wantStdoutSHA {
		t.Errorf("StdoutSHA256 = %s, want %s", receipt.StdoutSHA256, wantStdoutSHA)
	}
	if int64(len(stderrBytes)) != receipt.StderrBytes {
		t.Errorf("len(stderrBytes) = %d != StderrBytes = %d", len(stderrBytes), receipt.StderrBytes)
	}
	wantStderrSHA := sha256Hex(stderrBytes)
	if receipt.StderrSHA256 != wantStderrSHA {
		t.Errorf("StderrSHA256 = %s, want %s", receipt.StderrSHA256, wantStderrSHA)
	}
	if receipt.DurationNanos <= 0 {
		t.Errorf("DurationNanos = %d, want > 0", receipt.DurationNanos)
	}
	if got := receipt.WitnessOutcome(); got != abi.WitnessConfirmed {
		t.Errorf("WitnessOutcome = %v, want Confirmed", got)
	}
}

func TestRunPipeWitnessNonZeroExit(t *testing.T) {
	ctx := context.Background()
	// Invoke `go` with a nonexistent subcommand to guarantee a deterministic non-zero exit code.
	receipt, _, stderrBytes, err := RunPipeWitness(ctx, "", "go", "__fak_nonexistent_subcmd__")
	if err != nil {
		t.Fatalf("unexpected execution error on non-zero exit: %v", err)
	}
	if receipt.ExitCode == 0 {
		t.Errorf("ExitCode = %d, want non-zero", receipt.ExitCode)
	}
	if receipt.WitnessOutcome() != abi.WitnessRefuted {
		t.Errorf("WitnessOutcome = %v, want Refuted", receipt.WitnessOutcome())
	}
	if receipt.DurationNanos <= 0 {
		t.Errorf("DurationNanos = %d, want > 0", receipt.DurationNanos)
	}
	wantStderrSHA := sha256Hex(stderrBytes)
	if receipt.StderrSHA256 != wantStderrSHA {
		t.Errorf("StderrSHA256 = %s, want %s", receipt.StderrSHA256, wantStderrSHA)
	}
}

func TestRunPipeWitnessInvalidCommand(t *testing.T) {
	ctx := context.Background()
	// Empty argv
	receipt, _, _, err := RunPipeWitness(ctx, "")
	if err == nil {
		t.Error("expected error for empty command, got nil")
	}
	if receipt.WitnessOutcome() != abi.WitnessAbstain {
		t.Errorf("WitnessOutcome = %v, want Abstain", receipt.WitnessOutcome())
	}

	// Nonexistent executable
	receipt, _, _, err = RunPipeWitness(ctx, "", "__nonexistent_bin_fak_12345__")
	if err == nil {
		t.Error("expected error for nonexistent binary, got nil")
	}
	if receipt.Error == "" {
		t.Error("expected receipt.Error to be populated")
	}
	if receipt.WitnessOutcome() != abi.WitnessAbstain {
		t.Errorf("WitnessOutcome = %v, want Abstain", receipt.WitnessOutcome())
	}
}

func TestValidatePipeReceipt(t *testing.T) {
	stdoutSHA := sha256Hex([]byte("sample stdout"))
	stderrSHA := sha256Hex([]byte("sample stderr"))

	baseReceipt := &PipeReceipt{
		Schema:       PipeReceiptSchema,
		Command:      []string{"test"},
		ExitCode:     0,
		StdoutSHA256: stdoutSHA,
		StderrSHA256: stderrSHA,
	}
	baseReceipt.Attestation = SignPipeReceipt(baseReceipt, nil)

	// Valid matching spec
	outcome, err := ValidatePipeReceipt(baseReceipt, PipeClaimSpec{
		ExpectedExitCode:  0,
		ExpectedStdoutSHA: stdoutSHA,
		ExpectedStderrSHA: stderrSHA,
	})
	if err != nil || outcome != abi.WitnessConfirmed {
		t.Errorf("outcome = %v, err = %v, want Confirmed, nil", outcome, err)
	}

	// Exit code mismatch
	outcome, err = ValidatePipeReceipt(baseReceipt, PipeClaimSpec{
		ExpectedExitCode:  1,
		ExpectedStdoutSHA: stdoutSHA,
	})
	if err == nil || outcome != abi.WitnessRefuted {
		t.Errorf("outcome = %v, err = %v, want Refuted, error", outcome, err)
	}

	// Stdout SHA mismatch (tampered output)
	outcome, err = ValidatePipeReceipt(baseReceipt, PipeClaimSpec{
		ExpectedExitCode:  0,
		ExpectedStdoutSHA: "badsha0000000000000000000000000000000000000000000000000000000000",
	})
	if err == nil || outcome != abi.WitnessRefuted {
		t.Errorf("outcome = %v, err = %v, want Refuted, error", outcome, err)
	}

	// Stderr SHA mismatch (tampered error output)
	outcome, err = ValidatePipeReceipt(baseReceipt, PipeClaimSpec{
		ExpectedExitCode:  0,
		ExpectedStderrSHA: "badsha0000000000000000000000000000000000000000000000000000000000",
	})
	if err == nil || outcome != abi.WitnessRefuted {
		t.Errorf("outcome = %v, err = %v, want Refuted, error", outcome, err)
	}

	// Nil receipt
	outcome, err = ValidatePipeReceipt(nil, PipeClaimSpec{ExpectedExitCode: 0})
	if err == nil || outcome != abi.WitnessAbstain {
		t.Errorf("outcome = %v, err = %v, want Abstain, error", outcome, err)
	}

	// Receipt without execution or attestation is rejected
	unverifiedReceipt := &PipeReceipt{
		Schema:       PipeReceiptSchema,
		Command:      []string{"test"},
		ExitCode:     0,
		StdoutSHA256: stdoutSHA,
		StderrSHA256: stderrSHA,
	}
	outcome, err = ValidatePipeReceipt(unverifiedReceipt, PipeClaimSpec{ExpectedExitCode: 0})
	if err == nil || outcome != abi.WitnessAbstain {
		t.Errorf("outcome = %v, err = %v, want Abstain on unverified receipt", outcome, err)
	}

	// Receipt with error
	errReceipt := &PipeReceipt{
		Schema:   PipeReceiptSchema,
		ExitCode: -1,
		Error:    "spawn failed",
	}
	errReceipt.Attestation = SignPipeReceipt(errReceipt, nil)
	outcome, err = ValidatePipeReceipt(errReceipt, PipeClaimSpec{ExpectedExitCode: 0})
	if err == nil || outcome != abi.WitnessAbstain {
		t.Errorf("outcome = %v, err = %v, want Abstain, error", outcome, err)
	}

	// Matching non-zero expected exit code
	nonZeroReceipt := &PipeReceipt{
		Schema:   PipeReceiptSchema,
		ExitCode: 2,
	}
	nonZeroReceipt.Attestation = SignPipeReceipt(nonZeroReceipt, nil)
	outcome, err = ValidatePipeReceipt(nonZeroReceipt, PipeClaimSpec{ExpectedExitCode: 2})
	if err != nil || outcome != abi.WitnessConfirmed {
		t.Errorf("outcome = %v, err = %v, want Confirmed, nil", outcome, err)
	}
}

func TestResolverResolvePipeCommand(t *testing.T) {
	ctx := context.Background()
	r := New()

	// 1. Determine actual stdout SHA of `go version`
	_, stdoutBytes, _, err := RunPipeWitness(ctx, "", "go", "version")
	if err != nil {
		t.Fatalf("RunPipeWitness failed: %v", err)
	}
	actualSHA := sha256Hex(stdoutBytes)

	// 2. Resolve valid claim with matching command, exit code, and stdout SHA
	validSpec := PipeClaimSpec{
		Command:           []string{"go", "version"},
		ExpectedExitCode:  0,
		ExpectedStdoutSHA: actualSHA,
	}
	validJSON, _ := json.Marshal(validSpec)
	if got := r.Resolve(ctx, nil, "pipe:"+string(validJSON)); got != abi.WitnessConfirmed {
		t.Errorf("Resolve(valid) = %v, want Confirmed", got)
	}

	// 3. Resolve claim with tampered expected stdout SHA
	tamperedSpec := PipeClaimSpec{
		Command:           []string{"go", "version"},
		ExpectedExitCode:  0,
		ExpectedStdoutSHA: "deadbeef00000000000000000000000000000000000000000000000000000000",
	}
	tamperedJSON, _ := json.Marshal(tamperedSpec)
	if got := r.Resolve(ctx, nil, "pipe:"+string(tamperedJSON)); got != abi.WitnessRefuted {
		t.Errorf("Resolve(tampered SHA) = %v, want Refuted", got)
	}

	// 4. Resolve claim with unexpected exit code
	mismatchedExitSpec := PipeClaimSpec{
		Command:          []string{"go", "version"},
		ExpectedExitCode: 1,
	}
	mismatchedExitJSON, _ := json.Marshal(mismatchedExitSpec)
	if got := r.Resolve(ctx, nil, "pipe:"+string(mismatchedExitJSON)); got != abi.WitnessRefuted {
		t.Errorf("Resolve(exit mismatch) = %v, want Refuted", got)
	}

	// 5. Resolve claim with unexecutable binary
	unexecutableSpec := PipeClaimSpec{
		Command:          []string{"__fak_unexecutable_bin_404__"},
		ExpectedExitCode: 0,
	}
	unexecutableJSON, _ := json.Marshal(unexecutableSpec)
	if got := r.Resolve(ctx, nil, "pipe:"+string(unexecutableJSON)); got != abi.WitnessAbstain {
		t.Errorf("Resolve(unexecutable) = %v, want Abstain", got)
	}
}

func TestResolverResolvePipeReceipt(t *testing.T) {
	ctx := context.Background()
	r := New()

	validReceipt := &PipeReceipt{
		Schema:       PipeReceiptSchema,
		Command:      []string{"echo", "hi"},
		ExitCode:     0,
		StdoutSHA256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
	}

	// 1. Pre-recorded receipt without command execution is abstained (fail-closed anti-reward-hack)
	spec := PipeClaimSpec{
		ExpectedExitCode:  0,
		ExpectedStdoutSHA: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
		Receipt:           validReceipt,
	}
	raw, _ := json.Marshal(spec)
	if got := r.Resolve(ctx, nil, "pipe:"+string(raw)); got != abi.WitnessAbstain {
		t.Errorf("Resolve(bare self-reported receipt) = %v, want Abstain", got)
	}

	// 2. Direct receipt validation via ValidatePipeReceipt passes for verified receipt
	validReceipt.Attestation = SignPipeReceipt(validReceipt, nil)
	if outcome, err := ValidatePipeReceipt(validReceipt, spec); outcome != abi.WitnessConfirmed || err != nil {
		t.Errorf("ValidatePipeReceipt(valid) = %v, %v, want Confirmed, nil", outcome, err)
	}

	// 3. Direct receipt validation rejects tampered receipt
	tamperedReceipt := &PipeReceipt{
		Schema:       PipeReceiptSchema,
		ExitCode:     0,
		StdoutSHA256: "0000000000000000000000000000000000000000000000000000000000000000",
	}
	tamperedReceipt.Attestation = SignPipeReceipt(tamperedReceipt, nil)
	specTampered := PipeClaimSpec{
		ExpectedExitCode:  0,
		ExpectedStdoutSHA: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
	}
	if outcome, err := ValidatePipeReceipt(tamperedReceipt, specTampered); outcome != abi.WitnessRefuted || err == nil {
		t.Errorf("ValidatePipeReceipt(tampered) = %v, %v, want Refuted, error", outcome, err)
	}

	// 4. Malformed JSON claim
	if got := r.Resolve(ctx, nil, "pipe:{invalid_json"); got != abi.WitnessAbstain {
		t.Errorf("Resolve(malformed JSON) = %v, want Abstain", got)
	}

	// 5. Empty spec (no command, no receipt)
	if got := r.Resolve(ctx, nil, "pipe:{}"); got != abi.WitnessAbstain {
		t.Errorf("Resolve(empty spec) = %v, want Abstain", got)
	}
}

const witLargeOutputEnv = "FAK_WITNESS_LARGE_OUTPUT"

// TestHelperWitnessLargeOutput emits ~5.12MB of output to stdout so RunPipeWitness
// stream buffer capping can be tested against a real OS process.
func TestHelperWitnessLargeOutput(t *testing.T) {
	if os.Getenv(witLargeOutputEnv) != "1" {
		return
	}
	chunk := bytes.Repeat([]byte("A"), 64*1024)
	for i := 0; i < 80; i++ { // 80 * 64KB = 5.12MB > MaxPipeBufferSize (4MB)
		_, _ = os.Stdout.Write(chunk)
	}
	os.Exit(0)
}

func TestRunPipeWitness_CapStreamBuffer(t *testing.T) {
	self, err := os.Executable()
	if err != nil {
		t.Fatalf("os.Executable: %v", err)
	}

	t.Setenv(witLargeOutputEnv, "1")
	ctx := context.Background()
	receipt, stdoutBytes, _, err := RunPipeWitness(ctx, "", self, "-test.run=^TestHelperWitnessLargeOutput$")
	if err != nil {
		t.Fatalf("RunPipeWitness failed: %v", err)
	}
	if receipt.ExitCode != 0 {
		t.Fatalf("ExitCode = %d, want 0", receipt.ExitCode)
	}
	// receipt.StdoutBytes records full bytes processed through the hash stream
	const wantMinBytes = 80 * 64 * 1024
	if receipt.StdoutBytes < wantMinBytes {
		t.Errorf("receipt.StdoutBytes = %d, want >= %d", receipt.StdoutBytes, wantMinBytes)
	}
	// Stream buffer retained in memory must be capped to MaxPipeBufferSize without crashing
	if len(stdoutBytes) != MaxPipeBufferSize {
		t.Errorf("len(stdoutBytes) = %d, want capped to MaxPipeBufferSize (%d)", len(stdoutBytes), MaxPipeBufferSize)
	}
	if receipt.StdoutSHA256 == "" {
		t.Error("expected non-empty StdoutSHA256 for streamed output")
	}
}

func TestPipeTamper_ForgedReceiptWithoutExecution(t *testing.T) {
	ctx := context.Background()
	r := New()

	// 1. Inlined forged receipt in claim string without execution returns WitnessAbstain
	forgedSpec := PipeClaimSpec{
		ExpectedExitCode:  0,
		ExpectedStdoutSHA: "deadbeef00000000000000000000000000000000000000000000000000000000",
		Receipt: &PipeReceipt{
			Schema:        PipeReceiptSchema,
			ExitCode:      0,
			StdoutSHA256:  "deadbeef00000000000000000000000000000000000000000000000000000000",
			StdoutBytes:   100,
			DurationNanos: 500,
		},
	}
	forgedJSON, err := json.Marshal(forgedSpec)
	if err != nil {
		t.Fatal(err)
	}
	if got := r.Resolve(ctx, nil, "pipe:"+string(forgedJSON)); got != abi.WitnessAbstain {
		t.Errorf("Resolve(forged receipt without execution) = %v, want Abstain", got)
	}

	// 2. Inlined forged receipt accompanying real command is refuted
	forgedWithCmd := PipeClaimSpec{
		Command:           []string{"go", "version"},
		ExpectedExitCode:  0,
		ExpectedStdoutSHA: "deadbeef00000000000000000000000000000000000000000000000000000000",
		Receipt: &PipeReceipt{
			Schema:       PipeReceiptSchema,
			Command:      []string{"go", "version"},
			ExitCode:     0,
			StdoutSHA256: "deadbeef00000000000000000000000000000000000000000000000000000000",
		},
	}
	forgedCmdJSON, _ := json.Marshal(forgedWithCmd)
	if got := r.Resolve(ctx, nil, "pipe:"+string(forgedCmdJSON)); got != abi.WitnessRefuted {
		t.Errorf("Resolve(forged inlined receipt with command) = %v, want Refuted", got)
	}

	// 3. Direct ValidatePipeReceipt on unexecuted, unattested receipt returns WitnessAbstain with error
	unverifiedReceipt := &PipeReceipt{
		Schema:       PipeReceiptSchema,
		Command:      []string{"echo", "hi"},
		ExitCode:     0,
		StdoutSHA256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
	}
	outcome, valErr := ValidatePipeReceipt(unverifiedReceipt, PipeClaimSpec{
		ExpectedExitCode:  0,
		ExpectedStdoutSHA: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
	})
	if outcome != abi.WitnessAbstain || valErr == nil {
		t.Errorf("ValidatePipeReceipt(unverified) = %v, %v, want Abstain, error", outcome, valErr)
	}

	// 4. PipeReceipt.WitnessOutcome() on unverified receipt returns WitnessAbstain
	if got := unverifiedReceipt.WitnessOutcome(); got != abi.WitnessAbstain {
		t.Errorf("unverifiedReceipt.WitnessOutcome() = %v, want Abstain", got)
	}

	// 5. Inlined receipt with invalid/forged signature is refuted
	forgedSigReceipt := &PipeReceipt{
		Schema:       PipeReceiptSchema,
		Command:      []string{"echo", "hi"},
		ExitCode:     0,
		StdoutSHA256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
		Attestation:  "forged-signature-invalid-hmac-123456",
	}
	outcome, valErr = ValidatePipeReceipt(forgedSigReceipt, PipeClaimSpec{ExpectedExitCode: 0})
	if outcome != abi.WitnessAbstain || valErr == nil {
		t.Errorf("ValidatePipeReceipt(forged attestation) = %v, %v, want Abstain, error", outcome, valErr)
	}
}

func TestExecutionEvidence_PopulatedFromPipe(t *testing.T) {
	ctx := context.Background()
	ev := NewExecutionVerifier("").runSelector(ctx, "", "HEAD", "test_kind", ExecutionSelector{
		ID:      "go-version-check",
		Command: []string{"go", "version"},
	})

	if ev.Outcome != "pass" {
		t.Fatalf("ev.Outcome = %q, want pass", ev.Outcome)
	}
	if ev.ExitCode != 0 {
		t.Errorf("ev.ExitCode = %d, want 0", ev.ExitCode)
	}
	if ev.StdoutSHA256 == "" {
		t.Error("expected non-empty ev.StdoutSHA256")
	}
	if ev.StdoutBytes <= 0 {
		t.Errorf("ev.StdoutBytes = %d, want > 0", ev.StdoutBytes)
	}
}
