package engine

// continuous_batcher_wire_test.go — the independent test-author witnesses for the
// oss-port-p0-batching-wire ticket. These probe the FROZEN contract
// (scratch/DESIGN-batching-wire.md) and the abi.LifecycleEngine seam, NOT the
// implementation's internals: M1 budgeted prefill-vs-decode selection, M2
// decode-residency in stable submission order, and the production "batcher"
// engine registration + admit/complete lifecycle.
//
// Every test drives steps deterministically (no time.Sleep for synchronization)
// and is named TestContinuousBatcher_Wire* so `-run TestContinuousBatcher`
// selects them alongside the pre-existing batcher tests.

import (
	"context"
	"encoding/json"
	"strconv"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/refutil"
)

// wireBatcher builds a fresh internal batcher with an explicit slot count and
// prefill budget (0 preserves the legacy decode-only path).
func wireBatcher(t *testing.T, maxSlots, prefillBudget int) *ContinuousBatcher {
	t.Helper()
	cfg := DefaultContinuousBatcherConfig()
	cfg.MaxSlots = maxSlots
	cfg.PrefillBudget = prefillBudget
	cb, err := NewContinuousBatcher(cfg)
	if err != nil {
		t.Fatalf("NewContinuousBatcher failed: %v", err)
	}
	t.Cleanup(func() { _ = cb.Close() })
	return cb
}

// wireStrictlyAscending reports whether every element is greater than the prior.
func wireStrictlyAscending(uids []uint64) bool {
	for i := 1; i < len(uids); i++ {
		if uids[i] <= uids[i-1] {
			return false
		}
	}
	return true
}

func wireCapsHave(caps []abi.Capability, want abi.Capability) bool {
	for _, c := range caps {
		if c == want {
			return true
		}
	}
	return false
}

// TestContinuousBatcher_WirePrefillBeatsDecode pins M1: with a positive budget and
// prefilling work present, a step MUST be a PREFILL step (consuming at most the
// budget) and MUST NOT advance the resident decode slot; the chunked request then
// reaches SlotStateActiveDecode and completes.
func TestContinuousBatcher_WirePrefillBeatsDecode(t *testing.T) {
	t.Parallel()
	const budget = 3
	cb := wireBatcher(t, 4, budget)
	ctx := context.Background()

	// Chunked request whose prompt (7) is longer than the budget (3).
	if _, err := cb.Submit(&SubagentRequest{
		SessionID:      "chunked",
		PromptTokens:   []int{10, 11, 12, 13, 14, 15, 16},
		TargetTokens:   2,
		ChunkedPrefill: true,
	}); err != nil {
		t.Fatalf("Submit(chunked) failed: %v", err)
	}
	// An eager decoding request resident at the same time.
	if _, err := cb.Submit(&SubagentRequest{
		SessionID:    "eager",
		PromptTokens: []int{99},
		TargetTokens: 3,
	}); err != nil {
		t.Fatalf("Submit(eager) failed: %v", err)
	}

	// Admission: chunked is prefilling; eager is already active-decode.
	chunked, _ := cb.GetSlot("chunked")
	if chunked.State != SlotStatePrefilling {
		t.Fatalf("chunked state after Submit = %s, want %s", chunked.State, SlotStatePrefilling)
	}
	eager, _ := cb.GetSlot("eager")
	if eager.State != SlotStateActiveDecode {
		t.Fatalf("eager state after Submit = %s, want %s", eager.State, SlotStateActiveDecode)
	}

	// Step 1: prefill is attempted first.
	res, err := cb.StepPhase(ctx)
	if err != nil {
		t.Fatalf("StepPhase failed: %v", err)
	}
	if res.Phase != PhasePrefill {
		t.Fatalf("first step phase = %q, want %q", res.Phase, PhasePrefill)
	}
	if res.PrefillTokens <= 0 || res.PrefillTokens > budget {
		t.Fatalf("PrefillTokens = %d, want 0 < x <= %d", res.PrefillTokens, budget)
	}
	if res.DecodeTokens != 0 {
		t.Fatalf("DecodeTokens = %d during prefill step, want 0", res.DecodeTokens)
	}
	if got := eager.TokensGenerated(); got != 0 {
		t.Fatalf("eager resident generated %d tokens during a prefill step, want 0", got)
	}

	// Keep stepping; the chunked request must become active-decode and complete.
	observedActive := false
	var finalChunked *Slot
	for i := 0; i < 30; i++ {
		if s, ok := cb.GetSlot("chunked"); ok {
			if s.State == SlotStateActiveDecode {
				observedActive = true
			}
			if s.TokensGenerated() >= 2 {
				finalChunked = s
				break
			}
		}
		if _, err := cb.StepPhase(ctx); err != nil {
			t.Fatalf("StepPhase failed at iteration %d: %v", i, err)
		}
	}
	if !observedActive {
		t.Fatalf("chunked request never reached %s", SlotStateActiveDecode)
	}
	if finalChunked == nil || finalChunked.TokensGenerated() != 2 {
		got := -1
		if finalChunked != nil {
			got = finalChunked.TokensGenerated()
		}
		t.Fatalf("chunked generated tokens = %d, want 2", got)
	}
}

// TestContinuousBatcher_WireDecodeResidentBetweenSteps pins M2: ordinary requests
// stay resident across >=3 decode steps, the per-step roster is strictly ascending
// and deterministic across fresh batchers.
func TestContinuousBatcher_WireDecodeResidentBetweenSteps(t *testing.T) {
	t.Parallel()
	ids := []string{"sub-1", "sub-2", "sub-3"}

	scenario := func() [][]uint64 {
		cb := wireBatcher(t, 4, 4) // budget>0 but no prefilling slots => decode arm
		ctx := context.Background()
		for _, id := range ids {
			if _, err := cb.Submit(&SubagentRequest{SessionID: id, PromptTokens: []int{1}, TargetTokens: 5}); err != nil {
				t.Fatalf("Submit(%s) failed: %v", id, err)
			}
		}
		var perStep [][]uint64
		for i := 0; i < 3; i++ {
			res, err := cb.StepPhase(ctx)
			if err != nil {
				t.Fatalf("StepPhase %d failed: %v", i, err)
			}
			if res.Phase != PhaseDecode {
				t.Fatalf("step %d phase = %q, want %q", i, res.Phase, PhaseDecode)
			}
			if len(res.DecodeResidentUIDs) != len(ids) {
				t.Fatalf("step %d DecodeResidentUIDs = %v, want %d residents", i, res.DecodeResidentUIDs, len(ids))
			}
			if !wireStrictlyAscending(res.DecodeResidentUIDs) {
				t.Fatalf("step %d DecodeResidentUIDs = %v, not strictly ascending", i, res.DecodeResidentUIDs)
			}
			if got := cb.ActiveSlotCount(); got != len(ids) {
				t.Fatalf("step %d ActiveSlotCount = %d, want %d", i, got, len(ids))
			}
			if got := len(cb.DecodeResident()); got < len(ids) {
				t.Fatalf("step %d DecodeResident count = %d, want >= %d", i, got, len(ids))
			}
			perStep = append(perStep, append([]uint64(nil), res.DecodeResidentUIDs...))
		}
		return perStep
	}

	first := scenario()
	second := scenario()
	if len(first) != 3 || len(second) != 3 {
		t.Fatalf("expected 3 steps each, got %d and %d", len(first), len(second))
	}
	for i := range first {
		if len(first[i]) != len(second[i]) {
			t.Fatalf("step %d resident length differs across runs: %v vs %v", i, first[i], second[i])
		}
		for j := range first[i] {
			if first[i][j] != second[i][j] {
				t.Fatalf("step %d UID %d differs across runs: %d vs %d", i, j, first[i][j], second[i][j])
			}
		}
	}
}

// TestContinuousBatcher_WireStableUIDOrder pins M2 ordering: the decode roster is
// submission order, independent of session-id lexical order, and deterministic.
func TestContinuousBatcher_WireStableUIDOrder(t *testing.T) {
	t.Parallel()
	// Deliberately NOT lexically sorted, so submission order != sort order.
	submission := []string{"zulu", "alpha", "mike", "bravo"}

	run := func() []uint64 {
		cb := wireBatcher(t, 4, 0)
		for _, id := range submission {
			if _, err := cb.Submit(&SubagentRequest{SessionID: id, PromptTokens: []int{7}, TargetTokens: 3}); err != nil {
				t.Fatalf("Submit(%s) failed: %v", id, err)
			}
		}
		return cb.DecodeResidentUIDs()
	}

	got := run()
	want := []uint64{1, 2, 3, 4}
	if len(got) != len(want) {
		t.Fatalf("DecodeResidentUIDs = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("DecodeResidentUIDs = %v, want %v", got, want)
		}
	}

	// The roster's session order must equal submission order.
	cb := wireBatcher(t, 4, 0)
	for _, id := range submission {
		if _, err := cb.Submit(&SubagentRequest{SessionID: id, PromptTokens: []int{7}, TargetTokens: 3}); err != nil {
			t.Fatalf("Submit(%s) failed: %v", id, err)
		}
	}
	roster := cb.DecodeResident()
	if len(roster) != len(submission) {
		t.Fatalf("DecodeResident len = %d, want %d", len(roster), len(submission))
	}
	for i, s := range roster {
		if s.SessionID != submission[i] {
			t.Fatalf("roster[%d] = %s, want %s (submission order)", i, s.SessionID, submission[i])
		}
	}

	repeat := run()
	if len(repeat) != len(got) {
		t.Fatalf("repeat DecodeResidentUIDs = %v, want %v", repeat, got)
	}
	for i := range got {
		if repeat[i] != got[i] {
			t.Fatalf("repeat DecodeResidentUIDs = %v, want %v (non-deterministic)", repeat, got)
		}
	}
}

// TestContinuousBatcher_WireBudgetExhaustionChunking pins the resumable-chunk
// arithmetic: PrefillPos advances by min(B, remaining) and the request takes
// ceil(L/B) prefill steps; PrefillPos never exceeds len(PendingPrompt).
func TestContinuousBatcher_WireBudgetExhaustionChunking(t *testing.T) {
	t.Parallel()
	const (
		budget = 3
		L      = 10
	)
	cb := wireBatcher(t, 2, budget)
	ctx := context.Background()

	prompt := make([]int, L)
	for i := range prompt {
		prompt[i] = 100 + i
	}
	if _, err := cb.Submit(&SubagentRequest{
		SessionID:      "chunk",
		PromptTokens:   prompt,
		TargetTokens:   2,
		ChunkedPrefill: true,
	}); err != nil {
		t.Fatalf("Submit failed: %v", err)
	}

	slot, ok := cb.GetSlot("chunk")
	if !ok {
		t.Fatal("chunk slot not found")
	}
	if slot.State != SlotStatePrefilling {
		t.Fatalf("initial state = %s, want %s", slot.State, SlotStatePrefilling)
	}
	if slot.PrefillPos != 0 {
		t.Fatalf("initial PrefillPos = %d, want 0", slot.PrefillPos)
	}

	prefillSteps := 0
	for slot.State == SlotStatePrefilling {
		res, err := cb.StepPhase(ctx)
		if err != nil {
			t.Fatalf("StepPhase failed: %v", err)
		}
		if res.Phase != PhasePrefill {
			t.Fatalf("state was still %s but phase = %q, want %q", slot.State, res.Phase, PhasePrefill)
		}
		prefillSteps++
		if slot.PrefillPos > len(slot.PendingPrompt) {
			t.Fatalf("PrefillPos = %d exceeds len(PendingPrompt) = %d", slot.PrefillPos, len(slot.PendingPrompt))
		}
		remaining := len(slot.PendingPrompt) - (slot.PrefillPos - res.PrefillTokens)
		wantChunk := budget
		if remaining < budget {
			wantChunk = remaining
		}
		if res.PrefillTokens != wantChunk {
			t.Fatalf("prefill step %d consumed %d tokens, want min(%d, remaining=%d)=%d",
				prefillSteps, res.PrefillTokens, budget, remaining, wantChunk)
		}
		if prefillSteps > 100 {
			t.Fatal("prefill did not terminate")
		}
	}

	wantSteps := (L + budget - 1) / budget // ceil(L/B)
	if prefillSteps != wantSteps {
		t.Fatalf("prefill steps = %d, want ceil(%d/%d) = %d", prefillSteps, L, budget, wantSteps)
	}
	if slot.PrefillPos != L {
		t.Fatalf("final PrefillPos = %d, want %d", slot.PrefillPos, L)
	}
	if slot.State != SlotStateActiveDecode {
		t.Fatalf("final state = %s, want %s", slot.State, SlotStateActiveDecode)
	}
}

// TestContinuousBatcher_WireDecodeResidentNotAdvancedDuringPrefill pins M2's
// "a resident request is NOT advanced during a prefill step".
func TestContinuousBatcher_WireDecodeResidentNotAdvancedDuringPrefill(t *testing.T) {
	t.Parallel()
	cb := wireBatcher(t, 4, 2) // budget 2
	ctx := context.Background()

	if _, err := cb.Submit(&SubagentRequest{
		SessionID:    "resident",
		PromptTokens: []int{5},
		TargetTokens: 5,
	}); err != nil {
		t.Fatalf("Submit(resident) failed: %v", err)
	}
	if _, err := cb.Submit(&SubagentRequest{
		SessionID:      "chunker",
		PromptTokens:   []int{1, 2, 3, 4, 5, 6},
		TargetTokens:   1,
		ChunkedPrefill: true,
	}); err != nil {
		t.Fatalf("Submit(chunker) failed: %v", err)
	}

	resident, _ := cb.GetSlot("resident")
	chunker, _ := cb.GetSlot("chunker")

	prefillSteps := 0
	for chunker.State == SlotStatePrefilling {
		before := resident.TokensGenerated()
		res, err := cb.StepPhase(ctx)
		if err != nil {
			t.Fatalf("StepPhase failed: %v", err)
		}
		if res.Phase != PhasePrefill {
			t.Fatalf("prefill step phase = %q, want %q", res.Phase, PhasePrefill)
		}
		if res.DecodeTokens != 0 {
			t.Fatalf("DecodeTokens = %d during prefill, want 0", res.DecodeTokens)
		}
		if after := resident.TokensGenerated(); after != before {
			t.Fatalf("resident advanced %d -> %d during a prefill step", before, after)
		}
		prefillSteps++
		if prefillSteps > 100 {
			t.Fatal("prefill did not terminate")
		}
	}
	if prefillSteps != 3 {
		t.Fatalf("prefill steps = %d, want 3 (ceil(6/2))", prefillSteps)
	}
	if resident.TokensGenerated() != 0 {
		t.Fatalf("resident generated %d tokens across prefill steps, want 0", resident.TokensGenerated())
	}
	if chunker.State != SlotStateActiveDecode {
		t.Fatalf("chunker state = %s, want %s", chunker.State, SlotStateActiveDecode)
	}
}

// TestContinuousBatcher_WireZeroBudgetEagerFallback pins the PrefillBudget==0
// edge: a ChunkedPrefill request must NOT get stuck prefilling; it is admitted
// eager/active per the legacy path and can decode to completion.
func TestContinuousBatcher_WireZeroBudgetEagerFallback(t *testing.T) {
	t.Parallel()
	cb := wireBatcher(t, 2, 0) // budget disabled
	ctx := context.Background()

	prompt := []int{4, 5, 6, 7, 8, 9, 10, 11, 12, 13}
	if _, err := cb.Submit(&SubagentRequest{
		SessionID:      "legacy-chunked",
		PromptTokens:   prompt,
		TargetTokens:   2,
		ChunkedPrefill: true,
	}); err != nil {
		t.Fatalf("Submit failed: %v", err)
	}

	slot, ok := cb.GetSlot("legacy-chunked")
	if !ok {
		t.Fatal("legacy-chunked slot not found")
	}
	if slot.State != SlotStateActiveDecode {
		t.Fatalf("state with zero budget = %s, want %s (eager legacy path)", slot.State, SlotStateActiveDecode)
	}
	if slot.LastToken != prompt[len(prompt)-1] {
		t.Fatalf("LastToken = %d, want last prompt token %d", slot.LastToken, prompt[len(prompt)-1])
	}

	for i := 0; i < 20 && slot.TokensGenerated() < 2; i++ {
		res, err := cb.StepPhase(ctx)
		if err != nil {
			t.Fatalf("StepPhase failed: %v", err)
		}
		if res.Phase == PhasePrefill {
			t.Fatalf("phase = %q with zero budget, want decode-only", res.Phase)
		}
	}
	if slot.TokensGenerated() != 2 {
		t.Fatalf("generated %d tokens, want 2", slot.TokensGenerated())
	}
}

// TestContinuousBatcher_WireChunkedFIFO pins FIFO service of oversized chunked
// requests by submission sequence.
func TestContinuousBatcher_WireChunkedFIFO(t *testing.T) {
	t.Parallel()
	cb := wireBatcher(t, 4, 2) // budget 2
	ctx := context.Background()

	if _, err := cb.Submit(&SubagentRequest{
		SessionID: "first", PromptTokens: []int{1, 2, 3, 4}, TargetTokens: 1, ChunkedPrefill: true,
	}); err != nil {
		t.Fatalf("Submit(first) failed: %v", err)
	}
	if _, err := cb.Submit(&SubagentRequest{
		SessionID: "second", PromptTokens: []int{5, 6, 7, 8}, TargetTokens: 1, ChunkedPrefill: true,
	}); err != nil {
		t.Fatalf("Submit(second) failed: %v", err)
	}

	first, _ := cb.GetSlot("first")
	second, _ := cb.GetSlot("second")

	// Step 1 consumes the whole budget from the FIRST (lowest uid) only.
	res, err := cb.StepPhase(ctx)
	if err != nil {
		t.Fatalf("StepPhase 1 failed: %v", err)
	}
	if res.PrefillTokens != 2 {
		t.Fatalf("step 1 PrefillTokens = %d, want 2", res.PrefillTokens)
	}
	if first.PrefillPos != 2 {
		t.Fatalf("first PrefillPos = %d, want 2", first.PrefillPos)
	}
	if second.PrefillPos != 0 {
		t.Fatalf("second PrefillPos = %d, want 0 (FIFO: first served first)", second.PrefillPos)
	}

	// Step 2 finishes the first's prefill; the second must still be untouched.
	if _, err := cb.StepPhase(ctx); err != nil {
		t.Fatalf("StepPhase 2 failed: %v", err)
	}
	if first.State != SlotStateActiveDecode {
		t.Fatalf("first state = %s, want %s after its 2 chunks", first.State, SlotStateActiveDecode)
	}
	if second.PrefillPos != 0 {
		t.Fatalf("second PrefillPos = %d, want 0 before first finishes", second.PrefillPos)
	}

	// Step 3 now serves the second.
	if _, err := cb.StepPhase(ctx); err != nil {
		t.Fatalf("StepPhase 3 failed: %v", err)
	}
	if second.PrefillPos != 2 {
		t.Fatalf("second PrefillPos = %d, want 2 after first completed", second.PrefillPos)
	}
}

// TestContinuousBatcher_WireClosedErrors pins the closed-batcher error surface.
func TestContinuousBatcher_WireClosedErrors(t *testing.T) {
	t.Parallel()
	cb := wireBatcher(t, 2, 4)
	if err := cb.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	if _, err := cb.Submit(&SubagentRequest{SessionID: "x", PromptTokens: []int{1}, TargetTokens: 1}); err != ErrBatcherClosed {
		t.Fatalf("Submit on closed batcher = %v, want ErrBatcherClosed", err)
	}
	if _, err := cb.StepPhase(context.Background()); err != ErrBatcherClosed {
		t.Fatalf("StepPhase on closed batcher = %v, want ErrBatcherClosed", err)
	}
	if _, err := cb.Step(context.Background()); err != ErrBatcherClosed {
		t.Fatalf("Step on closed batcher = %v, want ErrBatcherClosed", err)
	}
}

// TestContinuousBatcher_WireEngineRegistered pins the production registration:
// the "batcher" engine exists, advertises the lifecycle cap, and is a
// LifecycleEngine.
func TestContinuousBatcher_WireEngineRegistered(t *testing.T) {
	eng := abi.Engine(EngineIDBatcher)
	if eng == nil {
		t.Fatalf("abi.Engine(%q) is nil; production registration missing", EngineIDBatcher)
	}
	if !wireCapsHave(eng.Caps(), abi.EngineLifecycleCap) {
		t.Fatalf("batcher Caps() = %v, want %q", eng.Caps(), abi.EngineLifecycleCap)
	}
	if !abi.EngineSupportsLifecycle(eng) {
		t.Fatalf("abi.EngineSupportsLifecycle(batcher) = false, want true")
	}
}

// TestContinuousBatcher_WireEngineAdmitComplete pins the admit -> step -> stream
// -> result lifecycle end to end via the public constructor and Complete.
func TestContinuousBatcher_WireEngineAdmitComplete(t *testing.T) {
	const target = 4
	eng, err := NewBatchingEngine()
	if err != nil {
		t.Fatalf("NewBatchingEngine failed: %v", err)
	}
	defer func() { _ = eng.Close() }()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	args, err := json.Marshal(map[string]any{
		"session_id":    "s",
		"prompt_tokens": []int{1, 2, 3},
		"target_tokens": target,
	})
	if err != nil {
		t.Fatalf("marshal args failed: %v", err)
	}
	call := &abi.ToolCall{
		Tool: "t",
		Args: abi.Ref{Kind: abi.RefInline, Inline: args},
	}

	res, err := eng.Complete(ctx, call)
	if err != nil {
		t.Fatalf("Complete failed: %v", err)
	}
	if res == nil {
		t.Fatal("Complete returned a nil result")
	}
	if got := res.Meta["output_tokens"]; got != strconv.Itoa(target) {
		t.Fatalf("output_tokens = %q, want %d", got, target)
	}

	var body struct {
		GeneratedTokens []int `json:"generated_tokens"`
	}
	if raw := refutil.Bytes(ctx, res.Payload); len(raw) > 0 {
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Fatalf("unmarshal result payload failed: %v", err)
		}
		if len(body.GeneratedTokens) != target {
			t.Fatalf("generated_tokens len = %d, want %d", len(body.GeneratedTokens), target)
		}
	} else {
		t.Fatalf("result payload is empty; cannot witness generated_tokens")
	}
}

// wireModelConfig is the smallest valid Config that still exercises the real
// KV-cache forward path (Session.Prefill / PrefillNoLogits / Step). It mirrors the
// shape existing internal/model tests use with NewSynthetic.
func wireModelConfig() model.Config {
	return model.Config{
		Name:             "wire-synthetic",
		HiddenSize:       32,
		NumLayers:        2,
		NumHeads:         4,
		NumKVHeads:       2,
		HeadDim:          8,
		IntermediateSize: 64,
		VocabSize:        97,
		RMSNormEps:       1e-5,
		RopeTheta:        10000,
		EOSTokenID:       -1,
	}
}

// TestContinuousBatcher_WireModelChunkedPrefill is the BLOCKER regression for the
// real-model budgeted-prefill path. It configures an actual in-memory model and a
// chunked request whose prompt is longer than the budget, then drives StepPhase to
// completion. Against the pre-fix implementation (which did not build the session
// for chunked slots, or re-prefilled already-consumed chunks) this panics on a nil
// Session or corrupts PrefillPos; here it must run cleanly to SlotStateActiveDecode
// and emit exactly TargetTokens tokens.
//
// Since the spec never promised to expose the Session (the fields are unexported),
// the test proves the model path was genuinely exercised WITHOUT depending on
// private internals: the per-chunk prompt sliced from PendingPrompt is handed to
// Session.Prefill/PrefillNoLogits, so a zero-token step would silently skip the
// model and a nil session would panic. The test drives chunk-by-chunk with
// known-sized chunks, then further proves the real Step path ran to target with no
// panic.
func TestContinuousBatcher_WireModelChunkedPrefill(t *testing.T) {
	// Not t.Parallel(): this test holds no globals, but it runs the model forward
	// path and we keep it serial to stay independent of the process-wide engine.
	const (
		budget = 3
		target = 2
	)
	cfg := DefaultContinuousBatcherConfig()
	cfg.MaxSlots = 2
	cfg.PrefillBudget = budget
	cfg.Model = model.NewSynthetic(wireModelConfig())

	cb, err := NewContinuousBatcher(cfg)
	if err != nil {
		t.Fatalf("NewContinuousBatcher failed: %v", err)
	}
	defer func() { _ = cb.Close() }()
	ctx := context.Background()

	// Prompt (10) is longer than the budget (3) => resumable chunks.
	prompt := make([]int, 0, 10)
	for i := 0; i < 10; i++ {
		prompt = append(prompt, 1+i) // valid ids within VocabSize
	}
	if _, err := cb.Submit(&SubagentRequest{
		SessionID:      "model-chunk",
		PromptTokens:   prompt,
		TargetTokens:   target,
		ChunkedPrefill: true,
	}); err != nil {
		t.Fatalf("Submit failed: %v", err)
	}

	slot, ok := cb.GetSlot("model-chunk")
	if !ok {
		t.Fatal("model-chunk slot not found")
	}
	if slot.State != SlotStatePrefilling {
		t.Fatalf("initial state = %s, want %s", slot.State, SlotStatePrefilling)
	}

	// Drive prefill chunk-by-chunk. Each step must consume min(budget, remaining);
	// a zero-token prefill step would mean the model session never advanced.
	prefillSteps := 0
	for slot.State == SlotStatePrefilling {
		res, err := cb.StepPhase(ctx)
		if err != nil {
			t.Fatalf("StepPhase (model prefill) failed: %v", err)
		}
		if res.Phase != PhasePrefill {
			t.Fatalf("phase = %q while prefilling, want %q", res.Phase, PhasePrefill)
		}
		if res.PrefillTokens <= 0 || res.PrefillTokens > budget {
			t.Fatalf("PrefillTokens = %d, want 0 < x <= %d (real model forward must advance)", res.PrefillTokens, budget)
		}
		prefillSteps++
		if prefillSteps > 100 {
			t.Fatal("prefill did not terminate")
		}
	}
	if slot.State != SlotStateActiveDecode {
		t.Fatalf("state after prefill = %s, want %s", slot.State, SlotStateActiveDecode)
	}
	wantPrefillSteps := (len(prompt) + budget - 1) / budget
	if prefillSteps != wantPrefillSteps {
		t.Fatalf("prefill steps = %d, want ceil(%d/%d) = %d", prefillSteps, len(prompt), budget, wantPrefillSteps)
	}

	// In-package witness that the REAL model forward ran (not a silent no-op):
	// the slot owns a live Session and its KV cache grew in lockstep with every
	// prompt chunk consumed, i.e. Session.Prefill/PrefillNoLogits were called for
	// exactly the prompt. A missing session would have panicked in the step above;
	// an unadvanced/over-advanced cache would fail here.
	if slot.sess == nil {
		t.Fatal("chunked slot has no model Session; the real prefill path was not wired")
	}
	if slot.PrefillPos != len(prompt) {
		t.Fatalf("PrefillPos = %d, want %d", slot.PrefillPos, len(prompt))
	}
	if got := slot.sess.Cache.Len(); got != len(prompt) {
		t.Fatalf("session KV cache Len = %d, want %d (prefill must advance KV exactly once over the prompt)", got, len(prompt))
	}

	// Continue driving decode until the request completes; a nil session on the
	// decode path panics here, so reaching exactly TargetTokens proves the model
	// Session.Step path ran.
	for i := 0; i < 50 && slot.TokensGenerated() < target; i++ {
		if _, err := cb.StepPhase(ctx); err != nil {
			t.Fatalf("StepPhase (model decode) failed: %v", err)
		}
	}
	if got := slot.TokensGenerated(); got != target {
		t.Fatalf("generated tokens = %d, want %d", got, target)
	}
}

// TestContinuousBatcher_WireEngineAdmitAfterClose is the BLOCKER regression for the
// closed-engine admit path. Admit on a closed BatchingEngine must return a non-nil
// error promptly rather than register a handle whose producer (the stopped step
// loop) will never run, leaving Result() to block forever. A pre-close admitted
// request must also still terminate.
func TestContinuousBatcher_WireEngineAdmitAfterClose(t *testing.T) {
	// Uses its own engine instance (not the process-wide DefaultBatchingEngine).
	eng, err := NewBatchingEngine()
	if err != nil {
		t.Fatalf("NewBatchingEngine failed: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Admit one request before close, then close. It must still terminate.
	preArgs, err := json.Marshal(map[string]any{
		"session_id":    "before-close",
		"prompt_tokens": []int{1, 2, 3},
		"target_tokens": 3,
	})
	if err != nil {
		t.Fatalf("marshal pre-close args failed: %v", err)
	}
	pre, err := eng.Admit(ctx, &abi.ToolCall{Tool: "t", Args: abi.Ref{Kind: abi.RefInline, Inline: preArgs}})
	if err != nil {
		t.Fatalf("Admit(before close) failed: %v", err)
	}

	if err := eng.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	// Admit after close must fail fast with a non-nil error and must not hang.
	done := make(chan error, 1)
	go func() {
		postArgs, _ := json.Marshal(map[string]any{
			"session_id":    "after-close",
			"prompt_tokens": []int{1, 2, 3},
			"target_tokens": 3,
		})
		_, aerr := eng.Admit(ctx, &abi.ToolCall{Tool: "t", Args: abi.Ref{Kind: abi.RefInline, Inline: postArgs}})
		done <- aerr
	}()

	select {
	case aerr := <-done:
		if aerr == nil {
			t.Fatal("Admit after Close = nil error, want non-nil (engine is closed)")
		}
	case <-ctx.Done():
		t.Fatal("Admit after Close hung past the 5s deadline")
	}

	// The pre-close request must have terminated (result or error, but not hang).
	resCh := make(chan struct{})
	go func() {
		_, _ = pre.Result()
		close(resCh)
	}()
	select {
	case <-resCh:
	case <-ctx.Done():
		t.Fatal("pre-close request Result() hung after Close")
	}
}

// TestContinuousBatcher_WireEngineAdmitCompleteChunked drives chunked_prefill:true
// through the wire engine's Admit seam (short prompt, small budget), drains the
// token stream to closure, and asserts the assembled result: non-nil, output_tokens
// equal to target, and a generated-token payload of exactly the target length.
func TestContinuousBatcher_WireEngineAdmitCompleteChunked(t *testing.T) {
	const target = 4
	cfg := DefaultBatchingEngineConfig()
	cfg.Batcher.MaxSlots = 2
	cfg.Batcher.PrefillBudget = 2 // prompt (5) > budget => real chunked prefill
	cfg.StepInterval = 0
	eng, err := NewBatchingEngine(cfg)
	if err != nil {
		t.Fatalf("NewBatchingEngine failed: %v", err)
	}
	defer func() { _ = eng.Close() }()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	args, err := json.Marshal(map[string]any{
		"session_id":      "chunked-wire",
		"prompt_tokens":   []int{1, 2, 3, 4, 5},
		"target_tokens":   target,
		"chunked_prefill": true,
	})
	if err != nil {
		t.Fatalf("marshal args failed: %v", err)
	}
	req, err := eng.Admit(ctx, &abi.ToolCall{Tool: "t", Args: abi.Ref{Kind: abi.RefInline, Inline: args}})
	if err != nil {
		t.Fatalf("Admit(chunked) failed: %v", err)
	}

	// Drain the stream to closure before reading Result.
	drained := 0
	for range req.Tokens() {
		drained++
	}

	res, err := req.Result()
	if err != nil {
		t.Fatalf("Result failed: %v", err)
	}
	if res == nil {
		t.Fatal("Result returned nil")
	}
	if got := res.Meta["output_tokens"]; got != strconv.Itoa(target) {
		t.Fatalf("output_tokens = %q, want %d", got, target)
	}

	raw := refutil.Bytes(ctx, res.Payload)
	if len(raw) == 0 {
		t.Fatal("result payload is empty; cannot witness generated_tokens")
	}
	var body struct {
		Tool            string `json:"tool"`
		Engine          string `json:"engine"`
		GeneratedTokens []int  `json:"generated_tokens"`
	}
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatalf("unmarshal result payload failed: %v", err)
	}
	if len(body.GeneratedTokens) != target {
		t.Fatalf("generated_tokens len = %d, want %d (streamed %d)", len(body.GeneratedTokens), target, drained)
	}
	if body.Engine != EngineIDBatcher {
		t.Fatalf("payload engine = %q, want %q", body.Engine, EngineIDBatcher)
	}
}
