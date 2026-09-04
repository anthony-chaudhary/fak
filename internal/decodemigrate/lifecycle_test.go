package decodemigrate

import (
	"errors"
	"fmt"
	"sync"
	"testing"
)

func TestStateLifecycle_CloneAndMutationIsolation(t *testing.T) {
	tokens := []int64{1, 2, 3, 4, 5}
	meta := KVBufferMetadata{
		NumLayers:   16,
		NumHeads:    4,
		HeadDim:     64,
		BlockSize:   8,
		TotalTokens: 5,
	}
	payload := []byte(tagV1 + "-test-bytes-isolated")
	orig, err := NewDecodeState(VersionV1Legacy, "test-model", "session-isolation", tokens, meta, payload)
	if err != nil {
		t.Fatalf("failed to create original state: %v", err)
	}

	clone := orig.Clone()
	if clone == nil {
		t.Fatal("expected non-nil clone")
	}

	// Mutate clone
	clone.Tokens[0] = 99999
	clone.Payload[len(clone.Payload)-1] = 0x00
	clone.KVMetadata.BlockSize = 32

	// Verify original is untouched
	if orig.Tokens[0] == 99999 {
		t.Error("mutating clone tokens affected original state")
	}
	if orig.Payload[len(orig.Payload)-1] == 0x00 {
		t.Error("mutating clone payload affected original state")
	}
	if orig.KVMetadata.BlockSize != 8 {
		t.Error("mutating clone metadata affected original state")
	}
	if err := orig.Validate(); err != nil {
		t.Errorf("original state failed validation after clone mutation: %v", err)
	}
}

func TestStateLifecycle_SequentialMultiHopLifecycle(t *testing.T) {
	engine := NewMigrationRunner()
	tokens := []int64{42, 1337, 2026, 9999}
	meta := KVBufferMetadata{
		NumLayers:   24,
		NumHeads:    8,
		HeadDim:     128,
		BlockSize:   0,
		TotalTokens: len(tokens),
	}
	state, err := NewDecodeState(VersionV1Legacy, "qwen-spine", "seq-lifecycle", tokens, meta, []byte(tagV1+"-initial-weights"))
	if err != nil {
		t.Fatalf("NewDecodeState failed: %v", err)
	}

	// Transition 1: V1 -> V2
	state, err = engine.Migrate(state, VersionV2Paged)
	if err != nil {
		t.Fatalf("V1 -> V2 failed: %v", err)
	}
	if state.Version != VersionV2Paged || state.KVMetadata.BlockSize != 16 {
		t.Errorf("unexpected V2 state: ver=%s, block=%d", state.Version, state.KVMetadata.BlockSize)
	}

	// Transition 2: V2 -> V3
	state, err = engine.Migrate(state, VersionV3Quantized)
	if err != nil {
		t.Fatalf("V2 -> V3 failed: %v", err)
	}
	if state.Version != VersionV3Quantized {
		t.Errorf("unexpected V3 state version: %s", state.Version)
	}

	// Transition 3: V3 -> V4
	state, err = engine.Migrate(state, VersionV4Compressed)
	if err != nil {
		t.Fatalf("V3 -> V4 failed: %v", err)
	}
	if state.Version != VersionV4Compressed {
		t.Errorf("unexpected V4 state version: %s", state.Version)
	}

	// Transition 4: Reverse V4 -> V3
	state, err = engine.Migrate(state, VersionV3Quantized)
	if err != nil {
		t.Fatalf("V4 -> V3 failed: %v", err)
	}
	if state.Version != VersionV3Quantized {
		t.Errorf("unexpected reverse V3 state version: %s", state.Version)
	}

	// Transition 5: Reverse V3 -> V2
	state, err = engine.Migrate(state, VersionV2Paged)
	if err != nil {
		t.Fatalf("V3 -> V2 failed: %v", err)
	}
	if state.Version != VersionV2Paged {
		t.Errorf("unexpected reverse V2 state version: %s", state.Version)
	}

	// Transition 6: Reverse V2 -> V1
	state, err = engine.Migrate(state, VersionV1Legacy)
	if err != nil {
		t.Fatalf("V2 -> V1 failed: %v", err)
	}
	if state.Version != VersionV1Legacy {
		t.Errorf("unexpected final V1 state version: %s", state.Version)
	}

	// Verify complete token and structural preservation
	if len(state.Tokens) != len(tokens) {
		t.Fatalf("token length altered: got %d, want %d", len(state.Tokens), len(tokens))
	}
	for i := range tokens {
		if state.Tokens[i] != tokens[i] {
			t.Errorf("token %d mismatch: got %d, want %d", i, state.Tokens[i], tokens[i])
		}
	}
	if err := state.Validate(); err != nil {
		t.Errorf("final state failed validation: %v", err)
	}
}

func TestRegisterStep_CustomStepHop(t *testing.T) {
	engine := NewMigrationRunner()

	// Invalid from/to unknown
	err := engine.RegisterStep(MigrationStep{
		From:        VersionUnknown,
		To:          VersionV1Legacy,
		Description: "invalid step",
		Apply:       func(s *DecodeState) (*DecodeState, error) { return s, nil },
	})
	if !errors.Is(err, ErrIncompatibleFormat) {
		t.Errorf("expected ErrIncompatibleFormat, got %v", err)
	}

	// Same from/to
	err = engine.RegisterStep(MigrationStep{
		From:        VersionV1Legacy,
		To:          VersionV1Legacy,
		Description: "noop step",
		Apply:       func(s *DecodeState) (*DecodeState, error) { return s, nil },
	})
	if !errors.Is(err, ErrIncompatibleFormat) {
		t.Errorf("expected ErrIncompatibleFormat, got %v", err)
	}

	// Nil apply func
	err = engine.RegisterStep(MigrationStep{
		From:        VersionV1Legacy,
		To:          VersionV2Paged,
		Description: "nil apply step",
		Apply:       nil,
	})
	if !errors.Is(err, ErrMigrationFailed) {
		t.Errorf("expected ErrMigrationFailed, got %v", err)
	}

	// Valid custom step
	const customVersion FormatVersion = 50
	err = engine.RegisterStep(MigrationStep{
		From:        VersionV4Compressed,
		To:          customVersion,
		Description: "custom v4 to v50 experimental hop",
		Apply: func(s *DecodeState) (*DecodeState, error) {
			cp := s.Clone()
			cp.Version = customVersion
			return cp, nil
		},
	})
	if err != nil {
		t.Fatalf("RegisterStep failed: %v", err)
	}

	route, err := engine.Route(VersionV1Legacy, customVersion)
	if err != nil {
		t.Fatalf("Route(v1, customVersion) failed: %v", err)
	}
	// V1 -> V2 -> V3 -> V4 -> Custom = 4 steps
	if route.TotalSteps() != 4 {
		t.Errorf("expected 4 steps, got %d", route.TotalSteps())
	}
}

func TestConcurrentStateMigrations(t *testing.T) {
	engine := NewMigrationRunner()
	const numGoroutines = 20
	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		workerID := i
		go func() {
			defer wg.Done()
			tokens := []int64{int64(workerID), int64(workerID * 10), 999}
			meta := KVBufferMetadata{NumLayers: 8, NumHeads: 4, HeadDim: 64, BlockSize: 0, TotalTokens: len(tokens)}
			state, err := NewDecodeState(VersionV1Legacy, "concurrent-model", fmt.Sprintf("seq-%d", workerID), tokens, meta, []byte(tagV1+"-worker-bytes"))
			if err != nil {
				t.Errorf("worker %d: NewDecodeState failed: %v", workerID, err)
				return
			}

			// Forward migration V1 -> V4
			v4, err := engine.Migrate(state, VersionV4Compressed)
			if err != nil {
				t.Errorf("worker %d: Migrate V1->V4 failed: %v", workerID, err)
				return
			}
			if v4.Version != VersionV4Compressed {
				t.Errorf("worker %d: wrong target version %s", workerID, v4.Version)
				return
			}

			// Reverse migration V4 -> V1
			v1, err := engine.Migrate(v4, VersionV1Legacy)
			if err != nil {
				t.Errorf("worker %d: Migrate V4->V1 failed: %v", workerID, err)
				return
			}
			if v1.Version != VersionV1Legacy {
				t.Errorf("worker %d: wrong reverse target version %s", workerID, v1.Version)
				return
			}
			if err := v1.Validate(); err != nil {
				t.Errorf("worker %d: v1 Validate() failed: %v", workerID, err)
			}
		}()
	}

	wg.Wait()
}

func TestExecuteRoute_SourceMismatch(t *testing.T) {
	engine := NewMigrationRunner()
	state := sampleDecodeState(t, VersionV1Legacy, tagV1)

	// Create route from V2 -> V4
	route, err := engine.Route(VersionV2Paged, VersionV4Compressed)
	if err != nil {
		t.Fatalf("Route failed: %v", err)
	}

	// Attempt executing route with V1 state
	_, err = engine.ExecuteRoute(state, route)
	if !errors.Is(err, ErrIncompatibleFormat) {
		t.Errorf("expected ErrIncompatibleFormat on route source mismatch, got %v", err)
	}

	// Malformed route: Source != Target but zero steps
	emptyRoute := &MigrationRoute{
		Source: VersionV1Legacy,
		Target: VersionV2Paged,
		Steps:  nil,
	}
	_, err = engine.ExecuteRoute(state, emptyRoute)
	if !errors.Is(err, ErrMigrationFailed) {
		t.Errorf("expected ErrMigrationFailed on empty route with Source != Target, got %v", err)
	}

	// Malformed route: steps end at V2 but Target claimed V4
	stepV1V2, err := engine.Route(VersionV1Legacy, VersionV2Paged)
	if err != nil {
		t.Fatalf("Route failed: %v", err)
	}
	truncatedRoute := &MigrationRoute{
		Source: VersionV1Legacy,
		Target: VersionV4Compressed,
		Steps:  stepV1V2.Steps, // Only goes to V2
	}
	_, err = engine.ExecuteRoute(state, truncatedRoute)
	if !errors.Is(err, ErrMigrationFailed) {
		t.Errorf("expected ErrMigrationFailed on route ending prematurely, got %v", err)
	}
}
