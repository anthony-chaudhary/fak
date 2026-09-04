package decodemigrate

import (
	"errors"
	"strings"
	"testing"
)

func sampleDecodeState(t *testing.T, ver FormatVersion, tag string) *DecodeState {
	t.Helper()
	tokens := []int64{101, 2054, 2003, 1037, 3231, 102}
	meta := KVBufferMetadata{
		NumLayers:   32,
		NumHeads:    8,
		HeadDim:     128,
		BlockSize:   16,
		TotalTokens: len(tokens),
	}
	payload := []byte(tag + "-sample-kv-tensor-payload-block-bytes")
	state, err := NewDecodeState(ver, "fak-model-qwen38", "seq-test-001", tokens, meta, payload)
	if err != nil {
		t.Fatalf("failed to create sample decode state: %v", err)
	}
	return state
}

func TestFormatVersionString(t *testing.T) {
	tests := []struct {
		ver  FormatVersion
		want string
	}{
		{VersionV1Legacy, "v1_legacy"},
		{VersionV2Paged, "v2_paged"},
		{VersionV3Quantized, "v3_quantized"},
		{VersionV4Compressed, "v4_compressed"},
		{VersionUnknown, "version_unknown(0)"},
		{FormatVersion(999), "version_unknown(999)"},
	}
	for _, tc := range tests {
		if got := tc.ver.String(); got != tc.want {
			t.Errorf("FormatVersion(%d).String() = %q, want %q", tc.ver, got, tc.want)
		}
	}
}

func TestDecodeStateValidation(t *testing.T) {
	var nilState *DecodeState
	if err := nilState.Validate(); !errors.Is(err, ErrCorruptedState) {
		t.Errorf("nil state Validate() expected ErrCorruptedState, got %v", err)
	}

	state := sampleDecodeState(t, VersionV1Legacy, tagV1)
	if err := state.Validate(); err != nil {
		t.Fatalf("valid state Validate() unexpected error: %v", err)
	}

	// Unknown version
	badVer := state.Clone()
	badVer.Version = VersionUnknown
	if err := badVer.Validate(); !errors.Is(err, ErrIncompatibleFormat) {
		t.Errorf("bad version Validate() expected ErrIncompatibleFormat, got %v", err)
	}

	// Empty model ID
	badModel := state.Clone()
	badModel.ModelID = ""
	if err := badModel.Validate(); !errors.Is(err, ErrCorruptedState) {
		t.Errorf("empty model ID Validate() expected ErrCorruptedState, got %v", err)
	}

	// Empty sequence ID
	badSeq := state.Clone()
	badSeq.SequenceID = ""
	if err := badSeq.Validate(); !errors.Is(err, ErrCorruptedState) {
		t.Errorf("empty sequence ID Validate() expected ErrCorruptedState, got %v", err)
	}

	// Negative metadata
	badMeta := state.Clone()
	badMeta.KVMetadata.NumLayers = -1
	if err := badMeta.Validate(); !errors.Is(err, ErrCorruptedState) {
		t.Errorf("negative metadata Validate() expected ErrCorruptedState, got %v", err)
	}

	// Corrupted payload without checksum recalculation
	corrupted := state.Clone()
	corrupted.Payload[len(corrupted.Payload)-1] ^= 0xFF
	if err := corrupted.Validate(); !errors.Is(err, ErrCorruptedState) {
		t.Errorf("tampered payload Validate() expected ErrCorruptedState, got %v", err)
	}
}

func TestNewDecodeState_Errors(t *testing.T) {
	meta := KVBufferMetadata{NumLayers: 4, NumHeads: 2, HeadDim: 64}
	if _, err := NewDecodeState(VersionUnknown, "m", "s", nil, meta, nil); !errors.Is(err, ErrIncompatibleFormat) {
		t.Errorf("NewDecodeState with VersionUnknown expected ErrIncompatibleFormat, got %v", err)
	}
	if _, err := NewDecodeState(VersionV1Legacy, "", "s", nil, meta, nil); !errors.Is(err, ErrCorruptedState) {
		t.Errorf("NewDecodeState with empty modelID expected ErrCorruptedState, got %v", err)
	}
	if _, err := NewDecodeState(VersionV1Legacy, "m", "", nil, meta, nil); !errors.Is(err, ErrCorruptedState) {
		t.Errorf("NewDecodeState with empty sequenceID expected ErrCorruptedState, got %v", err)
	}
}

func TestMigrationPlan_DirectAndMultiHop(t *testing.T) {
	engine := NewMigrationRunner()

	// No-op plan
	noop, err := engine.Route(VersionV2Paged, VersionV2Paged)
	if err != nil {
		t.Fatalf("Plan(v2, v2) failed: %v", err)
	}
	if !noop.IsNoOp() || noop.TotalSteps() != 0 {
		t.Errorf("expected no-op plan, got total steps %d", noop.TotalSteps())
	}

	// Single-hop forward
	p1, err := engine.Route(VersionV1Legacy, VersionV2Paged)
	if err != nil {
		t.Fatalf("Plan(v1, v2) failed: %v", err)
	}
	if p1.TotalSteps() != 1 {
		t.Errorf("expected 1 step for v1->v2, got %d", p1.TotalSteps())
	}

	// Multi-hop forward: V1 -> V4
	p3, err := engine.Route(VersionV1Legacy, VersionV4Compressed)
	if err != nil {
		t.Fatalf("Plan(v1, v4) failed: %v", err)
	}
	if p3.TotalSteps() != 3 {
		t.Errorf("expected 3 steps for v1->v4, got %d", p3.TotalSteps())
	}
	if p3.Steps[0].From != VersionV1Legacy || p3.Steps[0].To != VersionV2Paged {
		t.Errorf("unexpected step 0: %s -> %s", p3.Steps[0].From, p3.Steps[0].To)
	}
	if p3.Steps[1].From != VersionV2Paged || p3.Steps[1].To != VersionV3Quantized {
		t.Errorf("unexpected step 1: %s -> %s", p3.Steps[1].From, p3.Steps[1].To)
	}
	if p3.Steps[2].From != VersionV3Quantized || p3.Steps[2].To != VersionV4Compressed {
		t.Errorf("unexpected step 2: %s -> %s", p3.Steps[2].From, p3.Steps[2].To)
	}

	// Multi-hop backward: V4 -> V1
	pBack, err := engine.Route(VersionV4Compressed, VersionV1Legacy)
	if err != nil {
		t.Fatalf("Plan(v4, v1) failed: %v", err)
	}
	if pBack.TotalSteps() != 3 {
		t.Errorf("expected 3 steps for v4->v1, got %d", pBack.TotalSteps())
	}

	// Incompatible / unknown version
	if _, err := engine.Route(VersionUnknown, VersionV1Legacy); !errors.Is(err, ErrIncompatibleFormat) {
		t.Errorf("Plan(unknown, v1) expected ErrIncompatibleFormat, got %v", err)
	}
	if _, err := engine.Route(FormatVersion(888), VersionV1Legacy); !errors.Is(err, ErrIncompatibleFormat) {
		t.Errorf("Plan(unregistered, v1) expected ErrIncompatibleFormat, got %v", err)
	}
}

func TestMigrationExecution_Success(t *testing.T) {
	engine := NewMigrationRunner()
	initial := sampleDecodeState(t, VersionV1Legacy, tagV1)

	// Step 1: V1 -> V2
	v2State, err := engine.Migrate(initial, VersionV2Paged)
	if err != nil {
		t.Fatalf("Migrate(v1, v2) failed: %v", err)
	}
	if v2State.Version != VersionV2Paged {
		t.Errorf("expected VersionV2Paged, got %s", v2State.Version)
	}
	if !strings.HasPrefix(string(v2State.Payload), tagV2) {
		t.Errorf("expected payload prefix %s, got %s", tagV2, string(v2State.Payload[:4]))
	}
	if err := v2State.Validate(); err != nil {
		t.Errorf("v2State Validate() failed: %v", err)
	}

	// Multi-hop: V2 -> V4
	v4State, err := engine.Migrate(v2State, VersionV4Compressed)
	if err != nil {
		t.Fatalf("Migrate(v2, v4) failed: %v", err)
	}
	if v4State.Version != VersionV4Compressed {
		t.Errorf("expected VersionV4Compressed, got %s", v4State.Version)
	}
	if !strings.HasPrefix(string(v4State.Payload), tagV4) {
		t.Errorf("expected payload prefix %s, got %s", tagV4, string(v4State.Payload[:4]))
	}
	if err := v4State.Validate(); err != nil {
		t.Errorf("v4State Validate() failed: %v", err)
	}

	// Multi-hop reverse: V4 -> V1
	v1Restored, err := engine.Migrate(v4State, VersionV1Legacy)
	if err != nil {
		t.Fatalf("Migrate(v4, v1) failed: %v", err)
	}
	if v1Restored.Version != VersionV1Legacy {
		t.Errorf("expected VersionV1Legacy, got %s", v1Restored.Version)
	}
	if !strings.HasPrefix(string(v1Restored.Payload), tagV1) {
		t.Errorf("expected payload prefix %s, got %s", tagV1, string(v1Restored.Payload[:4]))
	}
	if len(v1Restored.Tokens) != len(initial.Tokens) {
		t.Fatalf("token count mismatch: %d != %d", len(v1Restored.Tokens), len(initial.Tokens))
	}
	for i := range initial.Tokens {
		if v1Restored.Tokens[i] != initial.Tokens[i] {
			t.Errorf("token %d mismatch: got %d want %d", i, v1Restored.Tokens[i], initial.Tokens[i])
		}
	}
}

func TestMigrationFailClosed_OnCorruptedInput(t *testing.T) {
	engine := NewMigrationRunner()
	corrupted := sampleDecodeState(t, VersionV1Legacy, tagV1)
	// Invalidate checksum
	corrupted.Checksum[0] ^= 0xAA

	_, err := engine.Migrate(corrupted, VersionV2Paged)
	if !errors.Is(err, ErrCorruptedState) {
		t.Errorf("expected ErrCorruptedState on corrupted input, got %v", err)
	}
}

func TestMigrationFailClosed_StepFailureAndRollback(t *testing.T) {
	engine := NewMigrationRunner()

	// Register a step that returns an error
	errSimulated := errors.New("underlying kernel failure")
	_ = engine.RegisterStep(MigrationStep{
		From:        VersionV1Legacy,
		To:          FormatVersion(10),
		Description: "Failing step",
		Apply: func(s *DecodeState) (*DecodeState, error) {
			return nil, errSimulated
		},
	})

	state := sampleDecodeState(t, VersionV1Legacy, tagV1)
	_, err := engine.Migrate(state, FormatVersion(10))
	if !errors.Is(err, ErrMigrationFailed) {
		t.Errorf("expected ErrMigrationFailed, got %v", err)
	}

	// Register a step that violates token invariant
	_ = engine.RegisterStep(MigrationStep{
		From:        VersionV1Legacy,
		To:          FormatVersion(11),
		Description: "Token violating step",
		Apply: func(s *DecodeState) (*DecodeState, error) {
			cp := s.Clone()
			cp.Version = FormatVersion(11)
			cp.Tokens = append(cp.Tokens, 9999) // Mutated length!
			return cp, nil
		},
	})

	_, err = engine.Migrate(state, FormatVersion(11))
	if !errors.Is(err, ErrMigrationFailed) {
		t.Errorf("expected ErrMigrationFailed for token invariant violation, got %v", err)
	}

	// Register a step that mutates token content
	_ = engine.RegisterStep(MigrationStep{
		From:        VersionV1Legacy,
		To:          FormatVersion(12),
		Description: "Token content mutating step",
		Apply: func(s *DecodeState) (*DecodeState, error) {
			cp := s.Clone()
			cp.Version = FormatVersion(12)
			cp.Tokens[0] ^= 0x55
			return cp, nil
		},
	})

	_, err = engine.Migrate(state, FormatVersion(12))
	if !errors.Is(err, ErrMigrationFailed) {
		t.Errorf("expected ErrMigrationFailed for token content mutation, got %v", err)
	}

	// Register a step that mutates KV metadata
	_ = engine.RegisterStep(MigrationStep{
		From:        VersionV1Legacy,
		To:          FormatVersion(13),
		Description: "KV metadata mutating step",
		Apply: func(s *DecodeState) (*DecodeState, error) {
			cp := s.Clone()
			cp.Version = FormatVersion(13)
			cp.KVMetadata.NumLayers = 999
			return cp, nil
		},
	})

	_, err = engine.Migrate(state, FormatVersion(13))
	if !errors.Is(err, ErrMigrationFailed) {
		t.Errorf("expected ErrMigrationFailed for KV structural invariant violation, got %v", err)
	}
}

func TestMigrationFailClosed_PayloadTruncation(t *testing.T) {
	engine := NewMigrationRunner()

	// Truncated payload (<4 bytes)
	truncated, err := NewDecodeState(VersionV1Legacy, "m", "s", []int64{1}, KVBufferMetadata{NumLayers: 1, NumHeads: 1, HeadDim: 1}, []byte("AB"))
	if err != nil {
		t.Fatalf("failed to create state: %v", err)
	}

	_, err = engine.Migrate(truncated, VersionV2Paged)
	if !errors.Is(err, ErrTruncatedPayload) {
		t.Errorf("expected ErrTruncatedPayload, got %v", err)
	}

	// Mismatched tag
	badTag, err := NewDecodeState(VersionV1Legacy, "m", "s", []int64{1}, KVBufferMetadata{NumLayers: 1, NumHeads: 1, HeadDim: 1}, []byte("WRONGTAG"))
	if err != nil {
		t.Fatalf("failed to create state: %v", err)
	}

	_, err = engine.Migrate(badTag, VersionV2Paged)
	if !errors.Is(err, ErrCorruptedState) {
		t.Errorf("expected ErrCorruptedState for mismatched tag, got %v", err)
	}
}

func BenchmarkMigration(b *testing.B) {
	engine := NewMigrationRunner()
	tokens := make([]int64, 512)
	for i := range tokens {
		tokens[i] = int64(1000 + i)
	}
	meta := KVBufferMetadata{
		NumLayers:   32,
		NumHeads:    16,
		HeadDim:     128,
		BlockSize:   16,
		TotalTokens: len(tokens),
	}
	payload := make([]byte, 8192)
	copy(payload[:4], []byte(tagV1))

	state, err := NewDecodeState(VersionV1Legacy, "qwen-38b", "seq-bench", tokens, meta, payload)
	if err != nil {
		b.Fatalf("NewDecodeState failed: %v", err)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		// Migrate across multiple hops: V1 -> V4
		v4, err := engine.Migrate(state, VersionV4Compressed)
		if err != nil {
			b.Fatalf("Migrate failed: %v", err)
		}
		// Migrate back: V4 -> V1
		_, err = engine.Migrate(v4, VersionV1Legacy)
		if err != nil {
			b.Fatalf("Reverse migrate failed: %v", err)
		}
	}
}
