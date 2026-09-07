package compute

import (
	"strings"
	"testing"
)

func TestSpeculativeGraphReachabilityValidation(t *testing.T) {
	cfg := SpeculativeGraphConfig{
		PrimaryUBatchSize:   1024,
		SpecDraftUBatchSize: 512,
		DeviceTag:           "sm90",
		MaxActiveSeqs:       64,
		MaxTokensPerReq:     16,
	}

	planner, err := NewSpeculativeGraphPlanner(cfg)
	if err != nil {
		t.Fatalf("NewSpeculativeGraphPlanner failed: %v", err)
	}

	// 1. Valid keys pass reachability validation
	validKeys := []GraphCaptureKey{
		planner.SpeculativeCaptureKey(),
		planner.PrimaryCaptureKey(1024),
		planner.PrimaryCaptureKey(512),
		planner.PrimaryCaptureKey(64),
		{
			Kind:               GraphCaptureSpeculative,
			BatchSize:          512,
			Tag:                "sm90",
			NumSequences:       32,
			TokensPerReq:       8,
			SchedulerReachable: true,
		},
		{
			Kind:               GraphCaptureSpeculative,
			BatchSize:          512,
			Tag:                "sm90",
			NumSequences:       64,
			TokensPerReq:       8,
			SchedulerReachable: true,
		},
	}

	for _, k := range validKeys {
		if err := planner.ValidateKey(k); err != nil {
			t.Errorf("expected valid key %s, got validation error: %v", k, err)
		}
	}

	// 2. ShapeCaptureKey reachability
	shapeKey, err := planner.ShapeCaptureKey(16, 4)
	if err != nil {
		t.Fatalf("ShapeCaptureKey(16, 4) failed: %v", err)
	}
	if !shapeKey.SchedulerReachable {
		t.Errorf("ShapeCaptureKey should have SchedulerReachable: true")
	}

	// Impossible shape: 128 sequences exceeds MaxActiveSeqs (64)
	_, err = planner.ShapeCaptureKey(128, 4)
	if err == nil {
		t.Fatalf("expected error for ShapeCaptureKey(128, 4), got nil")
	}

	// Impossible shape: 32 tokens per req exceeds MaxTokensPerReq (16)
	_, err = planner.ShapeCaptureKey(16, 32)
	if err == nil {
		t.Fatalf("expected error for ShapeCaptureKey(16, 32), got nil")
	}

	// Impossible shape: 64 seqs * 16 tokens = 1024 > SpecDraftUBatchSize (512)
	_, err = planner.ShapeCaptureKey(64, 16)
	if err == nil {
		t.Fatalf("expected error for ShapeCaptureKey(64, 16), got nil")
	}

	// 3. Unreachable / synthetic impossible keys rejected
	invalidCases := []struct {
		name    string
		key     GraphCaptureKey
		wantErr string
	}{
		{
			name: "unknown key kind",
			key: GraphCaptureKey{
				Kind:      "unknown_kind",
				BatchSize: 512,
			},
			wantErr: "unknown capture key kind",
		},
		{
			name: "zero batch size",
			key: GraphCaptureKey{
				Kind:      GraphCaptureSpeculative,
				BatchSize: 0,
			},
			wantErr: "batch size must be positive",
		},
		{
			name: "negative batch size",
			key: GraphCaptureKey{
				Kind:      GraphCapturePrimary,
				BatchSize: -10,
			},
			wantErr: "batch size must be positive",
		},
		{
			name: "speculative batch size exceeds SpecDraftUBatchSize",
			key: GraphCaptureKey{
				Kind:      GraphCaptureSpeculative,
				BatchSize: 1024,
			},
			wantErr: "exceeds SpecDraftUBatchSize",
		},
		{
			name: "primary batch size exceeds PrimaryUBatchSize",
			key: GraphCaptureKey{
				Kind:      GraphCapturePrimary,
				BatchSize: 2048,
			},
			wantErr: "exceeds PrimaryUBatchSize",
		},
		{
			name: "speculative sequence count exceeds MaxActiveSeqs",
			key: GraphCaptureKey{
				Kind:         GraphCaptureSpeculative,
				BatchSize:    512,
				NumSequences: 128,
				TokensPerReq: 4,
			},
			wantErr: "exceeds MaxActiveSeqs",
		},
		{
			name: "speculative tokens per req exceeds MaxTokensPerReq",
			key: GraphCaptureKey{
				Kind:         GraphCaptureSpeculative,
				BatchSize:    512,
				NumSequences: 4,
				TokensPerReq: 32,
			},
			wantErr: "exceeds MaxTokensPerReq",
		},
		{
			name: "speculative total tokens exceeds SpecDraftUBatchSize",
			key: GraphCaptureKey{
				Kind:         GraphCaptureSpeculative,
				BatchSize:    512,
				NumSequences: 40,
				TokensPerReq: 15, // 40 * 15 = 600 > 512
			},
			wantErr: "exceeds SpecDraftUBatchSize",
		},
	}

	for _, tc := range invalidCases {
		t.Run(tc.name, func(t *testing.T) {
			err := planner.ValidateKey(tc.key)
			if err == nil {
				t.Fatalf("expected error for case %s, got nil", tc.name)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("error %q does not contain expected substring %q", err.Error(), tc.wantErr)
			}
		})
	}
}

func TestSpeculativeGraphRunnerAllocationGate(t *testing.T) {
	cfg := SpeculativeGraphConfig{
		PrimaryUBatchSize:   1024,
		SpecDraftUBatchSize: 512,
		MaxActiveSeqs:       64,
		MaxTokensPerReq:     16,
	}

	planner, err := NewSpeculativeGraphPlanner(cfg)
	if err != nil {
		t.Fatalf("NewSpeculativeGraphPlanner failed: %v", err)
	}

	runner := NewSpeculativeGraphRunner(planner)

	// Valid key successfully allocates graph
	validKey := planner.SpeculativeCaptureKey()
	graph, err := runner.AllocateGraph(validKey)
	if err != nil {
		t.Fatalf("AllocateGraph(validKey) failed: %v", err)
	}
	if graph == nil || graph.Key != validKey {
		t.Fatalf("unexpected graph returned: %+v", graph)
	}

	// Repeated allocation returns cached graph
	cachedGraph, err := runner.AllocateGraph(validKey)
	if err != nil {
		t.Fatalf("repeated AllocateGraph failed: %v", err)
	}
	if cachedGraph != graph {
		t.Errorf("repeated AllocateGraph did not return identical cached instance")
	}

	stats := runner.Stats()
	if stats.CapturesTotal != 1 {
		t.Errorf("CapturesTotal = %d, want 1", stats.CapturesTotal)
	}

	// Unreachable key fails validation BEFORE graph allocation
	unreachableKey := GraphCaptureKey{
		Kind:         GraphCaptureSpeculative,
		BatchSize:    512,
		NumSequences: 128, // > MaxActiveSeqs (64)
		TokensPerReq: 8,
	}

	_, err = runner.AllocateGraph(unreachableKey)
	if err == nil {
		t.Fatalf("expected AllocateGraph to fail for unreachable key, got nil")
	}
	if !strings.Contains(err.Error(), "exceeds MaxActiveSeqs") {
		t.Errorf("error %q does not contain 'exceeds MaxActiveSeqs'", err.Error())
	}

	// Verify no graph was allocated for the unreachable key
	if _, ok := runner.LookupGraph(unreachableKey); ok {
		t.Errorf("unreachable key was found in runner graph cache after failed allocation")
	}
	if runner.Stats().CapturesTotal != 1 {
		t.Errorf("CapturesTotal changed to %d after failed allocation, want 1", runner.Stats().CapturesTotal)
	}

	// ExecuteCaptureKey with unreachable key also fails before execution
	executed := false
	err = runner.ExecuteCaptureKey(unreachableKey, func(batchSize int) error {
		executed = true
		return nil
	})
	if err == nil {
		t.Fatalf("expected ExecuteCaptureKey to fail for unreachable key, got nil")
	}
	if executed {
		t.Fatalf("ExecuteCaptureKey executed callback for unreachable key")
	}
}
