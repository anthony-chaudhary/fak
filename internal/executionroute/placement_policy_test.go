package executionroute

import (
	"encoding/json"
	"testing"
)

func TestPlacementPolicy_DeterministicRouting(t *testing.T) {
	manifest := DefaultPlacementManifest()

	tests := []struct {
		name        string
		req         PlacementRequest
		wantSurface ExecutionSurface
		wantRule    string
	}{
		{
			name: "small auxiliary model to NPU",
			req: PlacementRequest{
				Model:          "Qwen2.5-1.5B",
				Role:           "tool_screen",
				ParameterSizeB: 1.5,
				ContextTokens:  2048,
			},
			wantSurface: SurfaceNPU,
			wantRule:    "npu-auxiliary-small",
		},
		{
			name: "primary medium model to GPU",
			req: PlacementRequest{
				Model:          "Qwen3-27B",
				Role:           "primary",
				ParameterSizeB: 27.0,
				ContextTokens:  8192,
			},
			wantSurface: SurfaceGPU,
			wantRule:    "gpu-primary-local",
		},
		{
			name: "massive model to Hosted",
			req: PlacementRequest{
				Model:          "DeepSeek-R1-405B",
				Role:           "primary",
				ParameterSizeB: 405.0,
			},
			wantSurface: SurfaceHosted,
			wantRule:    "hosted-massive-cloud",
		},
		{
			name: "massive model with offline constraint disqualified from hosted",
			req: PlacementRequest{
				Model:          "DeepSeek-R1-405B",
				Role:           "primary",
				ParameterSizeB: 405.0,
				OfflineOnly:    true,
			},
			wantSurface: SurfaceGPU,
			wantRule:    "gpu-primary-local",
		},
		{
			name: "classifier embedding to CPU",
			req: PlacementRequest{
				Model: "bge-small-en-v1.5",
				Role:  "embed",
			},
			wantSurface: SurfaceCPU,
			wantRule:    "cpu-offline-fallback",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			dec, err := manifest.RoutePlacement(tc.req)
			if err != nil {
				t.Fatalf("unexpected routing error: %v", err)
			}
			if dec.Surface != tc.wantSurface {
				t.Errorf("got surface %s, want %s (reason: %s)", dec.Surface, tc.wantSurface, dec.Reason)
			}
			if dec.RuleName != tc.wantRule {
				t.Errorf("got rule %q, want %q", dec.RuleName, tc.wantRule)
			}
			if len(dec.AuditLog) == 0 {
				t.Errorf("expected audit log entries, got none")
			}
			if dec.EvaluatedRules != len(dec.AuditLog) {
				t.Errorf("evaluated rules %d != audit log len %d", dec.EvaluatedRules, len(dec.AuditLog))
			}
		})
	}
}

func TestPlacementPolicy_Auditability(t *testing.T) {
	manifest := DefaultPlacementManifest()

	req := PlacementRequest{
		Model:          "Qwen3-27B",
		Role:           "primary",
		ParameterSizeB: 27.0,
	}

	dec, err := manifest.RoutePlacement(req)
	if err != nil {
		t.Fatalf("routing error: %v", err)
	}

	// Should have evaluated rule 1 (NPU - failed because role=primary/size=27B)
	// rule 2 (Hosted - failed because size 27B < 70B)
	// rule 3 (GPU - matched)
	if len(dec.AuditLog) < 3 {
		t.Fatalf("expected at least 3 audit log entries, got %d", len(dec.AuditLog))
	}

	r1 := dec.AuditLog[0]
	if r1.RuleName != "npu-auxiliary-small" || r1.Matched {
		t.Errorf("expected rule 1 npu-auxiliary-small to not match, got %+v", r1)
	}

	r2 := dec.AuditLog[1]
	if r2.RuleName != "hosted-massive-cloud" || r2.Matched {
		t.Errorf("expected rule 2 hosted-massive-cloud to not match, got %+v", r2)
	}

	r3 := dec.AuditLog[2]
	if r3.RuleName != "gpu-primary-local" || !r3.Matched {
		t.Errorf("expected rule 3 gpu-primary-local to match, got %+v", r3)
	}
}

func TestPlacementPolicy_DeterminismUnderRepeatedInvocations(t *testing.T) {
	manifest := DefaultPlacementManifest()

	req := PlacementRequest{
		Model:          "Qwen2.5-1.5B",
		Role:           "tool_screen",
		ParameterSizeB: 1.5,
		Tags:           []string{"agent", "auxiliary"},
	}

	firstDec, err := manifest.RoutePlacement(req)
	if err != nil {
		t.Fatalf("first route failed: %v", err)
	}
	firstJSON, _ := json.Marshal(firstDec)

	for i := 0; i < 100; i++ {
		nextDec, err := manifest.RoutePlacement(req)
		if err != nil {
			t.Fatalf("iter %d route failed: %v", i, err)
		}
		nextJSON, _ := json.Marshal(nextDec)
		if string(firstJSON) != string(nextJSON) {
			t.Fatalf("non-deterministic placement detected at iter %d:\nfirst: %s\nnext:  %s",
				i, string(firstJSON), string(nextJSON))
		}
	}
}

func TestPlacementPolicy_CustomManifestJSON(t *testing.T) {
	manifestJSON := []byte(`{
		"version": "fak-placement-policy/1",
		"default_surface": "cpu",
		"rules": [
			{
				"name": "npu-high-priority",
				"surface": "npu",
				"priority": 200,
				"required_tags": ["npu-pin"]
			}
		]
	}`)

	m, err := ParsePlacementManifest(manifestJSON)
	if err != nil {
		t.Fatalf("ParsePlacementManifest: %v", err)
	}

	// Request with tag matches rule
	dec1, err := m.RoutePlacement(PlacementRequest{Tags: []string{"npu-pin"}})
	if err != nil || dec1.Surface != SurfaceNPU {
		t.Errorf("expected npu, got %s (err: %v)", dec1.Surface, err)
	}

	// Request without tag falls back to cpu
	dec2, err := m.RoutePlacement(PlacementRequest{Tags: []string{"other"}})
	if err != nil || dec2.Surface != SurfaceCPU {
		t.Errorf("expected cpu default fallback, got %s (err: %v)", dec2.Surface, err)
	}
}

func TestPlacementPolicy_ParseManifestError(t *testing.T) {
	_, err := ParsePlacementManifest([]byte(`{not valid json`))
	if err == nil {
		t.Errorf("expected error on malformed JSON")
	}
}

func TestPlacementPolicy_ContextBounds(t *testing.T) {
	manifest := PlacementManifest{
		Version:        "fak-placement-policy/1",
		DefaultSurface: SurfaceCPU,
		Rules: []PlacementRule{
			{
				Name:       "npu-short-context",
				Surface:    SurfaceNPU,
				Priority:   10,
				MaxContext: 4096,
			},
		},
	}

	// Within bound
	dec1, err := manifest.RoutePlacement(PlacementRequest{ContextTokens: 2048})
	if err != nil || dec1.Surface != SurfaceNPU {
		t.Errorf("expected npu, got %s (err: %v)", dec1.Surface, err)
	}

	// Exceeds bound -> falls back to CPU
	dec2, err := manifest.RoutePlacement(PlacementRequest{ContextTokens: 8192})
	if err != nil || dec2.Surface != SurfaceCPU {
		t.Errorf("expected cpu fallback, got %s (err: %v)", dec2.Surface, err)
	}
}

func TestPlacementPolicy_AttributeMatching(t *testing.T) {
	manifest := PlacementManifest{
		Version:        "fak-placement-policy/1",
		DefaultSurface: SurfaceCPU,
		Rules: []PlacementRule{
			{
				Name:     "attr-matched-gpu",
				Surface:  SurfaceGPU,
				Priority: 10,
				Attributes: map[string]string{
					"accelerator": "vulkan",
					"tier":        "high",
				},
			},
		},
	}

	// Full match
	dec1, err := manifest.RoutePlacement(PlacementRequest{
		Attributes: map[string]string{
			"accelerator": "vulkan",
			"tier":        "high",
			"extra":       "ignored",
		},
	})
	if err != nil || dec1.Surface != SurfaceGPU {
		t.Errorf("expected gpu, got %s (err: %v)", dec1.Surface, err)
	}

	// Value mismatch
	dec2, err := manifest.RoutePlacement(PlacementRequest{
		Attributes: map[string]string{
			"accelerator": "metal",
			"tier":        "high",
		},
	})
	if err != nil || dec2.Surface != SurfaceCPU {
		t.Errorf("expected cpu fallback, got %s (err: %v)", dec2.Surface, err)
	}

	// Nil request attributes
	dec3, err := manifest.RoutePlacement(PlacementRequest{})
	if err != nil || dec3.Surface != SurfaceCPU {
		t.Errorf("expected cpu fallback, got %s (err: %v)", dec3.Surface, err)
	}
}

func TestPlacementPolicy_WildcardModelsAndRoles(t *testing.T) {
	manifest := PlacementManifest{
		Version:        "fak-placement-policy/1",
		DefaultSurface: SurfaceCPU,
		Rules: []PlacementRule{
			{
				Name:     "wildcard-npu",
				Surface:  SurfaceNPU,
				Priority: 10,
				Models:   []string{"*"},
				Roles:    []string{"*"},
			},
		},
	}

	dec, err := manifest.RoutePlacement(PlacementRequest{
		Model: "any-model-xyz",
		Role:  "any-role-abc",
	})
	if err != nil || dec.Surface != SurfaceNPU {
		t.Errorf("expected npu on wildcards, got %s (err: %v)", dec.Surface, err)
	}
}

func TestPlacementPolicy_HostedDefaultRefusedWhenRequireLocal(t *testing.T) {
	manifest := PlacementManifest{
		Version:        "fak-placement-policy/1",
		DefaultSurface: SurfaceHosted,
		Rules:          nil,
	}

	dec, err := manifest.RoutePlacement(PlacementRequest{RequireLocal: true})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if dec.Surface != SurfaceGPU {
		t.Errorf("expected fallback to gpu when hosted default is refused by require_local, got %s", dec.Surface)
	}
}
