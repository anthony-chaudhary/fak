package model

import (
	"testing"
)

func TestQuantCacheHints(t *testing.T) {
	t.Run("Constants", func(t *testing.T) {
		if DefaultMALLCapacityBlocks != 128 {
			t.Fatalf("DefaultMALLCapacityBlocks = %d, want 128", DefaultMALLCapacityBlocks)
		}
		if DefaultBlockTokens != 64 {
			t.Fatalf("DefaultBlockTokens = %d, want 64", DefaultBlockTokens)
		}
	})

	t.Run("DualCachePolicyHintDefaults", func(t *testing.T) {
		// Temporal pinned default
		if CacheHintTemporal.SLC != 0 || CacheHintTemporal.GLC != 0 || CacheHintTemporal.NT != 0 {
			t.Fatalf("CacheHintTemporal flags = (SLC:%d, GLC:%d, NT:%d), want (0,0,0)",
				CacheHintTemporal.SLC, CacheHintTemporal.GLC, CacheHintTemporal.NT)
		}
		if !CacheHintTemporal.Temporal || CacheHintTemporal.Bypass || !CacheHintTemporal.Prefetch {
			t.Fatalf("CacheHintTemporal booleans = (Temporal:%v, Bypass:%v, Prefetch:%v), want (true, false, true)",
				CacheHintTemporal.Temporal, CacheHintTemporal.Bypass, CacheHintTemporal.Prefetch)
		}
		if CacheHintTemporal.PolicyName != "TEMPORAL_PINNED" {
			t.Fatalf("CacheHintTemporal.PolicyName = %q, want TEMPORAL_PINNED", CacheHintTemporal.PolicyName)
		}

		// Streaming bypass default
		if CacheHintStreamingBypass.SLC != 1 || CacheHintStreamingBypass.GLC != 0 || CacheHintStreamingBypass.NT != 1 {
			t.Fatalf("CacheHintStreamingBypass flags = (SLC:%d, GLC:%d, NT:%d), want (1,0,1)",
				CacheHintStreamingBypass.SLC, CacheHintStreamingBypass.GLC, CacheHintStreamingBypass.NT)
		}
		if CacheHintStreamingBypass.Temporal || !CacheHintStreamingBypass.Bypass || CacheHintStreamingBypass.Prefetch {
			t.Fatalf("CacheHintStreamingBypass booleans = (Temporal:%v, Bypass:%v, Prefetch:%v), want (false, true, false)",
				CacheHintStreamingBypass.Temporal, CacheHintStreamingBypass.Bypass, CacheHintStreamingBypass.Prefetch)
		}
		if CacheHintStreamingBypass.PolicyName != "STREAMING_BYPASS" {
			t.Fatalf("CacheHintStreamingBypass.PolicyName = %q, want STREAMING_BYPASS", CacheHintStreamingBypass.PolicyName)
		}
	})

	t.Run("ClassifyKVBlockCacheHint", func(t *testing.T) {
		// Default zero args
		hDef := ClassifyKVBlockCacheHint()
		if !hDef.Temporal || hDef.Bypass || hDef.PinnedTokens != DefaultBlockTokens || hDef.BypassTokens != 0 {
			t.Fatalf("default ClassifyKVBlockCacheHint() = %+v", hDef)
		}

		// Negative block ID clamped to 0
		hNeg := ClassifyKVBlockCacheHint(-10)
		if !hNeg.Temporal || hNeg.Bypass || hNeg.PinnedTokens != DefaultBlockTokens || hNeg.BypassTokens != 0 {
			t.Fatalf("ClassifyKVBlockCacheHint(-10) = %+v", hNeg)
		}

		// Boundary tests around DefaultMALLCapacityBlocks (128)
		testCases := []struct {
			blockID      int
			customTokens int
			wantTemporal bool
			wantBypass   bool
			wantPinned   int
			wantBypassT  int
			policy       string
		}{
			{blockID: 0, wantTemporal: true, wantBypass: false, wantPinned: 64, wantBypassT: 0, policy: "TEMPORAL_PINNED"},
			{blockID: 63, wantTemporal: true, wantBypass: false, wantPinned: 64, wantBypassT: 0, policy: "TEMPORAL_PINNED"},
			{blockID: 127, wantTemporal: true, wantBypass: false, wantPinned: 64, wantBypassT: 0, policy: "TEMPORAL_PINNED"},
			{blockID: 128, wantTemporal: false, wantBypass: true, wantPinned: 0, wantBypassT: 64, policy: "STREAMING_BYPASS"},
			{blockID: 129, wantTemporal: false, wantBypass: true, wantPinned: 0, wantBypassT: 64, policy: "STREAMING_BYPASS"},
			{blockID: 500, wantTemporal: false, wantBypass: true, wantPinned: 0, wantBypassT: 64, policy: "STREAMING_BYPASS"},
			// With custom token counts
			{blockID: 5, customTokens: 32, wantTemporal: true, wantBypass: false, wantPinned: 32, wantBypassT: 0, policy: "TEMPORAL_PINNED"},
			{blockID: 200, customTokens: 128, wantTemporal: false, wantBypass: true, wantPinned: 0, wantBypassT: 128, policy: "STREAMING_BYPASS"},
		}

		for _, tc := range testCases {
			var h CachePolicyHint
			if tc.customTokens > 0 {
				h = ClassifyKVBlockCacheHint(tc.blockID, tc.customTokens)
			} else {
				h = ClassifyKVBlockCacheHint(tc.blockID)
			}
			if h.Temporal != tc.wantTemporal || h.Bypass != tc.wantBypass {
				t.Errorf("blockID=%d temporal=%v bypass=%v, want (%v, %v)",
					tc.blockID, h.Temporal, h.Bypass, tc.wantTemporal, tc.wantBypass)
			}
			if h.PinnedTokens != tc.wantPinned || h.BypassTokens != tc.wantBypassT {
				t.Errorf("blockID=%d tokens=(pinned:%d, bypass:%d), want (%d, %d)",
					tc.blockID, h.PinnedTokens, h.BypassTokens, tc.wantPinned, tc.wantBypassT)
			}
			if h.PolicyName != tc.policy {
				t.Errorf("blockID=%d policy=%q, want %q", tc.blockID, h.PolicyName, tc.policy)
			}
		}
	})

	t.Run("ClassifyTreeMaskCacheHint", func(t *testing.T) {
		// Zero-arg call
		hDef := ClassifyTreeMaskCacheHint()
		if !hDef.Temporal || hDef.Bypass || hDef.PinnedTokens != 0 || hDef.BypassTokens != 0 {
			t.Fatalf("ClassifyTreeMaskCacheHint() = %+v", hDef)
		}

		// Negative count
		hNeg := ClassifyTreeMaskCacheHint(-5)
		if !hNeg.Temporal || hNeg.Bypass || hNeg.PinnedTokens != 0 {
			t.Fatalf("ClassifyTreeMaskCacheHint(-5) = %+v, want pinned=0", hNeg)
		}

		// Node counts: 0, 1, 8, 32, 64, 128
		nodeCounts := []int{0, 1, 8, 32, 64, 128}
		for _, n := range nodeCounts {
			h := ClassifyTreeMaskCacheHint(n)
			if !h.Temporal || h.Bypass || h.Prefetch != true {
				t.Errorf("nodes=%d: temporal=%v, bypass=%v, prefetch=%v", n, h.Temporal, h.Bypass, h.Prefetch)
			}
			if h.SLC != 0 || h.GLC != 0 || h.NT != 0 {
				t.Errorf("nodes=%d: SLC=%d, GLC=%d, NT=%d", n, h.SLC, h.GLC, h.NT)
			}
			if h.PinnedTokens != n || h.BypassTokens != 0 {
				t.Errorf("nodes=%d: pinned=%d bypass=%d, want (%d, 0)", n, h.PinnedTokens, h.BypassTokens, n)
			}
			if h.PolicyName != "TEMPORAL_PINNED" {
				t.Errorf("nodes=%d: policy=%q, want TEMPORAL_PINNED", n, h.PolicyName)
			}
		}
	})

	t.Run("ClassifyWeightTensorCacheHint", func(t *testing.T) {
		// Without name
		hNoName := ClassifyWeightTensorCacheHint()
		if hNoName.Temporal || !hNoName.Bypass || hNoName.SLC != 1 || hNoName.NT != 1 {
			t.Fatalf("ClassifyWeightTensorCacheHint() = %+v", hNoName)
		}
		if hNoName.PolicyName != "STREAMING_BYPASS" {
			t.Fatalf("ClassifyWeightTensorCacheHint().PolicyName = %q", hNoName.PolicyName)
		}

		// With name
		hNamed := ClassifyWeightTensorCacheHint("model.layers.0.self_attn.q_proj.weight")
		if hNamed.Temporal || !hNamed.Bypass || hNamed.SLC != 1 || hNamed.NT != 1 {
			t.Fatalf("ClassifyWeightTensorCacheHint(named) = %+v", hNamed)
		}
		if hNamed.PolicyName != "STREAMING_BYPASS" {
			t.Fatalf("ClassifyWeightTensorCacheHint(named).PolicyName = %q", hNamed.PolicyName)
		}
	})

	t.Run("SessionReceiverMethods", func(t *testing.T) {
		s := &Session{}

		// Non-nil receiver
		hW := s.AttachWeightCacheHint("weight.tensor")
		if hW != ClassifyWeightTensorCacheHint("weight.tensor") {
			t.Errorf("s.AttachWeightCacheHint mismatch: %+v", hW)
		}
		hT := s.AttachTreeMaskCacheHint(64)
		if hT != ClassifyTreeMaskCacheHint(64) {
			t.Errorf("s.AttachTreeMaskCacheHint mismatch: %+v", hT)
		}
		hKV := s.AttachKVCacheHint(5)
		if hKV != ClassifyKVBlockCacheHint(5) {
			t.Errorf("s.AttachKVCacheHint mismatch: %+v", hKV)
		}

		// Nil receiver safety
		var nilSession *Session
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("nil Session call panicked: %v", r)
			}
		}()

		nilW := nilSession.AttachWeightCacheHint("weight.tensor")
		if nilW != CacheHintStreamingBypass {
			t.Errorf("nilSession.AttachWeightCacheHint = %+v, want CacheHintStreamingBypass", nilW)
		}
		nilT := nilSession.AttachTreeMaskCacheHint(32)
		if nilT.PinnedTokens != 32 || !nilT.Temporal {
			t.Errorf("nilSession.AttachTreeMaskCacheHint = %+v", nilT)
		}
		nilKV := nilSession.AttachKVCacheHint(10)
		if nilKV.PinnedTokens != 64 || !nilKV.Temporal {
			t.Errorf("nilSession.AttachKVCacheHint = %+v", nilKV)
		}
		nilKVBypass := nilSession.AttachKVCacheHint(150)
		if nilKVBypass.BypassTokens != 64 || !nilKVBypass.Bypass {
			t.Errorf("nilSession.AttachKVCacheHint(150) = %+v", nilKVBypass)
		}
	})
}
