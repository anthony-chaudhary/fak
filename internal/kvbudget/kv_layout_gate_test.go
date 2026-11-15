package kvbudget

import "testing"

// fak-test:runtime fast est=10ms lane=default

// TestKVLayoutAdmission pins the forward-admission gate for the Hopper MLA KV
// layout axes: it refuses an unsupported combination at plan time with a typed
// sentinel, and recomputes the per-block byte footprint from the declared
// layout for a supported combination.
func TestKVLayoutAdmission(t *testing.T) {
	// A fully-supported Hopper MLA layout: fp8_ds_mla, 128 tokens/block,
	// window 128, beam 1, compress ratio 1, dsv41_mla layout.
	supported := KVLayoutConfig{
		DType:          "fp8_ds_mla",
		TokensPerBlock: 128,
		WindowSize:     128,
		MaxBeamWidth:   1,
		CompressRatio:  1,
		Layout:         "dsv41_mla",
	}

	// (a) The refusal sentinel for an unsupported layout.
	t.Run("unsupported_layout_refusal", func(t *testing.T) {
		bad := supported
		bad.Layout = "not_a_layout"
		got := AdmitKVLayout(bad)
		if got.Admitted {
			t.Fatalf("admitted an unsupported layout: %+v", got)
		}
		if got.Reason != KVLayoutUnsupportedLayout {
			t.Fatalf("reason = %q, want %q", got.Reason, KVLayoutUnsupportedLayout)
		}
		if got.PerBlockBytes != 0 {
			t.Fatalf("refused verdict carried per-block bytes %d, want 0", got.PerBlockBytes)
		}
	})

	// (b) The per-block byte recompute for a supported layout. The dsv41_mla
	// geometry caches 61 layers × (512 latent + 64 rope) + 61 index layers ×
	// 128 = 42944 KV elements per token; at fp8_ds_mla (1 B/elem) that is
	// 42944 B/token, so a 128-token block is 42944 × 128 = 5496832 B.
	t.Run("per_block_byte_recompute", func(t *testing.T) {
		got := AdmitKVLayout(supported)
		if !got.Admitted {
			t.Fatalf("refused a supported layout: %+v", got)
		}
		if got.Reason != KVLayoutAdmitted {
			t.Fatalf("reason = %q, want empty (admitted)", got.Reason)
		}
		const want = 42944 * 128 // 5496832 B
		if got.PerBlockBytes != want {
			t.Fatalf("per-block bytes = %d, want %d", got.PerBlockBytes, want)
		}
	})

	// The recompute tracks the declared block size: a 256-token block is exactly
	// double the 128-token block, proving the figure is derived from the layout
	// axes rather than a hardcoded constant.
	t.Run("per_block_tracks_block_size", func(t *testing.T) {
		bigger := supported
		bigger.TokensPerBlock = 256
		got := AdmitKVLayout(bigger)
		if !got.Admitted {
			t.Fatalf("refused a supported 256-token layout: %+v", got)
		}
		const want = 42944 * 256 // 10993664 B
		if got.PerBlockBytes != want {
			t.Fatalf("per-block bytes = %d, want %d", got.PerBlockBytes, want)
		}
	})

	// Each remaining axis refuses with its own typed sentinel when it leaves the
	// supported set.
	t.Run("each_axis_refusal", func(t *testing.T) {
		cases := []struct {
			name   string
			mutate func(*KVLayoutConfig)
			want   KVLayoutRefusal
		}{
			{"dtype", func(c *KVLayoutConfig) { c.DType = "bf16" }, KVLayoutUnsupportedDType},
			{"block_size", func(c *KVLayoutConfig) { c.TokensPerBlock = 64 }, KVLayoutUnsupportedBlockSize},
			{"window", func(c *KVLayoutConfig) { c.WindowSize = 256 }, KVLayoutUnsupportedWindow},
			{"beam", func(c *KVLayoutConfig) { c.MaxBeamWidth = 4 }, KVLayoutUnsupportedBeam},
			{"compress", func(c *KVLayoutConfig) { c.CompressRatio = 8 }, KVLayoutUnsupportedCompress},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				cfg := supported
				tc.mutate(&cfg)
				got := AdmitKVLayout(cfg)
				if got.Admitted {
					t.Fatalf("admitted an unsupported %s: %+v", tc.name, got)
				}
				if got.Reason != tc.want {
					t.Fatalf("reason = %q, want %q", got.Reason, tc.want)
				}
			})
		}
	})
}
