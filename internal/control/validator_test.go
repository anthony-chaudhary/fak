package control

import (
	"strings"
	"testing"
)

func TestValidator_RelationalInvariants(t *testing.T) {
	t.Run("Invariant1_BatchTokensLessThanModelLen", func(t *testing.T) {
		cfg := DefaultConfig()
		cfg.MaxBatchTokens = 4096
		cfg.MaxModelLen = 8192

		errs := Validate(cfg)
		if !errs.HasErrors() {
			t.Fatalf("expected validation failure for max_batch_tokens < max_model_len, got none")
		}

		var found bool
		for _, e := range errs {
			if e.Code == ErrRelationalInvariantBatchTokens {
				found = true
				if !strings.Contains(e.Message, "max_batch_tokens (4096) must be >= max_model_len (8192)") {
					t.Errorf("unexpected error message: %s", e.Message)
				}
			}
		}
		if !found {
			t.Errorf("expected ErrRelationalInvariantBatchTokens in %v", errs)
		}
	})

	t.Run("Invariant2_DraftDepthExceedsPreallocatedSlots", func(t *testing.T) {
		cfg := DefaultConfig()
		cfg.SpeculativeDraftDepth = 10
		cfg.MaxPreallocatedDraftLimit = 8

		errs := Validate(cfg)
		if !errs.HasErrors() {
			t.Fatalf("expected validation failure for draft depth > draft slots, got none")
		}

		var found bool
		for _, e := range errs {
			if e.Code == ErrRelationalInvariantDraftDepth {
				found = true
				if !strings.Contains(e.Message, "speculative_draft_depth (10) must be <= max_preallocated_draft_slots (8)") {
					t.Errorf("unexpected error message: %s", e.Message)
				}
			}
		}
		if !found {
			t.Errorf("expected ErrRelationalInvariantDraftDepth in %v", errs)
		}
	})

	t.Run("Invariant3_KVBlocksExceedAvailableVRAMHeadroom", func(t *testing.T) {
		cfg := DefaultConfig()
		// 24GB VRAM, 14GB weights, 2GB headroom -> 8GB available for KV cache.
		// Setting target blocks to 10M with 2048 bytes block = ~20GB -> exceeds 8GB headroom.
		cfg.AvailableVRAMBytes = 24 * 1024 * 1024 * 1024
		cfg.ModelWeightsBytes = 14 * 1024 * 1024 * 1024
		cfg.ActivationHeadroomBytes = 2 * 1024 * 1024 * 1024
		cfg.TargetKVBlocks = 10000000
		cfg.BlockSizeBytes = 2048

		errs := Validate(cfg)
		if !errs.HasErrors() {
			t.Fatalf("expected validation failure for VRAM overcommit, got none")
		}

		var found bool
		for _, e := range errs {
			if e.Code == ErrRelationalInvariantVRAMOvercommit {
				found = true
				if !strings.Contains(e.Message, "exceeds available VRAM headroom") {
					t.Errorf("unexpected error message: %s", e.Message)
				}
			}
		}
		if !found {
			t.Errorf("expected ErrRelationalInvariantVRAMOvercommit in %v", errs)
		}
	})

	t.Run("DefaultConfig_PassesRelationalInvariants", func(t *testing.T) {
		cfg := DefaultConfig()
		errs := Validate(cfg)
		if errs.HasErrors() {
			t.Fatalf("default config failed validation: %v", errs)
		}
	})
}

func TestValidator_SyntacticBounds(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*ServingConfig)
		errCode string
	}{
		{
			name: "completion deadline ceiling",
			mutate: func(c *ServingConfig) {
				c.CompletionDeadlineMs = 4000000
			},
			errCode: ErrSyntacticInvalidRange,
		},
		{
			name: "stream timeout below minimum",
			mutate: func(c *ServingConfig) {
				c.StreamProgressTimeoutMs = 1000
			},
			errCode: ErrSyntacticInvalidRange,
		},
		{
			name: "max waiting seqs ceiling",
			mutate: func(c *ServingConfig) {
				c.MaxWaitingSeqs = 2000000
			},
			errCode: ErrSyntacticInvalidRange,
		},
		{
			name: "compact anchor head invalid",
			mutate: func(c *ServingConfig) {
				c.CompactAnchorHead = 2
			},
			errCode: ErrSyntacticInvalidRange,
		},
		{
			name: "log level invalid enum",
			mutate: func(c *ServingConfig) {
				c.LogLevel = "verbose"
			},
			errCode: ErrSyntacticInvalidEnum,
		},
		{
			name: "speculative acceptance threshold out of range",
			mutate: func(c *ServingConfig) {
				c.SpeculativeAcceptanceThreshold = 1.5
			},
			errCode: ErrSyntacticInvalidRange,
		},
		{
			name: "priority strategy invalid enum",
			mutate: func(c *ServingConfig) {
				c.PriorityStrategy = "random"
			},
			errCode: ErrSyntacticInvalidEnum,
		},
		{
			name: "preemption strategy invalid enum",
			mutate: func(c *ServingConfig) {
				c.PreemptionStrategy = "kill"
			},
			errCode: ErrSyntacticInvalidEnum,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := DefaultConfig()
			tc.mutate(&cfg)
			errs := Validate(cfg)
			if !errs.HasErrors() {
				t.Fatalf("expected validation failure, got none")
			}
			var found bool
			for _, e := range errs {
				if e.Code == tc.errCode {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("expected error code %s in %v", tc.errCode, errs)
			}
		})
	}
}
