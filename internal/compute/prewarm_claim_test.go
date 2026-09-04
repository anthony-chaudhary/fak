package compute

import (
	"sync"
	"testing"
	"time"
)

func sampleConfig() PrewarmEffectiveConfig {
	return PrewarmEffectiveConfig{
		ModelID:          "qwen3.8-35b-instruct",
		LayerCount:       64,
		HeadCount:        40,
		KVHeadCount:      8,
		HeadDimension:    128,
		PageSizeTokens:   64,
		Quantization:     "fp8",
		BackendType:      "metal",
		DevicePCI:        "0000:00:00.0",
		NUMANode:         0,
		Transport:        "shm",
		NICStripingCount: 2,
		PasswordPresent:  true,
		SecretClassified: false,
		IdentityClass:    "cluster-internal",
	}
}

func TestPrewarmEffectiveConfig_Digest(t *testing.T) {
	base := sampleConfig()
	d1 := base.Digest()
	d2 := base.Digest()

	if len(d1) != 64 {
		t.Fatalf("expected 64-character hex digest, got %d chars: %s", len(d1), d1)
	}
	if d1 != d2 {
		t.Fatalf("digest is not deterministic: %s != %s", d1, d2)
	}

	// Verify all fields are behavior-affecting.
	variants := []struct {
		name string
		mod  func(c *PrewarmEffectiveConfig)
	}{
		{"ModelID", func(c *PrewarmEffectiveConfig) { c.ModelID = "other-model" }},
		{"LayerCount", func(c *PrewarmEffectiveConfig) { c.LayerCount++ }},
		{"HeadCount", func(c *PrewarmEffectiveConfig) { c.HeadCount++ }},
		{"KVHeadCount", func(c *PrewarmEffectiveConfig) { c.KVHeadCount++ }},
		{"HeadDimension", func(c *PrewarmEffectiveConfig) { c.HeadDimension++ }},
		{"PageSizeTokens", func(c *PrewarmEffectiveConfig) { c.PageSizeTokens++ }},
		{"Quantization", func(c *PrewarmEffectiveConfig) { c.Quantization = "q4_k_m" }},
		{"BackendType", func(c *PrewarmEffectiveConfig) { c.BackendType = "cuda" }},
		{"DevicePCI", func(c *PrewarmEffectiveConfig) { c.DevicePCI = "0000:01:00.0" }},
		{"NUMANode", func(c *PrewarmEffectiveConfig) { c.NUMANode = 1 }},
		{"Transport", func(c *PrewarmEffectiveConfig) { c.Transport = "rdma" }},
		{"NICStripingCount", func(c *PrewarmEffectiveConfig) { c.NICStripingCount = 4 }},
		{"PasswordPresent", func(c *PrewarmEffectiveConfig) { c.PasswordPresent = false }},
		{"SecretClassified", func(c *PrewarmEffectiveConfig) { c.SecretClassified = true }},
		{"IdentityClass", func(c *PrewarmEffectiveConfig) { c.IdentityClass = "public" }},
	}

	for _, tc := range variants {
		cfg := base
		tc.mod(&cfg)
		diffDigest := cfg.Digest()
		if diffDigest == d1 {
			t.Errorf("field %s modification did not alter digest", tc.name)
		}
	}
}

func TestPrewarmClaimRegistry_PrepareSecretClassRefused(t *testing.T) {
	reg := NewPrewarmClaimRegistry(10, time.Minute)
	cfg := sampleConfig()
	cfg.SecretClassified = true

	status, err := reg.Prepare("key-secret", cfg, time.Minute)
	if status != ClaimStatusSecretClassRefused {
		t.Fatalf("expected ClaimStatusSecretClassRefused, got %s", status)
	}
	if err != ErrSecretClassRefused {
		t.Fatalf("expected ErrSecretClassRefused, got %v", err)
	}

	// Verify nothing was recorded.
	if _, ok := reg.GetState("key-secret"); ok {
		t.Fatalf("secret-classified entry should not be stored in registry")
	}

	// Empty key check.
	cfg.SecretClassified = false
	status, err = reg.Prepare("", cfg, time.Minute)
	if status != ClaimStatusNotFound || err != ErrWarmKeyRequired {
		t.Fatalf("expected ClaimStatusNotFound/ErrWarmKeyRequired for empty key, got %s, %v", status, err)
	}
}

func TestPrewarmClaimRegistry_LifecycleAndSingleClaim(t *testing.T) {
	now := time.Date(2026, 9, 3, 12, 0, 0, 0, time.UTC)
	reg := NewPrewarmClaimRegistry(10, time.Minute)
	reg.SetClock(func() time.Time { return now })

	cfg := sampleConfig()
	digest := cfg.Digest()
	key := "session-warm-123"

	// 1. Prepare
	status, err := reg.Prepare(key, cfg, 30*time.Second)
	if status != ClaimStatusOK || err != nil {
		t.Fatalf("Prepare failed: %s, %v", status, err)
	}

	state, ok := reg.GetState(key)
	if !ok || state != PrewarmStatePreparing {
		t.Fatalf("expected state PrewarmStatePreparing, got %s, ok=%t", state, ok)
	}

	// 2. Claim while preparing -> ClaimStatusNotReady
	claimStatus, receipt := reg.Claim(key, digest, "worker-1")
	if claimStatus != ClaimStatusNotReady || receipt != nil {
		t.Fatalf("expected ClaimStatusNotReady, got %s, receipt=%v", claimStatus, receipt)
	}

	// 3. CommitSuccess
	if err := reg.CommitSuccess(key, 1024); err != nil {
		t.Fatalf("CommitSuccess failed: %v", err)
	}

	state, ok = reg.GetState(key)
	if !ok || state != PrewarmStateReady {
		t.Fatalf("expected state PrewarmStateReady, got %s, ok=%t", state, ok)
	}

	// 4. First Claim succeeds
	now = now.Add(5 * time.Second)
	claimStatus, receipt = reg.Claim(key, digest, "worker-1")
	if claimStatus != ClaimStatusOK || receipt == nil {
		t.Fatalf("expected ClaimStatusOK with receipt, got %s, receipt=%v", claimStatus, receipt)
	}
	if receipt.WarmKey != key || receipt.ConfigDigest != digest || receipt.TokensWarmed != 1024 {
		t.Fatalf("receipt fields mismatch: %+v", receipt)
	}
	if receipt.ClaimantID != "worker-1" || !receipt.ClaimedAt.Equal(now) {
		t.Fatalf("receipt claimant or timestamp mismatch: %+v", receipt)
	}

	// 5. Subsequent Claim returns ClaimStatusAlreadyClaimed
	claimStatus, receipt2 := reg.Claim(key, digest, "worker-2")
	if claimStatus != ClaimStatusAlreadyClaimed || receipt2 != nil {
		t.Fatalf("expected ClaimStatusAlreadyClaimed on second claim, got %s, receipt=%v", claimStatus, receipt2)
	}
}

func TestPrewarmClaimRegistry_ConfigMismatch(t *testing.T) {
	reg := NewPrewarmClaimRegistry(10, time.Minute)
	cfg := sampleConfig()
	key := "key-mismatch"

	if _, err := reg.Prepare(key, cfg, time.Minute); err != nil {
		t.Fatalf("Prepare failed: %v", err)
	}
	if err := reg.CommitSuccess(key, 512); err != nil {
		t.Fatalf("CommitSuccess failed: %v", err)
	}

	status, receipt := reg.Claim(key, "mismatched-digest-000000000000000000000000000000000000000000000000", "worker-1")
	if status != ClaimStatusConfigMismatch || receipt != nil {
		t.Fatalf("expected ClaimStatusConfigMismatch, got %s, receipt=%v", status, receipt)
	}

	// Confirm entry was not claimed and still can be claimed with matching digest.
	status, receipt = reg.Claim(key, cfg.Digest(), "worker-2")
	if status != ClaimStatusOK || receipt == nil {
		t.Fatalf("expected ClaimStatusOK after previous mismatch, got %s, receipt=%v", status, receipt)
	}
}

func TestPrewarmClaimRegistry_Failure(t *testing.T) {
	reg := NewPrewarmClaimRegistry(10, time.Minute)
	cfg := sampleConfig()
	key := "key-fail"

	if _, err := reg.Prepare(key, cfg, time.Minute); err != nil {
		t.Fatalf("Prepare failed: %v", err)
	}
	if err := reg.CommitFailure(key, "device OOM during prefill"); err != nil {
		t.Fatalf("CommitFailure failed: %v", err)
	}

	state, ok := reg.GetState(key)
	if !ok || state != PrewarmStateFailed {
		t.Fatalf("expected PrewarmStateFailed, got %s", state)
	}

	status, receipt := reg.Claim(key, cfg.Digest(), "worker-1")
	if status != ClaimStatusFailed || receipt != nil {
		t.Fatalf("expected ClaimStatusFailed, got %s", status)
	}
}

func TestPrewarmClaimRegistry_Expiration(t *testing.T) {
	now := time.Date(2026, 9, 3, 12, 0, 0, 0, time.UTC)
	reg := NewPrewarmClaimRegistry(10, 10*time.Second)
	reg.SetClock(func() time.Time { return now })

	cfg := sampleConfig()
	key := "key-expire"

	if _, err := reg.Prepare(key, cfg, 10*time.Second); err != nil {
		t.Fatalf("Prepare failed: %v", err)
	}
	if err := reg.CommitSuccess(key, 256); err != nil {
		t.Fatalf("CommitSuccess failed: %v", err)
	}

	// Advance clock past TTL.
	now = now.Add(15 * time.Second)

	status, receipt := reg.Claim(key, cfg.Digest(), "worker-1")
	if status != ClaimStatusExpired || receipt != nil {
		t.Fatalf("expected ClaimStatusExpired, got %s", status)
	}

	state, ok := reg.GetState(key)
	if !ok || state != PrewarmStateExpired {
		t.Fatalf("expected PrewarmStateExpired, got %s", state)
	}
}

func TestPrewarmClaimRegistry_PreparingExpiration(t *testing.T) {
	now := time.Date(2026, 9, 3, 12, 0, 0, 0, time.UTC)
	reg := NewPrewarmClaimRegistry(10, 5*time.Second)
	reg.SetClock(func() time.Time { return now })

	cfg := sampleConfig()
	key := "key-prep-expire"

	if _, err := reg.Prepare(key, cfg, 5*time.Second); err != nil {
		t.Fatalf("Prepare failed: %v", err)
	}

	// Advance clock past TTL before commit.
	now = now.Add(10 * time.Second)

	status, receipt := reg.Claim(key, cfg.Digest(), "worker-1")
	if status != ClaimStatusExpired || receipt != nil {
		t.Fatalf("expected ClaimStatusExpired, got %s", status)
	}

	err := reg.CommitSuccess(key, 256)
	if err != ErrPrewarmExpired {
		t.Fatalf("expected ErrPrewarmExpired, got %v", err)
	}
}

func TestPrewarmClaimRegistry_Reap(t *testing.T) {
	now := time.Date(2026, 9, 3, 12, 0, 0, 0, time.UTC)
	reg := NewPrewarmClaimRegistry(10, time.Minute)
	reg.SetClock(func() time.Time { return now })

	cfg := sampleConfig()

	// Entry 1: claimed
	reg.Prepare("k1", cfg, time.Minute)
	reg.CommitSuccess("k1", 100)
	reg.Claim("k1", cfg.Digest(), "w1")

	// Entry 2: expired ready
	reg.Prepare("k2", cfg, 10*time.Second)
	reg.CommitSuccess("k2", 200)

	// Entry 3: unexpired ready
	reg.Prepare("k3", cfg, 20*time.Minute)
	reg.CommitSuccess("k3", 300)

	// Advance time past k2's TTL but before k3's.
	now = now.Add(30 * time.Second)

	purged := reg.Reap(now)
	if purged != 2 {
		t.Fatalf("expected 2 entries purged (claimed and expired), got %d", purged)
	}

	if reg.Len() != 1 {
		t.Fatalf("expected 1 remaining entry, got %d", reg.Len())
	}

	state, ok := reg.GetState("k3")
	if !ok || state != PrewarmStateReady {
		t.Fatalf("expected k3 to remain ready, got %s, ok=%t", state, ok)
	}
}

func TestPrewarmClaimRegistry_BoundedCapacity(t *testing.T) {
	now := time.Date(2026, 9, 3, 12, 0, 0, 0, time.UTC)
	reg := NewPrewarmClaimRegistry(3, 10*time.Second)
	reg.SetClock(func() time.Time { return now })

	cfg := sampleConfig()

	// Add 3 entries at capacity
	now = now.Add(1 * time.Second)
	reg.Prepare("k1", cfg, 5*time.Second)
	now = now.Add(1 * time.Second)
	reg.Prepare("k2", cfg, 20*time.Second)
	now = now.Add(1 * time.Second)
	reg.Prepare("k3", cfg, 20*time.Second)

	if reg.Len() != 3 {
		t.Fatalf("expected 3 entries, got %d", reg.Len())
	}

	// Advance time so k1 expires
	now = now.Add(10 * time.Second)

	// Preparing k4 should purge expired k1 and succeed within bound
	now = now.Add(1 * time.Second)
	status, err := reg.Prepare("k4", cfg, 20*time.Second)
	if status != ClaimStatusOK || err != nil {
		t.Fatalf("Prepare k4 failed: %s, %v", status, err)
	}
	if reg.Len() > 3 {
		t.Fatalf("registry exceeded maxEntries bound: %d", reg.Len())
	}

	// Preparing k5 when none are expired should evict oldest (k2)
	now = now.Add(1 * time.Second)
	status, err = reg.Prepare("k5", cfg, 20*time.Second)
	if status != ClaimStatusOK || err != nil {
		t.Fatalf("Prepare k5 failed: %s, %v", status, err)
	}
	if reg.Len() > 3 {
		t.Fatalf("registry exceeded maxEntries bound: %d", reg.Len())
	}
	if _, ok := reg.GetState("k2"); ok {
		t.Fatalf("expected oldest entry k2 to be evicted")
	}
}

func TestPrewarmClaimRegistry_Concurrency(t *testing.T) {
	reg := NewPrewarmClaimRegistry(100, 5*time.Minute)
	cfg := sampleConfig()
	digest := cfg.Digest()

	var wg sync.WaitGroup
	workers := 20
	rounds := 50

	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for r := 0; r < rounds; r++ {
				key := "concurrent-test-key"
				reg.Prepare(key, cfg, time.Minute)
				reg.CommitSuccess(key, 100)
				reg.Claim(key, digest, "concurrent-worker")
				reg.Reap(time.Now().UTC())
			}
		}(w)
	}

	wg.Wait()
	if l := reg.Len(); l < 0 {
		t.Fatalf("unexpected negative registry length: %d", l)
	}
}
