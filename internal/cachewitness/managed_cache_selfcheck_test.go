package cachewitness

import "testing"

// ttlLine renders one fak_gateway_cache_ttl_upgrade_total sample for an outcome.
func ttlLine(outcome string, n uint64) string {
	return "fak_gateway_cache_ttl_upgrade_total{outcome=\"" + outcome + "\"} " + itoa(n) + "\n"
}

func itoa(n uint64) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}

func TestSelfCheckManagedCacheConsistent(t *testing.T) {
	body := "# HELP fak_gateway_cache_ttl_upgrade_total WITNESSED\n" +
		"# TYPE fak_gateway_cache_ttl_upgrade_total counter\n" +
		ttlLine(ttlOutcomeUpgraded, 4) +
		ttlLine("volatile_head", 1) +
		"fak_gateway_inference_cached_prompt_tokens_total 12000\n" +
		"fak_vcache_cache_creation_tokens_total 800\n"

	rep := SelfCheckManagedCache("http://gw/metrics", body)
	if rep.Verdict != ManagedCacheConsistent {
		t.Fatalf("verdict = %q, want CONSISTENT (%s)", rep.Verdict, rep.Finding)
	}
	if !rep.OK {
		t.Fatal("CONSISTENT must be OK")
	}
	if rep.Authored != 4 {
		t.Fatalf("Authored = %d, want 4", rep.Authored)
	}
	if rep.ProviderCacheReadTokens != 12000 {
		t.Fatalf("ProviderCacheReadTokens = %d, want 12000", rep.ProviderCacheReadTokens)
	}
	if rep.ProviderCacheCreationTokens != 800 {
		t.Fatalf("ProviderCacheCreationTokens = %d, want 800", rep.ProviderCacheCreationTokens)
	}
	// Provenance fence: authoring WITNESSED, provider OBSERVED — never conflated.
	if rep.Provenance["authored"] != Witnessed {
		t.Fatalf("authored provenance = %q, want WITNESSED", rep.Provenance["authored"])
	}
	if rep.Provenance["provider_cache_read_tokens"] != Observed {
		t.Fatalf("provider read provenance = %q, want OBSERVED", rep.Provenance["provider_cache_read_tokens"])
	}
}

func TestSelfCheckManagedCachePlacedAndUpgradedCountsAsAuthored(t *testing.T) {
	body := ttlLine(ttlOutcomePlacedAndUpgraded, 3) +
		"fak_gateway_inference_cached_prompt_tokens_total 500\n"
	rep := SelfCheckManagedCache("gw", body)
	if rep.Verdict != ManagedCacheConsistent {
		t.Fatalf("verdict = %q, want CONSISTENT (%s)", rep.Verdict, rep.Finding)
	}
	if rep.Authored != 3 || rep.PlacedAndUpgraded != 3 {
		t.Fatalf("Authored=%d PlacedAndUpgraded=%d, want 3/3", rep.Authored, rep.PlacedAndUpgraded)
	}
}

func TestSelfCheckManagedCacheSuspectWhenNoProviderReuse(t *testing.T) {
	// fak authored upgrades but the provider served ZERO cache_read — the actionable
	// witnessed-authoring-without-observed-effect shape. OK (not fak's fault) but flagged.
	body := ttlLine(ttlOutcomeUpgraded, 5) +
		"fak_gateway_inference_cached_prompt_tokens_total 0\n"
	rep := SelfCheckManagedCache("gw", body)
	if rep.Verdict != ManagedCacheSuspect {
		t.Fatalf("verdict = %q, want SUSPECT (%s)", rep.Verdict, rep.Finding)
	}
	if !rep.OK {
		t.Fatal("SUSPECT falls open (OK) — a missing provider effect is not attributable to fak")
	}
	if rep.Authored != 5 {
		t.Fatalf("Authored = %d, want 5", rep.Authored)
	}
}

func TestSelfCheckManagedCacheBrokenOnSpliceFailed(t *testing.T) {
	// A fak-bug outcome fails HARD even when some upgrades also succeeded.
	body := ttlLine(ttlOutcomeUpgraded, 10) +
		ttlLine(ttlOutcomeSpliceFailed, 2) +
		"fak_gateway_inference_cached_prompt_tokens_total 9000\n"
	rep := SelfCheckManagedCache("gw", body)
	if rep.Verdict != ManagedCacheBroken {
		t.Fatalf("verdict = %q, want BROKEN (%s)", rep.Verdict, rep.Finding)
	}
	if rep.OK {
		t.Fatal("BROKEN is the one FAILING verdict — OK must be false")
	}
	if rep.CorruptionOutcomes != 2 {
		t.Fatalf("CorruptionOutcomes = %d, want 2", rep.CorruptionOutcomes)
	}
}

func TestSelfCheckManagedCacheBrokenOnRedecodeFailed(t *testing.T) {
	body := ttlLine(ttlOutcomeRedecodeFailed, 1)
	rep := SelfCheckManagedCache("gw", body)
	if rep.Verdict != ManagedCacheBroken || rep.OK {
		t.Fatalf("redecode_failed>0 must be BROKEN/not-OK, got %q OK=%t (%s)", rep.Verdict, rep.OK, rep.Finding)
	}
	if rep.CorruptionOutcomes != 1 {
		t.Fatalf("CorruptionOutcomes = %d, want 1", rep.CorruptionOutcomes)
	}
}

func TestSelfCheckManagedCacheNotEngagedNamesDominantBail(t *testing.T) {
	// Active lever, zero authored — every attempt bailed. The dominant NON-bug reason is named.
	body := ttlLine(ttlOutcomeUpgraded, 0) +
		ttlLine("no_stable_breakpoint", 7) +
		ttlLine("volatile_head", 2) +
		"fak_gateway_inference_cached_prompt_tokens_total 100\n"
	rep := SelfCheckManagedCache("gw", body)
	if rep.Verdict != ManagedCacheNotEngaged {
		t.Fatalf("verdict = %q, want NOT_ENGAGED (%s)", rep.Verdict, rep.Finding)
	}
	if !rep.OK {
		t.Fatal("NOT_ENGAGED falls open (OK)")
	}
	if rep.DominantBailReason != "no_stable_breakpoint" {
		t.Fatalf("DominantBailReason = %q, want no_stable_breakpoint", rep.DominantBailReason)
	}
	if rep.Authored != 0 {
		t.Fatalf("Authored = %d, want 0", rep.Authored)
	}
}

func TestSelfCheckManagedCacheBugBailAloneIsBrokenNotNotEngaged(t *testing.T) {
	// Authored 0 but the only bail is a fak bug → BROKEN wins over NOT_ENGAGED.
	body := ttlLine(ttlOutcomeUpgraded, 0) + ttlLine(ttlOutcomeSpliceFailed, 1)
	rep := SelfCheckManagedCache("gw", body)
	if rep.Verdict != ManagedCacheBroken || rep.OK {
		t.Fatalf("splice_failed alone must be BROKEN/not-OK, got %q OK=%t (%s)", rep.Verdict, rep.OK, rep.Finding)
	}
}

func TestSelfCheckManagedCacheInsufficientWhenLeverPassive(t *testing.T) {
	// No TTL-upgrade family at all (lever passive/off) — fall open, never a failure.
	body := "fak_gateway_inference_cached_prompt_tokens_total 4000\n" +
		"fak_gateway_kv_prefix_turns_total 10\n"
	rep := SelfCheckManagedCache("gw", body)
	if rep.Verdict != ManagedCacheInsufficient {
		t.Fatalf("verdict = %q, want INSUFFICIENT (%s)", rep.Verdict, rep.Finding)
	}
	if !rep.OK {
		t.Fatal("INSUFFICIENT falls open (OK)")
	}
}

func TestSelfCheckManagedCacheInsufficientWhenFamilyAllZero(t *testing.T) {
	// The family is present (active lever emits the upgraded row even at 0) but recorded no
	// attempt — nothing to reconcile, fall open rather than manufacturing a verdict.
	body := ttlLine(ttlOutcomeUpgraded, 0) +
		"fak_gateway_inference_cached_prompt_tokens_total 4000\n"
	rep := SelfCheckManagedCache("gw", body)
	if rep.Verdict != ManagedCacheInsufficient {
		t.Fatalf("verdict = %q, want INSUFFICIENT (%s)", rep.Verdict, rep.Finding)
	}
	if !rep.OK {
		t.Fatal("all-zero family falls open (OK)")
	}
}

func TestSelfCheckManagedCacheBailsCarryThrough(t *testing.T) {
	body := ttlLine(ttlOutcomeUpgraded, 2) +
		ttlLine("already_1h", 3) +
		"fak_gateway_inference_cached_prompt_tokens_total 6000\n"
	rep := SelfCheckManagedCache("gw", body)
	if rep.Bails["already_1h"] != 3 {
		t.Fatalf("Bails did not carry through: %v", rep.Bails)
	}
	// already_1h is not a bug and not a NOT_ENGAGED trigger here (upgrades exist), but it is
	// recorded for the operator.
	if rep.Verdict != ManagedCacheConsistent {
		t.Fatalf("verdict = %q, want CONSISTENT (%s)", rep.Verdict, rep.Finding)
	}
}
