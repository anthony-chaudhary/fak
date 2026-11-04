package ctxmmu

import (
	"errors"
	"testing"
)

// TestHiCacheRatioPolicy pins the borrowed SGLang HiCache host-pool lifecycle
// contract: ratio sizing arithmetic (default 2.0 cache / 1.2 buffer_only, fixed
// size overriding ratio) and the default write policy (write_through, three
// recognized policies, buffer_only resolving write_back to write_through).
func TestHiCacheRatioPolicy(t *testing.T) {
	t.Run("default_ratio_cache_mode", func(t *testing.T) {
		r, err := ResolveHiCacheTier(HiCacheTierOptions{})
		if err != nil {
			t.Fatalf("ResolveHiCacheTier(zero): %v", err)
		}
		if r.Mode != HiCacheHostModeCache {
			t.Errorf("zero options mode = %q, want %q", r.Mode, HiCacheHostModeCache)
		}
		if r.Ratio != DefaultHiCacheRatio || DefaultHiCacheRatio != 2.0 {
			t.Errorf("default cache-mode ratio = %v, want %v", r.Ratio, 2.0)
		}
		if r.SizedByFixed {
			t.Error("zero options must not be sized by fixed")
		}
		if got, err := r.HostPoolTokens(1_000_000, 1); err != nil || got != 2_000_000 {
			t.Errorf("HostPoolTokens(1e6) = %d, %v; want 2e6, nil", got, err)
		}
	})

	t.Run("default_ratio_buffer_only", func(t *testing.T) {
		r, err := ResolveHiCacheTier(HiCacheTierOptions{Mode: HiCacheHostModeBufferOnly})
		if err != nil {
			t.Fatalf("ResolveHiCacheTier(buffer_only): %v", err)
		}
		if r.Ratio != DefaultBufferOnlyHiCacheRatio || DefaultBufferOnlyHiCacheRatio != 1.2 {
			t.Errorf("buffer_only default ratio = %v, want %v", r.Ratio, 1.2)
		}
		if got, err := r.HostPoolTokens(1_000_000, 1); err != nil || got != 1_200_000 {
			t.Errorf("HostPoolTokens(1e6) = %d, %v; want 1.2e6, nil", got, err)
		}
	})

	t.Run("explicit_ratio_wins_over_default", func(t *testing.T) {
		r, err := ResolveHiCacheTier(HiCacheTierOptions{Ratio: 3.5})
		if err != nil {
			t.Fatalf("ResolveHiCacheTier(ratio=3.5): %v", err)
		}
		if r.Ratio != 3.5 {
			t.Errorf("ratio = %v, want 3.5", r.Ratio)
		}
		if got, _ := r.HostPoolTokens(1000, 1); got != 3500 {
			t.Errorf("HostPoolTokens(1000) = %d, want 3500", got)
		}
	})

	t.Run("fixed_size_overrides_ratio", func(t *testing.T) {
		fixed := int64(4 << 30) // 4 GiB
		r, err := ResolveHiCacheTier(HiCacheTierOptions{FixedBytes: fixed, Ratio: 9})
		if err != nil {
			t.Fatalf("ResolveHiCacheTier(fixed): %v", err)
		}
		if !r.SizedByFixed {
			t.Fatal("fixed bytes must mark the tier sized-by-fixed")
		}
		// The explicit ratio is still resolved and reported...
		if r.Ratio != 9 {
			t.Errorf("reported ratio = %v, want 9", r.Ratio)
		}
		// ...but the device size must not influence the host budget.
		for _, dev := range []int64{0, 1 << 30, 1 << 40} {
			if got, err := r.HostPoolBytes(dev); err != nil || got != fixed {
				t.Errorf("HostPoolBytes(%d) = %d, %v; want %d, nil", dev, got, err, fixed)
			}
		}
	})

	t.Run("ratio_sizing_rounds_half_away_from_zero", func(t *testing.T) {
		r, err := ResolveHiCacheTier(HiCacheTierOptions{Mode: HiCacheHostModeBufferOnly})
		if err != nil {
			t.Fatalf("ResolveHiCacheTier: %v", err)
		}
		// 101 * 1.2 = 121.2 -> 121; 102 * 1.2 = 122.4 -> 122.
		if got, _ := r.HostPoolTokens(101, 1); got != 121 {
			t.Errorf("HostPoolTokens(101) = %d, want 121", got)
		}
		if got, _ := r.HostPoolTokens(102, 1); got != 122 {
			t.Errorf("HostPoolTokens(102) = %d, want 122", got)
		}
	})

	t.Run("default_write_policy_is_write_through", func(t *testing.T) {
		for _, opts := range []HiCacheTierOptions{{}, {Mode: HiCacheHostModeBufferOnly}} {
			r, err := ResolveHiCacheTier(opts)
			if err != nil {
				t.Fatalf("ResolveHiCacheTier(%+v): %v", opts, err)
			}
			if r.WritePolicy != HiCacheWriteThrough {
				t.Errorf("zero WritePolicy resolved to %q, want %q", r.WritePolicy, HiCacheWriteThrough)
			}
		}
	})

	t.Run("all_three_write_policies_parse", func(t *testing.T) {
		want := map[string]HiCacheWritePolicy{
			"write_back":              HiCacheWriteBack,
			"write_through":           HiCacheWriteThrough,
			"write_through_selective": HiCacheWriteThroughSelective,
		}
		for s, p := range want {
			got, err := ParseHiCacheWritePolicy(s)
			if err != nil {
				t.Errorf("ParseHiCacheWritePolicy(%q) error: %v", s, err)
				continue
			}
			if got != p {
				t.Errorf("ParseHiCacheWritePolicy(%q) = %q, want %q", s, got, p)
			}
			if !got.IsValid() {
				t.Errorf("%q must report IsValid", got)
			}
		}
	})

	t.Run("unknown_write_policy_refused_with_sentinel", func(t *testing.T) {
		if _, err := ParseHiCacheWritePolicy("flush_always"); !errors.Is(err, ErrHiCacheUnknownWritePolicy) {
			t.Errorf("ParseHiCacheWritePolicy(flush_always) err = %v, want errors.Is %v", err, ErrHiCacheUnknownWritePolicy)
		}
		if _, err := ResolveHiCacheTier(HiCacheTierOptions{WritePolicy: "flush_always"}); !errors.Is(err, ErrHiCacheUnknownWritePolicy) {
			t.Errorf("ResolveHiCacheTier(bad policy) err = %v, want errors.Is %v", err, ErrHiCacheUnknownWritePolicy)
		}
	})

	t.Run("buffer_only_write_back_resolves_to_write_through", func(t *testing.T) {
		r, err := ResolveHiCacheTier(HiCacheTierOptions{Mode: HiCacheHostModeBufferOnly, WritePolicy: HiCacheWriteBack})
		if err != nil {
			t.Fatalf("ResolveHiCacheTier(buffer_only + write_back): %v", err)
		}
		if r.WritePolicy != HiCacheWriteThrough {
			t.Errorf("buffer_only write_back resolved to %q, want %q", r.WritePolicy, HiCacheWriteThrough)
		}
		// cache mode keeps write_back as requested.
		c, err := ResolveHiCacheTier(HiCacheTierOptions{WritePolicy: HiCacheWriteBack})
		if err != nil {
			t.Fatalf("ResolveHiCacheTier(cache + write_back): %v", err)
		}
		if c.WritePolicy != HiCacheWriteBack {
			t.Errorf("cache write_back resolved to %q, want %q", c.WritePolicy, HiCacheWriteBack)
		}
	})

	t.Run("invalid_inputs_refused_with_sentinels", func(t *testing.T) {
		if _, err := ResolveHiCacheTier(HiCacheTierOptions{Mode: "ramdisk"}); !errors.Is(err, ErrHiCacheInvalidHostMode) {
			t.Errorf("bad mode err = %v, want errors.Is %v", err, ErrHiCacheInvalidHostMode)
		}
		if _, err := ResolveHiCacheTier(HiCacheTierOptions{Ratio: -1}); !errors.Is(err, ErrHiCacheInvalidRatio) {
			t.Errorf("negative ratio err = %v, want errors.Is %v", err, ErrHiCacheInvalidRatio)
		}
	})

	t.Run("unsized_pool_never_allocated", func(t *testing.T) {
		unsized := ResolvedHiCacheTier{} // hand-built: no fixed size, no positive ratio
		if _, err := unsized.HostPoolBytes(1 << 20); !errors.Is(err, ErrHiCachePoolUnsized) {
			t.Errorf("HostPoolBytes(unsized) err = %v, want errors.Is %v", err, ErrHiCachePoolUnsized)
		}
		if _, err := unsized.HostPoolTokens(1000, 1); !errors.Is(err, ErrHiCachePoolUnsized) {
			t.Errorf("HostPoolTokens(unsized) err = %v, want errors.Is %v", err, ErrHiCachePoolUnsized)
		}
	})
}
