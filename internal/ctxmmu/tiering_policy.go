// tiering_policy.go — the HiCache-style host-tier pool sizing and write-policy contract
// for ctxmmu's device→host KV tiering.
//
// Adapted from sgl-project/sglang (Apache-2.0), pinned at
// 3c6a6c8068658b88e40bcaa5a0bf48b9a94ebdba:
//   - python/sglang/srt/arg_groups/fields/memory.py (hicache_ratio, hicache_size,
//     hicache_write_policy field definitions and defaults)
//   - python/sglang/srt/arg_groups/hicache_hook.py (handle_hicache_ratio_default,
//     validate_hicache_host_memory_mode resolution rules)
//
// The upstream contract this re-expresses:
//   - the host pool is sized by a RATIO against the device pool, not a fixed size:
//     default 2.0 in cache mode, 1.2 in buffer_only mode; an explicit ratio or a
//     positive fixed size overrides the default (fixed size wins over ratio);
//   - three write policies — write_back, write_through, write_through_selective —
//     defaulting to write_through;
//   - buffer_only has no persistent host tier to write back to, so a requested
//     write_back resolves to write_through rather than erroring;
//   - a host pool that ends up with neither a fixed size nor a positive ratio is
//     refused instead of silently built empty.
//
// Scope (honest, matching the sibling policy primitives' own): a standalone policy +
// sizing primitive, OFF the live serve path. It moves no bytes and wires no storage
// backend; the L3 storage tier behind it stays outside this package.

package ctxmmu

import (
	"errors"
	"fmt"
	"math"
)

var (
	// ErrHiCacheUnknownWritePolicy is returned when a write-policy string names none
	// of the recognized HiCache policies.
	ErrHiCacheUnknownWritePolicy = errors.New("ctxmmu: unknown hicache write policy")

	// ErrHiCacheInvalidHostMode is returned when the host memory mode is neither
	// cache nor buffer_only.
	ErrHiCacheInvalidHostMode = errors.New("ctxmmu: invalid hicache host memory mode")

	// ErrHiCacheInvalidRatio is returned when an explicitly set ratio is not a
	// finite positive number. A zero ratio means "unset", which is legal and
	// resolves to the mode default.
	ErrHiCacheInvalidRatio = errors.New("ctxmmu: hicache ratio must be positive when set")

	// ErrHiCachePoolUnsized is returned when a resolved tier carries neither a
	// fixed byte size nor a positive ratio, i.e. the host pool would be empty.
	// ResolveHiCacheTier cannot produce such a tier; the guard catches hand-built
	// ResolvedHiCacheTier values so an unsized pool is never silently allocated.
	ErrHiCachePoolUnsized = errors.New("ctxmmu: hicache host pool has neither a fixed size nor a ratio")
)

// HiCacheWritePolicy names how updated KV blocks reach the host tier.
type HiCacheWritePolicy string

const (
	// HiCacheWriteThrough writes every updated block to the host tier immediately.
	// It is the default policy in both host modes.
	HiCacheWriteThrough HiCacheWritePolicy = "write_through"
	// HiCacheWriteBack defers the host write until the block is evicted or
	// reclaimed from the device tier.
	HiCacheWriteBack HiCacheWritePolicy = "write_back"
	// HiCacheWriteThroughSelective writes through only the blocks a retention
	// signal selects, trading coverage for transfer cost.
	HiCacheWriteThroughSelective HiCacheWritePolicy = "write_through_selective"
)

// IsValid reports whether p is one of the recognized write policies.
func (p HiCacheWritePolicy) IsValid() bool {
	switch p {
	case HiCacheWriteThrough, HiCacheWriteBack, HiCacheWriteThroughSelective:
		return true
	default:
		return false
	}
}

// ParseHiCacheWritePolicy maps the wire spelling of a write policy onto its
// constant. An unrecognized string is refused with ErrHiCacheUnknownWritePolicy
// (wrapped, so errors.Is still matches the sentinel).
func ParseHiCacheWritePolicy(s string) (HiCacheWritePolicy, error) {
	p := HiCacheWritePolicy(s)
	if !p.IsValid() {
		return "", fmt.Errorf("%w: %q", ErrHiCacheUnknownWritePolicy, s)
	}
	return p, nil
}

// HiCacheHostMode names the role of the host memory pool.
type HiCacheHostMode string

const (
	// HiCacheHostModeCache makes the host pool a PERSISTENT tier: retained blocks
	// survive as long as they are not evicted. It is the default mode.
	HiCacheHostModeCache HiCacheHostMode = "cache"
	// HiCacheHostModeBufferOnly makes the host pool a TRANSIENT staging buffer
	// between the device tier and a storage backend; nothing is retained there.
	HiCacheHostModeBufferOnly HiCacheHostMode = "buffer_only"
)

// IsValid reports whether m is one of the recognized host modes. The zero value
// ("") is not valid on its own; options-level resolution treats it as cache.
func (m HiCacheHostMode) IsValid() bool {
	switch m {
	case HiCacheHostModeCache, HiCacheHostModeBufferOnly:
		return true
	default:
		return false
	}
}

// Per-mode default ratios. Upstream defaults the host/device ratio per host
// memory mode: buffer_only stages in flight rather than retaining, so it needs
// only enough to cover the write backlog plus parked prefetches.
const (
	// DefaultHiCacheRatio sizes the host pool at twice the device pool in cache mode.
	DefaultHiCacheRatio = 2.0
	// DefaultBufferOnlyHiCacheRatio sizes the host pool at 1.2x the device pool in
	// buffer_only mode.
	DefaultBufferOnlyHiCacheRatio = 1.2
)

// HiCacheTierOptions declares how a host tier should be sized and written. Zero
// values carry meaning: Mode "" resolves to cache, Ratio 0 to the mode default,
// and WritePolicy "" to write_through.
type HiCacheTierOptions struct {
	// Mode selects the host pool role. The zero value resolves to HiCacheHostModeCache.
	Mode HiCacheHostMode
	// Ratio is the host-to-device pool size ratio. Zero means "use the mode
	// default"; any other value must be a finite positive number.
	Ratio float64
	// FixedBytes is a fixed host pool size in bytes. When > 0 it OVERRIDES Ratio
	// in either mode; the ratio is still resolved and reported but not used for
	// sizing.
	FixedBytes int64
	// WritePolicy selects the write policy. The zero value resolves to
	// HiCacheWriteThrough.
	WritePolicy HiCacheWritePolicy
}

// ResolvedHiCacheTier is the validated, fully-defaulted form of
// HiCacheTierOptions: every field is concrete, so consumers never re-apply
// defaulting.
type ResolvedHiCacheTier struct {
	// Mode is always a valid host mode.
	Mode HiCacheHostMode
	// Ratio is always a finite positive number (the explicit one, or the mode
	// default). It is informational when SizedByFixed is true.
	Ratio float64
	// FixedBytes is > 0 exactly when the caller supplied a fixed size.
	FixedBytes int64
	// SizedByFixed reports whether FixedBytes, not Ratio, drives HostPool sizing.
	SizedByFixed bool
	// WritePolicy is always one of the three recognized policies.
	WritePolicy HiCacheWritePolicy
}

// ResolveHiCacheTier applies the upstream defaulting and lifecycle guards to o:
//
//  1. Mode "" becomes cache; any other invalid mode is refused.
//  2. Ratio 0 becomes the mode default (2.0 cache, 1.2 buffer_only); a negative
//     or non-finite ratio is refused.
//  3. FixedBytes > 0 marks the tier sized-by-fixed, overriding the ratio.
//  4. WritePolicy "" becomes write_through; an unrecognized policy is refused.
//  5. In buffer_only mode a requested write_back RESOLVES to write_through:
//     there is no persistent host tier to write back to, pages go to storage.
func ResolveHiCacheTier(o HiCacheTierOptions) (ResolvedHiCacheTier, error) {
	mode := o.Mode
	if mode == "" {
		mode = HiCacheHostModeCache
	}
	if !mode.IsValid() {
		return ResolvedHiCacheTier{}, fmt.Errorf("%w: %q", ErrHiCacheInvalidHostMode, mode)
	}

	ratio := o.Ratio
	if ratio == 0 {
		if mode == HiCacheHostModeBufferOnly {
			ratio = DefaultBufferOnlyHiCacheRatio
		} else {
			ratio = DefaultHiCacheRatio
		}
	}
	if math.IsNaN(ratio) || math.IsInf(ratio, 0) || ratio < 0 {
		return ResolvedHiCacheTier{}, fmt.Errorf("%w: %v", ErrHiCacheInvalidRatio, ratio)
	}

	writePolicy := o.WritePolicy
	if writePolicy == "" {
		writePolicy = HiCacheWriteThrough
	}
	if !writePolicy.IsValid() {
		return ResolvedHiCacheTier{}, fmt.Errorf("%w: %q", ErrHiCacheUnknownWritePolicy, writePolicy)
	}
	if mode == HiCacheHostModeBufferOnly && writePolicy == HiCacheWriteBack {
		// buffer_only has no host tier to write back to; resolve to write_through
		// instead of refusing the configuration.
		writePolicy = HiCacheWriteThrough
	}

	sizedByFixed := o.FixedBytes > 0
	return ResolvedHiCacheTier{
		Mode:         mode,
		Ratio:        ratio,
		FixedBytes:   o.FixedBytes,
		SizedByFixed: sizedByFixed,
		WritePolicy:  writePolicy,
	}, nil
}

// HostPoolBytes returns the host pool byte budget implied by t for a device pool
// of deviceBytes bytes. A sized-by-fixed tier ignores the device size entirely;
// otherwise the budget is deviceBytes × Ratio rounded half away from zero.
// deviceBytes must be >= 0.
func (t ResolvedHiCacheTier) HostPoolBytes(deviceBytes int64) (int64, error) {
	if t.SizedByFixed {
		return t.FixedBytes, nil
	}
	if !(t.Ratio > 0) {
		return 0, ErrHiCachePoolUnsized
	}
	if deviceBytes < 0 {
		deviceBytes = 0
	}
	return int64(math.Round(float64(deviceBytes) * t.Ratio)), nil
}

// HostPoolTokens is the token-unit form of HostPoolBytes for callers that count
// the device pool in tokens: it returns round(deviceTokens × Ratio), or the
// fixed size expressed in tokens when the tier is sized by fixed bytes
// (FixedBytes / bytesPerTok, truncated). deviceTokens must be >= 0.
func (t ResolvedHiCacheTier) HostPoolTokens(deviceTokens int, bytesPerTok int) (int, error) {
	if bytesPerTok <= 0 {
		bytesPerTok = 1
	}
	if t.SizedByFixed {
		return int(t.FixedBytes / int64(bytesPerTok)), nil
	}
	if !(t.Ratio > 0) {
		return 0, ErrHiCachePoolUnsized
	}
	if deviceTokens < 0 {
		deviceTokens = 0
	}
	return int(math.Round(float64(deviceTokens) * t.Ratio)), nil
}
