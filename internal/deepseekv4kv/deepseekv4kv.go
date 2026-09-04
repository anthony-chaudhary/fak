// Package deepseekv4kv is a pure, weight-free block-accounting fixture for the
// DeepSeek V4 heterogeneous KV plane and its on-disk prefix-reuse policies.
//
// It is the witness named by docs/notes/DEEPSEEK-V4-KV-LAYOUT-PLAN-2026-07-08.md
// (issue #3017, epic #3006). It loads no model, serves no context, and reuses no
// provider cache counter. It reasons in NORMALIZED units — one dense (uncompressed)
// KV entry, per token, per layer, is 1.0 — so the absolute per-entry byte size (which
// depends on unpublished V4 head dims) cancels out and is never fabricated. What the
// fixture asserts is the published, witnessed specifications:
//
//   - CSA is a rate-4 compressed KV block  (0.25 units/token/layer).
//   - HCA is a rate-128 compressed KV block (1/128 units/token/layer).
//   - SWA storage is bounded by a 128-token window, independent of sequence length.
//   - The uncompressed tail is dense (1.0 units/token/layer).
//
// and the storage / write / read / recompute amplification the three SWA on-disk
// policies trade off at 128K, 512K, and 1M contexts.
package deepseekv4kv

import (
	"fmt"
	"strings"
)

// Published V4 Pro KV constants (DeepSeek V4 technical report; carried in the
// attention seam map #3016 and the KV-layout plan #3017). These are the driving
// facts the block accounting validates against — not modeled guesses.
const (
	CSARate   = 4   // Compressed Sparse Attention: keep 1 KV entry's worth per 4 tokens.
	HCARate   = 128 // Hierarchical Compressed Attention: rate-128 KV block.
	SWAWindow = 128 // Sliding Window Attention window, in tokens (state is bounded to this).
)

// Nominal context sizes the amplification report covers (tokens).
const (
	Ctx128K = 128 * 1024
	Ctx512K = 512 * 1024
	Ctx1M   = 1024 * 1024
)

// ReportContexts is the fixed set of contexts the ticket's report must cover.
var ReportContexts = []int{Ctx128K, Ctx512K, Ctx1M}

// Kind is one of the four typed sub-caches that coexist in a single V4 forward pass.
type Kind int

const (
	// KindCSA identifies compressed rate-4 classical attention cache blocks.
	KindCSA Kind = iota
	// KindHCA identifies hierarchical compressed rate-128 attention cache blocks.
	KindHCA
	// KindSWA identifies sliding-window attention side state bounded to SWAWindow tokens.
	KindSWA
	// KindTail identifies uncompressed dense recent tokens.
	KindTail
)

// String returns the canonical human-readable abbreviation for the cache kind.
func (k Kind) String() string {
	switch k {
	case KindCSA:
		return "CSA"
	case KindHCA:
		return "HCA"
	case KindSWA:
		return "SWA"
	case KindTail:
		return "tail"
	default:
		return "?"
	}
}

// CompressionRatio reports how many dense KV entries one stored entry of the given
// per-token compressed kind stands in for. SWA is not a per-token compression (its
// storage is window-bounded, not rate-scaled), so it has no ratio here.
func CompressionRatio(k Kind) (ratio int, ok bool) {
	switch k {
	case KindCSA:
		return CSARate, true
	case KindHCA:
		return HCARate, true
	case KindTail:
		return 1, true // dense: one stored entry per token
	default:
		return 0, false // SWA: window-bounded, use SubCacheUnits
	}
}

// SubCacheUnits returns the normalized KV storage a sub-cache of the given kind holds
// for a prefix of seq tokens, per layer, in units of one dense KV entry. This is the
// block-accounting core:
//
//   - CSA  = seq / 4          (rate-4 compressed)
//   - HCA  = seq / 128        (rate-128 compressed)
//   - tail = seq              (dense)
//   - SWA  = min(seq, 128)    (window-bounded — the load-bearing invariant)
func SubCacheUnits(k Kind, seq int) float64 {
	if seq < 0 {
		seq = 0
	}
	switch k {
	case KindCSA:
		return float64(seq) / CSARate
	case KindHCA:
		return float64(seq) / HCARate
	case KindTail:
		return float64(seq)
	case KindSWA:
		return float64(minInt(seq, SWAWindow))
	default:
		return 0
	}
}

// MandatoryDurableUnits is the compressed KV that MUST be durable on disk to enable a
// prefix hit: the CSA + HCA blocks, per layer. It is the denominator the amplification
// figures are reported against (amplification "per mandatory compressed-KV byte").
func MandatoryDurableUnits(seq int) float64 {
	return SubCacheUnits(KindCSA, seq) + SubCacheUnits(KindHCA, seq)
}

// Policy is one of the three on-disk SWA-cache policies the ticket asks to separate.
type Policy int

const (
	// FullSWACache stores the sliding window at window cadence, eliminating recompute on hits.
	FullSWACache Policy = iota
	// PeriodicCheckpoint persists sliding window state every N tokens, recomputing trailing tokens.
	PeriodicCheckpoint
	// ZeroSWACache retains no durable sliding window state, recomputing the entire window on hits.
	ZeroSWACache
)

// String returns the descriptive identifier for the on-disk sliding window policy.
func (p Policy) String() string {
	switch p {
	case FullSWACache:
		return "full-swa-cache"
	case PeriodicCheckpoint:
		return "periodic-checkpoint"
	case ZeroSWACache:
		return "zero-swa-cache"
	default:
		return "?"
	}
}

// Amp is the amplification of one (context, policy) point, per layer, in normalized
// units. Storage/Write/Read are reported as multiples of MandatoryDurableUnits (so
// they are >= 1.0 and read as "per compressed-KV byte"); RecomputeTokens is the tail
// length rebuilt on a prefix hit (absolute tokens, since it is compute not bytes).
type Amp struct {
	Context         int
	Policy          Policy
	StorageAmp      float64 // durable on-disk bytes / mandatory compressed KV
	WriteAmp        float64 // bytes written over the run / mandatory compressed KV
	ReadAmp         float64 // bytes read from disk on a hit / mandatory compressed KV
	RecomputeTokens int     // SWA tail tokens recomputed on a hit
}

// Amplify computes the amplification for one context/policy. checkpointEvery is only
// consulted for PeriodicCheckpoint (tokens between SWA-state checkpoints); it is
// clamped to >= 1. The model is deliberately simple and monotonic, and it surfaces the
// real V4 finding: because the SWA window is only 128 tokens, storage barely differs
// across policies (SWA is O(window), not O(seq)) — the tradeoff lives in WRITE (full
// pays a large write cost) versus RECOMPUTE (zero/periodic rebuild the 128-token tail).
func Amplify(seq int, p Policy, checkpointEvery int) Amp {
	base := MandatoryDurableUnits(seq)
	if base <= 0 {
		return Amp{Context: seq, Policy: p, StorageAmp: 1, WriteAmp: 1, ReadAmp: 1}
	}
	if checkpointEvery < 1 {
		checkpointEvery = 1
	}
	window := float64(SWAWindow)

	var storedSWA, writtenSWA, readSWA float64
	var recompute int
	switch p {
	case FullSWACache:
		// Resident SWA window is bounded; it is (re)written at a window-cadence, so
		// the whole run writes seq/SWAWindow windows == seq units. Zero recompute.
		storedSWA = window
		writtenSWA = float64(seq) // (seq/SWAWindow) writes * SWAWindow units
		readSWA = window
		recompute = 0
	case PeriodicCheckpoint:
		// One bounded checkpoint window is resident; it is written every N tokens.
		// On a hit, restore the checkpoint and rebuild the tail — but SWA state only
		// depends on the last SWAWindow tokens, so recompute is min(N, SWAWindow).
		storedSWA = window
		writes := float64(seq) / float64(checkpointEvery)
		writtenSWA = writes * window
		readSWA = window
		recompute = minInt(checkpointEvery, SWAWindow)
	case ZeroSWACache:
		// Nothing durable for SWA; on a hit, recompute the whole 128-token window.
		storedSWA = 0
		writtenSWA = 0
		readSWA = 0
		recompute = SWAWindow
	}

	return Amp{
		Context:         seq,
		Policy:          p,
		StorageAmp:      (base + storedSWA) / base,
		WriteAmp:        (base + writtenSWA) / base,
		ReadAmp:         (base + readSWA) / base,
		RecomputeTokens: recompute,
	}
}

// ValidateBlockAccounting is the fail-closed self-check: it recomputes the CSA/HCA/tail
// compression ratios and the SWA window bound from SubCacheUnits and returns a non-nil
// error the instant any diverges from the published constants. A caller that trusts the
// report must call this first — a mismatch is a bug in the accounting, not a warning.
func ValidateBlockAccounting() error {
	const probe = 1 << 20 // 1M tokens: large enough that SWA is fully saturated.

	if got, want := SubCacheUnits(KindCSA, probe), float64(probe)/CSARate; got != want {
		return fmt.Errorf("deepseekv4kv: CSA accounting drift: got %g units, want %g (rate-%d)", got, want, CSARate)
	}
	if got, want := SubCacheUnits(KindHCA, probe), float64(probe)/HCARate; got != want {
		return fmt.Errorf("deepseekv4kv: HCA accounting drift: got %g units, want %g (rate-%d)", got, want, HCARate)
	}
	if got, want := SubCacheUnits(KindTail, probe), float64(probe); got != want {
		return fmt.Errorf("deepseekv4kv: tail must be dense: got %g units, want %g", got, want)
	}
	// SWA must be window-bounded: saturated above the window, exact below it.
	if got := SubCacheUnits(KindSWA, probe); got != float64(SWAWindow) {
		return fmt.Errorf("deepseekv4kv: SWA not window-bounded at seq=%d: got %g units, want %d", probe, got, SWAWindow)
	}
	if got := SubCacheUnits(KindSWA, SWAWindow/2); got != float64(SWAWindow/2) {
		return fmt.Errorf("deepseekv4kv: SWA below window must be exact: got %g, want %d", got, SWAWindow/2)
	}
	// Ratios must match the declared constants.
	if r, _ := CompressionRatio(KindCSA); r != CSARate {
		return fmt.Errorf("deepseekv4kv: CSA ratio %d != %d", r, CSARate)
	}
	if r, _ := CompressionRatio(KindHCA); r != HCARate {
		return fmt.Errorf("deepseekv4kv: HCA ratio %d != %d", r, HCARate)
	}
	return nil
}

// Report returns one Amp per (context, policy) over ReportContexts, using
// checkpointEvery for the periodic policy. The result is the storage/write/read
// amplification table the ticket's done-condition asks for (128K / 512K / 1M).
func Report(checkpointEvery int) []Amp {
	policies := []Policy{FullSWACache, PeriodicCheckpoint, ZeroSWACache}
	out := make([]Amp, 0, len(ReportContexts)*len(policies))
	for _, ctx := range ReportContexts {
		for _, p := range policies {
			out = append(out, Amplify(ctx, p, checkpointEvery))
		}
	}
	return out
}

// FormatReport renders Report as a fixed-width table for a runbook or CI log.
func FormatReport(checkpointEvery int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "DeepSeek V4 KV on-disk amplification (normalized units, per layer; checkpoint every %d tok)\n", checkpointEvery)
	fmt.Fprintf(&b, "%-8s %-20s %10s %10s %10s %12s\n", "context", "policy", "storage×", "write×", "read×", "recompute")
	for _, a := range Report(checkpointEvery) {
		fmt.Fprintf(&b, "%-8s %-20s %10.4f %10.4f %10.4f %12d\n",
			ctxLabel(a.Context), a.Policy, a.StorageAmp, a.WriteAmp, a.ReadAmp, a.RecomputeTokens)
	}
	return b.String()
}

func ctxLabel(seq int) string {
	switch seq {
	case Ctx128K:
		return "128K"
	case Ctx512K:
		return "512K"
	case Ctx1M:
		return "1M"
	default:
		return fmt.Sprintf("%d", seq)
	}
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
