package l3kv

// warmresume.go makes scale-to-zero cache-AWARE (#2853, Track D leaf of #2834). A
// serverless hibernation backend (Hermes' Modal/Daytona) snapshots the agent's
// filesystem/process on going-idle and wakes it on demand — "nearly nothing when
// idle" — but the PROVIDER-side KV cache is cold on wake, so the first post-hibernate
// turn re-pays the whole prefix. fak owns the KV cache, so it can deliver the second
// half only a kernel can: WARM on wake.
//
// This leaf adds the two hooks around l3kv's existing durable residency tier
// (StageSpan / RestoreSpan) plus the witness the issue's done-condition names:
//
//   - PersistPrefix — the GOING-IDLE hook: stage the session's warm-prefix spans
//     off-box and return a small manifest (digests + lengths) to page them back.
//   - RestorePrefix — the WAKE warm-restore hook: page the prefix back from the
//     durable tier and report how much came back warm.
//   - ResumeComparison — the cache-read fraction on the first resumed turn SIDE BY
//     SIDE with a cold-start baseline, so "materially warmer than cold" is checkable.
//
// Honest fences (the two confusion risks #2853 names):
//
//   - "Nearly nothing when idle" vs "full-KV persist." PersistPrefix stages the span
//     BYTES to the durable tier once, but the artifact kept BETWEEN sessions is the
//     PrefixManifest — digests + lengths only, not the bytes. On an FS-snapshotting
//     hibernation backend the staged bytes ride the snapshot for free and only the
//     tiny manifest need be re-read; that is the cheap half. This leaf does NOT
//     implement a hibernation/serverless backend (explicitly out of #2853's scope) —
//     only the KV persist/restore it wires around an existing idle/wake lifecycle.
//   - "Genuinely cold baseline." ColdBaseline models a wake WITHOUT the going-idle
//     hook: nothing staged, the whole prefix re-prefills cold, a cache-read fraction
//     of exactly 0. The warm arm's fraction comes from real RestoreSpan OUTCOMES over
//     the durable tier, never a warm number mislabeled as cold or vice versa.
//
// WITNESSED, not merely observed. A restored span is byte-identical by l3kv's
// integrity guarantee (store.Get re-verifies the sha256), so RestoredPositions is a
// WITNESSED cache_read — the kernel-owned half a provider prompt cache only OBSERVES.
// Landing those bytes back INTO the live decode cache at a fresh position (re-RoPE) is
// the reverse direction #1469 owns; this leaf witnesses the durable-tier warm
// AVAILABILITY that makes the first resumed turn a read instead of a cold write — the
// precondition #1469's re-RoPE consumes.

import (
	"context"
	"errors"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// Span identifies one contiguous run of a session's warm KV prefix by the content
// DIGEST the durable tier addresses it by and the [From, From+Positions) range it
// occupied in the live cache when it was persisted. It is the going-idle input: the
// addressable unit PersistPrefix stages off-box and RestorePrefix pages back on wake.
type Span struct {
	Digest    string // residency digest the durable tier keys the span by
	From      int    // start position in the live cache at persist time
	Positions int    // span length in positions
}

// ManifestSpan is one persisted span's reconstruct-cheaply metadata — the digest to
// page it back by and the positions it covers. It is deliberately NOT the span bytes
// (those live in the durable tier, keyed by Digest); a manifest of these is the small,
// serializable record that "costs nearly nothing" to keep between sessions (#2853).
type ManifestSpan struct {
	Digest    string `json:"digest"`
	Positions int    `json:"positions"`
}

// PrefixManifest is the going-idle artifact: the ordered digests+lengths a wake pages
// the warm prefix back from, plus TotalPositions — the WHOLE prefix the first resumed
// turn re-reads, counting even the spans that FAILED to stage. Keeping the full-prefix
// denominator here is what keeps the wake-time cache-read fraction honest: a span the
// going-idle stage could not durably write is still part of the prefix the resumed turn
// pays for, so it drags the fraction DOWN (cold), never silently vanishes from it.
type PrefixManifest struct {
	// Spans is only the spans that staged OK — the set a wake can page back.
	Spans []ManifestSpan `json:"spans"`
	// TotalPositions is the whole prefix, INCLUDING spans that failed to persist.
	TotalPositions int `json:"total_positions"`
}

// PersistPrefix is the GOING-IDLE hook (#2853): before a session scales to zero it
// stages each warm-prefix span off-box through the durable tier (backend.StageSpan) and
// returns the small PrefixManifest a later wake pages the prefix back from. It is
// fail-safe and honest, mirroring l3kv's residency typing:
//
//   - a span that stages OK is added to the manifest (pageable back on wake);
//   - a span that does not (a FAULT: no byte-source, serialize error, durable-write
//     error, or a per-span RestoreSpan-style transport error) is NOT added — the tier
//     could not durably hold it, so wake cannot page it back — but its positions still
//     count in TotalPositions, so the wake fraction reflects the loss rather than
//     hiding it.
//
// It never persists more than the span bytes the backend already owns: the manifest is
// digests + lengths only. A nil backend is the one hard error (there is nothing to
// stage through); every per-span failure is absorbed into a smaller, still-usable
// manifest so a partial going-idle never loses the spans that DID stage.
func PersistPrefix(ctx context.Context, backend abi.KVBackend, spans []Span) (PrefixManifest, error) {
	if backend == nil {
		return PrefixManifest{}, errors.New("l3kv: PersistPrefix nil backend")
	}
	var m PrefixManifest
	requests := make([]abi.KVResidencyRequest, len(spans))
	for i, sp := range spans {
		m.TotalPositions += sp.Positions
		requests[i] = abi.KVResidencyRequest{Digest: sp.Digest, From: sp.From, Positions: sp.Positions}
	}
	for i, res := range abi.StageSpans(ctx, backend, requests) {
		if res.Outcome != abi.KVResidencyOK {
			continue // not durably held — wake will re-prefill these positions cold.
		}
		sp := spans[i]
		m.Spans = append(m.Spans, ManifestSpan{Digest: sp.Digest, Positions: sp.Positions})
	}
	return m, nil
}

// ResumeReport is the WITNESS of a wake: how much of the prefix the durable tier could
// serve WARM on the first resumed turn. RestoredPositions are served from cache (a
// cache_read, byte-identical by l3kv's integrity guarantee); Missed positions (the tier
// no longer holds the span, or the span never staged) and Faulted positions (an I/O or
// integrity failure) both re-prefill cold (cache_creation). The three outcome buckets
// sum to TotalPositions by construction, so no position is unaccounted.
type ResumeReport struct {
	TotalPositions    int
	RestoredPositions int
	MissedPositions   int
	FaultedPositions  int
}

// CacheReadFraction is the #2853 headline: the fraction of the first resumed turn's
// prefix served WARM from the durable tier (RestoredPositions / TotalPositions). A
// zero-length prefix has no fraction to report and returns 0.
func (r ResumeReport) CacheReadFraction() float64 {
	if r.TotalPositions <= 0 {
		return 0
	}
	return float64(r.RestoredPositions) / float64(r.TotalPositions)
}

// RestorePrefix is the WAKE warm-restore hook (#2853): after a scale-to-zero wake it
// pages each manifest span back from the durable tier (backend.RestoreSpan) and reports
// how much of the prefix came back warm. It fences honestly on l3kv's residency
// trichotomy: an OK restore counts as a cache_read, a MISS (tier reaped/never held it)
// and a FAULT (I/O or integrity failure) both re-prefill cold — never a silent wrong
// hit. Spans that never made it into the manifest (going-idle could not durably stage
// them) are part of the prefix but page back cold; they are counted as MISSes so the
// three outcome buckets sum to TotalPositions and the fraction is never inflated.
func RestorePrefix(ctx context.Context, backend abi.KVBackend, m PrefixManifest) ResumeReport {
	rep := ResumeReport{TotalPositions: m.TotalPositions}
	accounted := 0
	requests := make([]abi.KVResidencyRequest, len(m.Spans))
	for i, sp := range m.Spans {
		accounted += sp.Positions
		requests[i] = abi.KVResidencyRequest{Digest: sp.Digest, Positions: sp.Positions}
	}
	for i, res := range abi.RestoreSpans(ctx, backend, requests) {
		positions := m.Spans[i].Positions
		switch res.Outcome {
		case abi.KVResidencyOK:
			rep.RestoredPositions += positions
		case abi.KVResidencyMiss:
			rep.MissedPositions += positions
		default:
			rep.FaultedPositions += positions
		}
	}
	if rem := m.TotalPositions - accounted; rem > 0 {
		rep.MissedPositions += rem // un-staged remainder of the prefix pages back cold.
	}
	return rep
}

// ColdBaseline is the no-warm-resume control: a scale-to-zero WITHOUT the going-idle
// persist hook wakes with nothing staged, so the whole prefix (totalPositions)
// re-prefills cold — a cache-read fraction of exactly 0. It is the genuinely-cold
// baseline the warm resume must beat for the #2853 done-condition to hold.
func ColdBaseline(totalPositions int) ResumeReport {
	if totalPositions < 0 {
		totalPositions = 0
	}
	return ResumeReport{TotalPositions: totalPositions, MissedPositions: totalPositions}
}

// ResumeComparison puts the warm-restore wake SIDE BY SIDE with the cold baseline — the
// #2853 witness shape. It never blends the two; it reports the cache-read fraction delta
// the going-idle persist + wake restore bought on the first resumed turn.
type ResumeComparison struct {
	Warm ResumeReport
	Cold ResumeReport
}

// CompareToCold builds the #2853 witness for a warm resume: the warm arm's report side
// by side with the cold baseline over the SAME prefix length, so the two fractions are
// apples-to-apples (the identical-workload guard #2853's confusion-risk fence names).
func CompareToCold(warm ResumeReport) ResumeComparison {
	return ResumeComparison{Warm: warm, Cold: ColdBaseline(warm.TotalPositions)}
}

// WarmerByFraction is the cache-read fraction points the warm resume gains over the
// cold baseline — positive when the going-idle persist + wake restore paid off.
func (c ResumeComparison) WarmerByFraction() float64 {
	return c.Warm.CacheReadFraction() - c.Cold.CacheReadFraction()
}

// MateriallyWarmer reports whether the warm resume beats the cold baseline by at least
// minDelta cache-read fraction points — the #2853 done-condition ("materially higher
// cache-read fraction than a cold-start baseline"). The caller names the bar; this leaf
// picks no magic threshold of its own.
func (c ResumeComparison) MateriallyWarmer(minDelta float64) bool {
	return c.WarmerByFraction() >= minDelta
}
