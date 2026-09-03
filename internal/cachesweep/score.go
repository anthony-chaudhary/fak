package cachesweep

import "math"

const (
	// DefaultHalfLifeSeconds is the default half-life for exponential hit decay
	// (6 hours = 21600 seconds), matching ds4_kvstore.c.
	DefaultHalfLifeSeconds = 21600.0

	// DefaultAnchorMultiplier is the retention boost factor for intentional turn
	// boundaries and cold anchors (2.0x).
	DefaultAnchorMultiplier = 2.0

	// DefaultSupersededFactor is the retention discount factor for intermediate
	// checkpoints superseded by newer exact prefixes (0.2x).
	DefaultSupersededFactor = 0.2
)

// TokenDensityHalfLifeScore computes the eviction priority score for a cached prefix.
// Higher scores indicate higher retention priority (less likely to be evicted).
//
// Formula:
//
//	Score = (EffectiveHits + 1.0) * (tokens / bytes) * AnchorFactor * SupersededFactor
//	where EffectiveHits = hits * 2^(-elapsedSeconds / 21600s)
//
// Parameters:
//   - hits: total access count of this prefix.
//   - elapsedSeconds: seconds elapsed since last access (negative values treated as 0).
//   - tokens: logical tokens represented by the cache entry.
//   - bytes: physical storage size in bytes.
//   - isAnchor: true for intentional turn boundaries / system prompt anchors.
//   - isSuperseded: true if an exact longer prefix supersedes this checkpoint.
func TokenDensityHalfLifeScore(hits int, elapsedSeconds float64, tokens, bytes int, isAnchor, isSuperseded bool) float64 {
	if tokens <= 0 {
		return 0.0
	}
	if bytes <= 0 {
		bytes = 1
	}

	effectiveHits := float64(hits)
	if elapsedSeconds > 0 && hits > 0 {
		effectiveHits = float64(hits) * math.Pow(2, -elapsedSeconds/DefaultHalfLifeSeconds)
	} else if hits < 0 {
		effectiveHits = 0
	}

	density := float64(tokens) / float64(bytes)

	anchorFactor := 1.0
	if isAnchor {
		anchorFactor = DefaultAnchorMultiplier
	}

	supersededFactor := 1.0
	if isSuperseded {
		supersededFactor = DefaultSupersededFactor
	}

	return (effectiveHits + 1.0) * density * anchorFactor * supersededFactor
}
