package headroom

import (
	"os"
	"strconv"
)

// The "when to compress" policy — the decision layer that is fak's real value-add
// over a bare compressor. A compressor knows HOW to shrink bytes; the GATE decides
// WHEN that is worth doing, per result, automatically, at the admission boundary.
//
// Compression is never free. It spends a content-addressed preserve write (so the
// original stays retrievable — the CCR promise) and it hands the model a codec
// annotation / elision marker to read. So the gate pays that cost only when the
// saving clears a floor: a marginal win on a small result is left RAW, the model
// reads it verbatim, and nothing is spent. This is the "knowing when NOT to" half.
//
// It composes with the other two "when not" rules already in the gate: it never
// compresses a result the security gates would quarantine (compressing would hide
// an injection/secret from detection — the load-bearing safety rule), and every
// saving it DOES take is reversible, so the decision is safe to get wrong — a wrong
// compress costs one demand-page, never a lost fact.
//
// Defaults are deliberately conservative: when in doubt, compress, because the
// original is preserved. A context-tight deployment raises the floor via the env
// knobs below.

var (
	// minCompressBytes: results smaller than this are left raw regardless — too
	// small for the preserve-write + annotation indirection to pay for itself.
	minCompressBytes = envInt("FAK_HEADROOM_MIN_BYTES", 48)
	// minSavedBytes: an absolute saving at or above this is always worth taking,
	// even at a low ratio — a large log that compresses only 8% still frees many
	// tokens, so a big ABSOLUTE win is never refused on ratio grounds.
	minSavedBytes = envInt("FAK_HEADROOM_MIN_SAVED_BYTES", 256)
	// minSavedRatio: a relative saving at or above this is worth taking even when
	// the absolute number is small — a short result cut by a meaningful fraction.
	minSavedRatio = envFloat("FAK_HEADROOM_MIN_SAVED_RATIO", 0.15)
)

// worthCompressing reports whether a genuine saving (orig -> neu, with neu < orig)
// clears the worth-it floor: the result is big enough to bother with, AND the
// saving is either a meaningful ABSOLUTE number OR a meaningful FRACTION. It is the
// pure core of the "when to compress" decision; the gate calls it after a
// compressor reports a real saving, and leaves the result raw when it returns false.
func worthCompressing(orig, neu int) bool {
	if orig < minCompressBytes {
		return false
	}
	saved := orig - neu
	if saved <= 0 {
		return false
	}
	if saved >= minSavedBytes {
		return true
	}
	return float64(saved)/float64(orig) >= minSavedRatio
}

func envInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			return n
		}
	}
	return def
}

func envFloat(key string, def float64) float64 {
	if v := os.Getenv(key); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil && f >= 0 {
			return f
		}
	}
	return def
}
