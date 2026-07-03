package main

// envprice.go — the one shared reader for per-MTok USD price env overrides
// (FAK_CACHEVALUE_*_PER_MTOK_USD, FAK_SPEND_*_PER_MTOK_USD). Both callers share
// the same set/unset and invalid-input contract; they differ only in whether a
// non-finite value (NaN/Inf) is rejected.

import (
	"math"
	"os"
	"strconv"
	"strings"
)

// priceFromEnv reads a price env var. The second return is whether the var was
// set to a non-empty value at all ("set"), independent of validity: unset or
// blank yields (0, false); a set-but-invalid value (unparseable, negative, or —
// when rejectNonFinite — NaN/Inf) yields (0, true), so an explicit override
// still suppresses the built-in default table rather than silently pricing.
func priceFromEnv(name string, rejectNonFinite bool) (float64, bool) {
	raw, ok := os.LookupEnv(name)
	raw = strings.TrimSpace(raw)
	if !ok || raw == "" {
		return 0, false
	}
	v, err := strconv.ParseFloat(raw, 64)
	if err != nil || v < 0 {
		return 0, true
	}
	if rejectNonFinite && (math.IsNaN(v) || math.IsInf(v, 0)) {
		return 0, true
	}
	return v, true
}
