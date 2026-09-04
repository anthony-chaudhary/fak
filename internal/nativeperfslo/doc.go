// Package nativeperfslo turns matched fak-native benchmark observations into
// stable time-series state. It refuses cross-envelope comparisons and preserves
// unavailable evidence instead of manufacturing zeroes.
//
// Invariant: native performance SLO evaluations are fail-closed and bounded.
// Guard: mismatched benchmark envelopes, missing evidence, or stale samples
// immediately produce unavailable series rather than reporting false health.
package nativeperfslo
