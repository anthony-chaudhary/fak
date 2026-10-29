// Package nativeperfslo turns matched fak-native benchmark observations into
// stable time-series state. It refuses cross-envelope comparisons and preserves
// unavailable evidence instead of manufacturing zeroes: mismatched envelopes,
// missing evidence, or stale samples produce unavailable series rather than
// reporting false health.
package nativeperfslo
