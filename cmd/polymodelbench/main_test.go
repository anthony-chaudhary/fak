package main

import "testing"

// TestSelfcheck runs the three witnesses in-process (quiet) so CI proves the
// host-many / decode-one / lossless-cache-led-MTP claims, not just `go build`.
func TestSelfcheck(t *testing.T) {
	if !hostMany(true) {
		t.Error("hostMany: residency budget/pin invariant failed")
	}
	if !decodeOne(true) {
		t.Error("decodeOne: serial decode-lane invariant failed")
	}
	if !cacheLedMTP(true) {
		t.Error("cacheLedMTP: greedy speculative decode is not lossless (bit-exact KV rollback regressed)")
	}
}
