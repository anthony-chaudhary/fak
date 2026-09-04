// Package dispatchcache provides in-memory and on-disk caching mechanisms
// for dispatch queue snapshots, routed backlog state, and delta watermarks.
//
// Invariant: dispatch queue cache persistence is fail-closed and bounded.
// Any schema mismatch, key deviation, or corrupted payload causes cache
// lookup to fail closed and return empty uncorrupted state.
//
// Guard: snapshot persistence operations must use atomic file renames to guarantee
// safe reader isolation and prevent half-written cache records.
package dispatchcache
