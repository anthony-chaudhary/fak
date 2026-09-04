// Package incidentrsi classifies operational incidents, maintains bounded
// burst debounce states, and emits durable content-free RSI trigger contracts.
//
// Invariant: incident debounce is fail-closed and bounded; invalid observations
// and persistence failures never hide the original product failure.
//
// Guard: automatic RSI trigger launches are restricted to trusted development
// checkouts and rate-limited by explicit cooldown windows.
package incidentrsi
