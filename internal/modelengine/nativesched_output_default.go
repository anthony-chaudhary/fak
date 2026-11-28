//go:build !fak_native_output_isolation_qualification

package modelengine

// Deferred-token outbox isolation is promoted to default on Engine/serve paths
// so delayed token consumers never stall concurrent batching across C=1..64 lanes.
const nativeOutputIsolationEnabled = true
