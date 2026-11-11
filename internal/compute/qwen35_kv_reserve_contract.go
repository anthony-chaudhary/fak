package compute

// Qwen35SequenceKVReserveBackend advertises that this backend can reserve
// additional positions on a session's growable Qwen3.5 sequence KV store ahead
// of demand. Cohort formation uses it to hold every lane's whole decode-episode
// footprint up front, so the episode's per-step reservations become capacity
// no-ops instead of mid-stream reallocation/copy churn. extraPositions counts
// FUTURE positions beyond the store's current length; the backend rounds the
// target up to its reservation chunk exactly like the per-step path. A fixed
// graph-capacity store already holds its whole capacity and reports success as
// a no-op; stores the backend cannot grow decline with an error callers may
// treat as best-effort (the per-step reservation remains the backstop).
type Qwen35SequenceKVReserveBackend interface {
	Qwen35ReserveSequenceKV(kv KVStore, extraPositions int) error
}
