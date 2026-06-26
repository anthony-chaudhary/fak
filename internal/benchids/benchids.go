// Package benchids generates a deterministic stream of synthetic token IDs for
// the benchmark command mains.
//
// It exists to retire the byte-identical lcgIDs helper that was copy-pasted
// across the benchmark mains (fleetserve, sessionbench, demorace, radixbench) —
// the duplication clone the code-slop scorecard named under #776, worst-first
// after parseInts (see internal/intlist).
package benchids

// LCG returns n synthetic token IDs in [0, vocab) drawn from a linear
// congruential generator seeded at 2463534242+seed. The same seed always yields
// the same sequence, so a benchmark's prompt is reproducible across runs and the
// seed alone distinguishes one synthetic prompt from another. This is the exact
// behaviour of the bench harnesses' former lcgIDs(n, vocab, seed).
func LCG(n, vocab int, seed uint64) []int {
	ids := make([]int, n)
	state := 2463534242 + seed
	for i := 0; i < n; i++ {
		state = (state*1103515245 + 12345) & 0x7fffffff
		ids[i] = int(state % uint64(vocab))
	}
	return ids
}
