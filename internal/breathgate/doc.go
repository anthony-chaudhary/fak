// Package breathgate provides turn pacing, pause control, debounce, and cooldown
// mechanisms for autonomous agent loops.
//
// In multi-turn agent execution, runaway loops can rapidly exhaust API rate limits,
// deplete token spend, or cause destructive thrashing. A breath gate introduces
// deliberate "breathing room" between autonomous operations, enforcing minimum
// intervals, bounded jitter to avoid thundering-herd synchronization, and automatic
// or explicit cooldown periods.
//
// Invariant: breathgate pacing decisions are fail-closed and bounded; invalid intervals clamp to non-negative bounds and cancelled contexts immediately abort waiting callers.
//
// Guard: Check and Wait guard against runaway loop execution by enforcing minimum inter-turn intervals and triggering un-bypassable cooldowns upon exceeding burst limits.
//
// Contract:
//   - All methods on Gate are safe for concurrent use across multiple goroutines.
//   - Pacing decisions are fail-closed and bounded: context cancellations or deadlines abort Wait immediately without admitting execution.
//   - Check reports admission status non-blockingly without mutating pacing state.
//   - Wait blocks callers until the minimum interval and active cooldowns have elapsed.
//   - RecordTurn tracks execution frequency and trips automatic cooldowns when bursts exceed configured thresholds.
//   - Reset clears all cooldowns and reservations, restoring the gate to its initial ready state.
package breathgate
