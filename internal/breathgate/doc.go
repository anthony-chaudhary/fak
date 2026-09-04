// Package breathgate provides turn pacing, pause control, debounce, and cooldown
// mechanisms for autonomous agent loops.
//
// In multi-turn agent execution, runaway loops can rapidly exhaust API rate limits,
// deplete token spend, or cause destructive thrashing. A breath gate introduces
// deliberate breathing room between autonomous operations, enforcing minimum
// intervals, bounded jitter to avoid thundering-herd synchronization, and automatic
// or explicit cooldown periods.
//
// All methods on Gate are safe for concurrent use across multiple goroutines.
package breathgate
