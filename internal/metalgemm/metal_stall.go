package metalgemm

import (
	"errors"
	"fmt"
	"math"
)

// DefaultCommandBufferWaitLimit is the package-wide bound applied to a native
// Metal command-buffer wait. Native code historically blocked without a bound at
// `[cmd waitUntilCompleted]`, so a stalled GPU surfaced as a silent hang rather
// than a diagnosable failure. Any wait at or over this limit is classified as a
// stall and must be reported as a typed error.
const DefaultCommandBufferWaitLimit = 10_000.0

// MetalCommandBufferStallError reports that a native Metal command-buffer wait
// exceeded its limit. The wait itself is not interruptible from Go; this type
// exists so the observation seam can convert an unbounded blocking call into an
// explicit, inspectable failure instead of a silent hang.
type MetalCommandBufferStallError struct {
	Operation          string
	WaitedMilliseconds float64
	LimitMilliseconds  float64
}

func (e MetalCommandBufferStallError) Error() string {
	return fmt.Sprintf("Metal command buffer stall during %s: waited %.3fms at/over %.3fms limit",
		e.Operation, e.WaitedMilliseconds, e.LimitMilliseconds)
}

// IsMetalCommandBufferStall reports whether err is the typed stall result.
func IsMetalCommandBufferStall(err error) bool {
	var stall MetalCommandBufferStallError
	return errors.As(err, &stall)
}

// CheckCommandBufferWait classifies one observed command-buffer wait against
// limitMS. It returns nil only while the wait is strictly under the limit and a
// *MetalCommandBufferStallError at or over it. The comparison is deliberately
// inclusive: a wait that reaches the limit has already consumed its entire
// budget and must not be reported as healthy.
//
// The contract is total and explicit at its boundaries:
//
//   - A non-finite waitedMS (NaN, +Inf, -Inf) or a negative waitedMS is treated
//     as a stall. Such a value is not a measurable wait, and an unmeasurable
//     wait must never be reported as healthy. -Inf and NaN in particular would
//     otherwise compare below the limit and slip through as healthy.
//   - limitMS must be finite and strictly positive. A non-finite or
//     non-positive limit is an invalid budget: it must not silently mean "never
//     stalls" (the old `waited < limit` comparison made NaN/0/negative limits
//     never fire). An invalid limit is reported as a stall whose
//     LimitMilliseconds records the invalid value, so the misconfiguration is
//     visible rather than swallowed.
func CheckCommandBufferWait(op string, waitedMS, limitMS float64) error {
	if waitedMS >= limitMS && isFinite(waitedMS) && isFinite(limitMS) && limitMS > 0 && waitedMS >= 0 {
		return MetalCommandBufferStallError{
			Operation:          op,
			WaitedMilliseconds: waitedMS,
			LimitMilliseconds:  limitMS,
		}
	}
	if isFinite(waitedMS) && waitedMS >= 0 && isFinite(limitMS) && limitMS > 0 {
		return nil
	}
	return MetalCommandBufferStallError{
		Operation:          op,
		WaitedMilliseconds: waitedMS,
		LimitMilliseconds:  limitMS,
	}
}

// isFinite reports whether f is neither NaN nor an infinity.
func isFinite(f float64) bool {
	return !math.IsNaN(f) && !math.IsInf(f, 0)
}
