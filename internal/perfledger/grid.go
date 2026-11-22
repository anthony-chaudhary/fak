// grid.go ports aiperf's log-spaced integer grid helper to Go.
//
// NOTICE: ported from ai-dynamo/aiperf, src/aiperf/search_recipes/builtins.py
// (_logspace_int_steps), pinned at commit ab610add4e8876fca223197596c728e17365f381
// (ai-dynamo/aiperf@ab610ad), Apache-2.0 licensed. The source builds the
// concurrency-ramp sweep grid: steps log-spaced points in [lo, hi], rounded
// half-to-even, floored at 1, deduplicated and sorted ascending.
package perfledger

import (
	"errors"
	"fmt"
	"math"
	"sort"
)

// ErrInvalidLogspaceGrid is wrapped by every LogspaceIntSteps refusal.
var ErrInvalidLogspaceGrid = errors.New("perfledger: invalid logspace grid")

// lgridMaxHi bounds hi to the exactly-representable integer range of float64
// so rounding to int can neither overflow nor lose integer precision.
const lgridMaxHi = 1 << 53

// LogspaceIntSteps returns up to steps log-spaced integers in [lo, hi],
// ascending and deduplicated (rounding can collapse neighbours, e.g. 1, 1, 2).
// Rounding is half-to-even to match Python's round. For integral lo and hi
// the result starts at lo and ends at hi.
//
// It refuses steps < 2 (a single-point ramp has no baseline), lo <= 0 (log
// spacing needs a positive bound), hi <= lo, and non-finite or oversized
// bounds.
func LogspaceIntSteps(lo, hi float64, steps int) ([]int, error) {
	if steps < 2 {
		return nil, fmt.Errorf("%w: steps must be >= 2 (got %d)", ErrInvalidLogspaceGrid, steps)
	}
	if math.IsNaN(lo) || math.IsInf(lo, 0) || math.IsNaN(hi) || math.IsInf(hi, 0) {
		return nil, fmt.Errorf("%w: bounds must be finite (lo=%v, hi=%v)", ErrInvalidLogspaceGrid, lo, hi)
	}
	if lo <= 0 {
		return nil, fmt.Errorf("%w: lo must be > 0 (got %v)", ErrInvalidLogspaceGrid, lo)
	}
	if hi <= lo {
		return nil, fmt.Errorf("%w: hi (%v) must be > lo (%v)", ErrInvalidLogspaceGrid, hi, lo)
	}
	if hi > lgridMaxHi {
		return nil, fmt.Errorf("%w: hi (%v) exceeds %d", ErrInvalidLogspaceGrid, hi, int64(lgridMaxHi))
	}
	logLo, logHi := math.Log(lo), math.Log(hi)
	seen := make(map[int]struct{}, steps)
	out := make([]int, 0, steps)
	for i := 0; i < steps; i++ {
		// Endpoints use lo and hi directly: exp(log(x)) is off by an ulp on
		// some libms, which flips half-way endpoints like 4.5 under rounding.
		v := math.Exp(logLo + (logHi-logLo)*float64(i)/float64(steps-1))
		switch i {
		case 0:
			v = lo
		case steps - 1:
			v = hi
		}
		n := int(math.RoundToEven(v))
		// Floors fractional lo (e.g. 0.3 rounds to 0) at the smallest usable
		// concurrency.
		if n < 1 {
			n = 1
		}
		if _, dup := seen[n]; dup {
			continue
		}
		seen[n] = struct{}{}
		out = append(out, n)
	}
	sort.Ints(out)
	return out, nil
}
