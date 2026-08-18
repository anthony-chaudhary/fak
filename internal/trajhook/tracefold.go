package trajhook

import "github.com/anthony-chaudhary/fak/internal/trajectory"

func traceAccumulator[T any](byTrace map[string]*T, order *[]string, turn trajectory.Turn, create func() *T) *T {
	a, ok := byTrace[turn.TraceID]
	if !ok {
		a = create()
		byTrace[turn.TraceID] = a
		*order = append(*order, turn.TraceID)
	}
	return a
}
