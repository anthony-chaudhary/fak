package harnessinstructions

import (
	"context"
	"errors"
	"sync"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

// OutcomeCounts is a point-in-time tally of dynamic-instruction resolution results.
type OutcomeCounts struct {
	Invocations  uint64                     `json:"invocations"`
	Succeeded    uint64                     `json:"succeeded"`
	Failed       uint64                     `json:"failed"`
	ByCode       map[harnesskit.Code]uint64 `json:"by_code"`
	Unclassified uint64                     `json:"unclassified"`
}

// OutcomeRecorder resolves dynamic instructions and retains content-free outcome counts.
// Its zero value is ready for concurrent use.
type OutcomeRecorder struct {
	mu     sync.Mutex
	counts OutcomeCounts
}

// Resolve runs the real instruction-composition path and records its terminal outcome.
func (r *OutcomeRecorder) Resolve(ctx context.Context, provider harnesskit.InstructionProvider, req harnesskit.InstructionRequest) (Realization, error) {
	realization, err := Resolve(ctx, provider, req)
	r.record(err)
	return realization, err
}

// Counts returns a caller-owned snapshot of the recorded outcomes.
func (r *OutcomeRecorder) Counts() OutcomeCounts {
	r.mu.Lock()
	defer r.mu.Unlock()

	counts := r.counts
	counts.ByCode = make(map[harnesskit.Code]uint64, len(r.counts.ByCode))
	for code, count := range r.counts.ByCode {
		counts.ByCode[code] = count
	}
	return counts
}

func (r *OutcomeRecorder) record(err error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.counts.Invocations++
	if err == nil {
		r.counts.Succeeded++
		return
	}
	r.counts.Failed++

	var contractErr *harnesskit.Error
	if !errors.As(err, &contractErr) {
		r.counts.Unclassified++
		return
	}
	if r.counts.ByCode == nil {
		r.counts.ByCode = make(map[harnesskit.Code]uint64)
	}
	r.counts.ByCode[contractErr.Code]++
}
