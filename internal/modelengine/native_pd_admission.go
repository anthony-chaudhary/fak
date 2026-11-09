package modelengine

import (
	"context"
	"errors"
	"fmt"
	"time"
)

// ErrNativePDCostAdmission identifies a refusal before native prefill work starts.
var ErrNativePDCostAdmission = errors.New("modelengine: native P/D cost admission refused")

// NativePDCostEstimate estimates time to first token as queue delay plus KV
// transfer and full prompt prefill. Routing locality does not imply KV reuse.
type NativePDCostEstimate struct {
	Queue    time.Duration
	Transfer time.Duration
	Prefill  time.Duration
}

// NativePDCostEstimator supplies caller-calibrated costs for the actual prompt
// length and selected decode worker. Concurrent Admit calls may call it in
// parallel; implementations must be concurrency-safe. No prompt text is passed.
type NativePDCostEstimator func(context.Context, int, string) (NativePDCostEstimate, error)

type nativePDCostAdmission struct {
	ceiling  time.Duration
	estimate NativePDCostEstimator
}

// WithNativePDCostAdmission enables a pre-prefill cost ceiling. It implements
// the queue + transfer + prefill admission rule described in Mooncake FAST25
// section 4.1, not its predictor or global scheduler. Callers supply calibrated
// estimates; this package makes no hardware throughput assumptions. Invalid
// enabled configurations fail closed at admission. Omitting the option preserves
// existing behavior. Configure the option only while constructing the cluster.
func WithNativePDCostAdmission(ceiling time.Duration, estimate NativePDCostEstimator) NativePDOption {
	return func(c *NativePDCluster) {
		c.costAdmission = &nativePDCostAdmission{ceiling: ceiling, estimate: estimate}
	}
}

func (c *NativePDCluster) checkAdmissionCost(ctx context.Context, promptTokens int, worker string) error {
	p := c.costAdmission
	if p == nil {
		return nil
	}
	if p.ceiling <= 0 || p.estimate == nil {
		return fmt.Errorf("%w: positive ceiling and calibrated estimator required", ErrNativePDCostAdmission)
	}
	cost, err := p.estimate(ctx, promptTokens, worker)
	if err != nil {
		return fmt.Errorf("%w: estimate: %w", ErrNativePDCostAdmission, err)
	}
	if cost.Queue < 0 || cost.Transfer < 0 || cost.Prefill < 0 {
		return fmt.Errorf("%w: negative cost component", ErrNativePDCostAdmission)
	}
	// Compare each nonnegative component against the remaining budget, rather
	// than overflowing a duration sum and accidentally admitting costly work.
	remaining := p.ceiling
	for _, component := range [...]time.Duration{cost.Queue, cost.Transfer, cost.Prefill} {
		if component > remaining {
			return fmt.Errorf("%w: estimated time to first token exceeds %s", ErrNativePDCostAdmission, p.ceiling)
		}
		remaining -= component
	}
	return nil
}
