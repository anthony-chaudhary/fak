package ops

import (
	"math"
	"time"
)

// DefaultMaxDutyPercent is the declared share of one core the ops loop may
// consume (#14480). 2% is the midpoint of the 1-5% range: a 2s quick phase
// at the 60s base cadence is 3.3% and gets stretched, while a typical
// sub-second tick (<1.2s) is untouched.
const DefaultMaxDutyPercent = 2.0

// DefaultDutyWindow is the number of recent ticks the cost mean spans.
const DefaultDutyWindow = 8

// DutyCyclePolicy bounds the ops loop's own share of host time. The loop never
// stretches beyond MaxInterval, because doing so would silently lose
// maintenance coverage.
type DutyCyclePolicy struct {
	MaxDutyPercent float64
	MinInterval    time.Duration
	MaxInterval    time.Duration
	Window         int
}

// DutyDecision is the pure output of one duty-cycle observation.
type DutyDecision struct {
	NextInterval  time.Duration
	Stretched     bool
	Overloaded    bool // required interval exceeded MaxInterval and was clamped
	Yielded       bool // disk below refuse watermark: the cap yields
	PredictedDuty float64
	ReasonClass   PauseReasonClass // DUTY_CYCLE_CAP when Stretched
}

// DutyCycleController derives the next interval from the rolling mean of
// witnessed self_cpu_ms. It is the ops analogue of
// systembaseline.CadenceController (#14480).
type DutyCycleController struct {
	policy DutyCyclePolicy
	costs  []time.Duration
}

// NewDutyCycleController normalises the policy and returns a controller.
func NewDutyCycleController(p DutyCyclePolicy) *DutyCycleController {
	if p.MaxDutyPercent <= 0 || p.MaxDutyPercent > 100 {
		p.MaxDutyPercent = DefaultMaxDutyPercent
	}
	if p.MinInterval <= 0 {
		p.MinInterval = time.Second
	}
	if p.MaxInterval < p.MinInterval {
		p.MaxInterval = p.MinInterval
	}
	if p.Window <= 0 {
		p.Window = DefaultDutyWindow
	}
	return &DutyCycleController{policy: p}
}

// Observe folds one completed tick and returns the next interval. base is the
// cadence the phase would otherwise use. diskBelowRefuse makes the cap yield so
// a pressure-relief phase is never stretched away. A cycle without a measured
// self CPU is not observed (an unmeasured tick is not a zero-cost tick).
func (c *DutyCycleController) Observe(cycle LoadCycle, base time.Duration, diskBelowRefuse bool) DutyDecision {
	if base < c.policy.MinInterval {
		base = c.policy.MinInterval
	}
	if cycle.HaveSelfCPU {
		c.costs = append(c.costs, time.Duration(cycle.SelfCPUMillis)*time.Millisecond)
		if len(c.costs) > c.policy.Window {
			c.costs = c.costs[len(c.costs)-c.policy.Window:]
		}
	}
	cost := c.meanCost()
	d := DutyDecision{NextInterval: base}
	if diskBelowRefuse {
		d.Yielded = true
		d.PredictedDuty = predictedDuty(cost, base)
		return d
	}
	// 25% reserve: a controller targeting the ceiling oscillates just above it.
	required := time.Duration(math.Ceil(float64(cost) * 125 / c.policy.MaxDutyPercent))
	if required > base {
		d.NextInterval = required
		d.Stretched = true
		d.ReasonClass = PauseReasonDutyCycleCap
	}
	if d.NextInterval > c.policy.MaxInterval {
		d.NextInterval = c.policy.MaxInterval
		d.Overloaded = true
		if d.NextInterval < base {
			d.NextInterval = base
			d.Stretched = false
			d.ReasonClass = ""
		}
	}
	d.PredictedDuty = predictedDuty(cost, d.NextInterval)
	return d
}

func (c *DutyCycleController) meanCost() time.Duration {
	if len(c.costs) == 0 {
		return 0
	}
	var sum time.Duration
	for _, x := range c.costs {
		sum += x
	}
	return sum / time.Duration(len(c.costs))
}

func predictedDuty(cost, interval time.Duration) float64 {
	if interval <= 0 {
		return 0
	}
	return float64(cost) * 100 / float64(interval)
}
