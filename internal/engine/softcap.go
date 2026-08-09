package engine

import "fmt"

// SoftCapState is the controller's stable admission state.
type SoftCapState uint8

const (
	SoftCapNormal SoftCapState = iota
	SoftCapPressure
	SoftCapHard
)

func (s SoftCapState) String() string {
	switch s {
	case SoftCapNormal:
		return "normal"
	case SoftCapPressure:
		return "soft-pressure"
	case SoftCapHard:
		return "hard-pressure"
	default:
		return "unknown"
	}
}

// SoftCapConfig defines the two capacity boundaries and the hysteresis needed
// before a noisy sample may change the stable state. Hard pressure is never
// delayed: the Samples value only applies to soft entry and recovery.
type SoftCapConfig struct {
	SoftLimitBytes int64
	HardLimitBytes int64
	Samples        int
}

// softCapResult records one observation and whether it changed the stable
// state. Pending is the number of consecutive observations accumulated toward
// a hysteresis-gated transition.
type softCapResult struct {
	Previous SoftCapState
	State    SoftCapState
	Changed  bool
	Pending  int
	Reason   string
}

// SoftCapController turns noisy usage observations into stable capacity
// pressure. A hard-limit observation enters SoftCapHard immediately. All other
// changes require Samples consecutive observations in the same target band.
type SoftCapController struct {
	config  SoftCapConfig
	state   SoftCapState
	pending SoftCapState
	streak  int
}

// NewSoftCapController validates config and returns a controller in the normal
// state. SoftLimitBytes must be lower than HardLimitBytes.
func NewSoftCapController(config SoftCapConfig) (*SoftCapController, error) {
	if config.SoftLimitBytes <= 0 {
		return nil, fmt.Errorf("soft cap: soft limit must be positive")
	}
	if config.HardLimitBytes <= config.SoftLimitBytes {
		return nil, fmt.Errorf("soft cap: hard limit must exceed soft limit")
	}
	if config.Samples <= 0 {
		return nil, fmt.Errorf("soft cap: samples must be positive")
	}
	return &SoftCapController{config: config, state: SoftCapNormal}, nil
}

// Observe incorporates one non-negative usage sample.
func (c *SoftCapController) Observe(usedBytes int64) softCapResult {
	previous := c.state
	if usedBytes < 0 {
		c.resetPending()
		return softCapResult{Previous: previous, State: c.state, Reason: "invalid-sample"}
	}

	target := c.band(usedBytes)
	if target == SoftCapHard {
		c.state = SoftCapHard
		c.resetPending()
		return softCapResult{
			Previous: previous,
			State:    c.state,
			Changed:  previous != c.state,
			Reason:   "hard-limit",
		}
	}
	if target == c.state {
		c.resetPending()
		return softCapResult{Previous: previous, State: c.state, Reason: "stable"}
	}
	if target != c.pending {
		c.pending = target
		c.streak = 0
	}
	c.streak++
	if c.streak < c.config.Samples {
		return softCapResult{
			Previous: previous,
			State:    c.state,
			Pending:  c.streak,
			Reason:   "hysteresis",
		}
	}

	c.state = target
	c.resetPending()
	return softCapResult{
		Previous: previous,
		State:    c.state,
		Changed:  true,
		Reason:   "sustained",
	}
}

func (c *SoftCapController) band(usedBytes int64) SoftCapState {
	switch {
	case usedBytes >= c.config.HardLimitBytes:
		return SoftCapHard
	case usedBytes >= c.config.SoftLimitBytes:
		return SoftCapPressure
	default:
		return SoftCapNormal
	}
}

func (c *SoftCapController) resetPending() {
	c.pending = SoftCapNormal
	c.streak = 0
}
