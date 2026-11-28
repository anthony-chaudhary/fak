package model

import (
	"fmt"
	"math"
)

// Inkling2LoopPassesError refuses a per-request loop pass count the loaded
// checkpoint cannot run.
type Inkling2LoopPassesError struct {
	Requested, Max int
}

func (e *Inkling2LoopPassesError) Error() string {
	return fmt.Sprintf("inkling2_loop: %d loop passes requested, checkpoint supports 1..%d", e.Requested, e.Max)
}

// WithPasses returns hp running requested core passes for one request; 0 keeps
// the checkpoint's count. The bound is the loaded loop.count: the controllers
// are shared across passes, so more passes would run, but past the trained
// count the output is unvalidated. KV and short-conv state are per request and
// sized from the returned Passes, so N passes cost Layers+(N-1)*core cache
// slots and as many layer visits per token.
func (hp Inkling2LoopHParams) WithPasses(requested int) (Inkling2LoopHParams, error) {
	if requested == 0 {
		return hp, nil
	}
	if requested < 1 || requested > hp.Passes {
		return hp, &Inkling2LoopPassesError{Requested: requested, Max: hp.Passes}
	}
	hp.Passes = requested
	return hp, nil
}

// WithPassesUnbounded overrides hp.Passes for diagnostic pass-sweep or probe runs
// beyond the checkpoint's trained loop count (still requiring requested >= 1).
func (hp Inkling2LoopHParams) WithPassesUnbounded(requested int) (Inkling2LoopHParams, error) {
	if requested < 1 {
		return hp, &Inkling2LoopPassesError{Requested: requested, Max: hp.Passes}
	}
	hp.Passes = requested
	return hp, nil
}

// FixedPointReadout compares R, 2R, and 4R pass logit vectors to check
// fixed-point contraction (conceptually inspired by mbzuai-oryx/LoopedWSP LWSP-2).
type FixedPointReadout struct {
	DriftRTo2R  float64 `json:"drift_r_to_2r"`
	Drift2RTo4R float64 `json:"drift_2r_to_4r"`
	Converged   bool    `json:"converged"`
}

// CheckFixedPointReadout measures L2 drift between R->2R and 2R->4R readouts.
func CheckFixedPointReadout(r, r2, r4 []float32, tol float64) (FixedPointReadout, error) {
	if len(r) == 0 || len(r) != len(r2) || len(r) != len(r4) {
		return FixedPointReadout{}, fmt.Errorf("inkling2_loop: mismatched readout lengths (%d, %d, %d)", len(r), len(r2), len(r4))
	}
	d12 := sliceL2Diff(r, r2)
	d24 := sliceL2Diff(r2, r4)
	return FixedPointReadout{
		DriftRTo2R:  d12,
		Drift2RTo4R: d24,
		Converged:   d24 <= d12 && d24 <= tol,
	}, nil
}

func sliceL2Diff(a, b []float32) float64 {
	var sum float64
	for i := range a {
		d := float64(a[i]) - float64(b[i])
		sum += d * d
	}
	return math.Sqrt(sum)
}

// Inkling2LoopRKConfig configures optional damped Euler/RK intra-layer
// sub-stepping (clean-room INSPIRE-ONLY adaptation of rse-looper-1).
type Inkling2LoopRKConfig struct {
	Window int     `json:"rk_window"`
	K      int     `json:"rk_k"`
	Beta   float32 `json:"rk_beta"`
}

func (c *Inkling2LoopRKConfig) Enabled() bool {
	return c != nil && c.K > 1 && c.Window > 0 && c.Beta > 0 && c.Beta <= 1
}

// Inkling2LoopPassProbeConfig configures the opt-in pass-shift diagnostic probe
// for non-default pass overrides (clean-room INSPIRE-ONLY adaptation of rse-looper-3).
type Inkling2LoopPassProbeConfig struct {
	Enabled           bool    `json:"enabled"`
	CollapseThreshold float64 `json:"collapse_threshold"`
}

// Inkling2LoopPassProbeReport records consecutive-pass exit hidden-state cosines.
type Inkling2LoopPassProbeReport struct {
	PassCosines []float64 `json:"pass_cosines"`
	MinCosine   float64   `json:"min_cosine"`
	SignFlip    bool      `json:"sign_flip"`
	Collapse    bool      `json:"collapse"`
}

func evaluatePassProbe(exits [][]float32, threshold float64) *Inkling2LoopPassProbeReport {
	if len(exits) < 2 {
		return &Inkling2LoopPassProbeReport{MinCosine: 1}
	}
	if threshold <= 0 {
		threshold = 0.2
	}
	rep := &Inkling2LoopPassProbeReport{
		PassCosines: make([]float64, 0, len(exits)-1),
		MinCosine:   1,
	}
	for i := 1; i < len(exits); i++ {
		c := vecCosine(exits[i-1], exits[i])
		rep.PassCosines = append(rep.PassCosines, c)
		if c < rep.MinCosine {
			rep.MinCosine = c
		}
		if c < 0 {
			rep.SignFlip = true
		}
		if c < threshold {
			rep.Collapse = true
		}
	}
	return rep
}

func vecCosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := 0; i < len(a) && i < len(b); i++ {
		x, y := float64(a[i]), float64(b[i])
		dot += x * y
		na += x * x
		nb += y * y
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / math.Sqrt(na*nb)
}
