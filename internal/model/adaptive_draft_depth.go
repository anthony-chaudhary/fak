package model

import (
	"errors"
	"math"
)

// AdaptiveDraftDepthController bounds speculative depth using observed acceptance with
// hysteresis. Target verification remains authoritative; this controls proposed work only.
type AdaptiveDraftDepthController struct {
	MinDepth, MaxDepth               int
	RaiseAcceptance, LowerAcceptance float64
	HysteresisWindows                int
	depth, high, low                 int
}

type AdaptiveDraftSample struct {
	Accepted, Proposed           int
	VerifierBytes, VerifierFLOPs uint64
	LatencyMilliseconds, Joules  float64
}

type AdaptiveDraftReceipt struct {
	Schema, Engine, Artifact, Selector, QualityConstraint, Rollback string
	FixedDepth, FinalDepth, AcceptedTokens, ProposedTokens          int
	VerifierBytes, VerifierFLOPs                                    uint64
	LatencyMilliseconds, Joules, AcceptanceRate                     float64
	Depths                                                          []int
}

func (c *AdaptiveDraftDepthController) init() error {
	if c.MinDepth <= 0 || c.MaxDepth < c.MinDepth || c.HysteresisWindows <= 0 || c.LowerAcceptance < 0 || c.RaiseAcceptance > 1 || c.LowerAcceptance >= c.RaiseAcceptance {
		return errors.New("model: invalid adaptive draft-depth policy")
	}
	if c.depth == 0 {
		c.depth = c.MinDepth
	}
	return nil
}
func (c *AdaptiveDraftDepthController) Observe(s AdaptiveDraftSample) (int, error) {
	if err := c.init(); err != nil {
		return 0, err
	}
	if s.Proposed <= 0 || s.Accepted < 0 || s.Accepted > s.Proposed || math.IsNaN(s.LatencyMilliseconds) || s.LatencyMilliseconds < 0 || math.IsNaN(s.Joules) || s.Joules < 0 {
		return 0, errors.New("model: invalid adaptive draft sample")
	}
	rate := float64(s.Accepted) / float64(s.Proposed)
	switch {
	case rate >= c.RaiseAcceptance:
		c.high++
		c.low = 0
		if c.high >= c.HysteresisWindows && c.depth < c.MaxDepth {
			c.depth++
			c.high = 0
		}
	case rate <= c.LowerAcceptance:
		c.low++
		c.high = 0
		if c.low >= c.HysteresisWindows && c.depth > c.MinDepth {
			c.depth--
			c.low = 0
		}
	default:
		c.high = 0
		c.low = 0
	}
	return c.depth, nil
}
func EvaluateAdaptiveDraftDepth(c AdaptiveDraftDepthController, fixed int, samples []AdaptiveDraftSample, artifact string, dflashCandidate bool) (AdaptiveDraftReceipt, error) {
	if fixed <= 0 || artifact == "" || len(samples) == 0 {
		return AdaptiveDraftReceipt{}, errors.New("model: incomplete adaptive draft evaluation")
	}
	r := AdaptiveDraftReceipt{Schema: "fak-adaptive-draft-depth/1", Engine: "fak-native", Artifact: artifact, Selector: "acceptance-hysteresis", QualityConstraint: "target verifier remains exactness boundary", Rollback: "set adaptive draft policy off and retain fixed depth", FixedDepth: fixed}
	if dflashCandidate {
		r.Selector = "dflash2-algorithmic-candidate-rejected-no-trained-layer"
	}
	for _, s := range samples {
		d, e := c.Observe(s)
		if e != nil {
			return AdaptiveDraftReceipt{}, e
		}
		r.Depths = append(r.Depths, d)
		r.AcceptedTokens += s.Accepted
		r.ProposedTokens += s.Proposed
		r.VerifierBytes += s.VerifierBytes
		r.VerifierFLOPs += s.VerifierFLOPs
		r.LatencyMilliseconds += s.LatencyMilliseconds
		r.Joules += s.Joules
	}
	r.FinalDepth = r.Depths[len(r.Depths)-1]
	r.AcceptanceRate = float64(r.AcceptedTokens) / float64(r.ProposedTokens)
	return r, nil
}
