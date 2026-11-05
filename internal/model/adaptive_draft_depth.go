package model

import (
	"errors"
	"fmt"
	"math"
)

// AdaptiveDraftDepthController bounds speculative depth using observed acceptance with
// hysteresis. Target verification remains authoritative; this controls proposed work only.
type AdaptiveDraftDepthController struct {
	MinDepth, MaxDepth               int
	RaiseAcceptance, LowerAcceptance float64
	HysteresisWindows                int
	depth, high, low                 int
	generation                       int // advances on every fitted-depth move
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
			c.generation++
		}
	case rate <= c.LowerAcceptance:
		c.low++
		c.high = 0
		if c.low >= c.HysteresisWindows && c.depth > c.MinDepth {
			c.depth--
			c.low = 0
			c.generation++
		}
	default:
		c.high = 0
		c.low = 0
	}
	return c.depth, nil
}

// Generation reports how many times the fitted depth has moved since
// construction. A serving path that caches the fitted draft limit compares
// generations before each draft: an advanced counter means the cached limit
// is stale and must be re-read via Limit; an unchanged one means the cache
// stays valid.
func (c *AdaptiveDraftDepthController) Generation() int { return c.generation }

// Limit reports the currently fitted draft depth - MinDepth until the first
// fit moves it. It is the value a drafter refreshes when Generation advances.
func (c *AdaptiveDraftDepthController) Limit() int {
	if c.depth == 0 {
		return c.MinDepth
	}
	return c.depth
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

// ErrInvalidAdaptiveDraftModule fails closed on empty artifacts, non-positive
// draft lengths, or empty compiled modules.
var ErrInvalidAdaptiveDraftModule = errors.New("model: invalid adaptive draft module")

// AdaptiveDraftModuleKey names a compiled spec-decode module by its base
// artifact plus the draft length the compiler observed. Keying the name by
// draft length keeps a length-mismatched request from reusing an artifact
// built for another draft length.
func AdaptiveDraftModuleKey(artifact string, draftLength int) (string, error) {
	if artifact == "" || draftLength <= 0 {
		return "", ErrInvalidAdaptiveDraftModule
	}
	return fmt.Sprintf("%s_d%d", artifact, draftLength), nil
}

// AdaptiveDraftModuleRegistry caches compiled spec-decode artifacts under
// their draft-length keys. A lookup at any other draft length is a miss,
// never a hit: the caller recompiles instead of binding a mismatched module.
type AdaptiveDraftModuleRegistry struct {
	modules map[string]string
}

// NewAdaptiveDraftModuleRegistry returns an empty registry.
func NewAdaptiveDraftModuleRegistry() *AdaptiveDraftModuleRegistry {
	return &AdaptiveDraftModuleRegistry{modules: make(map[string]string)}
}

// Compile records a freshly compiled module under its draft-length key.
func (r *AdaptiveDraftModuleRegistry) Compile(artifact string, draftLength int, module string) error {
	if r.modules == nil {
		r.modules = make(map[string]string)
	}
	if module == "" {
		return ErrInvalidAdaptiveDraftModule
	}
	key, err := AdaptiveDraftModuleKey(artifact, draftLength)
	if err != nil {
		return err
	}
	r.modules[key] = module
	return nil
}

// Lookup returns the compiled module bound to exactly (artifact, draftLength);
// every other draft length reports a miss so the caller recompiles.
func (r *AdaptiveDraftModuleRegistry) Lookup(artifact string, draftLength int) (string, bool) {
	key, err := AdaptiveDraftModuleKey(artifact, draftLength)
	if err != nil {
		return "", false
	}
	module, ok := r.modules[key]
	return module, ok
}
