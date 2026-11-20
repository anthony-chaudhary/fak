package engine

import (
	"errors"
	"math"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// SamplingStream is a named namespace for coordinate-keyed sampling draws
// (issue #14493). Three distinct streams -- draft / accept / residual -- keep
// draws for different purposes from colliding for the same (seed, position,
// token). The constants are the upstream stream salts (hellojiaru/
// deepseek-v4.1-flash-gb10, spec_sampling.h @ 7a87dfb4, Apache-2.0), preserved
// verbatim so independent implementations agree bit-for-bit.
type SamplingStream uint64

const (
	// SamplingStreamDraft namespaces draft-token draws.
	SamplingStreamDraft SamplingStream = 0xD2B74407B1CE6E93
	// SamplingStreamAccept namespaces accept/reject draws.
	SamplingStreamAccept SamplingStream = 0xCA5A826395121157
	// SamplingStreamResidual namespaces residual (excess-probability) draws.
	SamplingStreamResidual SamplingStream = 0x9E6C63D0676A9A99
)

// SamplingCoord is the coordinate key of a probabilistic sampling draw
// (issue #14493): the request seed, the predicted absolute token position, the
// stream namespace, and the token id. A draw is a pure function of this
// coordinate alone, so it is invariant to batch entry, CUDA launch identity,
// rank, and accepted-draft count -- the topology-invariance property the
// upstream contract (spec_sampling.h:27-51 @ 7a87dfb4) requires.
type SamplingCoord struct {
	Seed     uint64         // request seed
	Position int64          // predicted absolute token position
	Stream   SamplingStream // draft / accept / residual namespace
	Token    uint32         // token id
}

// mix is the splitmix64 finalizer (upstream spec_sampling::mix): two
// multiply-xor rounds that avalanche a 64-bit input.
func mix(x uint64) uint64 {
	x ^= x >> 31
	x *= 0xBF58476D1CE4E5B9
	x ^= x >> 29
	x *= 0x94D049BB133111EB
	x ^= x >> 32
	return x
}

// unit maps a 64-bit hash to a 23-bit midpoint strictly inside (0,1)
// (upstream spec_sampling::unit): the top 23 bits, offset by a half step and
// scaled by 1/2^23. Every result is exactly representable in binary32 and never
// a boundary value, so float32 consumers see the same midpoint.
func unit(h uint64) float64 {
	return (float64(h>>41) + 0.5) * (1.0 / 8388608.0)
}

// Draw returns the deterministic 23-bit midpoint in (0,1) for this coordinate
// (upstream spec_sampling::uniform). It is a pure function of
// (Seed, Position, Stream, Token) alone: permuting batch entry, CUDA launch
// identity, rank, or accepted-draft count leaves the draw bit-identical.
// Position is folded in through its low 32 bits, matching the upstream uint32
// truncation of the position argument.
func (c SamplingCoord) Draw() float64 {
	pos := uint64(uint32(c.Position))
	tok := uint64(c.Token)
	return unit(mix(mix(c.Seed^uint64(c.Stream)) ^ (pos * 0x9E3779B97F4A7C15) ^ (tok * 0xD1B54A32D192ED03)))
}

// ErrNonFiniteInput is returned by LogPositiveDifference when either input is
// NaN or ±Inf: a non-finite log-probability has no defined positive difference.
var ErrNonFiniteInput = errors.New("engine: non-finite log-probability input")

// LogPositiveDifference computes log(max(p-q, 0)) given the log-probabilities
// lp = log(p) and lq = log(q), WITHOUT exponentiating p and q separately
// (upstream spec_sampling::log_positive_difference @ 7a87dfb4). This is the
// numerically safe log-space comparison for two nearly-equal probabilities:
// the raw difference p - q can underflow or cancel in float, while the
// identity log(p-q) = lp + log(1 - exp(lq-lp)) -- evaluated as
// lp + log(-expm1(lq-lp)) -- stays exact. When p <= q (lp <= lq) the positive
// difference is empty and the result is -Inf. Non-finite inputs are refused
// with ErrNonFiniteInput.
func LogPositiveDifference(lp, lq float64) (float64, error) {
	if math.IsNaN(lp) || math.IsInf(lp, 0) || math.IsNaN(lq) || math.IsInf(lq, 0) {
		return 0, ErrNonFiniteInput
	}
	if lp <= lq {
		return math.Inf(-1), nil
	}
	return lp + math.Log(-math.Expm1(lq-lp)), nil
}

// samplingCoordinatesCap is the negotiable capability token advertising the
// coordinate-keyed sampling ABI, in the same style as the determinism tokens
// (engine.vllm.determinism.*), so consumers can probe for it.
const samplingCoordinatesCap abi.Capability = "engine.sampling.coordinates"

// SamplingCoordinatesCapability returns the negotiable capability token for the
// coordinate-keyed sampling ABI.
func SamplingCoordinatesCapability() abi.Capability {
	return samplingCoordinatesCap
}

func init() {
	// Register the sampling-coordinate capability token as a negotiable feature
	// so it appears in the capability registry alongside the determinism tokens.
	abi.RegisterCapability(samplingCoordinatesCap)
}
