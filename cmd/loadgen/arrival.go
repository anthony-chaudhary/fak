// Seeded arrival-pattern interval generators for request pacing.
//
// DIRECT-PORT of the interval generators from ai-dynamo/aiperf
// (src/aiperf/timing/intervals.py), pinned at commit
// ab610add4e8876fca223197596c728e17365f381 (aiperf v0.14.0):
//
//	https://github.com/ai-dynamo/aiperf/blob/ab610add4e8876fca223197596c728e17365f381/src/aiperf/timing/intervals.py
//
// The upstream distributions are preserved exactly: Poisson uses exponential
// inter-arrivals with mean 1/rate; Gamma uses shape=smoothness and
// scale=1/(rate*smoothness) so the mean stays 1/rate while smoothness tunes
// burstiness (the same value as vLLM's burstiness parameter yields the same
// distribution); Constant returns the fixed period 1/rate; ConcurrencyBurst
// returns zero delay, so throughput is governed by the concurrency semaphore
// rather than a rate. All four support setRate for mid-run ramping, applied on
// the next nextInterval call.
//
// Deviation from upstream, forced by this harness: upstream derives a named
// substream (rng.derive) from a process-global generator; loadgen is a single
// stdlib-only binary with an injectable clock, so each generator is seeded from
// ArrivalSeed mixed with its pattern name (same intent: independent, reproducible
// streams per distribution, stable across runs for a fixed seed).
package main

import (
	"fmt"
	"hash/fnv"
	"math"
	"math/rand"
	"time"
)

// ArrivalPattern names an inter-arrival distribution. The empty string means
// "no pattern": issuance falls back to the constant-rate pacer (or unpaced
// worker drain), preserving today's behavior byte-for-byte.
type ArrivalPattern string

const (
	arrivalPoisson  ArrivalPattern = "poisson"
	arrivalGamma    ArrivalPattern = "gamma"
	arrivalConstant ArrivalPattern = "constant"
	arrivalBurst    ArrivalPattern = "concurrency_burst"
)

// arrivalPatterns lists every accepted pattern name, in a stable order.
var arrivalPatterns = []ArrivalPattern{arrivalPoisson, arrivalGamma, arrivalConstant, arrivalBurst}

// parseArrivalPattern accepts the pattern names (case-insensitively) and the
// empty string. Anything else is refused with the closed set of valid names.
func parseArrivalPattern(s string) (ArrivalPattern, error) {
	if s == "" {
		return "", nil
	}
	for _, p := range arrivalPatterns {
		if string(p) == s {
			return p, nil
		}
	}
	return "", fmt.Errorf("unknown arrival pattern %q (valid: poisson, gamma, constant, concurrency_burst)", s)
}

// intervalGenerator yields inter-arrival delays in SECONDS and supports live
// rate changes for rate ramping. It mirrors aiperf's
// IntervalGeneratorProtocol: next_interval() plus set_rate().
type intervalGenerator interface {
	// nextInterval returns the delay until the next dispatch, in seconds.
	// ConcurrencyBurst always returns 0.
	nextInterval() float64
	// setRate updates the request rate for ramping; takes effect on the next
	// nextInterval call. It is a no-op for ConcurrencyBurst.
	setRate(newRate float64) error
	// rate reports the current request rate (0 for ConcurrencyBurst).
	rate() float64
}

// deriveSource builds a reproducible *rand.Rand for a named stream: the pattern
// name is hashed with FNV-1a and XOR-mixed into the seed, so two generators with
// the same seed but different patterns draw independent sequences, and a fixed
// seed reproduces the same sequence across runs.
func deriveSource(seed int64, name string) *rand.Rand {
	h := fnv.New64a()
	_, _ = h.Write([]byte(name))
	return rand.New(rand.NewSource(seed ^ int64(h.Sum64()))) //nolint:gosec // reproducibility, not secrecy
}

// errNonPositiveRate is the closed sentinel for a rate that cannot produce
// arrivals.
var errNonPositiveRate = fmt.Errorf("loadgen: arrival rate must be > 0")

// rateValid reports whether r can seed or set a rate. NaN/Inf and <=0 are
// rejected so a bad flag fails loudly instead of producing a silent zero-rate
// stream.
func rateValid(r float64) bool { return r > 0 && !math.IsNaN(r) && !math.IsInf(r, 0) }

// poissonGenerator is a Poisson process (exponential inter-arrival times).
type poissonGenerator struct {
	rng   *rand.Rand
	rateS float64
}

func newPoissonGenerator(rate float64, seed int64) (*poissonGenerator, error) {
	if !rateValid(rate) {
		return nil, fmt.Errorf("%w: poisson got %g", errNonPositiveRate, rate)
	}
	return &poissonGenerator{rng: deriveSource(seed, "poisson"), rateS: rate}, nil
}

func (g *poissonGenerator) nextInterval() float64 { return g.rng.ExpFloat64() / g.rateS }

func (g *poissonGenerator) setRate(r float64) error {
	if !rateValid(r) {
		return fmt.Errorf("%w: poisson got %g", errNonPositiveRate, r)
	}
	g.rateS = r
	return nil
}

func (g *poissonGenerator) rate() float64 { return g.rateS }

// gammaGenerator generalizes Poisson with a tunable smoothness (Gamma shape):
// smoothness == 1 is exponential (Poisson-equivalent); <1 is burstier; >1 is
// more regular. The mean stays 1/rate for every smoothness.
type gammaGenerator struct {
	rng        *rand.Rand
	rateS      float64
	smoothness float64
}

func newGammaGenerator(rate, smoothness float64, seed int64) (*gammaGenerator, error) {
	if !rateValid(rate) {
		return nil, fmt.Errorf("%w: gamma got %g", errNonPositiveRate, rate)
	}
	if !rateValid(smoothness) {
		return nil, fmt.Errorf("loadgen: arrival smoothness must be > 0, got %g", smoothness)
	}
	return &gammaGenerator{rng: deriveSource(seed, "gamma"), rateS: rate, smoothness: smoothness}, nil
}

// nextInterval draws Gamma(shape=smoothness, scale=1/(rate*smoothness)); the
// scale keeps the mean at 1/rate regardless of smoothness.
func (g *gammaGenerator) nextInterval() float64 {
	return gammaVariate(g.rng, g.smoothness, 1.0/(g.rateS*g.smoothness))
}

func (g *gammaGenerator) setRate(r float64) error {
	if !rateValid(r) {
		return fmt.Errorf("%w: gamma got %g", errNonPositiveRate, r)
	}
	g.rateS = r
	return nil
}

func (g *gammaGenerator) rate() float64 { return g.rateS }

// constantGenerator emits perfectly regular intervals (period = 1/rate):
// deterministic baselines with no RNG.
type constantGenerator struct {
	rateS float64
}

func newConstantGenerator(rate float64) (*constantGenerator, error) {
	if !rateValid(rate) {
		return nil, fmt.Errorf("%w: constant got %g", errNonPositiveRate, rate)
	}
	return &constantGenerator{rateS: rate}, nil
}

func (g *constantGenerator) nextInterval() float64 { return 1.0 / g.rateS }

func (g *constantGenerator) setRate(r float64) error {
	if !rateValid(r) {
		return fmt.Errorf("%w: constant got %g", errNonPositiveRate, r)
	}
	g.rateS = r
	return nil
}

func (g *constantGenerator) rate() float64 { return g.rateS }

// burstGenerator issues credits immediately; throughput is limited only by the
// concurrency semaphore. rate() is 0 (rate is not applicable) and setRate is a
// no-op, matching upstream ConcurrencyBurstIntervalGenerator.
type burstGenerator struct{}

func newBurstGenerator() *burstGenerator { return &burstGenerator{} }

func (g *burstGenerator) nextInterval() float64 { return 0 }

func (g *burstGenerator) setRate(float64) error { return nil }

func (g *burstGenerator) rate() float64 { return 0 }

// newIntervalGenerator builds the generator named by pattern. It refuses the
// empty pattern (that is the "no pattern" sentinel, handled by the caller), an
// unknown name, and any non-positive rate/smoothness.
func newIntervalGenerator(pattern ArrivalPattern, rate, smoothness float64, seed int64) (intervalGenerator, error) {
	switch pattern {
	case arrivalPoisson:
		return newPoissonGenerator(rate, seed)
	case arrivalGamma:
		return newGammaGenerator(rate, smoothness, seed)
	case arrivalConstant:
		return newConstantGenerator(rate)
	case arrivalBurst:
		return newBurstGenerator(), nil
	case "":
		return nil, fmt.Errorf("loadgen: no arrival pattern selected")
	default:
		return nil, fmt.Errorf("unknown arrival pattern %q", pattern)
	}
}

// delaySource adapts an intervalGenerator into the pacer's per-step delay
// function (seconds -> time.Duration).
func delaySource(g intervalGenerator) func() time.Duration {
	return func() time.Duration {
		return time.Duration(g.nextInterval() * float64(time.Second))
	}
}

// CPython random.gammavariate constants (Lib/random.py): LOG4 = log(4),
// SG_MAGICCONST = 1 + log(4.5), E = e.
const (
	gammaLOG4         = 1.3862943611198906
	gammaSGMagicConst = 2.5040773967762742
)

// gammaVariate draws a Gamma(alpha, beta) sample using the exact algorithm
// CPython's random.gammavariate uses - the function aiperf's
// rng.gammavariate(shape, scale) ultimately calls. Go's stdlib has no Gamma
// sampler, so porting the upstream algorithm keeps the distribution identical
// to aiperf's for the same (shape, scale), including its branch structure:
//
//	alpha > 1: R.C.H. Cheng (1977) rejection scheme, "The generation of Gamma
//	           variables with non-integral shape parameters";
//	alpha == 1: exponential via -log(1-u)*beta (Poisson-equivalent);
//	0 < alpha < 1: Algorithm GS (Kennedy & Gentle).
//
// alpha and beta must be > 0; callers validate before reaching here.
func gammaVariate(rng *rand.Rand, alpha, beta float64) float64 {
	if alpha > 1.0 {
		ainv := math.Sqrt(2.0*alpha - 1.0)
		bbb := alpha - gammaLOG4
		ccc := alpha + ainv
		for {
			u1 := rng.Float64()
			if !(1e-7 < u1 && u1 < 0.9999999) {
				continue
			}
			u2 := 1.0 - rng.Float64()
			v := math.Log(u1/(1.0-u1)) / ainv
			x := alpha * math.Exp(v)
			z := u1 * u1 * u2
			r := bbb + ccc*v - x
			if r+gammaSGMagicConst-4.5*z >= 0.0 || r >= math.Log(z) {
				return x * beta
			}
		}
	}
	if alpha == 1.0 {
		return -math.Log(1.0-rng.Float64()) * beta
	}
	// 0 < alpha < 1: Algorithm GS (Kennedy & Gentle).
	for {
		u := rng.Float64()
		b := (math.E + alpha) / math.E
		p := b * u
		var x float64
		if p <= 1.0 {
			x = math.Pow(p, 1.0/alpha)
		} else {
			x = -math.Log((b - p) / alpha)
		}
		u1 := rng.Float64()
		if p > 1.0 {
			if u1 <= math.Pow(x, alpha-1.0) {
				return x * beta
			}
		} else if u1 <= math.Exp(-x) {
			return x * beta
		}
	}
}
