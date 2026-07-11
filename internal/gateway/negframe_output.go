package gateway

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/negframe"
)

// negframeOutputAudit is the #3567 output-side shadow: a sampled, observe-only
// counter of negatively-framed spans in the MODEL's own outbound prose. It is the
// #3566 emit-time Reframe pass (which flips fak's INBOUND guard notes to positive
// voice) pointed the other way -- at what the model emits -- so the #3546 A/B can
// ask whether reframing the input shifts the framing of the output. It never
// mutates the assistant content or the wire: the only effect is the counter.
//
// It is DEFAULT-OFF (rate 0): a shadow/ablation instrument must not perturb the
// serve hot path in production, so the classification only runs once an operator or
// the #3546 A/B (via #3568's lever) opts in with FAK_NEGFRAME_OUTPUT_SAMPLE > 0.
// When off, observe() short-circuits on a single float compare -- the vDSO serve
// latency floor (TestSyscallServeLatencyDistribution) is untouched. When on,
// negframe.Classify is a bounded regex pass over prose (code fences skipped), and
// the sample rate bounds how many responses pay it.
type negframeOutputAudit struct {
	rate      float64       // [0,1]: 0 = off (default), 1 = every sampled response
	calls     atomic.Uint64 // responses considered (drives fractional sampling)
	scanned   atomic.Uint64 // responses actually classified (post-sampling)
	negatives atomic.Uint64 // total negative-framing findings (fak_negframe_output_negatives_total)
}

// outputNegframeAudit is the process-wide singleton the serve paths fold into and
// renderMetrics publishes. Package-level (not a Server field) because the counter
// is process-global telemetry, like the other fak_*_total families -- and so the
// #3567 seam adds no surface to the actively-contended Server struct/constructor.
var outputNegframeAudit = newNegframeOutputAudit()

func newNegframeOutputAudit() *negframeOutputAudit {
	return &negframeOutputAudit{rate: resolveNegframeOutputRate(os.Getenv("FAK_NEGFRAME_OUTPUT_SAMPLE"))}
}

// resolveNegframeOutputRate parses the sample-rate knob, clamping to [0,1] and
// defaulting to OFF (0) on an empty or malformed value -- so the shadow stays off
// the hot path unless explicitly enabled.
func resolveNegframeOutputRate(v string) float64 {
	v = strings.TrimSpace(v)
	if v == "" {
		return 0
	}
	f, err := strconv.ParseFloat(v, 64)
	if err != nil || f < 0 {
		return 0
	}
	if f > 1 {
		return 1.0
	}
	return f
}

// observe folds one model-output response into the shadow counter: it samples per
// the configured rate, and on a sampled response adds the negframe finding count.
// Pure telemetry -- the argument is never modified and nothing is returned.
func (a *negframeOutputAudit) observe(text string) {
	if a == nil || a.rate <= 0 {
		return // shadow disabled: a single compare, zero hot-path cost
	}
	n := a.calls.Add(1)
	if !a.shouldScan(n) {
		return
	}
	a.scanned.Add(1)
	if neg := len(negframe.Classify("model-output", text)); neg > 0 {
		a.negatives.Add(uint64(neg))
	}
}

// shouldScan is the deterministic (rand-free) sampling gate: rate>=1 scans every
// response, rate<=0 none, and a fractional rate scans when the running rate*count
// product crosses the next integer -- an even 1-in-N spread with no shared RNG, so
// it is reproducible in a test and cheap on the hot path.
func (a *negframeOutputAudit) shouldScan(n uint64) bool {
	if a.rate >= 1 {
		return true
	}
	if a.rate <= 0 {
		return false
	}
	return int64(float64(n)*a.rate) > int64(float64(n-1)*a.rate)
}

// writeMetrics renders the shadow's Prometheus fragment onto the shared /metrics
// surface. Always emitted (a real, always-armed family): the counter reads 0 until
// the first sampled negative.
func (a *negframeOutputAudit) writeMetrics(b *strings.Builder) {
	if a == nil || b == nil {
		return
	}
	writeHelpType(b, "fak_negframe_output_negatives_total",
		"Negative-framing spans found in model OUTPUT prose (sampled shadow, observe-only; #3567).", "counter")
	fmt.Fprintf(b, "fak_negframe_output_negatives_total{surface=\"model_output\"} %d\n", a.negatives.Load())
	writeHelpType(b, "fak_negframe_output_scanned_total",
		"Model-output responses classified by the negframe output shadow (post-sampling).", "counter")
	fmt.Fprintf(b, "fak_negframe_output_scanned_total{surface=\"model_output\"} %d\n", a.scanned.Load())
}
