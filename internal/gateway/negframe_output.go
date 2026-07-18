package gateway

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/negframe"
)

const negframeOutputQueueDepth = 32

// negframeOutputAudit is the #3567 output-side shadow: a sampled, observe-only
// counter of negatively-framed spans in the MODEL's own outbound prose. It is the
// #3566 emit-time Reframe pass pointed the other way, at what the model emits.
// It never mutates assistant content or waits for classification: sampled prose is
// offered to one bounded background worker and a full queue drops telemetry rather
// than adding response latency.
type negframeOutputAudit struct {
	rate      float64       // [0,1]: 0 = off (default), 1 = every response
	calls     atomic.Uint64 // responses considered (drives fractional sampling)
	scanned   atomic.Uint64 // responses classified by the background worker
	negatives atomic.Uint64 // total negative-framing findings
	dropped   atomic.Uint64 // sampled responses omitted because the queue was full

	start    sync.Once
	queue    chan string
	classify func(string) int // test seam; production uses classifyModelOutput
}

// outputNegframeAudit is the process-wide singleton the serve paths fold into and
// renderMetrics publishes.
var outputNegframeAudit = newNegframeOutputAudit()

func newNegframeOutputAudit() *negframeOutputAudit {
	return &negframeOutputAudit{rate: resolveNegframeOutputRate(os.Getenv("FAK_NEGFRAME_OUTPUT_SAMPLE"))}
}

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
		return 1
	}
	return f
}

// classifyModelOutput deliberately calls the same pure detector used by the tree
// card. Keeping the call here makes detector drift mechanically visible.
func classifyModelOutput(text string) int {
	return len(negframe.Classify("model-output", text))
}

// observe samples and enqueues one immutable model-output string. It never runs
// classification inline and never blocks: if the bounded shadow queue is full,
// telemetry is dropped and counted instead of delaying the response.
func (a *negframeOutputAudit) observe(text string) {
	if a == nil || a.rate <= 0 {
		return
	}
	n := a.calls.Add(1)
	if !a.shouldScan(n) {
		return
	}
	a.startWorker()
	select {
	case a.queue <- text:
	default:
		a.dropped.Add(1)
	}
}

func (a *negframeOutputAudit) startWorker() {
	a.start.Do(func() {
		if a.queue == nil {
			a.queue = make(chan string, negframeOutputQueueDepth)
		}
		classify := a.classify
		if classify == nil {
			classify = classifyModelOutput
		}
		go func() {
			for text := range a.queue {
				if count := classify(text); count > 0 {
					a.negatives.Add(uint64(count))
				}
				a.scanned.Add(1)
			}
		}()
	})
}

func (a *negframeOutputAudit) shouldScan(n uint64) bool {
	if a.rate >= 1 {
		return true
	}
	if a.rate <= 0 {
		return false
	}
	return int64(float64(n)*a.rate) > int64(float64(n-1)*a.rate)
}

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
	writeHelpType(b, "fak_negframe_output_dropped_total",
		"Sampled model-output responses omitted to keep the shadow off the response path.", "counter")
	fmt.Fprintf(b, "fak_negframe_output_dropped_total{surface=\"model_output\"} %d\n", a.dropped.Load())
	writePositiveComplementMetrics(b)
}
