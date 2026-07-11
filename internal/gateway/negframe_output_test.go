package gateway

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/negframe"
)

// TestOutputNegationCounter is the #3567 witness: a sampled-shadow audit over a
// canned model response folds exactly the negframe finding count into the counter,
// never touches the response bytes, and is fully skipped when sampling is off.
func TestOutputNegationCounter(t *testing.T) {
	// A canned model "response" carrying unambiguous negative idioms plus
	// positive/neutral lines that must not inflate the count.
	const modelOutput = "Here is the plan.\n" +
		"Do not forget to run the tests.\n" + // mechanical negative
		"Don't hesitate to ask for help.\n" + // mechanical negative
		"I will ship it now." // positive/neutral

	wantN := len(negframe.Classify("model-output", modelOutput))
	if wantN == 0 {
		t.Fatalf("fixture sanity: canned output must contain at least one negative; got 0")
	}

	// rate=1: every response scanned; counter == the finding count.
	on := &negframeOutputAudit{rate: 1}
	on.observe(modelOutput)
	if got := on.negatives.Load(); got != uint64(wantN) {
		t.Fatalf("sampled counter = %d, want %d", got, wantN)
	}
	if got := on.scanned.Load(); got != 1 {
		t.Fatalf("scanned = %d, want 1", got)
	}

	// The counter accumulates across responses.
	on.observe(modelOutput)
	if got := on.negatives.Load(); got != uint64(2*wantN) {
		t.Fatalf("counter after 2 observes = %d, want %d", got, 2*wantN)
	}

	// rate=0: shadow fully off -- no scan, no count (DoD: off the hot path).
	off := &negframeOutputAudit{rate: 0}
	off.observe(modelOutput)
	if got := off.negatives.Load(); got != 0 {
		t.Fatalf("off-shadow counter = %d, want 0", got)
	}
	if got := off.scanned.Load(); got != 0 {
		t.Fatalf("off-shadow scanned = %d, want 0", got)
	}

	// The content channel is byte-identical regardless of the shadow: observe is
	// pure telemetry, and the adjudication-note seam returns content unchanged when
	// there is no note (nil adjs). Proves the shadow never mutates the wire.
	if got := prependAdjudicationContentNote(modelOutput, nil); got != modelOutput {
		t.Fatalf("content channel mutated: got %q want %q", got, modelOutput)
	}
}

// TestOutputNegationSampleRate covers the rate knob parse+clamp, the deterministic
// fractional-sampling spread, and the Prometheus fragment shape.
func TestOutputNegationSampleRate(t *testing.T) {
	for _, tc := range []struct {
		in   string
		want float64
	}{
		{"", 0}, {"1", 1.0}, {"0", 0}, {"0.5", 0.5}, {"bad", 0}, {"-1", 0}, {"2", 1.0},
	} {
		if got := resolveNegframeOutputRate(tc.in); got != tc.want {
			t.Errorf("resolveNegframeOutputRate(%q) = %v, want %v", tc.in, got, tc.want)
		}
	}

	// rate=0.5 gives an even, deterministic 1-in-2 spread: 5 scanned of 10.
	half := &negframeOutputAudit{rate: 0.5}
	for i := 0; i < 10; i++ {
		half.observe("Do not forget to x.")
	}
	if got := half.scanned.Load(); got != 5 {
		t.Fatalf("rate 0.5 over 10 observes = %d scanned, want 5", got)
	}

	// The metrics fragment renders the labeled counter with HELP/TYPE, house-style.
	a := &negframeOutputAudit{rate: 1}
	a.observe("Do not forget to run.")
	var b strings.Builder
	a.writeMetrics(&b)
	out := b.String()
	for _, want := range []string{
		"# TYPE fak_negframe_output_negatives_total counter",
		`fak_negframe_output_negatives_total{surface="model_output"}`,
		`fak_negframe_output_scanned_total{surface="model_output"} 1`,
	} {
		if !strings.Contains(out, want) {
			t.Errorf("metrics fragment missing %q; got:\n%s", want, out)
		}
	}
}
