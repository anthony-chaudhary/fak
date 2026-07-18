package gateway

import (
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/negframe"
)

func waitOutputAudit(t *testing.T, a *negframeOutputAudit, want uint64) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for a.scanned.Load() < want && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if got := a.scanned.Load(); got != want {
		t.Fatalf("scanned = %d, want %d", got, want)
	}
}

// TestOutputNegationCounter is the #3567 witness: a sampled shadow over canned
// model prose folds exactly the shared detector's findings into the counter while
// leaving response content byte-identical.
func TestOutputNegationCounter(t *testing.T) {
	const modelOutput = "Here is the plan.\n" +
		"Do not forget to run the tests.\n" +
		"Don't hesitate to ask for help.\n" +
		"I will ship it now."

	wantN := len(negframe.Classify("model-output", modelOutput))
	if wantN == 0 {
		t.Fatal("fixture sanity: canned output must contain a negative")
	}

	on := &negframeOutputAudit{rate: 1}
	on.observe(modelOutput)
	waitOutputAudit(t, on, 1)
	if got := on.negatives.Load(); got != uint64(wantN) {
		t.Fatalf("sampled counter = %d, want %d", got, wantN)
	}
	on.observe(modelOutput)
	waitOutputAudit(t, on, 2)
	if got := on.negatives.Load(); got != uint64(2*wantN) {
		t.Fatalf("counter after 2 observes = %d, want %d", got, 2*wantN)
	}

	off := &negframeOutputAudit{rate: 0}
	off.observe(modelOutput)
	if got := off.scanned.Load(); got != 0 {
		t.Fatalf("off-shadow scanned = %d, want 0", got)
	}
	if got := prependAdjudicationContentNote(modelOutput, nil); got != modelOutput {
		t.Fatalf("content channel mutated: got %q want %q", got, modelOutput)
	}
}

// TestOutputNegationAuditReturnsBeforeClassification proves sampled-on classification remains off
// the response path: observe returns while a deliberately blocked detector is
// still pending. A bounded full queue also returns immediately and records drops.
func TestOutputNegationAuditReturnsBeforeClassification(t *testing.T) {
	entered := make(chan struct{})
	release := make(chan struct{})
	a := &negframeOutputAudit{rate: 1, classify: func(string) int {
		select {
		case <-entered:
		default:
			close(entered)
		}
		<-release
		return 1
	}}

	start := time.Now()
	a.observe("Do not wait.")
	if elapsed := time.Since(start); elapsed > 50*time.Millisecond {
		t.Fatalf("sampled observe waited %s for classification", elapsed)
	}
	select {
	case <-entered:
	case <-time.After(time.Second):
		t.Fatal("background classifier did not start")
	}

	start = time.Now()
	for i := 0; i < negframeOutputQueueDepth+2; i++ {
		a.observe("Do not queue forever.")
	}
	if elapsed := time.Since(start); elapsed > 50*time.Millisecond {
		t.Fatalf("full shadow queue delayed response path by %s", elapsed)
	}
	if got := a.dropped.Load(); got == 0 {
		t.Fatal("full queue did not record dropped telemetry")
	}
	close(release)
}

func TestOutputNegationSampleRate(t *testing.T) {
	for _, tc := range []struct {
		in   string
		want float64
	}{
		{"", 0}, {"1", 1}, {"0", 0}, {"0.5", 0.5}, {"bad", 0}, {"-1", 0}, {"2", 1},
	} {
		if got := resolveNegframeOutputRate(tc.in); got != tc.want {
			t.Errorf("resolveNegframeOutputRate(%q) = %v, want %v", tc.in, got, tc.want)
		}
	}

	half := &negframeOutputAudit{rate: 0.5}
	for i := 0; i < 10; i++ {
		half.observe("Do not forget to x.")
	}
	waitOutputAudit(t, half, 5)

	a := &negframeOutputAudit{rate: 1}
	a.observe("Do not forget to run.")
	waitOutputAudit(t, a, 1)
	var b strings.Builder
	a.writeMetrics(&b)
	out := b.String()
	for _, want := range []string{
		"# TYPE fak_negframe_output_negatives_total counter",
		`fak_negframe_output_negatives_total{surface="model_output"}`,
		`fak_negframe_output_scanned_total{surface="model_output"} 1`,
		`fak_negframe_output_dropped_total{surface="model_output"} 0`,
	} {
		if !strings.Contains(out, want) {
			t.Errorf("metrics fragment missing %q; got:\n%s", want, out)
		}
	}
}

func TestOutputNegframeWiredIntoRenderMetrics(t *testing.T) {
	srv := newTestServer(t)
	out := srv.renderMetrics()
	for _, want := range []string{
		"# TYPE fak_negframe_output_negatives_total counter",
		`fak_negframe_output_negatives_total{surface="model_output"} 0`,
		"# TYPE fak_negframe_output_scanned_total counter",
		`fak_negframe_output_scanned_total{surface="model_output"} 0`,
		`fak_negframe_output_dropped_total{surface="model_output"} 0`,
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("negframe output family not wired into live /metrics render (missing %q):\n%s", want, out)
		}
	}
}
