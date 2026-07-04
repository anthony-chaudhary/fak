package headroom

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// TestRunBenchNativeSavesAggregate witnesses the native compressor's realized
// savings on the representative corpus: a large aggregate win, the expected codec
// firing per sample shape, and the incompressible prose control left untouched.
func TestRunBenchNativeSavesAggregate(t *testing.T) {
	native, ok := Lookup(NativeName)
	if !ok {
		t.Fatal("native compressor not registered")
	}
	r := RunBench(native, BenchCorpus())
	if r.OrigTotal <= 0 {
		t.Fatal("empty corpus")
	}
	if r.Owner != "fak" || r.Dependency != "in_process" || r.Fidelity != "recoverable" ||
		r.Evidence != "witnessed" || r.Status != "measured" {
		t.Fatalf("native attribution/status = %+v", r)
	}
	if r.Saved < 0.5 {
		t.Fatalf("aggregate saving = %.3f, want >= 0.5 on the representative corpus\n%s", r.Saved, r.Render())
	}

	// Each codec the native transforms can fire should appear somewhere, and the
	// prose control must record no saving.
	codecs := map[string]string{} // sample -> codec
	for _, s := range r.Samples {
		codecs[s.Name] = s.Codec
	}
	allCodecs := strings.Join(valuesOf(codecs), " ")
	for _, want := range []string{"ansi-strip", "cr-collapse", "line-fold", "line-dedup", "json-min"} {
		if !strings.Contains(allCodecs, want) {
			t.Fatalf("expected codec %q to fire somewhere in the corpus; codecs=%q", want, allCodecs)
		}
	}
	if c := codecs["plain-prose"]; c != "(none)" {
		t.Fatalf("plain prose should not compress, got codec %q", c)
	}
	for _, s := range r.Samples {
		if s.Name == "plain-prose" && s.Saved != 0 {
			t.Fatalf("plain prose saved=%.3f, want 0", s.Saved)
		}
		if s.Name == "plain-prose" && s.Status != "no_effect" {
			t.Fatalf("plain prose status=%q, want no_effect", s.Status)
		}
		if s.Saved > 0 && s.Status != "saved" {
			t.Fatalf("sample %q saved %.3f but status=%q", s.Name, s.Saved, s.Status)
		}
		if s.NewLen > s.OrigLen {
			t.Fatalf("sample %q expanded: %d -> %d", s.Name, s.OrigLen, s.NewLen)
		}
	}
}

// TestRunBenchNoopZero: the noop compressor saves nothing — the honest off baseline.
func TestRunBenchNoopZero(t *testing.T) {
	noop, ok := Lookup(NoopName)
	if !ok {
		t.Fatal("noop not registered")
	}
	r := RunBench(noop, BenchCorpus())
	if r.Saved != 0 || r.NewTotal != r.OrigTotal {
		t.Fatalf("noop must save nothing: saved=%.3f orig=%d new=%d", r.Saved, r.OrigTotal, r.NewTotal)
	}
	if r.Status != "no-op" || r.Owner != "fak" || r.Dependency != "none" || r.Fidelity != "no-op" {
		t.Fatalf("noop attribution/status = %+v", r)
	}
	for _, s := range r.Samples {
		if s.Codec != "(none)" || s.Status != "no-op" {
			t.Fatalf("noop sample %q codec/status=%q/%q, want (none)/no-op", s.Name, s.Codec, s.Status)
		}
	}
}

func TestRunBenchHeadroomUnavailableIsNotNoSaving(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer srv.Close()
	comp := headroomBridge{url: srv.URL, client: srv.Client()}
	r := RunBench(comp, []BenchInput{{Name: "sample", Bytes: []byte("a compressible looking sample that cannot reach headroom")}})
	if r.Status != "unavailable" || r.Owner != "external" || r.Dependency != "external_http_sidecar" {
		t.Fatalf("headroom unavailable report = %+v", r)
	}
	if len(r.Samples) != 1 || r.Samples[0].Status != "unavailable" ||
		!strings.Contains(r.Samples[0].Reason, "HTTP 503") {
		t.Fatalf("headroom unavailable sample = %+v", r.Samples)
	}
}

// TestRunBenchDeterministic: same compressor + corpus => identical report.
func TestRunBenchDeterministic(t *testing.T) {
	native, _ := Lookup(NativeName)
	a := RunBench(native, BenchCorpus())
	b := RunBench(native, BenchCorpus())
	if a.Render() != b.Render() {
		t.Fatalf("bench not deterministic:\n--- a ---\n%s\n--- b ---\n%s", a.Render(), b.Render())
	}
}

// TestBenchRender: the rendered table carries the compressor name, the TOTAL row,
// and a row per sample.
func TestBenchRender(t *testing.T) {
	native, _ := Lookup(NativeName)
	r := RunBench(native, BenchCorpus())
	s := r.Render()
	if !strings.Contains(s, "compressor: native") {
		t.Fatalf("render missing compressor name:\n%s", s)
	}
	if !strings.Contains(s, "status:") || !strings.Contains(s, "owner=fak") || !strings.Contains(s, "no_effect") {
		t.Fatalf("render missing status/attribution details:\n%s", s)
	}
	if !strings.Contains(s, "TOTAL") {
		t.Fatalf("render missing TOTAL row:\n%s", s)
	}
	for _, in := range BenchCorpus() {
		if !strings.Contains(s, in.Name) {
			t.Fatalf("render missing sample %q:\n%s", in.Name, s)
		}
	}
}

func valuesOf(m map[string]string) []string {
	out := make([]string, 0, len(m))
	for _, v := range m {
		out = append(out, v)
	}
	return out
}
