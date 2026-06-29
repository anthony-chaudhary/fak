package headroom

import (
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
	for _, s := range r.Samples {
		if s.Codec != "(none)" {
			t.Fatalf("noop sample %q codec=%q, want (none)", s.Name, s.Codec)
		}
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
