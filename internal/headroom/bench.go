package headroom

import (
	"context"
	"fmt"
	"strings"
)

// This file is the measurement surface for the context-savings AREA: a
// deterministic, no-model, no-clock A/B that replays a built-in corpus of the
// tool-output SHAPES a coding agent actually streams into model context
// (colorized test output, a progress bar, scattered warnings, pretty JSON, a
// retry-spam log, a CRLF build log, and a plain-prose control) through a chosen
// Compressor and reports the realized byte savings per sample and in aggregate.
//
// It exists so the compressor's wins are WITNESSED, not asserted — `fak headroom
// bench` prints the realized ratio on a representative corpus. HONEST SCOPE: this
// is the realized saving on THIS corpus (a representative mix, not all traffic);
// the per-sample rows show where the win concentrates (colorized / progress /
// duplicate-heavy output) and where it is ~0 (unique prose), so the number is
// never read as a blanket ratio.

// BenchInput is one named corpus sample fed to the compressor.
type BenchInput struct {
	Name  string `json:"name"`
	Bytes []byte `json:"-"`
}

// BenchSample is the compression result for one corpus sample.
type BenchSample struct {
	Name    string  `json:"name"`
	Kind    string  `json:"kind"`
	Status  string  `json:"status"`
	Codec   string  `json:"codec"`
	OrigLen int     `json:"orig_len"`
	NewLen  int     `json:"new_len"`
	Saved   float64 `json:"saved_ratio"`
	Reason  string  `json:"reason,omitempty"`
}

// BenchReport is the aggregate compression result over a corpus.
type BenchReport struct {
	Compressor string        `json:"compressor"`
	Owner      string        `json:"owner,omitempty"`
	Dependency string        `json:"dependency,omitempty"`
	Fidelity   string        `json:"fidelity,omitempty"`
	Evidence   string        `json:"evidence,omitempty"`
	Status     string        `json:"status,omitempty"`
	Reason     string        `json:"reason,omitempty"`
	Samples    []BenchSample `json:"samples"`
	OrigTotal  int           `json:"orig_total"`
	NewTotal   int           `json:"new_total"`
	Saved      float64       `json:"saved_ratio"`
}

// savedRatio is the fraction of bytes removed, in [0,1] (0 when nothing helped).
func savedRatio(orig, neu int) float64 {
	if orig <= 0 || neu >= orig {
		return 0
	}
	return float64(orig-neu) / float64(orig)
}

// RunBench runs each sample through comp and aggregates the savings. It is
// deterministic for a given (comp, inputs): no clock, no RNG, pure over the
// compressor's pure Compress.
func RunBench(comp Compressor, inputs []BenchInput) BenchReport {
	r := BenchReport{Compressor: comp.Name()}
	r.Owner, r.Dependency, r.Fidelity, r.Evidence = benchAttribution(comp.Name())
	for _, in := range inputs {
		out, err := comp.Compress(context.Background(), Input{Bytes: in.Bytes})
		if err != nil {
			out = passthroughWithReason(Input{Bytes: in.Bytes}, "error", err.Error())
		}
		if out.NewLen == 0 && len(in.Bytes) > 0 {
			out.NewLen = len(in.Bytes)
		}
		codec := out.Codec
		if !out.Compressed {
			codec = "(none)"
		}
		status := out.Status
		if status == "" {
			status = benchSampleStatus(comp.Name(), out)
		}
		reason := out.Reason
		if reason == "" {
			reason = benchSampleReason(comp.Name(), out)
		}
		r.Samples = append(r.Samples, BenchSample{
			Name:    in.Name,
			Kind:    Detect(in.Bytes).String(),
			Status:  status,
			Codec:   codec,
			OrigLen: len(in.Bytes),
			NewLen:  out.NewLen,
			Saved:   savedRatio(len(in.Bytes), out.NewLen),
			Reason:  reason,
		})
		r.OrigTotal += len(in.Bytes)
		r.NewTotal += out.NewLen
	}
	r.Saved = savedRatio(r.OrigTotal, r.NewTotal)
	r.Status, r.Reason = benchAggregateStatus(comp.Name(), r.Samples, r.Saved)
	return r
}

// Render returns a human-readable table + aggregate of the report.
func (r BenchReport) Render() string {
	var b strings.Builder
	fmt.Fprintf(&b, "compressor: %s\n\n", r.Compressor)
	if r.Status != "" || r.Owner != "" || r.Dependency != "" || r.Fidelity != "" {
		fmt.Fprintf(&b, "status:     %s owner=%s dependency=%s fidelity=%s evidence=%s\n", dash(r.Status), dash(r.Owner), dash(r.Dependency), dash(r.Fidelity), dash(r.Evidence))
		if r.Reason != "" {
			fmt.Fprintf(&b, "reason:     %s\n", r.Reason)
		}
		fmt.Fprintf(&b, "\n")
	}
	fmt.Fprintf(&b, "%-20s %-8s %-13s %-26s %9s %9s %7s %s\n", "sample", "kind", "status", "codec", "orig", "new", "saved", "reason")
	for _, s := range r.Samples {
		fmt.Fprintf(&b, "%-20s %-8s %-13s %-26s %9d %9d %6.1f%% %s\n", s.Name, s.Kind, dash(s.Status), s.Codec, s.OrigLen, s.NewLen, s.Saved*100, s.Reason)
	}
	fmt.Fprintf(&b, "%-20s %-8s %-13s %-26s %9d %9d %6.1f%%\n", "TOTAL", "", dash(r.Status), "", r.OrigTotal, r.NewTotal, r.Saved*100)
	return b.String()
}

func benchAttribution(compressor string) (owner, dependency, fidelity, evidence string) {
	switch strings.ToLower(strings.TrimSpace(compressor)) {
	case HeadroomName:
		return "external", "external_http_sidecar", "recoverable", "observed"
	case NativeName:
		return "fak", "in_process", "recoverable", "witnessed"
	case LinguaName:
		return "external", "llmlingua_http_service", "lossy_original_cas_preserved", "observed"
	case NoopName:
		return "fak", "none", "no-op", "configured"
	default:
		return "unknown", "registered_plugin", "unknown", "configured"
	}
}

func benchSampleStatus(compressor string, out Output) string {
	if out.Compressed {
		return "saved"
	}
	if strings.EqualFold(compressor, NoopName) {
		return "no-op"
	}
	return "no_effect"
}

func benchSampleReason(compressor string, out Output) string {
	if out.Compressed {
		return "compressed with " + out.Codec
	}
	if strings.EqualFold(compressor, NoopName) {
		return "noop compressor selected"
	}
	return "compressor returned no smaller rendering"
}

func benchAggregateStatus(compressor string, samples []BenchSample, saved float64) (status, reason string) {
	if len(samples) == 0 {
		return "missing", "no samples were supplied"
	}
	counts := map[string]int{}
	for _, s := range samples {
		counts[s.Status]++
	}
	switch {
	case counts["error"] > 0:
		return "error", fmt.Sprintf("%d sample(s) returned compressor errors", counts["error"])
	case counts["unavailable"] == len(samples):
		return "unavailable", "all samples passed through because the selected compressor dependency was unavailable"
	case counts["unavailable"] > 0:
		return "partial", fmt.Sprintf("%d sample(s) could not reach the selected compressor dependency", counts["unavailable"])
	case strings.EqualFold(compressor, NoopName):
		return "no-op", "noop compressor selected; context compression is intentionally disabled"
	case saved > 0:
		return "measured", "compressor produced realized byte savings on this corpus"
	default:
		return "no_saving", "compressor ran but produced no aggregate saving on this corpus"
	}
}

func dash(s string) string {
	if strings.TrimSpace(s) == "" {
		return "-"
	}
	return s
}

// BenchCorpus returns the deterministic, representative corpus. Each sample is a
// real tool-output SHAPE; together they span the codecs the native compressor
// fires plus an incompressible control.
func BenchCorpus() []BenchInput {
	return []BenchInput{
		{Name: "colorized-test", Bytes: []byte(corpusColorizedTest())},
		{Name: "progress-bar", Bytes: []byte(corpusProgressBar())},
		{Name: "scattered-warnings", Bytes: []byte(corpusScatteredWarnings())},
		{Name: "pretty-json", Bytes: []byte(corpusPrettyJSON())},
		{Name: "repeated-retries", Bytes: []byte(corpusRepeatedRetries())},
		{Name: "crlf-build-log", Bytes: []byte(corpusCRLFBuildLog())},
		{Name: "plain-prose", Bytes: []byte(corpusPlainProse())},
	}
}

func corpusColorizedTest() string {
	const e = "\x1b"
	pkgs := []string{"adjudicator", "gateway", "headroom", "gitgate", "engine", "vdso", "ctxmmu", "policy"}
	var b strings.Builder
	for _, p := range pkgs {
		fmt.Fprintf(&b, e+"[32mok"+e+"[0m  \tgithub.com/anthony-chaudhary/fak/internal/%s\t%s\n", p, "0.12s")
	}
	b.WriteString(e + "[31m--- FAIL: TestSomething (0.03s)" + e + "[0m\n")
	b.WriteString(e + "[31mFAIL" + e + "[0m\tgithub.com/anthony-chaudhary/fak/internal/flaky\t0.03s\n")
	return b.String()
}

func corpusProgressBar() string {
	var b strings.Builder
	b.WriteString("pulling model weights ")
	for i := 0; i <= 100; i++ {
		fmt.Fprintf(&b, "\rpulling model weights %3d%% [%-20s]", i, strings.Repeat("=", i/5))
	}
	b.WriteString("\rpulling model weights done           \n")
	return b.String()
}

func corpusScatteredWarnings() string {
	var b strings.Builder
	b.WriteString("starting vet over 12 packages\n")
	for i := 0; i < 12; i++ {
		fmt.Fprintf(&b, "vet: internal/pkg%02d/file.go\n", i)
		b.WriteString("warning: deprecated call to LegacyAPI, use the v2 surface instead\n")
	}
	b.WriteString("vet complete\n")
	return b.String()
}

func corpusPrettyJSON() string {
	return "{\n" +
		"    \"model\": \"claude-opus-4-8\",\n" +
		"    \"usage\": {\n" +
		"        \"input_tokens\": 12345,\n" +
		"        \"output_tokens\": 678,\n" +
		"        \"cache_read_input_tokens\": 9000\n" +
		"    },\n" +
		"    \"choices\": [\n" +
		"        {\n" +
		"            \"index\": 0,\n" +
		"            \"finish_reason\": \"stop\"\n" +
		"        }\n" +
		"    ]\n" +
		"}\n"
}

func corpusRepeatedRetries() string {
	var b strings.Builder
	b.WriteString("connecting to upstream\n")
	for i := 0; i < 40; i++ {
		b.WriteString("retry: connection refused, backing off 500ms\n")
	}
	b.WriteString("connected\n")
	return b.String()
}

func corpusCRLFBuildLog() string {
	lines := []string{
		"[build] compiling module a   ",
		"[build] compiling module b   ",
		"[build] linking              ",
		"[build] done                 ",
	}
	return strings.Join(lines, "\r\n") + "\r\n"
}

func corpusPlainProse() string {
	return "The kernel adjudicates each tool call before it runs, denying by structure, " +
		"repairing malformed calls, and quarantining poisoned results so the model never " +
		"sees them. This paragraph is unique prose with no repetition or control bytes, so " +
		"the native compressor correctly leaves it essentially untouched.\n"
}
