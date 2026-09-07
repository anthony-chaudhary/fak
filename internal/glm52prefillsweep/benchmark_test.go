package glm52prefillsweep

import (
	"fmt"
	"io"
	"path/filepath"
	"testing"
)

var (
	benchPlanSink     []PlanStep
	benchPromptSink   string
	benchReportSink   DryRunReport
	benchRecordSink   Record
	benchManifestSink Manifest
	benchStringSink   string
	benchBytesSink    []byte
	benchLengthsSink  []int
	benchExitCodeSink int
	benchJSONMapSink  map[string]any
)

func BenchmarkSyntheticPrompt(b *testing.B) {
	for _, length := range PrefillLengths {
		b.Run(fmt.Sprintf("tokens_%d", length), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchPromptSink = syntheticPrompt(length)
			}
		})
	}
}

func BenchmarkPrefillPayload(b *testing.B) {
	b.Run("streaming", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = prefillPayload("zai-org/GLM-5.2", 2048, DefaultMaxTokens, true)
		}
	})
	b.Run("blocking", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = prefillPayload("zai-org/GLM-5.2", 2048, DefaultMaxTokens, false)
		}
	})
}

func BenchmarkBuildPlan(b *testing.B) {
	b.Run("default_5_lengths", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchPlanSink = BuildPlan("zai-org/GLM-5.2", "experiments/benchmark/runs/test", nil, DefaultMaxTokens, true, FragileMinLen)
		}
	})
	for _, length := range PrefillLengths {
		b.Run(fmt.Sprintf("single_len_%d", length), func(b *testing.B) {
			lengths := []int{length}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchPlanSink = BuildPlan("zai-org/GLM-5.2", "experiments/benchmark/runs/test", lengths, DefaultMaxTokens, true, FragileMinLen)
			}
		})
	}
}

func BenchmarkBuildDryRunReport(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchReportSink = BuildDryRunReport("zai-org/GLM-5.2", "http://localhost:8000/v1", "experiments/benchmark/runs/test", PrefillLengths, DefaultMaxTokens, true)
	}
}

func BenchmarkRecordForLength(b *testing.B) {
	b.Run("ok_status", func(b *testing.B) {
		m := measurement{
			ok:               true,
			httpStatus:       200,
			ttftSeconds:      fptr(0.352),
			totalSeconds:     fptr(0.360),
			promptTokens:     iptr(2048),
			completionTokens: iptr(1),
			source:           "stream-ttft",
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchRecordSink = RecordForLength(m, "zai-org/GLM-5.2", "http://localhost:8000/v1", 2048, 1, true)
		}
	})
	b.Run("fail_status", func(b *testing.B) {
		m := measurement{
			ok:         false,
			httpStatus: 500,
			errMsg:     "CUDA illegal memory access at 0x7f001234",
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchRecordSink = RecordForLength(m, "zai-org/GLM-5.2", "http://localhost:8000/v1", 8192, 1, true)
		}
	})
}

func BenchmarkRunID(b *testing.B) {
	lineage := Lineage{
		LineageSchema: LineageSchema,
		AppVersion:    "1.0.0",
		UTC:           "2026-09-06T12:34:56.789012Z",
		GitCommit:     "abcdef1234567890",
		GoVersion:     "go1.23.0",
		Node:          "strix1",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchStringSink = runID(lineage, 4096)
	}
}

func BenchmarkBuildManifest(b *testing.B) {
	lineage := Lineage{
		LineageSchema: LineageSchema,
		AppVersion:    "1.0.0",
		UTC:           "2026-09-06T12:34:56.789012Z",
		GitCommit:     "abcdef1234567890",
		GoVersion:     "go1.23.0",
		Node:          "strix1",
	}
	m := measurement{
		ok:               true,
		httpStatus:       200,
		ttftSeconds:      fptr(0.125),
		totalSeconds:     fptr(0.130),
		promptTokens:     iptr(512),
		completionTokens: iptr(1),
		source:           "stream-ttft",
	}
	record := RecordForLength(m, "zai-org/GLM-5.2", "http://localhost:8000/v1", 512, 1, true)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchManifestSink = BuildManifest(record, lineage, 512)
	}
}

func BenchmarkResultsMarkdown(b *testing.B) {
	m := measurement{
		ok:               true,
		httpStatus:       200,
		ttftSeconds:      fptr(0.250),
		totalSeconds:     fptr(0.255),
		promptTokens:     iptr(2048),
		completionTokens: iptr(1),
		source:           "stream-ttft",
	}
	record := RecordForLength(m, "zai-org/GLM-5.2", "http://localhost:8000/v1", 2048, 1, true)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchStringSink = resultsMarkdown(record)
	}
}

func BenchmarkMarshalIndent(b *testing.B) {
	lineage := Lineage{
		LineageSchema: LineageSchema,
		AppVersion:    "1.0.0",
		UTC:           "2026-09-06T12:34:56.789012Z",
		GitCommit:     "abcdef1234567890",
		GoVersion:     "go1.23.0",
		Node:          "strix1",
	}
	m := measurement{
		ok:               true,
		httpStatus:       200,
		ttftSeconds:      fptr(0.250),
		totalSeconds:     fptr(0.255),
		promptTokens:     iptr(2048),
		completionTokens: iptr(1),
		source:           "stream-ttft",
	}
	record := RecordForLength(m, "zai-org/GLM-5.2", "http://localhost:8000/v1", 2048, 1, true)
	manifest := BuildManifest(record, lineage, 2048)

	b.Run("manifest", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchBytesSink = marshalIndent(manifest)
		}
	})
	b.Run("record", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchBytesSink = marshalIndent(record)
		}
	})
}

func BenchmarkWriteLedgerArtifact(b *testing.B) {
	lineage := Lineage{
		LineageSchema: LineageSchema,
		AppVersion:    "1.0.0",
		UTC:           "2026-09-06T12:34:56.789012Z",
		GitCommit:     "abcdef1234567890",
		GoVersion:     "go1.23.0",
		Node:          "strix1",
	}
	m := measurement{
		ok:               true,
		httpStatus:       200,
		ttftSeconds:      fptr(0.250),
		totalSeconds:     fptr(0.255),
		promptTokens:     iptr(2048),
		completionTokens: iptr(1),
		source:           "stream-ttft",
	}
	record := RecordForLength(m, "zai-org/GLM-5.2", "http://localhost:8000/v1", 2048, 1, true)

	dir := b.TempDir()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		target := filepath.Join(dir, fmt.Sprintf("p2048_%d", i))
		path, err := WriteLedgerArtifact(target, record, lineage)
		if err != nil {
			b.Fatal(err)
		}
		benchStringSink = path
	}
}

func BenchmarkParseLengths(b *testing.B) {
	raw := "128, 512, 2048, 4096, 8192"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		lengths, err := parseLengths(raw)
		if err != nil {
			b.Fatal(err)
		}
		benchLengthsSink = lengths
	}
}

func BenchmarkParseJSONObject(b *testing.B) {
	chunk := `{"id":"chatcmpl-bench","choices":[{"delta":{"content":"word"}}],"usage":{"prompt_tokens":2048,"completion_tokens":1}}`
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchJSONMapSink = parseJSONObject(chunk)
	}
}

func BenchmarkRun_DryRun(b *testing.B) {
	dir := b.TempDir()
	out := filepath.Join(dir, "plan.json")
	args := []string{"--dry-run", "--out", out}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		code := Run(io.Discard, io.Discard, args)
		if code != 0 {
			b.Fatalf("Run returned exit code %d", code)
		}
		benchExitCodeSink = code
	}
}
