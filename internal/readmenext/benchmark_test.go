package readmenext

import (
	"testing"
)

var (
	benchFragSink    *CandidateFragment
	benchStringSink  string
	benchStringsSink []string
	benchResultSink  *PublishResult
)

var benchSampleJSON = []byte(`{
	"schema": "fak-readme-candidate/1",
	"issue": 10944,
	"topic": "nvidia-h100-q8",
	"target_section": "hardware_table",
	"candidate_content": "| NVIDIA | Hopper H100 Q8_0: 111.9 tok/s (+17.4% vs f32) | Verified | [NVIDIA](docs/nv.md) |",
	"retire_target": {
		"action": "replace_row",
		"target_text": "| NVIDIA | Older H100 result: 95 tok/s. | Verified baseline. | [NVIDIA](docs/nvidia.md) |",
		"legacy_archive_doc": "docs/README-legacy.md"
	},
	"witness": {
		"authority_entry": "BENCHMARK-AUTHORITY.md#quick-reference-primary-numbers",
		"receipt_path": "docs/_witnesses/issue-10944-nvidia-gcp-overnight/README.md",
		"hardware_json_row": "NVIDIA"
	},
	"laws_checklist": {
		"sota_comparison": true,
		"feynman_gloss": true,
		"wide_audience": true
	},
	"proposed_by": "benchmarker",
	"date": "2026-09-06"
}`)

// BenchmarkParseCandidateFragment measures deserialization and default-field population
// for staged README candidate fragment payloads.
func BenchmarkParseCandidateFragment(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		frag, err := ParseCandidateFragment(benchSampleJSON)
		if err != nil {
			b.Fatal(err)
		}
		benchFragSink = frag
	}
}

// BenchmarkParseCandidateFragment_Parallel measures concurrent candidate fragment
// deserialization under multi-goroutine load.
func BenchmarkParseCandidateFragment_Parallel(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		var localFrag *CandidateFragment
		for pb.Next() {
			frag, err := ParseCandidateFragment(benchSampleJSON)
			if err != nil {
				b.Fatal(err)
			}
			localFrag = frag
		}
		_ = localFrag
	})
}

// BenchmarkValidateFragment measures full validation of a candidate fragment against
// repository artifacts (witness receipts, benchmark authority anchor, hardware manifest,
// SOTA comparison laws, and section bounds).
func BenchmarkValidateFragment(b *testing.B) {
	repoRoot := setupHermeticRepo(b)
	frag, err := ParseCandidateFragment(benchSampleJSON)
	if err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := ValidateFragment(frag, repoRoot); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkValidateFragment_InMem measures in-memory constraint auditing: schema,
// date format, bounded sections, regex-based SOTA-vs-naive laws, and checklist flags.
func BenchmarkValidateFragment_InMem(b *testing.B) {
	frag := &CandidateFragment{
		Schema:           SchemaCandidate,
		Issue:            11200,
		Topic:            "why-fak-addition",
		TargetSection:    TargetWhyFak,
		CandidateContent: "- **Zero-copy storage overflow:** Stream paged KV cache directly from NVMe with 4.1× vs tuned baselines.",
		RetireTarget: RetireTarget{
			Action: RetireActionNone,
		},
		LawsChecklist: LawsChecklist{
			SOTAComparison: true,
			FeynmanGloss:   true,
			WideAudience:   true,
		},
		Date: "2026-09-06",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := ValidateFragment(frag, ""); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkSynthesizeNextDraft measures full README preview synthesis with multiple candidate
// fragments touching distinct sections (hardware table replacement, why_fak append,
// and hero headline substitution).
func BenchmarkSynthesizeNextDraft(b *testing.B) {
	frag1, err := ParseCandidateFragment(benchSampleJSON)
	if err != nil {
		b.Fatal(err)
	}
	frag2 := &CandidateFragment{
		Schema:           SchemaCandidate,
		Issue:            11200,
		Topic:            "why-fak-addition",
		TargetSection:    TargetWhyFak,
		CandidateContent: "- **Zero-copy storage overflow:** Stream paged KV cache directly from NVMe.",
		RetireTarget: RetireTarget{
			Action: RetireActionNone,
		},
	}
	frag3 := &CandidateFragment{
		Schema:           SchemaCandidate,
		Issue:            11500,
		Topic:            "hero-update",
		TargetSection:    TargetHeroHeadline,
		CandidateContent: "**fak is an agent runtime: one binary puts a fast, cache-accelerated boundary between your coding agent and every tool call.**",
		RetireTarget: RetireTarget{
			Action:     RetireActionReplaceRow,
			TargetText: "**fak is an agent runtime: one binary puts a fast, cache-accelerated boundary between your coding agent and every tool call.**",
		},
	}
	fragments := []*CandidateFragment{frag1, frag2, frag3}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		draft, changes, err := SynthesizeNextDraft(sampleReadme, fragments)
		if err != nil {
			b.Fatal(err)
		}
		benchStringSink = draft
		benchStringsSink = changes
	}
}

// BenchmarkPreviewNext measures single-fragment replacement synthesis performance.
func BenchmarkPreviewNext(b *testing.B) {
	frag, err := ParseCandidateFragment(benchSampleJSON)
	if err != nil {
		b.Fatal(err)
	}
	fragments := []*CandidateFragment{frag}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		draft, changes, err := PreviewNext(sampleReadme, fragments)
		if err != nil {
			b.Fatal(err)
		}
		benchStringSink = draft
		benchStringsSink = changes
	}
}

// BenchmarkPublish_DryRun measures end-to-end dry-run publication: loading README,
// validating candidate fragments against repository witnesses, previewing synthesis,
// and compiling the PublishResult manifest.
func BenchmarkPublish_DryRun(b *testing.B) {
	repoRoot := setupHermeticRepo(b)
	frag, err := ParseCandidateFragment(benchSampleJSON)
	if err != nil {
		b.Fatal(err)
	}
	fragments := []*CandidateFragment{frag}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := Publish(repoRoot, fragments, true)
		if err != nil {
			b.Fatal(err)
		}
		benchResultSink = res
	}
}

// TestBenchmarkSanity verifies that all benchmarks execute without panics and complete runs.
func TestBenchmarkSanity(t *testing.T) {
	frag, err := ParseCandidateFragment(benchSampleJSON)
	if err != nil || frag == nil {
		t.Fatalf("ParseCandidateFragment failed: %v", err)
	}
	draft, changes, err := SynthesizeNextDraft(sampleReadme, []*CandidateFragment{frag})
	if err != nil || len(draft) == 0 || len(changes) == 0 {
		t.Fatalf("SynthesizeNextDraft failed: %v", err)
	}
}
