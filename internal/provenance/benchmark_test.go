package provenance

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

var (
	benchSinkLabel    abi.TaintLabel
	benchSinkBool     bool
	benchSinkSource   Source
	benchSinkSources  map[string]Source
	benchSinkVersion  ToolVersion
	benchSinkPinKind  PinKind
	benchSinkWitness  VersionWitness
	benchSinkBytes    []byte
	benchSinkManifest RunManifest
	benchSinkString   string
	benchSinkDiv      *Divergence
	benchSinkArtifact ReplayArtifact
	benchSinkError    error
)

var (
	benchCallTrusted   = &abi.ToolCall{Tool: "Read"}
	benchCallUntrusted = &abi.ToolCall{Tool: "read_webpage"}
	benchCallForged    = &abi.ToolCall{
		Tool: "read_webpage",
		Meta: map[string]string{"provenance": "trusted_local"},
	}
	benchResultNormal = &abi.Result{
		Status:  abi.StatusOK,
		Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte("content payload")},
	}
	benchResultSealed = &abi.Result{
		Status: abi.StatusOK,
		Meta:   map[string]string{"quarantine_id": "qid-1234"},
	}

	benchProbeMatch = VersionProbe{
		Tool:  "mytool",
		Raw:   "mytool version 1.2.3\n",
		Found: true,
	}
	benchProbePathConfusion = VersionProbe{
		Tool:  "mytool",
		Path:  "/opt/toolchain-1.2.3/bin/mytool",
		Raw:   "/opt/toolchain-1.2.3/bin/mytool version 2.0.0 (x86_64-linux)\n",
		Found: true,
	}
	benchProbeMismatch = VersionProbe{
		Tool:  "mytool",
		Raw:   "mytool version 1.2.30\n",
		Found: true,
	}

	benchManifestBaseline = RunManifest{
		Case:           "decode-parity/greedy-eos@v3",
		Model:          "glm-4.6@sha256:1111",
		Tokenizer:      "glm-bpe@sha256:2222",
		Backend:        "fak-engine",
		LoadProvenance: "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
		Seed:           42,
		CodeRev:        "internal/provenance@r120+gabcdef",
		Baseline:       "nightly-2026-09-01/greedy",
		Tolerance:      "exact",
		BinaryRev:      "fak-0.9.0+abc1234",
		Hardware:       "cpu-avx512",
		DecodeParams: map[string]string{
			"temperature": "0",
			"top_p":       "1",
			"max_tokens":  "256",
		},
		CacheState:    "cold",
		Normalization: "nfc+trim",
		Env: map[string]string{
			"OMP_NUM_THREADS": "8",
			"FAK_API_KEY":     "sk-secret-token",
		},
		Tier:    TierNightly,
		Cost:    "3.2s / 1xcpu / 4.1k tok",
		Secrets: []string{"OMP_NUM_THREADS"},
	}

	benchManifestEquivalent = RunManifest{
		Case:           "  decode-parity/greedy-eos@v3  ",
		Model:          "glm-4.6@sha256:1111 ",
		Tokenizer:      " glm-bpe@sha256:2222",
		Backend:        "fak-engine",
		LoadProvenance: "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
		Seed:           42,
		CodeRev:        "internal/provenance@r120+gabcdef",
		Baseline:       "nightly-2026-09-01/greedy",
		Tolerance:      "exact",
		BinaryRev:      "fak-0.9.0+abc1234",
		Hardware:       "cpu-avx512",
		DecodeParams: map[string]string{
			"max_tokens":  "256",
			"temperature": "0",
			"top_p":       "1",
		},
		CacheState:    "cold",
		Normalization: "nfc+trim",
		Env: map[string]string{
			"FAK_API_KEY":     "sk-secret-token",
			"OMP_NUM_THREADS": "8",
		},
		Tier:    TierNightly,
		Cost:    "3.2s / 1xcpu / 4.1k tok",
		Secrets: []string{"OMP_NUM_THREADS"},
	}

	benchManifestDivergent = func() RunManifest {
		m := benchManifestBaseline
		m.DecodeParams = map[string]string{
			"temperature": "0.7",
			"top_p":       "1",
			"max_tokens":  "256",
		}
		return m
	}()

	benchWitnessAccepted = VerifyToolVersion("1.2.3", benchProbeMatch)
	benchReplayBytes     = Compare(benchManifestBaseline, benchManifestBaseline).JSON()
)

func BenchmarkTaint(b *testing.B) {
	b.Run("TrustedLocal", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkLabel = Taint(benchCallTrusted, benchResultNormal)
		}
	})

	b.Run("UntrustedTool", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkLabel = Taint(benchCallUntrusted, benchResultNormal)
		}
	})

	b.Run("KernelSealed", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkLabel = Taint(benchCallTrusted, benchResultSealed)
		}
	})

	b.Run("ForgedSelfTrust", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkLabel = Taint(benchCallForged, benchResultNormal)
		}
	})
}

func BenchmarkAttemptedSelfTrust(b *testing.B) {
	b.Run("Present", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkBool = AttemptedSelfTrust(benchCallForged)
		}
	})

	b.Run("Absent", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkBool = AttemptedSelfTrust(benchCallUntrusted)
		}
	})
}

func BenchmarkSourceRegistry(b *testing.B) {
	b.Run("SourceOf_Hit", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkSource = SourceOf("Read")
		}
	})

	b.Run("SourceOf_Miss", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkSource = SourceOf("unregistered_extension_tool")
		}
	})

	b.Run("Sources_Snapshot", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkSources = Sources()
		}
	})
}

func BenchmarkParseToolVersion(b *testing.B) {
	b.Run("Standard", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkVersion, benchSinkBool = ParseToolVersion("1.2.3")
		}
	})

	b.Run("Prefixed", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkVersion, benchSinkBool = ParseToolVersion("v20.11.1")
		}
	})

	b.Run("Prerelease", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkVersion, benchSinkBool = ParseToolVersion("1.2.3-rc1")
		}
	})

	b.Run("Decorated", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkVersion, benchSinkBool = ParseToolVersion("(2.43.0)")
		}
	})

	b.Run("Invalid", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkVersion, benchSinkBool = ParseToolVersion("go1.22.3")
		}
	})
}

func BenchmarkClassifyPin(b *testing.B) {
	b.Run("Exact", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkPinKind = ClassifyPin("1.2.3")
		}
	})

	b.Run("Constraint", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkPinKind = ClassifyPin(">=1.2.0")
		}
	})

	b.Run("Wildcard", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkPinKind = ClassifyPin("1.2.*")
		}
	})
}

func BenchmarkVerifyToolVersion(b *testing.B) {
	b.Run("ExactMatch", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkWitness = VerifyToolVersion("1.2.3", benchProbeMatch)
		}
	})

	b.Run("PathConfusionRefusal", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkWitness = VerifyToolVersion("1.2.3", benchProbePathConfusion)
		}
	})

	b.Run("Mismatch", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkWitness = VerifyToolVersion("1.2.3", benchProbeMismatch)
		}
	})
}

func BenchmarkVersionWitnessReceipt(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkBytes = benchWitnessAccepted.Receipt()
	}
}

func BenchmarkRunManifestNormalize(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkManifest = benchManifestEquivalent.Normalize()
	}
}

func BenchmarkRunManifestFingerprint(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkString = benchManifestBaseline.Fingerprint()
	}
}

func BenchmarkRunManifestValidate(b *testing.B) {
	b.Run("Valid", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkError = benchManifestBaseline.Validate()
		}
	})

	b.Run("MissingField", func(b *testing.B) {
		inval := benchManifestBaseline
		inval.Tokenizer = ""
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkError = inval.Validate()
		}
	})
}

func BenchmarkRunManifestFirstDivergence(b *testing.B) {
	b.Run("Equivalent", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkDiv = benchManifestBaseline.FirstDivergence(benchManifestEquivalent)
		}
	})

	b.Run("Divergent", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkDiv = benchManifestBaseline.FirstDivergence(benchManifestDivergent)
		}
	})
}

func BenchmarkRunManifestCompare(b *testing.B) {
	b.Run("Pass", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkArtifact = Compare(benchManifestBaseline, benchManifestEquivalent)
		}
	})

	b.Run("Fail", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkArtifact = Compare(benchManifestBaseline, benchManifestDivergent)
		}
	})
}

func BenchmarkReplayArtifactJSON(b *testing.B) {
	art := Compare(benchManifestBaseline, benchManifestBaseline)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkBytes = art.JSON()
	}
}

func BenchmarkReplayFrom(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkArtifact, benchSinkError = ReplayFrom(benchReplayBytes)
	}
}

func TestBenchmarkSanity(t *testing.T) {
	if Taint(benchCallTrusted, benchResultNormal) != abi.TaintTrusted {
		t.Fatal("expected trusted taint for Read")
	}
	if Taint(benchCallUntrusted, benchResultNormal) != abi.TaintTainted {
		t.Fatal("expected tainted taint for read_webpage")
	}
	if !AttemptedSelfTrust(benchCallForged) {
		t.Fatal("expected attempted self trust")
	}
	if !benchWitnessAccepted.Satisfied() {
		t.Fatal("expected benchWitnessAccepted to be satisfied")
	}
	if benchManifestBaseline.Fingerprint() != benchManifestEquivalent.Fingerprint() {
		t.Fatal("expected baseline and equivalent manifests to match fingerprint")
	}
	art := Compare(benchManifestBaseline, benchManifestEquivalent)
	if !art.Pass() {
		t.Fatalf("expected pass, got %s: %s", art.Verdict, art.Reason)
	}
	replayed, err := ReplayFrom(benchReplayBytes)
	if err != nil || !replayed.Pass() {
		t.Fatalf("replayed failed: %v", err)
	}
}
