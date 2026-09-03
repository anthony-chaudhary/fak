package issue9513witness

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/qwen38quantrun"
)

const (
	expectedArtifactSHA256 = "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
	expectedArtifactBytes  = int64(17106775008)
	expectedModelID        = "unsloth/Qwen3.8-27B-GGUF@f1bfb127c64f7072bdd2cad55f258b9c8b2910fe"

	expectedLlamaBuild    = "9828"
	expectedLlamaRevision = "ebd048fc5e4b43ec4e0b4abe0b9bf66e1724dad0"
	expectedLlamaLicense  = "MIT"
	expectedLlamaBinsum   = "12df97ffa9d48545e96cd3237a71f78efd1cc0222f971cbd65f7ab57e793b128"

	expectedCandidateEngine  = "fak-native"
	expectedCandidateBackend = "metal"
	expectedCandidateForward = "metal/qwen35-hybrid-session-v1"

	expectedReferenceEngine  = "llama.cpp"
	expectedReferenceBackend = "metal"
	expectedReferenceForward = "llama.cpp/metal"

	parityThreshold = 0.95
	logitTolerance  = 0.001
)

type parityReport struct {
	Schema      string `json:"schema"`
	Issue       int    `json:"issue"`
	ParentIssue int    `json:"parent_issue"`
	Phase       string `json:"phase"`
	Verdict     string `json:"verdict"`
	Artifact    struct {
		Model  string `json:"model"`
		Bytes  int64  `json:"bytes"`
		SHA256 string `json:"sha256"`
	} `json:"artifact"`
	Hardware struct {
		Host        string `json:"host"`
		GPUCores    int    `json:"gpu_cores"`
		MemoryGiB   int    `json:"memory_gib"`
		MemoryBytes uint64 `json:"memory_bytes"`
		OS          string `json:"os"`
		Arch        string `json:"arch"`
	} `json:"hardware"`
	Workload struct {
		PrefillTokens int     `json:"prefill_tokens"`
		DecodeTokens  int     `json:"decode_tokens"`
		Temperature   float64 `json:"temperature"`
		Repetitions   int     `json:"repetitions"`
		PromptSHA256  string  `json:"prompt_sha256"`
	} `json:"workload"`
	Candidate struct {
		Name      string `json:"name"`
		Execution struct {
			Engine         string `json:"engine"`
			Backend        string `json:"backend"`
			ForwardPath    string `json:"forward_path"`
			Q4K            bool   `json:"q4k"`
			FallbackActive bool   `json:"fallback_active"`
			ComparatorOnly bool   `json:"comparator_only"`
		} `json:"execution"`
		PrefillTokensPerSecondMean         float64                                  `json:"prefill_tokens_per_second_mean"`
		DecodeTokensPerSecondMean          float64                                  `json:"decode_tokens_per_second_mean"`
		DecodeTokensPerSecondGeometricMean float64                                  `json:"decode_tokens_per_second_geometric_mean"`
		TTFTMSMean                         float64                                  `json:"ttft_ms_mean"`
		RSSBytesPeak                       uint64                                   `json:"rss_bytes_peak"`
		OSFootprintBytesPeak               uint64                                   `json:"os_footprint_bytes_peak"`
		Repetitions                        []qwen38quantrun.OracleMatchedRepetition `json:"repetitions"`
	} `json:"candidate"`
	Reference struct {
		Name      string `json:"name"`
		Execution struct {
			Engine         string `json:"engine"`
			Backend        string `json:"backend"`
			ForwardPath    string `json:"forward_path"`
			Q4K            bool   `json:"q4k"`
			FallbackActive bool   `json:"fallback_active"`
			ComparatorOnly bool   `json:"comparator_only"`
			Build          string `json:"build"`
			Revision       string `json:"revision"`
			License        string `json:"license"`
			BinarySHA256   string `json:"binary_sha256"`
		} `json:"execution"`
		PrefillTokensPerSecondMean         float64                                  `json:"prefill_tokens_per_second_mean"`
		DecodeTokensPerSecondMean          float64                                  `json:"decode_tokens_per_second_mean"`
		DecodeTokensPerSecondGeometricMean float64                                  `json:"decode_tokens_per_second_geometric_mean"`
		TTFTMSMean                         float64                                  `json:"ttft_ms_mean"`
		RSSBytesPeak                       uint64                                   `json:"rss_bytes_peak"`
		OSFootprintBytesPeak               uint64                                   `json:"os_footprint_bytes_peak"`
		Repetitions                        []qwen38quantrun.OracleMatchedRepetition `json:"repetitions"`
	} `json:"reference"`
	Parity struct {
		TokenEquality       bool    `json:"token_equality"`
		DeterministicTokens bool    `json:"deterministic_tokens"`
		TokenCount          int     `json:"token_count"`
		LogitTolerance      float64 `json:"logit_tolerance"`
		MaxAbsLogitDiff     float64 `json:"max_abs_logit_diff"`
		LogitParity         bool    `json:"logit_parity"`
	} `json:"parity"`
	Ratios struct {
		ArithmeticThroughputRatio float64 `json:"arithmetic_throughput_ratio"`
		GeometricThroughputRatio  float64 `json:"geometric_throughput_ratio"`
		MinimumThreshold          float64 `json:"minimum_threshold"`
		MeetsParity               bool    `json:"meets_parity"`
	} `json:"ratios"`
	Provenance struct {
		ConfigSHA256  string `json:"config_sha256"`
		ArchiveSHA256 string `json:"archive_sha256"`
	} `json:"provenance"`
}

func witnessPath(filename string) string {
	if _, err := os.Stat("oracle-report.json"); err == nil {
		return filename
	}
	return filepath.Join("docs/_witnesses/issue-9513-qwen38-m10-parity", filename)
}

func findRepoRoot() (string, error) {
	wd, err := os.Getwd()
	if err != nil {
		return "", err
	}
	curr := wd
	for {
		if _, err := os.Stat(filepath.Join(curr, "go.mod")); err == nil {
			return curr, nil
		}
		parent := filepath.Dir(curr)
		if parent == curr {
			break
		}
		curr = parent
	}
	return wd, nil
}

func readJSONStrictFile(t *testing.T, path string, dst any) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	if err := json.Unmarshal(data, dst); err != nil {
		t.Fatalf("unmarshal %s: %v", path, err)
	}
}

func fileSHA256Hex(t *testing.T, path string) string {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read file for sha256 %s: %v", path, err)
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

// TestMatchedParityReceipt validates the complete immutable bundle,
// recomputing artifact/config/report hashes, strict runtime identities,
// token/logit parity, repetition counts, and throughput ratios from raw evidence.
func TestMatchedParityReceipt(t *testing.T) {
	// 1. Validate SHA256SUMS
	t.Run("Checksums", func(t *testing.T) {
		sumsPath := witnessPath("SHA256SUMS")
		data, err := os.ReadFile(sumsPath)
		if err != nil {
			t.Fatalf("read SHA256SUMS: %v", err)
		}
		lines := strings.Split(strings.TrimSpace(string(data)), "\n")
		if len(lines) < 10 {
			t.Fatalf("SHA256SUMS has only %d entries, want >= 10", len(lines))
		}
		for _, line := range lines {
			fields := strings.Fields(line)
			if len(fields) != 2 {
				t.Fatalf("malformed SHA256SUMS line: %q", line)
			}
			expectedHash, fname := fields[0], fields[1]
			actualHash := fileSHA256Hex(t, witnessPath(fname))
			if actualHash != expectedHash {
				t.Errorf("%s sha256 mismatch: got %s, want %s", fname, actualHash, expectedHash)
			}
		}
	})

	// 2. Validate oracle-report.json structure and hashes
	var report parityReport
	readJSONStrictFile(t, witnessPath("oracle-report.json"), &report)

	t.Run("ReportMetadataAndEnvelope", func(t *testing.T) {
		if report.Schema != "fak.qwen38-m10-parity-report/1" {
			t.Errorf("schema = %q, want fak.qwen38-m10-parity-report/1", report.Schema)
		}
		if report.Issue != 9513 || report.ParentIssue != 9430 || report.Phase != "M10" {
			t.Errorf("unexpected issue/phase: issue=%d parent=%d phase=%s", report.Issue, report.ParentIssue, report.Phase)
		}
		if report.Verdict != "PASS" {
			t.Errorf("verdict = %q, want PASS", report.Verdict)
		}

		if report.Artifact.SHA256 != expectedArtifactSHA256 || report.Artifact.Bytes != expectedArtifactBytes || report.Artifact.Model != expectedModelID {
			t.Errorf("artifact mismatch: %+v", report.Artifact)
		}

		if report.Hardware.GPUCores != 18 || report.Hardware.MemoryGiB != 36 {
			t.Errorf("hardware mismatch: %+v", report.Hardware)
		}

		if report.Workload.PrefillTokens != 32 || report.Workload.DecodeTokens != 64 || report.Workload.Temperature != 0.0 || report.Workload.Repetitions != 3 {
			t.Errorf("workload shape mismatch: %+v", report.Workload)
		}

		// Verify provenance hashes
		actualConfigSHA := fileSHA256Hex(t, witnessPath("oracle-config.json"))
		if report.Provenance.ConfigSHA256 != actualConfigSHA {
			t.Errorf("config SHA mismatch: report=%s actual=%s", report.Provenance.ConfigSHA256, actualConfigSHA)
		}
		actualArchiveSHA := fileSHA256Hex(t, witnessPath("oracle-archive.json"))
		if report.Provenance.ArchiveSHA256 != actualArchiveSHA {
			t.Errorf("archive SHA mismatch: report=%s actual=%s", report.Provenance.ArchiveSHA256, actualArchiveSHA)
		}
	})

	// 3. Validate runtime and adapter configurations
	t.Run("StrictRuntimeIdentities", func(t *testing.T) {
		var fakAdapter, llamaAdapter qwen38quantrun.AdapterConfig
		readJSONStrictFile(t, witnessPath("fak-adapter.json"), &fakAdapter)
		readJSONStrictFile(t, witnessPath("llama-adapter.json"), &llamaAdapter)

		if fakAdapter.ExecutionEngine != "fak-native" {
			t.Errorf("fak adapter execution engine = %q, want fak-native", fakAdapter.ExecutionEngine)
		}
		if llamaAdapter.ExecutionEngine != "llama.cpp" {
			t.Errorf("llama adapter execution engine = %q, want llama.cpp", llamaAdapter.ExecutionEngine)
		}

		// Candidate identity
		c := report.Candidate.Execution
		if c.Engine != expectedCandidateEngine || c.Backend != expectedCandidateBackend || c.ForwardPath != expectedCandidateForward || !c.Q4K || c.FallbackActive || c.ComparatorOnly {
			t.Errorf("candidate execution identity mismatch: %+v", c)
		}

		// Reference comparator identity
		r := report.Reference.Execution
		if r.Engine != expectedReferenceEngine || r.Backend != expectedReferenceBackend || r.ForwardPath != expectedReferenceForward || !r.Q4K || r.FallbackActive || !r.ComparatorOnly {
			t.Errorf("reference execution identity mismatch: %+v", r)
		}
		if r.Build != expectedLlamaBuild || r.Revision != expectedLlamaRevision || r.License != expectedLlamaLicense || r.BinarySHA256 != expectedLlamaBinsum {
			t.Errorf("comparator build/revision mismatch: %+v", r)
		}
	})

	// 4. Raw measurement evidence: token equality, logit parity, and repetitions
	var fakRun, llamaRun qwen38quantrun.OracleMeasurementRun
	readJSONStrictFile(t, witnessPath("fak-measurement.json"), &fakRun)
	readJSONStrictFile(t, witnessPath("llama-measurement.json"), &llamaRun)

	t.Run("RawEvidenceAndTokenParity", func(t *testing.T) {
		if fakRun.Matched == nil || llamaRun.Matched == nil {
			t.Fatal("missing matched envelope in measurement runs")
		}
		if len(fakRun.Matched.Repetitions) != 3 || len(llamaRun.Matched.Repetitions) != 3 {
			t.Fatalf("expected 3 repetitions per arm, got fak=%d llama=%d", len(fakRun.Matched.Repetitions), len(llamaRun.Matched.Repetitions))
		}

		var firstFakTokens, firstLlamaTokens []string
		for i := 0; i < 3; i++ {
			fRep := fakRun.Matched.Repetitions[i]
			lRep := llamaRun.Matched.Repetitions[i]

			if len(fRep.Tokens) != 64 || len(lRep.Tokens) != 64 {
				t.Fatalf("repetition %d tokens length: fak=%d llama=%d, want 64", i+1, len(fRep.Tokens), len(lRep.Tokens))
			}

			for idx, tok := range fRep.Tokens {
				if tok == "" {
					t.Errorf("fak rep %d token %d is empty", i+1, idx)
				}
			}
			for idx, tok := range lRep.Tokens {
				if tok == "" {
					t.Errorf("llama rep %d token %d is empty", i+1, idx)
				}
			}

			// Determinism across repetitions
			if firstFakTokens == nil {
				firstFakTokens = slices.Clone(fRep.Tokens)
			} else if !slices.Equal(firstFakTokens, fRep.Tokens) {
				t.Errorf("fak rep %d tokens drift from rep 1", i+1)
			}
			if firstLlamaTokens == nil {
				firstLlamaTokens = slices.Clone(lRep.Tokens)
			} else if !slices.Equal(firstLlamaTokens, lRep.Tokens) {
				t.Errorf("llama rep %d tokens drift from rep 1", i+1)
			}

			// Cross-arm token equality
			if !slices.Equal(fRep.Tokens, lRep.Tokens) {
				t.Errorf("rep %d token equality failed between fak and llama", i+1)
			}

			// Logit parity
			if len(fRep.Logits) == 0 || len(lRep.Logits) == 0 {
				t.Fatalf("rep %d has empty logits", i+1)
			}
			for j := range fRep.Logits {
				diff := math.Abs(fRep.Logits[j] - lRep.Logits[j])
				if diff > logitTolerance {
					t.Errorf("rep %d logit %d diff=%g > tol=%g", i+1, j, diff, logitTolerance)
				}
			}

			// Positive timing and memory metrics
			if fRep.TTFTMS <= 0 || fRep.PrefillSeconds <= 0 || fRep.PrefillTokensPerSecond <= 0 || fRep.DecodeSeconds <= 0 || fRep.DecodeTokensPerSecond <= 0 || fRep.RSSBytes == 0 || fRep.OSFootprintBytes == 0 {
				t.Errorf("fak rep %d has zero or negative timing/memory: %+v", i+1, fRep)
			}
			if lRep.TTFTMS <= 0 || lRep.PrefillSeconds <= 0 || lRep.PrefillTokensPerSecond <= 0 || lRep.DecodeSeconds <= 0 || lRep.DecodeTokensPerSecond <= 0 || lRep.RSSBytes == 0 || lRep.OSFootprintBytes == 0 {
				t.Errorf("llama rep %d has zero or negative timing/memory: %+v", i+1, lRep)
			}
		}
	})

	// 5. Recompute throughput metrics and enforce >= 95% parity threshold
	t.Run("RecomputeThroughputRatios", func(t *testing.T) {
		var fakRates, llamaRates []float64
		for _, r := range fakRun.Matched.Repetitions {
			fakRates = append(fakRates, r.DecodeTokensPerSecond)
		}
		for _, r := range llamaRun.Matched.Repetitions {
			llamaRates = append(llamaRates, r.DecodeTokensPerSecond)
		}

		arithMean := func(vals []float64) float64 {
			s := 0.0
			for _, v := range vals {
				s += v
			}
			return s / float64(len(vals))
		}
		geoMean := func(vals []float64) float64 {
			p := 1.0
			for _, v := range vals {
				p *= v
			}
			return math.Pow(p, 1.0/float64(len(vals)))
		}

		recomputedFakArith := arithMean(fakRates)
		recomputedFakGeo := geoMean(fakRates)
		recomputedLlamaArith := arithMean(llamaRates)
		recomputedLlamaGeo := geoMean(llamaRates)

		recomputedArithRatio := recomputedFakArith / recomputedLlamaArith
		recomputedGeoRatio := recomputedFakGeo / recomputedLlamaGeo

		t.Logf("Recomputed FAK Decode Throughput: Arithmetic=%.4f tok/s, Geometric=%.4f tok/s", recomputedFakArith, recomputedFakGeo)
		t.Logf("Recomputed Llama Decode Throughput: Arithmetic=%.4f tok/s, Geometric=%.4f tok/s", recomputedLlamaArith, recomputedLlamaGeo)
		t.Logf("Recomputed Parity Ratios: Arithmetic=%.4f, Geometric=%.4f (Threshold=%.2f)", recomputedArithRatio, recomputedGeoRatio, parityThreshold)

		if recomputedArithRatio < parityThreshold {
			t.Fatalf("arithmetic throughput ratio %g < threshold %g", recomputedArithRatio, parityThreshold)
		}
		if recomputedGeoRatio < parityThreshold {
			t.Fatalf("geometric throughput ratio %g < threshold %g", recomputedGeoRatio, parityThreshold)
		}

		// Compare against report values
		if math.Abs(report.Ratios.ArithmeticThroughputRatio-recomputedArithRatio) > 1e-6 {
			t.Errorf("report arithmetic ratio %g != recomputed %g", report.Ratios.ArithmeticThroughputRatio, recomputedArithRatio)
		}
		if math.Abs(report.Ratios.GeometricThroughputRatio-recomputedGeoRatio) > 1e-6 {
			t.Errorf("report geometric ratio %g != recomputed %g", report.Ratios.GeometricThroughputRatio, recomputedGeoRatio)
		}
		if !report.Ratios.MeetsParity {
			t.Errorf("report meets_parity is false")
		}
	})

	// 6. Optional MLX observation check
	t.Run("MLXObservationClassification", func(t *testing.T) {
		mlxFile := witnessPath("mlx-observation.json")
		if _, err := os.Stat(mlxFile); os.IsNotExist(err) {
			t.Skip("mlx-observation.json is optional and not present")
		}
		var mlx struct {
			Schema         string `json:"schema"`
			Runtime        string `json:"runtime"`
			Comparability  string `json:"comparability"`
			ArtifactSHA256 string `json:"artifact_sha256"`
		}
		readJSONStrictFile(t, mlxFile, &mlx)
		if mlx.Comparability != "equivalent-model-only" {
			t.Errorf("MLX comparability = %q, want equivalent-model-only", mlx.Comparability)
		}
		if mlx.ArtifactSHA256 == expectedArtifactSHA256 {
			t.Errorf("MLX artifact SHA unexpectedly matches GGUF artifact SHA")
		}
	})

	// 7. Oracle validation check: RunOracle produces expected archive
	t.Run("OracleReplayReadback", func(t *testing.T) {
		repoRoot, err := findRepoRoot()
		if err != nil {
			t.Fatalf("find repo root: %v", err)
		}
		configRel := "docs/_witnesses/issue-9513-qwen38-m10-parity/oracle-config.json"
		corpusRel := "docs/benchmarks/qwen38-quant/corpus.json"

		origWd, err := os.Getwd()
		if err != nil {
			t.Fatal(err)
		}
		if err := os.Chdir(repoRoot); err != nil {
			t.Fatal(err)
		}
		defer os.Chdir(origWd)

		tmpDir := t.TempDir()
		replayReportPath := filepath.Join(tmpDir, "replay-report.json")
		replayArchivePath := filepath.Join(tmpDir, "replay-archive.json")

		_, err = qwen38quantrun.RunOracle(context.Background(), configRel, corpusRel, replayReportPath, replayArchivePath)
		if err != nil {
			t.Fatalf("RunOracle replay failed: %v", err)
		}

		// Replay archive must byte-match committed oracle-archive.json
		committedArchive, err := os.ReadFile("docs/_witnesses/issue-9513-qwen38-m10-parity/oracle-archive.json")
		if err != nil {
			t.Fatalf("read committed oracle-archive.json: %v", err)
		}
		replayedArchive, err := os.ReadFile(replayArchivePath)
		if err != nil {
			t.Fatalf("read replayed archive: %v", err)
		}
		if !bytes.Equal(committedArchive, replayedArchive) {
			t.Errorf("replayed archive does not byte-match committed oracle-archive.json")
		}
	})
}
