package qwen38quant

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

const q5KMWitnessDir = "../../docs/_witnesses/issue-8311-qwen38-q5km"

var q5KMPinnedArtifacts = map[string]struct {
	model  string
	sha256 string
	bytes  uint64
}{
	"q5_k_m": {"qwen38:27b-q5_k_m", "07deb7fa91bf751d3000774fe5bb8afae5ffb41255fd19980147468052e07177", 19834055648},
	"q4_k_m": {"qwen38:27b-q4_k_m", "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169", 17106775008},
}

type q5KMWitnessSummary struct {
	Schema       string                   `json:"schema"`
	Issue        int                      `json:"issue"`
	CorpusSHA256 string                   `json:"corpus_sha256"`
	Verdict      string                   `json:"verdict"`
	Rollback     string                   `json:"rollback"`
	Arms         map[string]q5KMArmMetric `json:"arms"`
	Comparison   q5KMComparison           `json:"comparison"`
}

type q5KMArmMetric struct {
	ReportSHA256              string  `json:"report_sha256"`
	ArchiveSHA256             string  `json:"archive_sha256"`
	ArtifactBytes             uint64  `json:"artifact_bytes"`
	Trials                    int     `json:"trials"`
	QualityPasses             int     `json:"quality_passes"`
	P50LatencyMS              float64 `json:"p50_latency_ms"`
	P95LatencyMS              float64 `json:"p95_latency_ms"`
	ThroughputTokensPerSecond float64 `json:"throughput_tokens_per_second"`
	PeakMemoryBytes           uint64  `json:"peak_memory_bytes"`
	MeanPowerWatts            float64 `json:"mean_power_watts"`
	CacheColdLatencyMS        float64 `json:"cache_cold_latency_ms"`
	CacheWarmLatencyMS        float64 `json:"cache_warm_latency_ms"`
	CacheDeltaPercent         float64 `json:"cache_delta_percent"`
	Verdict                   string  `json:"verdict"`
}

type q5KMComparison struct {
	MemoryPremiumPercent         float64 `json:"memory_premium_percent"`
	QualityDeltaPercentagePoints float64 `json:"quality_delta_percentage_points"`
	ThroughputDeltaPercent       float64 `json:"throughput_delta_percent"`
	CredibleQualityGain          bool    `json:"credible_quality_gain"`
	Decision                     string  `json:"decision"`
}

type q5KMArchive struct {
	Schema       string       `json:"schema"`
	CorpusID     string       `json:"corpus_id"`
	Arm          string       `json:"arm"`
	Before       q5KMResource `json:"before"`
	After        q5KMResource `json:"after"`
	Results      []q5KMResult `json:"results"`
	RestartReady bool         `json:"restart_ready"`
	CleanupOK    bool         `json:"cleanup_ok"`
}

type q5KMResource struct {
	MemoryBytes uint64  `json:"memory_bytes"`
	PowerWatts  float64 `json:"power_watts"`
}

type q5KMResult struct {
	Workload          string         `json:"workload"`
	Repeat            int            `json:"repeat"`
	LatencyMS         float64        `json:"latency_ms"`
	Usage             map[string]int `json:"usage"`
	Quality           string         `json:"quality"`
	Phase             string         `json:"phase"`
	CachedInputTokens int            `json:"cached_input_tokens"`
	Resource          q5KMResource   `json:"resource"`
}

func TestQ5KMCampaignWitness(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	witnessDir := filepath.Join(root, "docs", "_witnesses", "issue-8311-qwen38-q5km")
	corpusBytes, err := os.ReadFile(filepath.Join(root, "docs", "benchmarks", "qwen38-quant", "corpus.json"))
	if err != nil {
		t.Fatal(err)
	}
	corpus, err := DecodeCorpus(corpusBytes)
	if err != nil {
		t.Fatal(err)
	}
	var summary struct {
		Schema               string `json:"schema"`
		Issue                int    `json:"issue"`
		Status               string `json:"status"`
		CorpusSHA256         string `json:"corpus_sha256"`
		ComparisonAdmissible bool   `json:"comparison_admissible"`
		Arms                 map[string]struct {
			Status                   string `json:"status"`
			SuccessfulAPICompletions int    `json:"successful_api_completions"`
			Report                   string `json:"report"`
			Archive                  string `json:"archive"`
		} `json:"arms"`
	}
	decodeWitnessJSON(t, filepath.Join(witnessDir, "summary.json"), &summary)
	if summary.Schema != "fak.qwen38-platform-qualification/1" || summary.Issue != 8311 || summary.Status != "INVALID_API_CONTRACT" || summary.ComparisonAdmissible {
		t.Fatalf("invalid qualification summary: %+v", summary)
	}
	if summary.CorpusSHA256 != CorpusDigest(corpus) {
		t.Fatalf("corpus digest=%s want %s", summary.CorpusSHA256, CorpusDigest(corpus))
	}
	for _, armName := range []string{"q5_k_m", "q4_k_m"} {
		arm, ok := summary.Arms[armName]
		if !ok || arm.Status != "INVALID_API_CONTRACT" || arm.SuccessfulAPICompletions != 0 {
			t.Fatalf("arm %s qualification=%+v", armName, arm)
		}
		var report Report
		decodeWitnessJSON(t, filepath.Join(witnessDir, arm.Report), &report)
		if err := Validate(report, corpus); err == nil {
			t.Fatalf("arm %s invalid report unexpectedly validated", armName)
		}
		var archive q5KMArchive
		decodeWitnessJSON(t, filepath.Join(witnessDir, arm.Archive), &archive)
		if archive.Schema != "fak.qwen38-quant-raw/1" || len(archive.Results) != 18 {
			t.Fatalf("arm %s archive metadata invalid: schema=%q results=%d", armName, archive.Schema, len(archive.Results))
		}
	}
	text := string(readWitnessFile(t, filepath.Join(witnessDir, "README.md")))
	for _, phrase := range []string{"INVALID", "does not rank Q5_K_M against Q4_K_M", "first text request as an API canary", "platform-support question remains unanswered"} {
		if !strings.Contains(text, phrase) {
			t.Fatalf("README missing %q", phrase)
		}
	}
	for _, forbidden := range []string{"Q5_K_M does not justify", "Rollback is to retain Q4_K_M"} {
		if strings.Contains(text, forbidden) {
			t.Fatalf("README retains inadmissible comparison %q", forbidden)
		}
	}
}

func validateQ5KMArm(t *testing.T, arm string, want q5KMArmMetric, corpus Corpus) q5KMArmMetric {
	t.Helper()
	prefix := map[string]string{"q5_k_m": "q5", "q4_k_m": "q4"}[arm]
	reportPath := filepath.Join(q5KMWitnessDir, prefix+"-report.json")
	archivePath := filepath.Join(q5KMWitnessDir, prefix+"-archive.json")
	reportBytes, archiveBytes := readWitnessFile(t, reportPath), readWitnessFile(t, archivePath)

	var report Report
	if err := json.Unmarshal(reportBytes, &report); err != nil {
		t.Fatalf("decode %s: %v", reportPath, err)
	}
	if report.Arm != arm {
		t.Fatalf("%s report arm = %q", arm, report.Arm)
	}
	pinned := q5KMPinnedArtifacts[arm]
	if report.Identity.Model != pinned.model || report.Identity.ArtifactSHA256 != pinned.sha256 || report.Identity.CheckpointSHA256 != pinned.sha256 || want.ArtifactBytes != pinned.bytes {
		t.Fatalf("%s artifact identity = %#v, summary bytes=%d", arm, report.Identity, want.ArtifactBytes)
	}
	if err := Validate(report, corpus); err != nil {
		t.Fatalf("validate %s report: %v", arm, err)
	}
	if got := digestWitness(archiveBytes); got != report.RawArchiveSHA256 {
		t.Fatalf("%s archive digest = %s, report binds %s", arm, got, report.RawArchiveSHA256)
	}

	var archive q5KMArchive
	if err := json.Unmarshal(archiveBytes, &archive); err != nil {
		t.Fatalf("decode %s: %v", archivePath, err)
	}
	if archive.Schema != "fak.qwen38-quant-raw/1" || archive.CorpusID != corpus.ID || archive.Arm != arm || !archive.RestartReady || !archive.CleanupOK {
		t.Fatalf("%s archive lifecycle = %#v", arm, archive)
	}

	got := computeQ5KMMetrics(archive.Results, archive.Before, archive.After)
	got.ReportSHA256 = digestWitness(reportBytes)
	got.ArchiveSHA256 = digestWitness(archiveBytes)
	got.ArtifactBytes = want.ArtifactBytes
	got.Verdict = report.Verdict
	if want.ArtifactBytes == 0 || want.ReportSHA256 != got.ReportSHA256 || want.ArchiveSHA256 != got.ArchiveSHA256 || want.Verdict != got.Verdict {
		t.Fatalf("%s bound metrics = %#v, want %#v", arm, got, want)
	}
	if got.Trials != len(corpus.Workloads)*corpus.MinimumRepetitions || got.PeakMemoryBytes == 0 || got.MeanPowerWatts <= 0 || got.ThroughputTokensPerSecond < 0 || got.CacheColdLatencyMS <= 0 || got.CacheWarmLatencyMS <= 0 {
		t.Fatalf("%s incomplete metrics = %#v", arm, got)
	}
	assertArmMetrics(t, arm, want, got)
	return got
}

func computeQ5KMMetrics(results []q5KMResult, observations ...q5KMResource) q5KMArmMetric {
	got := q5KMArmMetric{Trials: len(results)}
	latencies := make([]float64, 0, len(results))
	totalTokens, totalLatencyMS, powerSamples := 0, 0.0, 0
	for _, observation := range observations {
		if observation.MemoryBytes > got.PeakMemoryBytes {
			got.PeakMemoryBytes = observation.MemoryBytes
		}
		if observation.PowerWatts > 0 {
			got.MeanPowerWatts += observation.PowerWatts
			powerSamples++
		}
	}
	for _, result := range results {
		latencies = append(latencies, result.LatencyMS)
		totalLatencyMS += result.LatencyMS
		totalTokens += result.Usage["completion_tokens"]
		if result.Quality == "PASS" {
			got.QualityPasses++
		}
		if result.Resource.MemoryBytes > got.PeakMemoryBytes {
			got.PeakMemoryBytes = result.Resource.MemoryBytes
		}
		if result.Resource.PowerWatts > 0 {
			got.MeanPowerWatts += result.Resource.PowerWatts
			powerSamples++
		}
		if result.Workload == "repeated_workflow_cache" {
			switch result.Phase {
			case "cold":
				got.CacheColdLatencyMS = result.LatencyMS
			case "warm":
				got.CacheWarmLatencyMS = result.LatencyMS
			}
		}
	}
	if len(latencies) > 0 {
		slices.Sort(latencies)
		mid := len(latencies) / 2
		got.P50LatencyMS = latencies[mid]
		if len(latencies)%2 == 0 {
			got.P50LatencyMS = (latencies[mid-1] + latencies[mid]) / 2
		}
		got.P95LatencyMS = latencies[int(math.Ceil(0.95*float64(len(latencies))))-1]
	}
	if totalLatencyMS > 0 {
		got.ThroughputTokensPerSecond = float64(totalTokens) / (totalLatencyMS / 1000)
	}
	if powerSamples > 0 {
		got.MeanPowerWatts /= float64(powerSamples)
	}
	if got.CacheColdLatencyMS > 0 {
		got.CacheDeltaPercent = percentDelta(got.CacheWarmLatencyMS, got.CacheColdLatencyMS)
	}
	return got
}

func assertArmMetrics(t *testing.T, arm string, want, got q5KMArmMetric) {
	t.Helper()
	if want.Trials != got.Trials || want.QualityPasses != got.QualityPasses || want.PeakMemoryBytes != got.PeakMemoryBytes {
		t.Fatalf("%s counts/resources = %#v, want %#v", arm, got, want)
	}
	for name, values := range map[string][2]float64{
		"p50 latency": {want.P50LatencyMS, got.P50LatencyMS},
		"p95 latency": {want.P95LatencyMS, got.P95LatencyMS},
		"throughput":  {want.ThroughputTokensPerSecond, got.ThroughputTokensPerSecond},
		"mean power":  {want.MeanPowerWatts, got.MeanPowerWatts},
		"cache cold":  {want.CacheColdLatencyMS, got.CacheColdLatencyMS},
		"cache warm":  {want.CacheWarmLatencyMS, got.CacheWarmLatencyMS},
		"cache delta": {want.CacheDeltaPercent, got.CacheDeltaPercent},
	} {
		assertClose(t, arm+" "+name, values[0], values[1])
	}
}

func decodeWitnessJSON(t *testing.T, path string, dst any) {
	t.Helper()
	data := readWitnessFile(t, path)
	if err := json.Unmarshal(data, dst); err != nil {
		t.Fatalf("decode %s: %v", path, err)
	}
}

func readWitnessFile(t *testing.T, path string) []byte {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return data
}

func digestWitness(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func percentDelta(value, baseline float64) float64 {
	if baseline == 0 {
		return 0
	}
	return (value/baseline - 1) * 100
}

func assertClose(t *testing.T, name string, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-6*math.Max(1, math.Abs(want)) {
		t.Fatalf("%s = %.9f, want %.9f", name, got, want)
	}
}
