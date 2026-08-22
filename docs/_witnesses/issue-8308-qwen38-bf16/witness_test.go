package bf16witness_test

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/qwen38quant"
)

const (
	artifactSHA = "f5e9eb121362eac4d112889bcdbd53ac49d4a2a7a2fe6b1b9bad21c3efeaaf22"
	templateSHA = "20158f9b80605efa9f0794b988d970861f5020d9f10b82bc65f1b38e0abf59bc"
)

var pinnedFiles = map[string]string{
	"archive.json":                "ed057a1c492bb5117b8b1be578f2cf59a60d67cdb3fe0035bcf681bcfe35d4fe",
	"failure-before-archive.json": "e2d156e4520ef47a56259afe6151e86f6802793e2548234ce265fee2922fab2a",
	"failure-before-report.json":  "cd13468b4562eab26ddf88197a9bfb94b24c02596a962e87a3015347ec246f19",
	"failure-before-summary.json": "973e2ca28a346b94d8bf3b813964fd59ec4e87c3932db108b1b57d0d4db466bf",
	"provenance.json":             "2bca830518006bbcd39758a215f18788af30f8bf9688240e89e984d90eb93d67",
	"report.json":                 "8baf30202b2d0d3594b80cc48402a19cfed2173815a49bc5e5f8ab563cc9ce7f",
	"runtime.json":                "febfdaf56542d2ad8dc40e1410e85b8c7206f4b1e82c06bca7ba3ca82ce4b779",
	"smaller-topology-oom.json":   "238210e53d43fae247ef85635119e2d3831fd71fc9b6c2ddb7bbf341e9e04682",
	"summary.json":                "5e3ec92bdf1c1b8791646eb55dfdfc6b997a5a2c07e224151550452e3e4b724c",
}

type archive struct {
	Schema       string      `json:"schema"`
	CorpusID     string      `json:"corpus_id"`
	Arm          string      `json:"arm"`
	Before       observation `json:"before"`
	After        observation `json:"after"`
	Results      []result    `json:"results"`
	RestartReady bool        `json:"restart_ready"`
	CleanupOK    bool        `json:"cleanup_ok"`
}

type observation struct {
	Identity       qwen38quant.Identity `json:"identity"`
	Hardware       string               `json:"hardware"`
	Software       string               `json:"software"`
	Device         string               `json:"device"`
	ContextTokens  int                  `json:"context_tokens"`
	CacheMode      string               `json:"cache_mode"`
	Resident       bool                 `json:"resident"`
	FallbackActive bool                 `json:"fallback_active"`
	MemoryBytes    uint64               `json:"memory_bytes"`
	PowerWatts     float64              `json:"power_watts"`
}

type result struct {
	Workload          string         `json:"workload"`
	Repeat            int            `json:"repeat"`
	LatencyMS         float64        `json:"latency_ms"`
	Usage             map[string]int `json:"usage"`
	Quality           string         `json:"quality"`
	Failure           string         `json:"failure"`
	Phase             string         `json:"phase"`
	CachedInputTokens int            `json:"cached_input_tokens"`
	Resource          resource       `json:"resource"`
}

type resource struct {
	MemoryBytes uint64  `json:"memory_bytes"`
	PowerWatts  float64 `json:"power_watts"`
}

type summary struct {
	Schema              string                   `json:"schema"`
	Verdict             string                   `json:"verdict"`
	EvidenceClass       string                   `json:"evidence_class"`
	Trials              int                      `json:"trials"`
	Quality             map[string]qualityMetric `json:"quality"`
	LoadLatencyMS       int64                    `json:"load_latency_ms"`
	Throughput          float64                  `json:"throughput_completion_tokens_per_second_p50"`
	PeakMemoryBytes     uint64                   `json:"peak_memory_bytes"`
	PeakPowerWatts      float64                  `json:"peak_power_watts"`
	Cache               []cacheMetric            `json:"cache"`
	RestartReady        bool                     `json:"restart_ready"`
	CleanupOK           bool                     `json:"cleanup_ok"`
	PostCleanupMemoryMB float64                  `json:"post_cleanup_memory_mib"`
	ArchiveSHA256       string                   `json:"raw_archive_sha256"`
	ReportSHA256        string                   `json:"report_sha256"`
	Rollback            string                   `json:"rollback"`
}

type qualityMetric struct {
	Pass       int     `json:"pass"`
	Fail       int     `json:"fail"`
	LatencyP50 float64 `json:"latency_p50_ms"`
}

type cacheMetric struct {
	Phase             string  `json:"phase"`
	LatencyMS         float64 `json:"latency_ms"`
	CachedInputTokens int     `json:"cached_input_tokens"`
}

type provenance struct {
	Schema           string     `json:"schema"`
	Repository       string     `json:"repository"`
	Revision         string     `json:"revision"`
	CheckpointSHA256 string     `json:"checkpoint_sha256"`
	ArtifactSHA256   string     `json:"artifact_sha256"`
	TokenizerSHA256  string     `json:"tokenizer_sha256"`
	TemplateSHA256   string     `json:"template_sha256"`
	FileCount        int        `json:"file_count"`
	WeightBytes      uint64     `json:"weight_bytes"`
	Files            []provFile `json:"files"`
}

type provFile struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
	Bytes  uint64 `json:"bytes"`
}

type runtime struct {
	CUDA                    string `json:"cuda"`
	Driver                  string `json:"driver"`
	ExecutionTemplateSHA256 string `json:"execution_template_sha256"`
	ImageID                 string `json:"image_id"`
	ModelRevision           string `json:"model_revision"`
	Versions                string `json:"versions"`
}

type oomControl struct {
	Schema                  string  `json:"schema"`
	Verdict                 string  `json:"verdict"`
	Reason                  string  `json:"reason"`
	ModelRevision           string  `json:"model_revision"`
	CheckpointSHA256        string  `json:"checkpoint_sha256"`
	ArtifactSHA256          string  `json:"artifact_sha256"`
	Precision               string  `json:"precision"`
	EffectiveMemoryLimitMiB int     `json:"effective_memory_limit_mib"`
	AllocatorFraction       float64 `json:"allocator_fraction"`
	LoadCompleted           bool    `json:"load_completed"`
	FallbackActive          bool    `json:"fallback_active"`
	ContainerExitCode       int     `json:"container_exit_code"`
	ServiceLogSHA256        string  `json:"service_log_sha256"`
}

func TestBF16ReferenceWitness(t *testing.T) {
	for name, want := range pinnedFiles {
		if got := digest(readFile(t, name)); got != want {
			t.Fatalf("%s digest = %s, want %s", name, got, want)
		}
	}

	corpusBytes := readFile(t, filepath.Join("..", "..", "benchmarks", "qwen38-quant", "corpus.json"))
	corpus, err := qwen38quant.DecodeCorpus(corpusBytes)
	if err != nil {
		t.Fatal(err)
	}

	reportBytes := readFile(t, "report.json")
	var report qwen38quant.Report
	decode(t, reportBytes, &report)
	if err := qwen38quant.Validate(report, corpus); err != nil {
		t.Fatalf("validate report: %v", err)
	}
	if report.Arm != "bf16" || report.Verdict != "PROMOTE" || report.EvidenceClass != "CAMPAIGN" {
		t.Fatalf("report disposition = %s/%s/%s", report.Arm, report.Verdict, report.EvidenceClass)
	}
	if report.Identity.CheckpointSHA256 != artifactSHA || report.Identity.ArtifactSHA256 != artifactSHA || report.Identity.TemplateSHA256 != templateSHA {
		t.Fatalf("report identity = %#v", report.Identity)
	}
	if report.Environment.ContextTokens != 65536 || !report.Environment.DenyFallback || report.Environment.RequireDevice != "A100-SXM4-80GB" {
		t.Fatalf("report environment = %#v", report.Environment)
	}

	archiveBytes := readFile(t, "archive.json")
	if got := digest(archiveBytes); got != report.RawArchiveSHA256 {
		t.Fatalf("archive digest = %s, report binds %s", got, report.RawArchiveSHA256)
	}
	var gotArchive archive
	decode(t, archiveBytes, &gotArchive)
	validateFinalArchive(t, gotArchive, corpus)

	var gotSummary summary
	decode(t, readFile(t, "summary.json"), &gotSummary)
	validateSummary(t, gotSummary, gotArchive, reportBytes, archiveBytes, corpus)
	validateProvenance(t, report)
	validateFailureBefore(t, corpus)
	validateOOMControl(t)
}

func validateFinalArchive(t *testing.T, got archive, corpus qwen38quant.Corpus) {
	t.Helper()
	wantTrials := len(corpus.Workloads) * corpus.MinimumRepetitions
	if got.Schema != "fak.qwen38-quant-raw/1" || got.CorpusID != corpus.ID || got.Arm != "bf16" || len(got.Results) != wantTrials {
		t.Fatalf("archive identity/count = %#v, results=%d", got, len(got.Results))
	}
	if !got.RestartReady || !got.CleanupOK || !got.Before.Resident || !got.After.Resident || got.Before.FallbackActive || got.After.FallbackActive {
		t.Fatalf("archive lifecycle = before=%#v after=%#v restart=%v cleanup=%v", got.Before, got.After, got.RestartReady, got.CleanupOK)
	}
	counts := make(map[string]int, len(corpus.Workloads))
	for _, row := range got.Results {
		counts[row.Workload]++
		if row.Quality != "PASS" || row.Failure != "" || row.LatencyMS <= 0 {
			t.Fatalf("non-passing result = %#v", row)
		}
	}
	for _, workload := range corpus.Workloads {
		if counts[workload] != corpus.MinimumRepetitions {
			t.Fatalf("%s repetitions = %d", workload, counts[workload])
		}
	}
}

func validateSummary(t *testing.T, got summary, raw archive, reportBytes, archiveBytes []byte, corpus qwen38quant.Corpus) {
	t.Helper()
	if got.Schema != "fak.qwen38-bf16-summary/1" || got.Verdict != "PROMOTE" || got.EvidenceClass != "CAMPAIGN" || got.Trials != len(raw.Results) {
		t.Fatalf("summary identity = %#v", got)
	}
	if !got.RestartReady || !got.CleanupOK || got.LoadLatencyMS <= 0 || got.Rollback == "" || got.PostCleanupMemoryMB >= float64(got.PeakMemoryBytes)/(1<<20) {
		t.Fatalf("summary lifecycle = %#v", got)
	}
	if got.ReportSHA256 != digest(reportBytes) || got.ArchiveSHA256 != digest(archiveBytes) {
		t.Fatalf("summary digests = report %s archive %s", got.ReportSHA256, got.ArchiveSHA256)
	}

	latencies := make(map[string][]float64, len(corpus.Workloads))
	throughputs := make([]float64, 0, len(raw.Results))
	var peakPower float64
	peakMemory := max(raw.Before.MemoryBytes, raw.After.MemoryBytes)
	for _, row := range raw.Results {
		latencies[row.Workload] = append(latencies[row.Workload], row.LatencyMS)
		throughputs = append(throughputs, float64(row.Usage["completion_tokens"])/(row.LatencyMS/1000))
		peakMemory = max(peakMemory, row.Resource.MemoryBytes)
		peakPower = math.Max(peakPower, row.Resource.PowerWatts)
	}
	if peakMemory != got.PeakMemoryBytes || !closeFloat(peakPower, got.PeakPowerWatts) {
		t.Fatalf("summary resources = %d/%.3f, want %d/%.3f", got.PeakMemoryBytes, got.PeakPowerWatts, peakMemory, peakPower)
	}
	sort.Float64s(throughputs)
	wantThroughput := (throughputs[len(throughputs)/2-1] + throughputs[len(throughputs)/2]) / 2
	if !closeFloat(got.Throughput, wantThroughput) {
		t.Fatalf("throughput = %.9f, want %.9f", got.Throughput, wantThroughput)
	}
	for workload, values := range latencies {
		sort.Float64s(values)
		metric := got.Quality[workload]
		if metric.Pass != corpus.MinimumRepetitions || metric.Fail != 0 || !closeFloat(metric.LatencyP50, values[len(values)/2]) {
			t.Fatalf("%s summary metric = %#v, latencies=%v", workload, metric, values)
		}
	}
	cache := make(map[string]cacheMetric, len(got.Cache))
	for _, row := range got.Cache {
		cache[row.Phase] = row
	}
	if cache["cold"].LatencyMS <= cache["warm"].LatencyMS || cache["warm_after_restart"].LatencyMS <= 0 {
		t.Fatalf("cache evidence = %#v", cache)
	}
	if cache["cold"].CachedInputTokens != 0 || cache["warm"].CachedInputTokens != 0 {
		t.Fatal("unexpected nonzero cached-token telemetry")
	}
}

func validateProvenance(t *testing.T, report qwen38quant.Report) {
	t.Helper()
	var p provenance
	decode(t, readFile(t, "provenance.json"), &p)
	if p.Schema != "fak.qwen38-bf16-provenance/1" || p.Repository != "Qwen/Qwen3.8-27B" || p.Revision != "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0" {
		t.Fatalf("provenance source = %#v", p)
	}
	if p.FileCount != len(p.Files) || p.FileCount != 32 || p.WeightBytes != 55563006776 || manifestDigest(p.Files) != artifactSHA {
		t.Fatalf("provenance manifest = count %d/%d bytes %d digest %s", p.FileCount, len(p.Files), p.WeightBytes, manifestDigest(p.Files))
	}
	if p.CheckpointSHA256 != artifactSHA || p.ArtifactSHA256 != artifactSHA || p.TokenizerSHA256 != report.Identity.TokenizerSHA256 || p.TemplateSHA256 != "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041" {
		t.Fatalf("provenance identity = %#v", p)
	}
	var rt runtime
	decode(t, readFile(t, "runtime.json"), &rt)
	if rt.ExecutionTemplateSHA256 != templateSHA || rt.ModelRevision != p.Revision || rt.CUDA != "13.0" || rt.Driver == "" || rt.ImageID == "" || rt.Versions == "" {
		t.Fatalf("runtime identity = %#v", rt)
	}
}

func validateFailureBefore(t *testing.T, corpus qwen38quant.Corpus) {
	t.Helper()
	reportBytes := readFile(t, "failure-before-report.json")
	var report qwen38quant.Report
	decode(t, reportBytes, &report)
	if err := qwen38quant.Validate(report, corpus); err != nil {
		t.Fatalf("validate failure-before report: %v", err)
	}
	if report.Verdict != "HOLD" || report.Identity.ArtifactSHA256 != artifactSHA {
		t.Fatalf("failure-before report = %#v", report)
	}
	archiveBytes := readFile(t, "failure-before-archive.json")
	if digest(archiveBytes) != report.RawArchiveSHA256 {
		t.Fatal("failure-before archive is not bound by its report")
	}
	var before archive
	decode(t, archiveBytes, &before)
	passes, failures := 0, 0
	for _, row := range before.Results {
		if row.Quality == "PASS" {
			passes++
		} else {
			failures++
		}
	}
	if passes != 3 || failures != 15 || !before.RestartReady || !before.CleanupOK {
		t.Fatalf("failure-before results = pass %d fail %d restart %v cleanup %v", passes, failures, before.RestartReady, before.CleanupOK)
	}
	var beforeSummary summary
	decode(t, readFile(t, "failure-before-summary.json"), &beforeSummary)
	if beforeSummary.Verdict != "HOLD" || beforeSummary.Trials != 18 || beforeSummary.ReportSHA256 != digest(reportBytes) || beforeSummary.ArchiveSHA256 != digest(archiveBytes) {
		t.Fatalf("failure-before summary = %#v", beforeSummary)
	}
}

func validateOOMControl(t *testing.T) {
	t.Helper()
	var got oomControl
	decode(t, readFile(t, "smaller-topology-oom.json"), &got)
	if got.Schema != "fak.qwen38-bf16-smaller-topology/1" || got.Verdict != "EXCLUDE" || got.Reason != "CUDA_OUT_OF_MEMORY" {
		t.Fatalf("OOM disposition = %#v", got)
	}
	if got.ModelRevision != "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0" || got.CheckpointSHA256 != artifactSHA || got.ArtifactSHA256 != artifactSHA || got.Precision != "bfloat16" {
		t.Fatalf("OOM identity = %#v", got)
	}
	if got.EffectiveMemoryLimitMiB != 39322 || !closeFloat(got.AllocatorFraction, 0.48) || got.LoadCompleted || got.FallbackActive || got.ContainerExitCode == 0 || got.ServiceLogSHA256 == "" {
		t.Fatalf("OOM result = %#v", got)
	}
}

func manifestDigest(files []provFile) string {
	files = append([]provFile(nil), files...)
	sort.Slice(files, func(i, j int) bool { return files[i].Path < files[j].Path })
	var manifest bytes.Buffer
	for _, file := range files {
		fmt.Fprintf(&manifest, "%s\x00%s\x00%d\n", file.Path, file.SHA256, file.Bytes)
	}
	return digest(manifest.Bytes())
}

func readFile(t *testing.T, path string) []byte {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func decode(t *testing.T, data []byte, dst any) {
	t.Helper()
	if err := json.Unmarshal(data, dst); err != nil {
		t.Fatal(err)
	}
}

func digest(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func closeFloat(got, want float64) bool {
	return math.Abs(got-want) <= 1e-6*math.Max(1, math.Abs(want))
}
