package model

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestQwen35MTPProductionReceiptInstrumentation(t *testing.T) {
	m := qwen35MTPEnabledSyntheticModel(t)
	s := m.NewSession()
	defer s.Close()
	baseline := m.NewSession()
	defer baseline.Close()
	want := baseline.Generate([]int{0, 1}, 4)
	run, receipt, err := SpecDecodeGreedyQwen35MTPDepthNWithReceipt(s, []int{0, 1}, 4, 1)
	if err != nil || !reflect.DeepEqual(run.Output, want) {
		t.Fatalf("synthetic instrumentation: %v, output %v want %v", err, run.Output, want)
	}
	var total int64
	for _, ns := range receipt.Stages {
		total += ns
	}
	if total != receipt.TotalNanoseconds || total <= 0 || len(receipt.Transactions) != run.Rounds {
		t.Fatalf("incomplete instrumentation: %+v", receipt)
	}
	for _, tx := range receipt.Transactions {
		if tx.Engine != "fak-native" || tx.OneOperation || tx.TargetVerificationOperations != 0 || tx.TargetDecodeSteps != 1 || tx.DowngradeReason == "" {
			t.Fatalf("transaction: %+v", tx)
		}
	}
	_, failed, err := SpecDecodeGreedyQwen35MTPDepthNWithReceipt(nil, []int{0}, 1, 1)
	if err == nil || failed.Error == "" || failed.TotalNanoseconds <= 0 {
		t.Fatalf("failed admission omitted: %+v %v", failed, err)
	}
	t.Log("synthetic instrumentation only; no real-artifact performance acceptance")
}

// The opt-in manifest pins a caller-supplied exported checkpoint. Source identity
// is an operator attestation; hashes verify the exact local bytes, not upstream
// authenticity. Run in a dedicated Linux process with a reviewed timeout/budget:
// FAK_MTP_ACCEPTANCE=/absolute/manifest.json go test ./internal/model -run '^TestQwen38MTPRealArtifactAcceptance$' -count=1 -v -timeout=120s
// Manifest: {"source":"Qwen/Qwen3.8-...","revision":"<immutable revision>",
// "directory":"/artifact","sha256":{"config.json":"...","manifest.json":"...","weights.f32":"..."},
// "prompt":[1,2],"new_tokens":8,"memory_budget_bytes":137438953472}
func TestQwen38MTPRealArtifactAcceptance(t *testing.T) {
	path := os.Getenv("FAK_MTP_ACCEPTANCE")
	if path == "" {
		t.Skip("real artifact acceptance requires FAK_MTP_ACCEPTANCE; synthetic fixtures do not qualify")
	}
	var candidate struct {
		Source       string            `json:"source"`
		Revision     string            `json:"revision"`
		Directory    string            `json:"directory"`
		SHA256       map[string]string `json:"sha256"`
		Prompt       []int             `json:"prompt"`
		NewTokens    int               `json:"new_tokens"`
		MemoryBudget int64             `json:"memory_budget_bytes"`
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &candidate); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(strings.ToLower(candidate.Source), "qwen3.8") || strings.Contains(strings.ToLower(candidate.Source), "synthetic") || len(candidate.Revision) < 12 || len(candidate.Prompt) == 0 || len(candidate.Prompt) > 16 || candidate.NewTokens < 1 || candidate.NewTokens > 8 || candidate.MemoryBudget <= 0 {
		t.Fatal("unrecognized identity or unreviewed workload: require real Qwen3.8 source, immutable revision, 1..16 prompt IDs, 1..8 generated tokens and memory budget")
	}
	setupStart := time.Now()
	for _, name := range []string{"config.json", "manifest.json", "weights.f32"} {
		want, err := hex.DecodeString(candidate.SHA256[name])
		if err != nil || len(want) != sha256.Size {
			t.Fatalf("missing or malformed SHA256 for %s", name)
		}
		f, err := os.Open(filepath.Join(candidate.Directory, name))
		if err != nil {
			t.Fatal(err)
		}
		stat, err := f.Stat()
		if err != nil {
			f.Close()
			t.Fatal(err)
		}
		// Reserve conservative room before materializing weights, activations and
		// snapshots. The eventual process peak remains the acceptance fence.
		if name == "weights.f32" && stat.Size() > candidate.MemoryBudget/3 {
			f.Close()
			t.Fatal("weight materialization exceeds conservative memory budget")
		}
		h := sha256.New()
		_, err = io.Copy(h, f)
		f.Close()
		if err != nil || !reflect.DeepEqual(h.Sum(nil), want) {
			t.Fatalf("artifact hash mismatch for %s: %v", name, err)
		}
	}
	peakBefore := mtpProcessPeakRSS(t)
	m, err := Load(candidate.Directory)
	if err != nil {
		t.Fatal(err)
	}
	defer m.CloseWeights()
	setupNS := time.Since(setupStart).Nanoseconds()
	type trial struct {
		Warmup         bool                `json:"warmup"`
		Order          string              `json:"order"`
		BaselineNS     int64               `json:"baseline_nanoseconds"`
		MTPNS          int64               `json:"mtp_nanoseconds"`
		BaselineOutput []int               `json:"baseline_output"`
		MTPOutput      []int               `json:"mtp_output"`
		Receipt        Qwen35MTPRunReceipt `json:"receipt"`
		Equal          bool                `json:"equal"`
		OneOperation   bool                `json:"one_operation_per_block"`
	}
	var trials []trial
	baselineTotal, mtpTotal := setupNS, setupNS
	eligible := true
	for i := 0; i < 4; i++ {
		row := trial{Warmup: i == 0, Order: "baseline,mtp"}
		base := func() {
			start := time.Now()
			s := m.NewSession()
			row.BaselineOutput = s.Generate(candidate.Prompt, candidate.NewTokens)
			s.Close()
			row.BaselineNS = time.Since(start).Nanoseconds()
		}
		mtp := func() {
			start := time.Now()
			s := m.NewSession()
			run, receipt, _ := SpecDecodeGreedyQwen35MTPDepthNWithReceipt(s, candidate.Prompt, candidate.NewTokens, 1)
			s.Close()
			row.MTPNS = time.Since(start).Nanoseconds()
			row.MTPOutput, row.Receipt = run.Output, receipt
			row.OneOperation = len(receipt.Transactions) > 0 && len(receipt.Transactions) == run.Rounds
			for _, tx := range receipt.Transactions {
				row.OneOperation = row.OneOperation && tx.OneOperation && tx.TargetVerificationOperations == 1 && tx.Engine == "fak-native"
			}
		}
		if i%2 == 0 {
			base()
			mtp()
		} else {
			row.Order = "mtp,baseline"
			mtp()
			base()
		}
		row.Equal = reflect.DeepEqual(row.BaselineOutput, row.MTPOutput)
		baselineTotal += row.BaselineNS
		mtpTotal += row.MTPNS
		trials = append(trials, row)
		if row.Receipt.Error != "" || !row.Equal || !row.OneOperation || mtpProcessPeakRSS(t) > candidate.MemoryBudget {
			eligible = false
			break
		}
		if !row.Warmup && row.MTPNS >= row.BaselineNS {
			eligible = false
			break
		}
	}
	peak := mtpProcessPeakRSS(t)
	verdict, reason := "KEEP", "matched output, one target operation per block, net latency and memory within envelope"
	if !eligible || len(trials) != 4 || mtpTotal >= baselineTotal {
		verdict, reason = "DOWNGRADE_ORDINARY_FAK_NATIVE", "execution, equality, verification, memory or net-latency gate failed"
	}
	report := map[string]any{"schema": "fak-mtp-real-acceptance/1", "candidate": candidate, "artifact_identity_basis": "operator source attestation plus locally verified SHA256", "engine": "fak-native", "target": "f32", "depth": 1, "setup_nanoseconds": setupNS, "baseline_inclusive_nanoseconds": baselineTotal, "mtp_inclusive_nanoseconds": mtpTotal, "memory_scope": "process lifetime peak RSS; includes both arms and setup, not per-arm incremental", "peak_rss_before_load_bytes": peakBefore, "peak_rss_bytes": peak, "verdict": verdict, "reason": reason, "trials": trials, "envelope": "one warmup and three alternating paired trials; inclusive totals charge setup and warmup; KEEP applies only to this pinned workload"}
	b, err := json.Marshal(report)
	if err != nil {
		t.Fatal(err)
	}
	t.Log(string(b))
	if verdict != "KEEP" {
		t.Fatalf("%s: %s", verdict, reason)
	}
}

func mtpProcessPeakRSS(t *testing.T) int64 {
	t.Helper()
	b, err := os.ReadFile("/proc/self/status")
	if err != nil {
		t.Fatal("acceptance needs Linux process peak RSS:", err)
	}
	for _, line := range strings.Split(string(b), "\n") {
		if strings.HasPrefix(line, "VmHWM:") {
			var kb int64
			if _, err := fmt.Sscanf(line, "VmHWM: %d kB", &kb); err == nil && kb > 0 {
				return kb * 1024
			}
		}
	}
	t.Fatal("process peak RSS unavailable")
	return 0
}
