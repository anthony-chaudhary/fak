package model

import (
	"crypto/sha256"
	"encoding/binary"
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
	if err == nil || failed.Error == "" || failed.TotalNanoseconds < 0 {
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
// Optional format "hf-safetensors" instead requires hashes for config.json,
// model.safetensors.index.json and every indexed shard; loads retained BF16/F16/F32
// tensors through the native loader into F32. Omitted format or "fak-f32" keeps
// the exported format. Use an immutable snapshot and a reviewed memory budget.
func TestQwen38MTPRealArtifactAcceptance(t *testing.T) {
	path := os.Getenv("FAK_MTP_ACCEPTANCE")
	if path == "" {
		t.Skip("real artifact acceptance requires FAK_MTP_ACCEPTANCE; synthetic fixtures do not qualify")
	}
	var candidate struct {
		Format       string            `json:"format"`
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
	if candidate.Format != "" && candidate.Format != "fak-f32" && candidate.Format != "hf-safetensors" {
		t.Fatal("unsupported artifact format")
	}
	var safeCfg Config
	var decodedBytes int64
	if candidate.Format == "hf-safetensors" {
		safeCfg, decodedBytes, err = mtpSafetensorsPreflight(candidate.Directory, candidate.SHA256, candidate.MemoryBudget)
		if err != nil {
			t.Fatal(err)
		}
	} else {
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
	}
	peakBefore := mtpProcessPeakRSS(t)
	var m *Model
	if candidate.Format == "hf-safetensors" {
		m, err = mtpLoadRetainedSafetensors(candidate.Directory, safeCfg)
	} else {
		m, err = Load(candidate.Directory)
	}
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
	if candidate.Format == "hf-safetensors" {
		report["loader"] = "LoadSafetensorsDir"
		report["source_storage"] = "safetensors BF16/F16/F32; validated per tensor"
		report["resident_dtype"] = "F32"
		report["retained_decoded_bytes"] = decodedBytes
		report["memory_admission"] = "3x retained decoded bytes is a conservative reservation, not measured RSS"
	}
	b, err := json.Marshal(report)
	if err != nil {
		t.Fatal(err)
	}
	t.Log(string(b))
	if verdict != "KEEP" {
		t.Fatalf("%s: %s", verdict, reason)
	}
}

func TestMTPDirectSafetensorsPreflight(t *testing.T) {
	for _, fault := range []string{"valid", "hash", "path", "symlink", "membership", "shape", "storage", "budget", "header"} {
		t.Run(fault, func(t *testing.T) {
			cfg := qwen35MTPLoadConfig()
			_, tensors := qwen35MTPLoadTensors(t, cfg)
			// This is a tiny synthetic storage fixture, not real-model acceptance.
			for name, tensor := range tensors {
				b := make([]byte, len(tensor.data)/2)
				for i := 0; i < len(b); i += 2 {
					copy(b[i:i+2], tensor.data[2*i+2:2*i+4])
				}
				tensor.dtype, tensor.data = "BF16", b
				tensors[name] = tensor
			}
			if fault == "shape" {
				tensor := tensors["mtp.norm.weight"]
				tensor.shape = []int{2, 2}
				tensors["mtp.norm.weight"] = tensor
			}
			if fault == "storage" {
				tensor := tensors["mtp.norm.weight"]
				tensor.data = tensor.data[:len(tensor.data)-1]
				tensors["mtp.norm.weight"] = tensor
			}
			path := writeTinySafetensors(t, tensors)
			dir, shard := filepath.Dir(path), filepath.Base(path)
			weights := map[string]string{}
			for name := range tensors {
				weights[name] = shard
			}
			if fault == "path" {
				weights["mtp.norm.weight"] = "../escape.safetensors"
			}
			if fault == "membership" {
				delete(weights, "mtp.norm.weight")
			}
			writeJSON := func(name string, v any) {
				b, err := json.Marshal(v)
				if err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(dir, name), b, 0600); err != nil {
					t.Fatal(err)
				}
			}
			writeJSON("config.json", cfg)
			writeJSON("model.safetensors.index.json", map[string]any{"weight_map": weights})
			if fault == "header" {
				f, err := os.OpenFile(path, os.O_WRONLY, 0600)
				if err != nil {
					t.Fatal(err)
				}
				var prefix [8]byte
				binary.LittleEndian.PutUint64(prefix[:], (1<<20)+1)
				_, err = f.Write(prefix[:])
				f.Close()
				if err != nil {
					t.Fatal(err)
				}
			}
			hashes := map[string]string{}
			for _, name := range []string{"config.json", "model.safetensors.index.json", shard} {
				b, err := os.ReadFile(filepath.Join(dir, name))
				if err != nil {
					t.Fatal(err)
				}
				sum := sha256.Sum256(b)
				hashes[name] = hex.EncodeToString(sum[:])
			}
			if fault == "hash" {
				delete(hashes, shard)
			}
			if fault == "symlink" {
				outside := filepath.Join(t.TempDir(), "outside.safetensors")
				if err := os.Rename(path, outside); err != nil {
					t.Fatal(err)
				}
				if err := os.Symlink(outside, path); err != nil {
					t.Skipf("symlink creation not permitted in this environment: %v", err)
				}
			}
			budget := int64(1 << 20)
			if fault == "budget" {
				budget = 3
			}
			gotCfg, size, err := mtpSafetensorsPreflight(dir, hashes, budget)
			if fault != "valid" {
				if err == nil {
					t.Fatalf("accepted %s defect", fault)
				}
				return
			}
			if err != nil || size <= 0 {
				t.Fatalf("preflight size=%d err=%v", size, err)
			}
			old := RetainMTP
			m, err := mtpLoadRetainedSafetensors(dir, gotCfg)
			if err != nil {
				t.Fatal(err)
			}
			defer m.CloseWeights()
			if RetainMTP != old || len(m.raw) != int(size) || m.manifest["mtp.norm.weight"].Dtype != "f32" {
				t.Fatal("retention restoration or decoded storage mismatch")
			}
			if _, err := mtpLoadRetainedSafetensors(filepath.Join(dir, "absent"), gotCfg); err == nil || RetainMTP != old {
				t.Fatal("failed load did not preserve retention flag")
			}
		})
	}
}

func mtpLoadRetainedSafetensors(dir string, cfg Config) (*Model, error) {
	previous := RetainMTP
	RetainMTP = true
	defer func() { RetainMTP = previous }()
	return LoadSafetensorsDir(dir, cfg)
}

// mtpSafetensorsPreflight inspects bounded headers before the loader allocates
// decoded weights. Hashes attest local bytes; upstream identity remains explicit
// operator provenance. Run only against an immutable snapshot in a dedicated process.
func mtpSafetensorsPreflight(dir string, hashes map[string]string, budget int64) (Config, int64, error) {
	var cfg Config
	root, err := filepath.EvalSymlinks(dir)
	if err != nil {
		return cfg, 0, err
	}
	root, err = filepath.Abs(root)
	if err != nil {
		return cfg, 0, err
	}
	verified := func(name string) (string, error) {
		if name == "" || name == "." || name == ".." || strings.ContainsAny(name, "/\\:") || filepath.IsAbs(name) {
			return "", fmt.Errorf("unsafe artifact path %q", name)
		}
		path, err := filepath.EvalSymlinks(filepath.Join(root, name))
		if err != nil {
			return "", err
		}
		rel, err := filepath.Rel(root, path)
		if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) || filepath.IsAbs(rel) {
			return "", fmt.Errorf("artifact escapes snapshot: %s", name)
		}
		want, err := hex.DecodeString(hashes[name])
		if err != nil || len(want) != sha256.Size {
			return "", fmt.Errorf("missing SHA256: %s", name)
		}
		f, err := os.Open(path)
		if err != nil {
			return "", err
		}
		defer f.Close()
		st, err := f.Stat()
		if err != nil || !st.Mode().IsRegular() {
			return "", fmt.Errorf("artifact is not regular: %s", name)
		}
		h := sha256.New()
		if _, err := io.Copy(h, f); err != nil {
			return "", err
		}
		if !reflect.DeepEqual(h.Sum(nil), want) {
			return "", fmt.Errorf("SHA256 mismatch: %s", name)
		}
		return path, nil
	}
	readMetadata := func(name string) ([]byte, error) {
		path, err := verified(name)
		if err != nil {
			return nil, err
		}
		f, err := os.Open(path)
		if err != nil {
			return nil, err
		}
		defer f.Close()
		b, err := io.ReadAll(io.LimitReader(f, (1<<20)+1))
		if len(b) > 1<<20 {
			return nil, fmt.Errorf("metadata exceeds 1MiB: %s", name)
		}
		return b, err
	}
	config, err := readMetadata("config.json")
	if err != nil {
		return cfg, 0, err
	}
	if err := json.Unmarshal(config, &cfg); err != nil {
		return cfg, 0, err
	}
	if !cfg.isQwen35TextFamily() || cfg.NumMTPLayers() != 1 || cfg.MTPUseDedicatedEmbeddings {
		return cfg, 0, fmt.Errorf("unsupported retained MTP configuration")
	}
	expected, err := qwen35MTPExpectedShapes(cfg)
	if err != nil {
		return cfg, 0, err
	}
	index, err := readMetadata("model.safetensors.index.json")
	if err != nil {
		return cfg, 0, err
	}
	var idx struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := json.Unmarshal(index, &idx); err != nil {
		return cfg, 0, err
	}
	if len(idx.WeightMap) == 0 {
		return cfg, 0, fmt.Errorf("empty safetensors index")
	}
	shards := map[string]bool{}
	for _, shard := range idx.WeightMap {
		if !strings.HasSuffix(shard, ".safetensors") {
			return cfg, 0, fmt.Errorf("invalid shard name %q", shard)
		}
		shards[shard] = true
	}
	seen := map[string]bool{}
	var decoded int64
	for shard := range shards {
		path, err := verified(shard)
		if err != nil {
			return cfg, 0, err
		}
		err = func() error {
			f, err := os.Open(path)
			if err != nil {
				return err
			}
			defer f.Close()
			st, err := f.Stat()
			if err != nil {
				return err
			}
			var prefix [8]byte
			if _, err := io.ReadFull(f, prefix[:]); err != nil {
				return err
			}
			if n := binary.LittleEndian.Uint64(prefix[:]); n > 1<<20 || st.Size() < 8 || n > uint64(st.Size()-8) {
				return fmt.Errorf("invalid or oversized shard header")
			}
			sf, err := newSafetensorsFile(f, st.Size(), nil)
			if err != nil {
				return err
			}
			for _, name := range safetensorsTensorNames(sf.hdr) {
				if seen[name] || idx.WeightMap[name] != shard {
					return fmt.Errorf("index/header mismatch: %s", name)
				}
				seen[name] = true
				var e stEntry
				if err := json.Unmarshal(sf.hdr[name], &e); err != nil {
					return err
				}
				n, ok := checkedShapeProduct(e.Shape...)
				if !ok || n <= 0 || n > int(^uint(0)>>1)/4 {
					return fmt.Errorf("invalid tensor shape: %s", name)
				}
				width := 2
				switch e.Dtype {
				case "F32":
					width = 4
				case "BF16", "F16":
				default:
					return fmt.Errorf("unsupported dtype: %s", name)
				}
				start, end, err := safetensorsDataBounds(sf.dataBase, sf.size, e)
				if err != nil || end-start != int64(n)*int64(width) {
					return fmt.Errorf("invalid tensor storage: %s", name)
				}
				if strings.HasPrefix(name, "mtp.") {
					if shape, ok := expected[name]; !ok || !sameIntShape(shape, e.Shape) {
						return fmt.Errorf("incompatible MTP shape: %s", name)
					}
				}
				if strings.HasPrefix(name, "model.visual.") {
					continue
				}
				size := int64(n) * 4
				if budget <= 0 || size > budget/3-decoded {
					return fmt.Errorf("decoded weights exceed conservative memory reservation")
				}
				decoded += size
			}
			return nil
		}()
		if err != nil {
			return cfg, 0, err
		}
	}
	if len(seen) != len(idx.WeightMap) {
		return cfg, 0, fmt.Errorf("index references missing tensor")
	}
	for name := range expected {
		if !seen[name] {
			return cfg, 0, fmt.Errorf("missing retained tensor: %s", name)
		}
	}
	return cfg, decoded, nil
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
