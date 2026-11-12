package servereceipt

import (
	"path/filepath"
	"strings"
	"testing"
)

// loadTinyFacts loads the committed tiny fixture for exact-assertion tests.
func loadTinyFacts(t *testing.T) Facts {
	t.Helper()
	f, err := LoadFacts(filepath.Join("testdata", "tiny.facts.json"))
	if err != nil {
		t.Fatalf("LoadFacts(testdata/tiny.facts.json): %v", err)
	}
	return f
}

func loadKimiFacts(t *testing.T) Facts {
	t.Helper()
	f, err := LoadFacts(filepath.Join("..", "..", "examples", "serve-receipt", "kimi-k31-pro-nvfp4.facts.json"))
	if err != nil {
		t.Fatalf("LoadFacts(examples/serve-receipt/kimi-k31-pro-nvfp4.facts.json): %v", err)
	}
	return f
}

func testGPU(t *testing.T, key string) GPUProfile {
	t.Helper()
	p, ok := LookupProfile(key)
	if !ok {
		t.Fatalf("LookupProfile(%q): not found", key)
	}
	return p
}

// TestKVBytesPerTokenPerGPU pins the Kimi constant: 24 × ((512+64)×1 + 128×1 + 4) = 16,992 B.
func TestKVBytesPerTokenPerGPU(t *testing.T) {
	kimi := KVShape{FullAttnLayers: 24, KVLoraRank: 512, QKRopeHeadDim: 64, IndexerKeyDim: 128,
		LatentDTypeBytes: 1, IndexerKeyDTypeBytes: 1, IndexerScaleBytes: 4, TPReplicated: true}
	if got := KVBytesPerTokenPerGPU(kimi, 8); got != 16992 {
		t.Fatalf("Kimi kv/token = %d, want 16992", got)
	}
	if got := KVBytesPerTokenPerGPU(kimi, 16); got != 16992 {
		t.Fatalf("Kimi kv/token at tp16 (replicated) = %d, want 16992 (no TP divide)", got)
	}

	tiny := KVShape{FullAttnLayers: 10, KVLoraRank: 64, QKRopeHeadDim: 32, IndexerKeyDim: 16,
		LatentDTypeBytes: 2, IndexerKeyDTypeBytes: 2, IndexerScaleBytes: 4, TPReplicated: false}
	// 10 × ((64+32)×2 + 16×2 + 4) = 2,280 per rank-holder; sharded → /4 = 570.
	if got := KVBytesPerTokenPerGPU(tiny, 4); got != 570 {
		t.Fatalf("tiny kv/token at tp4 (sharded) = %d, want 570", got)
	}
	replicated := tiny
	replicated.TPReplicated = true
	if got := KVBytesPerTokenPerGPU(replicated, 4); got != 2280 {
		t.Fatalf("tiny kv/token at tp4 (replicated) = %d, want 2280", got)
	}
}

func TestKVFormula(t *testing.T) {
	kimi := KVShape{FullAttnLayers: 24, KVLoraRank: 512, QKRopeHeadDim: 64, IndexerKeyDim: 128,
		LatentDTypeBytes: 1, IndexerKeyDTypeBytes: 1, IndexerScaleBytes: 4, TPReplicated: true}
	want := "24 × ((512+64)×1 + 128×1 + 4)"
	if got := KVFormula(kimi, 8); got != want {
		t.Fatalf("KVFormula = %q, want %q", got, want)
	}
	sharded := kimi
	sharded.TPReplicated = false
	if got := KVFormula(sharded, 8); got != want+" / tp 8" {
		t.Fatalf("KVFormula sharded = %q, want %q", got, want+" / tp 8")
	}
}

// TestPlanTinyExactArithmetic asserts every intermediate byte for the tiny
// fixture at tp=4 (all divisions exact; util budget floored once).
func TestPlanTinyExactArithmetic(t *testing.T) {
	facts := loadTinyFacts(t)
	receipt, err := Plan(facts, testGPU(t, "rtx-pro-6000-blackwell-96gb"),
		Config{TP: 4, PP: 1, MaxNumSeqs: 32, GPUMemUtil: 0.90, ActivationReserveGB: 2})
	if err != nil {
		t.Fatalf("Plan: %v", err)
	}
	ar := receipt.Arithmetic
	want := Arithmetic{
		Units:                  UnitsNote,
		GPUsTotal:              4,
		WeightsPerGPUBytes:     250_000_000, // ceil(1e9 / 4)
		KVBytesPerTokenPerGPU:  570,         // 2,280 / tp 4 (not tp_replicated)
		KVFormula:              "10 × ((64+32)×2 + 16×2 + 4) / tp 4",
		StatePerGPUPerSeqBytes: 2_000_000,       // 8,000,000 / tp 4 (sharded)
		VRAMBytesPerGPU:        103_079_215_104, // 96 GiB
		UtilBudgetBytes:        92_771_293_593,  // floor(103,079,215,104 × 0.90)
		UsablePerGPUBytes:      92_521_293_593,  // budget − 250,000,000
		ReservedStateBytes:     64_000_000,      // 2,000,000 × 32 seqs
		ActivationReserveBytes: 2_000_000_000,   // 2 decimal GB
		MaxCachedTokens:        158_697_006,     // floor(90,457,293,593 / 570)
		RequestedMaxModelLen:   0,
		HeadroomTokens:         158_697_006,
	}
	if ar != want {
		t.Fatalf("arithmetic mismatch:\n got %+v\nwant %+v", ar, want)
	}
	if receipt.Verdict != VerdictOK {
		t.Fatalf("verdict = %s, want OK", receipt.Verdict)
	}
}

// TestPlanTPReplicationVsSharding: flipping the two TP flags must move the
// per-GPU numbers by exactly the TP factor, nothing else.
func TestPlanTPReplicationVsSharding(t *testing.T) {
	facts := loadTinyFacts(t)
	gpu := testGPU(t, "rtx-pro-6000-blackwell-96gb")
	cfg := Config{TP: 4, PP: 1, MaxNumSeqs: 32, GPUMemUtil: 0.90, ActivationReserveGB: 2}

	sharded, err := Plan(facts, gpu, cfg)
	if err != nil {
		t.Fatalf("Plan sharded: %v", err)
	}
	rep := facts
	rep.KV.TPReplicated = true
	replicated, err := Plan(rep, gpu, cfg)
	if err != nil {
		t.Fatalf("Plan replicated: %v", err)
	}
	if replicated.Arithmetic.KVBytesPerTokenPerGPU != 4*sharded.Arithmetic.KVBytesPerTokenPerGPU {
		t.Fatalf("replicated kv/token %d, want 4 × sharded %d",
			replicated.Arithmetic.KVBytesPerTokenPerGPU, sharded.Arithmetic.KVBytesPerTokenPerGPU)
	}

	unsharded := facts
	unsharded.StateTPSharded = false
	plain, err := Plan(unsharded, gpu, cfg)
	if err != nil {
		t.Fatalf("Plan unsharded state: %v", err)
	}
	if plain.Arithmetic.StatePerGPUPerSeqBytes != 4*sharded.Arithmetic.StatePerGPUPerSeqBytes {
		t.Fatalf("unsharded state/seq %d, want 4 × sharded %d",
			plain.Arithmetic.StatePerGPUPerSeqBytes, sharded.Arithmetic.StatePerGPUPerSeqBytes)
	}
	if plain.Arithmetic.ReservedStateBytes != 4*sharded.Arithmetic.ReservedStateBytes {
		t.Fatalf("unsharded reserved_state %d, want 4 × sharded %d",
			plain.Arithmetic.ReservedStateBytes, sharded.Arithmetic.ReservedStateBytes)
	}
}

// TestPlanVerdicts covers OK / TIGHT / both INFEASIBLE causes.
func TestPlanVerdicts(t *testing.T) {
	facts := loadTinyFacts(t)
	gpu := testGPU(t, "rtx-pro-6000-blackwell-96gb")
	base := Config{TP: 4, PP: 1, MaxNumSeqs: 32, GPUMemUtil: 0.90, ActivationReserveGB: 2}

	tests := []struct {
		name       string
		mutate     func(*Facts, *Config)
		want       Verdict
		wantReason string // substring
	}{
		{
			name:   "ok without a request",
			mutate: func(f *Facts, c *Config) {},
			want:   VerdictOK,
		},
		{
			name: "ok when the request fits exactly",
			mutate: func(f *Facts, c *Config) {
				c.MaxModelLen = 158_697_006
			},
			want: VerdictOK,
		},
		{
			name: "tight when the request exceeds the cache budget",
			mutate: func(f *Facts, c *Config) {
				c.MaxModelLen = 1_000_000_000
			},
			want:       VerdictTight,
			wantReason: "max_cached_tokens 158697006 < requested max-model-len 1000000000",
		},
		{
			name: "infeasible when weights exceed the util budget",
			mutate: func(f *Facts, c *Config) {
				f.WeightsBytes = 1_000_000_000_000
				c.TP = 1
			},
			want:       VerdictInfeasible,
			wantReason: "usable_per_gpu ≤ 0 after weights",
		},
		{
			name: "infeasible when heads do not divide TP",
			mutate: func(f *Facts, c *Config) {
				f.NumAttentionHeads = 33
			},
			want:       VerdictInfeasible,
			wantReason: "num_attention_heads 33 not divisible by tp 4",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			f, c := facts, base
			tc.mutate(&f, &c)
			receipt, err := Plan(f, gpu, c)
			if err != nil {
				t.Fatalf("Plan: %v", err)
			}
			if receipt.Verdict != tc.want {
				t.Fatalf("verdict = %s (%s), want %s", receipt.Verdict, receipt.Reason, tc.want)
			}
			if tc.wantReason != "" && !strings.Contains(receipt.Reason, tc.wantReason) {
				t.Fatalf("reason = %q, want substring %q", receipt.Reason, tc.wantReason)
			}
		})
	}
}

// TestPlanWarnings: the two non-fatal warnings fire exactly per condition.
func TestPlanWarnings(t *testing.T) {
	facts := loadTinyFacts(t)
	base := Config{TP: 8, PP: 1, MaxNumSeqs: 32, GPUMemUtil: 0.90, ActivationReserveGB: 2}

	tests := []struct {
		name      string
		gpuKey    string
		tp        int
		wantCount int
		wantSub   []string
	}{
		{name: "ada at tp8 fires both", gpuKey: "rtx-6000-ada-48gb", tp: 8, wantCount: 2,
			wantSub: []string{"NVFP4 checkpoints require Blackwell SM≥10.0", "wide TP all-reduce over PCIe"}},
		{name: "blackwell at tp2 fires none", gpuKey: "rtx-pro-6000-blackwell-96gb", tp: 2, wantCount: 0},
		{name: "b200 nvlink at tp8 fires none", gpuKey: "b200-180gb", tp: 8, wantCount: 0},
		{name: "ada at tp2 fires fp4 only", gpuKey: "rtx-6000-ada-48gb", tp: 2, wantCount: 1,
			wantSub: []string{"NVFP4 checkpoints require Blackwell SM≥10.0"}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := base
			cfg.TP = tc.tp
			receipt, err := Plan(facts, testGPU(t, tc.gpuKey), cfg)
			if err != nil {
				t.Fatalf("Plan: %v", err)
			}
			if len(receipt.Warnings) != tc.wantCount {
				t.Fatalf("warnings = %v, want %d", receipt.Warnings, tc.wantCount)
			}
			for _, sub := range tc.wantSub {
				found := false
				for _, w := range receipt.Warnings {
					if strings.Contains(w, sub) {
						found = true
					}
				}
				if !found {
					t.Fatalf("warnings %v: no entry contains %q", receipt.Warnings, sub)
				}
			}
		})
	}
}

// TestProfiles pins the built-in catalog shape and lookup behavior.
func TestProfiles(t *testing.T) {
	profiles := AllProfiles()
	if len(profiles) != 4 {
		t.Fatalf("AllProfiles = %d entries, want 4", len(profiles))
	}
	want := map[string]GPUProfile{
		"rtx-pro-6000-blackwell-96gb": {Key: "rtx-pro-6000-blackwell-96gb", VRAMGB: 96, SM: "12.0", FP4Native: true, NVLink: false, PCIe: "5.0 x16"},
		"b200-180gb":                  {Key: "b200-180gb", VRAMGB: 180, SM: "10.0", FP4Native: true, NVLink: true},
		"gb300-288gb":                 {Key: "gb300-288gb", VRAMGB: 288, SM: "10.3", FP4Native: true, NVLink: true},
		"rtx-6000-ada-48gb":           {Key: "rtx-6000-ada-48gb", VRAMGB: 48, SM: "8.9", FP4Native: false, NVLink: false},
	}
	for _, p := range profiles {
		w, ok := want[p.Key]
		if !ok || p != w {
			t.Fatalf("profile %q = %+v, want %+v", p.Key, p, w)
		}
	}
	keys := ProfileKeys()
	for i := 1; i < len(keys); i++ {
		if keys[i-1] >= keys[i] {
			t.Fatalf("ProfileKeys not sorted: %v", keys)
		}
	}
	if _, ok := LookupProfile("no-such-gpu"); ok {
		t.Fatal("LookupProfile(no-such-gpu) = ok, want false")
	}
	if p, ok := LookupProfile("B200-180GB"); !ok || p.Key != "b200-180gb" {
		t.Fatalf("LookupProfile case-insensitive: (%+v, %v)", p, ok)
	}
}

// TestPlanInputErrors: structurally impossible inputs refuse with a
// usage-class error (CLI exit 2), not a verdict.
func TestPlanInputErrors(t *testing.T) {
	facts := loadTinyFacts(t)
	gpu := testGPU(t, "rtx-pro-6000-blackwell-96gb")
	base := Config{TP: 4, PP: 1, MaxNumSeqs: 32, GPUMemUtil: 0.90, ActivationReserveGB: 2}

	tests := []struct {
		name   string
		mutate func(*Facts, *Config)
	}{
		{name: "wrong schema", mutate: func(f *Facts, c *Config) { f.Schema = "fak.serve-receipt.facts/0" }},
		{name: "empty model", mutate: func(f *Facts, c *Config) { f.Model = " " }},
		{name: "zero weights", mutate: func(f *Facts, c *Config) { f.WeightsBytes = 0 }},
		{name: "zero kv layers", mutate: func(f *Facts, c *Config) { f.KV.FullAttnLayers = 0 }},
		{name: "tp 0", mutate: func(f *Facts, c *Config) { c.TP = 0 }},
		{name: "zero seqs", mutate: func(f *Facts, c *Config) { c.MaxNumSeqs = 0 }},
		{name: "util out of range", mutate: func(f *Facts, c *Config) { c.GPUMemUtil = 1.5 }},
		{name: "negative reserve", mutate: func(f *Facts, c *Config) { c.ActivationReserveGB = -1 }},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			f, c := facts, base
			tc.mutate(&f, &c)
			if _, err := Plan(f, gpu, c); err == nil {
				t.Fatal("Plan = nil error, want usage-class error")
			}
		})
	}
}
