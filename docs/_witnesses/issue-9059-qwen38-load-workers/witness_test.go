package issue9059witness

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"reflect"
	"strings"
	"testing"
)

type witness struct {
	Schema         string    `json:"schema"`
	Issue          int       `json:"issue"`
	RunID          string    `json:"run_id"`
	CapturedUTC    string    `json:"captured_utc"`
	Scope          string    `json:"scope"`
	Host           host      `json:"host"`
	FrozenIdentity identity  `json:"frozen_identity"`
	Isolation      isolation `json:"isolation"`
	Arms           []arm     `json:"arms"`
	Deltas         deltas    `json:"exact_deltas_candidate_minus_control"`
	Decision       decision  `json:"decision"`
}

type host struct {
	OS                 string `json:"os"`
	Arch               string `json:"arch"`
	Machine            string `json:"machine"`
	GPUCores           int    `json:"gpu_cores"`
	UnifiedMemoryBytes int64  `json:"unified_memory_bytes"`
}

type identity struct {
	Model             string `json:"model"`
	ModelBytes        int64  `json:"model_bytes"`
	ModelSourceSHA256 string `json:"model_source_sha256"`
	BinarySHA256      string `json:"binary_sha256"`
	BaseRevision      string `json:"base_revision"`
	SourceDiffSHA256  string `json:"source_diff_sha256"`
	GoVersion         string `json:"go_version"`
	AppVersion        string `json:"app_version"`
}

type isolation struct {
	GPULease struct {
		PersistentHelper   bool   `json:"persistent_helper"`
		HeldAcrossBothArms bool   `json:"held_across_both_arms"`
		HelperBinarySHA256 string `json:"helper_binary_sha256"`
	} `json:"gpu_lease"`
	Port8090 struct {
		ExactOwnerCommandSHA256 string `json:"exact_owner_command_sha256"`
		WatcherScriptSHA256     string `json:"watcher_script_sha256"`
		QuiescedBeforeArms      bool   `json:"quiesced_before_arms"`
		WatcherActiveAcrossArms bool   `json:"watcher_active_across_arms"`
		ListenerOverlapObserved bool   `json:"listener_overlap_observed"`
	} `json:"port_8090"`
	CleanupRestoration struct {
		Watcher struct {
			Stopped          bool `json:"stopped"`
			MatchedTerms     int  `json:"matched_terms"`
			UnmatchedSignals int  `json:"unmatched_signals"`
			ReusedPIDSignals int  `json:"reused_pid_signals"`
		} `json:"watcher"`
		OriginalLlamaOwner struct {
			Restored      bool   `json:"restored"`
			PID           int    `json:"pid"`
			CommandSHA256 string `json:"command_sha256"`
			ModelIdentity string `json:"model_identity"`
			StableSeconds int    `json:"stable_seconds"`
		} `json:"original_llama_owner"`
		StaleGPULeaseHelper struct {
			VerifiedAbsent         bool `json:"verified_absent"`
			ConsecutiveLockSamples int  `json:"consecutive_lock_samples"`
		} `json:"stale_gpu_lease_helper"`
	} `json:"cleanup_restoration"`
	ArmsSerialized                        bool  `json:"arms_serialized"`
	CacheDisplacementBytesPerArm          int64 `json:"cache_displacement_bytes_per_arm"`
	CacheDisplacementExceedsUnifiedMemory bool  `json:"cache_displacement_exceeds_unified_memory"`
}

type displacement struct {
	Name   string `json:"name"`
	Bytes  int64  `json:"bytes"`
	SHA256 string `json:"sha256"`
}

type arm struct {
	Name                  string         `json:"name"`
	IdentityRef           string         `json:"identity_ref"`
	LoadWorkersLiteral    string         `json:"load_workers_literal"`
	LoadWorkersEffective  int            `json:"load_workers_effective"`
	GOMAXPROCS            int            `json:"gomaxprocs"`
	LoadMS                float64        `json:"load_ms"`
	PeakRSSBytes          int64          `json:"peak_rss_bytes"`
	PeakFootprintBytes    int64          `json:"peak_footprint_bytes"`
	SwapBeforeMiB         float64        `json:"swap_before_mib"`
	SwapPeakMiB           float64        `json:"swap_peak_mib"`
	SwapAfterMiB          float64        `json:"swap_after_mib"`
	PositiveSwapGrowthMiB float64        `json:"positive_swap_growth_mib"`
	ExitStatus            int            `json:"exit_status"`
	CheckpointOwnerClosed bool           `json:"checkpoint_owner_closed"`
	Displacement          []displacement `json:"displacement"`
}

type deltas struct {
	LoadMS                  float64 `json:"load_ms"`
	LoadPercent             float64 `json:"load_percent"`
	PeakRSSBytes            int64   `json:"peak_rss_bytes"`
	PeakRSSPercent          float64 `json:"peak_rss_percent"`
	PeakFootprintBytes      int64   `json:"peak_footprint_bytes"`
	PeakFootprintPercent    float64 `json:"peak_footprint_percent"`
	SwapAfterMinusBeforeMiB float64 `json:"swap_after_minus_before_mib"`
}

type decision struct {
	Verdict                         string `json:"verdict"`
	KeepRequiresLowerPeakPressure   bool   `json:"keep_requires_lower_peak_pressure"`
	CandidateLowerPeakRSS           bool   `json:"candidate_lower_peak_rss"`
	CandidateLowerPeakFootprint     bool   `json:"candidate_lower_peak_footprint"`
	CandidateZeroPositiveSwapGrowth bool   `json:"candidate_zero_positive_swap_growth"`
	BothCleanTeardown               bool   `json:"both_clean_teardown"`
	NativeProfileRun                bool   `json:"native_profile_run"`
	NativeProfileSkipReason         string `json:"native_profile_skip_reason"`
	Claim                           string `json:"claim"`
}

func TestQwen38LoadWorkerAblationWitness(t *testing.T) {
	raw, err := os.ReadFile("witness.json")
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(raw, []byte("/Users/")) {
		t.Fatal("public witness leaked a workstation path")
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	var w witness
	if err := dec.Decode(&w); err != nil {
		t.Fatalf("decode witness: %v", err)
	}

	if w.Schema != "fak.qwen38-load-worker-ablation-witness/1" || w.Issue != 9059 || w.RunID != "run-20260826T1201Z" {
		t.Fatalf("unexpected witness identity: schema=%q issue=%d run=%q", w.Schema, w.Issue, w.RunID)
	}
	if w.Scope != "fak-native streamed dense Q4_K load-only pressure ablation" {
		t.Fatalf("scope = %q", w.Scope)
	}
	if w.Host.OS != "darwin" || w.Host.Arch != "arm64" || w.Host.Machine != "Apple M3 Pro" || w.Host.GPUCores != 18 || w.Host.UnifiedMemoryBytes != 38654705664 {
		t.Fatalf("host envelope = %+v", w.Host)
	}

	identity := w.FrozenIdentity
	if identity.Model != "unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf" || identity.ModelBytes != 17106775008 {
		t.Fatalf("model identity = %+v", identity)
	}
	assertSHA256(t, "model source", identity.ModelSourceSHA256, "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169")
	assertSHA256(t, "binary", identity.BinarySHA256, "e6021596691ea9e9dc3c767deb7b7b0703e59cd8620d7f924939a3e7db84bb35")
	assertSHA256(t, "source diff", identity.SourceDiffSHA256, "dd5ff3c094ce9e38788748ebeb625a49a3b935d7dfc452ec0d99a5aaf5f02910")
	assertHex(t, "base revision", identity.BaseRevision, 20)
	if identity.BaseRevision != "83ee601a42554f98cf6e7f3d1cb0b4d9b5c540ec" || identity.GoVersion != "go1.26.6" || identity.AppVersion != "0.45.0" {
		t.Fatalf("build identity = %+v", identity)
	}

	iso := w.Isolation
	if !iso.GPULease.PersistentHelper || !iso.GPULease.HeldAcrossBothArms {
		t.Fatal("persistent machine-wide GPU lease was not bound across both arms")
	}
	assertSHA256(t, "GPU lease helper", iso.GPULease.HelperBinarySHA256, "611c3439921c4f38c44d7d308583ca0f949f30db4a565a5b080e89bd3144d2a2")
	assertSHA256(t, "8090 owner command", iso.Port8090.ExactOwnerCommandSHA256, "a0c13c8d7e84fa92c76db532db0a6fed13b3b42313a75e10b8f69e252f76a91d")
	assertSHA256(t, "8090 watcher", iso.Port8090.WatcherScriptSHA256, "7c52b59f0dee026e189cad5e5480e53325e29c3b03bc1e7dc11fb349e938d1ff")
	if !iso.Port8090.QuiescedBeforeArms || !iso.Port8090.WatcherActiveAcrossArms || iso.Port8090.ListenerOverlapObserved || !iso.ArmsSerialized {
		t.Fatalf("unsafe or overlapping arm isolation: %+v", iso)
	}
	cleanup := iso.CleanupRestoration
	if !cleanup.Watcher.Stopped || cleanup.Watcher.MatchedTerms != 168 || cleanup.Watcher.UnmatchedSignals != 0 || cleanup.Watcher.ReusedPIDSignals != 0 {
		t.Fatalf("watcher cleanup receipt = %+v", cleanup.Watcher)
	}
	owner := cleanup.OriginalLlamaOwner
	if !owner.Restored || owner.PID != 62326 || owner.ModelIdentity != "qwen3.6-27b" || owner.StableSeconds != 90 {
		t.Fatalf("original llama restoration receipt = %+v", owner)
	}
	assertSHA256(t, "restored 8090 owner command", owner.CommandSHA256, iso.Port8090.ExactOwnerCommandSHA256)
	if !cleanup.StaleGPULeaseHelper.VerifiedAbsent || cleanup.StaleGPULeaseHelper.ConsecutiveLockSamples != 30 {
		t.Fatalf("stale GPU lease helper receipt = %+v", cleanup.StaleGPULeaseHelper)
	}
	if !iso.CacheDisplacementExceedsUnifiedMemory || iso.CacheDisplacementBytesPerArm <= w.Host.UnifiedMemoryBytes {
		t.Fatalf("cache displacement %d does not exceed unified memory %d", iso.CacheDisplacementBytesPerArm, w.Host.UnifiedMemoryBytes)
	}

	if len(w.Arms) != 2 {
		t.Fatalf("arms = %d, want 2", len(w.Arms))
	}
	control := armByName(t, w.Arms, "control-workers12")
	candidate := armByName(t, w.Arms, "candidate-workers1")
	assertArmIdentity(t, control, "12", 12)
	assertArmIdentity(t, candidate, "1", 1)
	if control.LoadMS != 16685.636875 || control.PeakRSSBytes != 16621256704 || control.PeakFootprintBytes != 28454507112 {
		t.Fatalf("control measurements = %+v", control)
	}
	if candidate.LoadMS != 18465.682875 || candidate.PeakRSSBytes != 17109860352 || candidate.PeakFootprintBytes != 30846490920 {
		t.Fatalf("candidate measurements = %+v", candidate)
	}
	if control.SwapBeforeMiB != 2569.25 || control.SwapPeakMiB != 2569.25 || control.SwapAfterMiB != 2569.25 {
		t.Fatalf("control swap window = before %.2f peak %.2f after %.2f", control.SwapBeforeMiB, control.SwapPeakMiB, control.SwapAfterMiB)
	}
	if candidate.SwapBeforeMiB != 2513.25 || candidate.SwapPeakMiB != 2513.25 || candidate.SwapAfterMiB != 2473.25 {
		t.Fatalf("candidate swap window = before %.2f peak %.2f after %.2f", candidate.SwapBeforeMiB, candidate.SwapPeakMiB, candidate.SwapAfterMiB)
	}
	assertSwapGrowth(t, control)
	assertSwapGrowth(t, candidate)
	if !reflect.DeepEqual(control.Displacement, candidate.Displacement) {
		t.Fatal("arms did not use the identical cache-displacement pair")
	}
	var displacementBytes int64
	wantDisplacementSHA := map[string]string{
		"cache-displacement-a": "33c45709dc2638426f1b86abc71a4dc4ecf6aae94fea583308b38459d99fee71",
		"cache-displacement-b": "0d003f6662faee786ed5da3e31b29c978de5ae5d275c8794c606a7f3c01aa8f5",
	}
	for _, artifact := range control.Displacement {
		displacementBytes += artifact.Bytes
		want, ok := wantDisplacementSHA[artifact.Name]
		if !ok {
			t.Fatalf("unexpected displacement artifact %q", artifact.Name)
		}
		assertSHA256(t, artifact.Name, artifact.SHA256, want)
	}
	if displacementBytes != 48368448672 || displacementBytes != iso.CacheDisplacementBytesPerArm {
		t.Fatalf("displacement bytes = %d", displacementBytes)
	}

	assertClose(t, "load delta", w.Deltas.LoadMS, candidate.LoadMS-control.LoadMS, 1e-9)
	assertClose(t, "load percent", w.Deltas.LoadPercent, 100*(candidate.LoadMS-control.LoadMS)/control.LoadMS, 1e-9)
	if w.Deltas.PeakRSSBytes != candidate.PeakRSSBytes-control.PeakRSSBytes || w.Deltas.PeakFootprintBytes != candidate.PeakFootprintBytes-control.PeakFootprintBytes {
		t.Fatalf("byte deltas = %+v", w.Deltas)
	}
	assertClose(t, "RSS percent", w.Deltas.PeakRSSPercent, 100*float64(w.Deltas.PeakRSSBytes)/float64(control.PeakRSSBytes), 1e-9)
	assertClose(t, "footprint percent", w.Deltas.PeakFootprintPercent, 100*float64(w.Deltas.PeakFootprintBytes)/float64(control.PeakFootprintBytes), 1e-9)
	assertClose(t, "candidate swap delta", w.Deltas.SwapAfterMinusBeforeMiB, candidate.SwapAfterMiB-candidate.SwapBeforeMiB, 1e-9)
	if w.Deltas.LoadMS != 1780.046 || w.Deltas.PeakRSSBytes != 488603648 || w.Deltas.PeakFootprintBytes != 2391983808 || w.Deltas.SwapAfterMinusBeforeMiB != -40 {
		t.Fatalf("unexpected exact deltas: %+v", w.Deltas)
	}

	lowerRSS := candidate.PeakRSSBytes < control.PeakRSSBytes
	lowerFootprint := candidate.PeakFootprintBytes < control.PeakFootprintBytes
	zeroPositiveSwap := candidate.PositiveSwapGrowthMiB == 0
	cleanTeardown := control.ExitStatus == 0 && candidate.ExitStatus == 0 && control.CheckpointOwnerClosed && candidate.CheckpointOwnerClosed
	keep := lowerRSS && lowerFootprint && zeroPositiveSwap && cleanTeardown
	if keep || w.Decision.Verdict != "REJECT" {
		t.Fatalf("verdict mismatch: keep=%v recorded=%q", keep, w.Decision.Verdict)
	}
	if !w.Decision.KeepRequiresLowerPeakPressure || w.Decision.CandidateLowerPeakRSS != lowerRSS || w.Decision.CandidateLowerPeakFootprint != lowerFootprint || w.Decision.CandidateZeroPositiveSwapGrowth != zeroPositiveSwap || w.Decision.BothCleanTeardown != cleanTeardown {
		t.Fatalf("decision predicates do not match measurements: %+v", w.Decision)
	}
	if w.Decision.NativeProfileRun || !strings.Contains(w.Decision.NativeProfileSkipReason, "failed the load-only pressure gate") {
		t.Fatalf("native-profile conditional was not enforced: %+v", w.Decision)
	}
	if !strings.Contains(w.Decision.Claim, "did not lower measured startup pressure") {
		t.Fatalf("claim does not state the bounded REJECT result: %q", w.Decision.Claim)
	}
}

func armByName(t *testing.T, arms []arm, name string) arm {
	t.Helper()
	for _, a := range arms {
		if a.Name == name {
			return a
		}
	}
	t.Fatalf("arm %q missing", name)
	return arm{}
}

func assertArmIdentity(t *testing.T, a arm, literal string, effective int) {
	t.Helper()
	if a.IdentityRef != "frozen_identity" || a.LoadWorkersLiteral != literal || a.LoadWorkersEffective != effective || a.GOMAXPROCS != 12 {
		t.Fatalf("arm identity/control = %+v", a)
	}
	if a.ExitStatus != 0 || !a.CheckpointOwnerClosed {
		t.Fatalf("arm did not exit and tear down cleanly: %+v", a)
	}
}

func assertSwapGrowth(t *testing.T, a arm) {
	t.Helper()
	want := math.Max(0, a.SwapPeakMiB-a.SwapBeforeMiB)
	if a.PositiveSwapGrowthMiB != want {
		t.Fatalf("%s positive swap growth = %.2f, recomputed %.2f", a.Name, a.PositiveSwapGrowthMiB, want)
	}
}

func assertSHA256(t *testing.T, label, got, want string) {
	t.Helper()
	assertHex(t, label, got, 32)
	if got != want {
		t.Fatalf("%s SHA-256 = %s, want %s", label, got, want)
	}
}

func assertHex(t *testing.T, label, value string, bytes int) {
	t.Helper()
	decoded, err := hex.DecodeString(value)
	if err != nil || len(decoded) != bytes || value != strings.ToLower(value) {
		t.Fatalf("%s is not %d-byte lowercase hex: %q", label, bytes, value)
	}
}

func assertClose(t *testing.T, label string, got, want, tolerance float64) {
	t.Helper()
	if math.Abs(got-want) > tolerance {
		t.Fatalf("%s = %.12f, recomputed %.12f", label, got, want)
	}
}
