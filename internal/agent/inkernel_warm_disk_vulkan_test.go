//go:build vulkan && (windows || linux) && cgo

package agent

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"runtime/debug"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

const (
	vulkanWarmDiskOptInEnv      = "FAK_TEST_VULKAN_WARM_DISK_RESTART"
	vulkanWarmDiskChildEnv      = "FAK_TEST_VULKAN_WARM_DISK_CHILD"
	vulkanWarmDiskChildMarker   = "FAK_VULKAN_WARM_DISK_CHILD="
	vulkanWarmDiskReceiptMarker = "FAK_VULKAN_WARM_DISK_RECEIPT="
	vulkanWarmDiskSchema        = "fak-vulkan-warm-disk-restart/1"
)

// The source-bound builder sets both stamps from its admitted immutable archive:
//
//	-ldflags "-X github.com/anthony-chaudhary/fak/internal/agent.vulkanWarmDiskSourceRevisionStamp=<full-commit> -X github.com/anthony-chaudhary/fak/internal/agent.vulkanWarmDiskSourceArchiveSHA256Stamp=<archive-sha256>"
//
// They deliberately have no environment fallback: an executing process cannot
// promote itself into source-bound evidence by choosing environment variables.
var (
	vulkanWarmDiskSourceRevisionStamp      string
	vulkanWarmDiskSourceArchiveSHA256Stamp string
	vulkanWarmDiskOptInFlag                = flag.Bool("fak-vulkan-warm-disk-restart", false, "run the physical Vulkan warm-disk restart witness")
)

type vulkanWarmDiskTransferReceipt struct {
	ComputeDispatches uint64 `json:"compute_dispatches"`
	DispatchSubmits   uint64 `json:"dispatch_submits"`
	H2DCount          uint64 `json:"h2d_count"`
	H2DBytes          uint64 `json:"h2d_bytes"`
	D2HCount          uint64 `json:"d2h_count"`
	D2HBytes          uint64 `json:"d2h_bytes"`
}

type vulkanWarmDiskChildReceipt struct {
	Schema                   string                        `json:"schema"`
	Phase                    string                        `json:"phase"`
	PID                      int                           `json:"pid"`
	SourceRevision           string                        `json:"source_revision"`
	SourceArchiveSHA256      string                        `json:"source_archive_sha256"`
	SourceModified           bool                          `json:"source_modified"`
	ExecutableSHA256         string                        `json:"executable_sha256"`
	ModelIdentity            string                        `json:"model_identity"`
	Backend                  string                        `json:"backend"`
	Device                   string                        `json:"device"`
	Driver                   string                        `json:"driver"`
	Runtime                  string                        `json:"runtime"`
	DiskOutcome              string                        `json:"disk_outcome"`
	DiskTier                 string                        `json:"disk_tier"`
	WriteBytes               int64                         `json:"write_bytes,omitempty"`
	ReadBytes                int64                         `json:"read_bytes,omitempty"`
	Prefilled                int                           `json:"prefilled_tokens"`
	Stable                   int                           `json:"stable_tokens"`
	Matched                  int                           `json:"matched_tokens"`
	Generated                []int                         `json:"generated_tokens"`
	LogitsDigest             string                        `json:"logits_digest"`
	ContinuationLogitsDigest string                        `json:"continuation_logits_digest"`
	Transfers                vulkanWarmDiskTransferReceipt `json:"transfers"`
}

type vulkanWarmDiskRestartReceipt struct {
	Schema                  string                     `json:"schema"`
	PID                     int                        `json:"pid"`
	CacheMode               string                     `json:"cache_mode"`
	SourceRevision          string                     `json:"source_revision"`
	SourceArchiveSHA256     string                     `json:"source_archive_sha256"`
	SourceModified          bool                       `json:"source_modified"`
	ExecutableSHA256        string                     `json:"executable_sha256"`
	ModelIdentity           string                     `json:"model_identity"`
	ExpectedKVTransferBytes uint64                     `json:"expected_kv_transfer_bytes"`
	ExpectedColdD2HBytes    uint64                     `json:"expected_cold_d2h_bytes"`
	ExpectedKVTransferCount uint64                     `json:"expected_kv_transfer_count"`
	Cold                    vulkanWarmDiskChildReceipt `json:"cold"`
	Warm                    vulkanWarmDiskChildReceipt `json:"warm"`
	TokenParity             bool                       `json:"token_parity"`
	LogitParity             bool                       `json:"logit_parity"`
	ContinuationLogitParity bool                       `json:"continuation_logit_parity"`
}

// fak-test:justify why=integration when=changed:internal/agent/**
// fak-test:runtime slow est=30s lane=optin
// Runtime justification: the witness starts two fresh required-device processes so no
// process memory can satisfy the restart; the synthetic model bounds each child to 45s.
func TestInKernelWarmPrefixDiskAcrossProcessesVulkan(t *testing.T) {
	if os.Getenv(vulkanWarmDiskOptInEnv) != "1" && !*vulkanWarmDiskOptInFlag {
		t.Skip("set FAK_TEST_VULKAN_WARM_DISK_RESTART=1 for the physical restart witness")
	}
	requireVulkanWarmDiskEnvironment(t)

	cacheRoot := t.TempDir()
	run := func(phase string) vulkanWarmDiskChildReceipt {
		t.Helper()
		ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
		defer cancel()
		cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestInKernelWarmPrefixDiskVulkanProcessHelper$", "-test.count=1")
		cmd.Env = append(warmDiskProcessEnv(os.Environ(), cacheRoot, phase), vulkanWarmDiskChildEnv+"="+phase)
		output, err := cmd.CombinedOutput()
		if ctx.Err() != nil {
			t.Fatalf("%s Vulkan child timed out: %v", phase, ctx.Err())
		}
		if err != nil {
			t.Fatalf("%s Vulkan child failed: %v output=%q", phase, err, boundedVulkanWarmDiskOutput(output))
		}
		for _, line := range bytes.Split(output, []byte{'\n'}) {
			line = bytes.TrimSpace(line)
			if !bytes.HasPrefix(line, []byte(vulkanWarmDiskChildMarker)) {
				continue
			}
			var result vulkanWarmDiskChildReceipt
			if err := json.Unmarshal(bytes.TrimPrefix(line, []byte(vulkanWarmDiskChildMarker)), &result); err != nil {
				t.Fatalf("%s Vulkan child result decode: %v", phase, err)
			}
			return result
		}
		t.Fatalf("%s Vulkan child emitted no structured result: %q", phase, boundedVulkanWarmDiskOutput(output))
		return vulkanWarmDiskChildReceipt{}
	}

	cold := run("cold")
	warm := run("warm")
	cfg := warmCfg()
	expectedKVBytes := uint64(cold.Stable * 3 * cfg.NumLayers * cfg.NumKVHeads * cfg.HeadDim * compute.F32.Bytes())
	expectedKVTransfers := uint64(3 * cfg.NumLayers)
	expectedColdD2HBytes := expectedKVBytes + uint64(cfg.VocabSize*compute.F32.Bytes())

	if cold.Schema != vulkanWarmDiskSchema || warm.Schema != vulkanWarmDiskSchema || cold.Phase != "cold" || warm.Phase != "warm" || cold.PID == warm.PID {
		t.Fatalf("invalid child identity: cold_schema=%q cold_phase=%q cold_pid=%d warm_schema=%q warm_phase=%q warm_pid=%d", cold.Schema, cold.Phase, cold.PID, warm.Schema, warm.Phase, warm.PID)
	}
	if cold.SourceRevision == "" || cold.SourceRevision != warm.SourceRevision || cold.SourceArchiveSHA256 == "" || cold.SourceArchiveSHA256 != warm.SourceArchiveSHA256 || cold.SourceModified || warm.SourceModified || cold.ExecutableSHA256 == "" || cold.ExecutableSHA256 != warm.ExecutableSHA256 {
		t.Fatalf("source/artifact identity differs across restart: cold_rev=%q cold_archive=%q cold_modified=%t cold_exe=%q warm_rev=%q warm_archive=%q warm_modified=%t warm_exe=%q", cold.SourceRevision, cold.SourceArchiveSHA256, cold.SourceModified, cold.ExecutableSHA256, warm.SourceRevision, warm.SourceArchiveSHA256, warm.SourceModified, warm.ExecutableSHA256)
	}
	if cold.ModelIdentity == "" || cold.ModelIdentity != warm.ModelIdentity || cold.Backend != "vulkan" || warm.Backend != "vulkan" || cold.Device == "" || cold.Device != warm.Device || cold.Driver != warm.Driver || cold.Runtime != warm.Runtime {
		t.Fatalf("model/backend identity differs: cold_model=%q warm_model=%q cold_backend=%q warm_backend=%q cold_device=%q warm_device=%q cold_driver=%q warm_driver=%q cold_runtime=%q warm_runtime=%q", cold.ModelIdentity, warm.ModelIdentity, cold.Backend, warm.Backend, cold.Device, warm.Device, cold.Driver, warm.Driver, cold.Runtime, warm.Runtime)
	}
	if cold.DiskOutcome != string(WarmDiskPersisted) || cold.DiskTier != string(WarmDiskTierLocalSSD) || cold.WriteBytes <= 0 || cold.Prefilled != cold.Stable {
		t.Fatalf("cold disk=%s/%s write=%d prefilled=%d stable=%d", cold.DiskOutcome, cold.DiskTier, cold.WriteBytes, cold.Prefilled, cold.Stable)
	}
	if warm.DiskOutcome != string(WarmDiskRestored) || warm.DiskTier != string(WarmDiskTierLocalSSD) || warm.ReadBytes <= 0 || warm.Prefilled != 0 {
		t.Fatalf("warm disk=%s/%s read=%d prefilled=%d", warm.DiskOutcome, warm.DiskTier, warm.ReadBytes, warm.Prefilled)
	}
	if cold.Matched < cold.Stable || warm.Matched < warm.Stable || cold.Stable != warm.Stable {
		t.Fatalf("matched cold=%d/%d warm=%d/%d", cold.Matched, cold.Stable, warm.Matched, warm.Stable)
	}
	if len(cold.Generated) != 4 || len(warm.Generated) != 4 || !eqInts(cold.Generated, warm.Generated) || cold.LogitsDigest == "" || cold.LogitsDigest != warm.LogitsDigest || cold.ContinuationLogitsDigest == "" || cold.ContinuationLogitsDigest != warm.ContinuationLogitsDigest {
		t.Fatalf("cross-process parity: cold_generated=%d warm_generated=%d tokens=%t initial_logits=%t continuation_logits=%t", len(cold.Generated), len(warm.Generated), eqInts(cold.Generated, warm.Generated), cold.LogitsDigest == warm.LogitsDigest, cold.ContinuationLogitsDigest == warm.ContinuationLogitsDigest)
	}
	if got := cold.Transfers; got.D2HBytes != expectedColdD2HBytes || got.D2HCount != expectedKVTransfers+1 || got.ComputeDispatches == 0 || got.DispatchSubmits == 0 {
		t.Fatalf("cold transfers: compute=%d submits=%d h2d_count=%d h2d_bytes=%d d2h_count=%d d2h_bytes=%d want_d2h_count=%d want_d2h_bytes=%d", got.ComputeDispatches, got.DispatchSubmits, got.H2DCount, got.H2DBytes, got.D2HCount, got.D2HBytes, expectedKVTransfers+1, expectedColdD2HBytes)
	}
	if got := warm.Transfers; got.H2DBytes != expectedKVBytes || got.H2DCount != expectedKVTransfers || got.D2HBytes != 0 || got.D2HCount != 0 || got.ComputeDispatches != 0 || got.DispatchSubmits == 0 {
		t.Fatalf("warm transfers: compute=%d submits=%d h2d_count=%d h2d_bytes=%d d2h_count=%d d2h_bytes=%d want_h2d_count=%d want_h2d_bytes=%d", got.ComputeDispatches, got.DispatchSubmits, got.H2DCount, got.H2DBytes, got.D2HCount, got.D2HBytes, expectedKVTransfers, expectedKVBytes)
	}

	source := vulkanWarmDiskSourceIdentity(t)
	executable := vulkanWarmDiskExecutableSHA256(t)
	if source.Revision != cold.SourceRevision || source.ArchiveSHA256 != cold.SourceArchiveSHA256 || source.Modified != cold.SourceModified || executable != cold.ExecutableSHA256 {
		t.Fatalf("parent source/artifact identity differs from children: parent_rev=%q parent_archive=%q parent_modified=%t parent_exe=%q child_rev=%q child_archive=%q child_modified=%t child_exe=%q", source.Revision, source.ArchiveSHA256, source.Modified, executable, cold.SourceRevision, cold.SourceArchiveSHA256, cold.SourceModified, cold.ExecutableSHA256)
	}
	receipt := vulkanWarmDiskRestartReceipt{
		Schema: vulkanWarmDiskSchema, PID: os.Getpid(), CacheMode: "isolated-default",
		SourceRevision: source.Revision, SourceArchiveSHA256: source.ArchiveSHA256, SourceModified: source.Modified, ExecutableSHA256: executable,
		ModelIdentity: cold.ModelIdentity, ExpectedKVTransferBytes: expectedKVBytes,
		ExpectedColdD2HBytes: expectedColdD2HBytes, ExpectedKVTransferCount: expectedKVTransfers,
		Cold: cold, Warm: warm, TokenParity: true, LogitParity: true, ContinuationLogitParity: true,
	}
	payload, err := json.Marshal(receipt)
	if err != nil {
		t.Fatalf("marshal Vulkan restart receipt: %v", err)
	}
	fmt.Printf("%s%s\n", vulkanWarmDiskReceiptMarker, payload)
}

// fak-test:justify why=integration when=changed:internal/agent/**
// fak-test:runtime slow est=15s lane=optin
// Runtime justification: direct execution is skipped; a parent invokes one bounded
// physical phase in a fresh process to make process-local cache reuse impossible.
func TestInKernelWarmPrefixDiskVulkanProcessHelper(t *testing.T) {
	phase := os.Getenv(vulkanWarmDiskChildEnv)
	if phase == "" {
		t.Skip("Vulkan restart subprocess helper")
	}
	if phase != "cold" && phase != "warm" {
		t.Fatalf("unknown Vulkan restart phase %q", phase)
	}
	requireVulkanWarmDiskEnvironment(t)

	backend, ok := compute.Lookup("vulkan")
	if !ok || backend == nil || backend.Name() != "vulkan" || !backend.Caps().DeviceMemory {
		t.Fatal("required device-backed Vulkan backend is unavailable")
	}
	if expected := strings.TrimSpace(os.Getenv("FAK_VULKAN_EXPECT_DEVICE")); !strings.Contains(strings.ToLower(backend.Tier()), strings.ToLower(expected)) {
		t.Fatalf("Vulkan device %q does not match required device %q", backend.Tier(), expected)
	}

	cfg := warmCfg()
	p := NewInKernelPlannerWithConfig(model.NewSynthetic(cfg), loadProbeTok(t), "synthetic-vulkan-process-disk", false, backend, false, InKernelPlannerConfig{
		RequireDeviceExecution: true,
		DenseGPULayers:         cfg.NumLayers,
	})
	p.quant = false
	modelIdentity := vulkanWarmDiskModelIdentity(t)
	p.SetWarmDiskModelDigest(modelIdentity)
	in := warmFixtureInputs()
	spec, err := p.DeriveWarmPrefix("tenant-vulkan-process", "agent-1", in)
	if err != nil {
		t.Fatalf("derive Vulkan warm prefix: %v", err)
	}
	p.SetWarmPrefixInputs(in)

	window, available, err := compute.BeginBackendExecutionObservation(backend)
	if err != nil || !available {
		t.Fatalf("begin Vulkan startup observation: available=%t err=%v", available, err)
	}
	receipt, err := p.WarmPrefix(WithPrefixCacheIdentity(context.Background(), spec.Scope.Tenant, spec.Scope.Agent), spec)
	if err != nil || !receipt.Usable() || receipt.Disk == nil {
		_, _ = window.End()
		t.Fatalf("Vulkan WarmPrefix: err=%v status=%s reason=%s", err, receipt.Status, receipt.Reason)
	}
	observation, err := window.End()
	if err != nil {
		t.Fatalf("end Vulkan startup observation: %v", err)
	}
	if !observation.TransferCountersObserved || !observation.DeviceAllocationObserved || observation.Identity.Backend != "vulkan" {
		t.Fatalf("incomplete Vulkan observation: backend=%q transfers=%t allocation=%t device=%q driver=%q runtime=%q", observation.Identity.Backend, observation.TransferCountersObserved, observation.DeviceAllocationObserved, observation.Identity.Device, observation.Identity.Driver, observation.Identity.Runtime)
	}

	messages := make([]Message, 0, len(in.SystemBlocks)+1)
	messages = append(messages, Message{Role: RoleSystem, Content: string(in.Instructions)})
	for _, block := range in.SystemBlocks {
		messages = append(messages, Message{Role: RoleSystem, Content: string(block)})
	}
	encoded, err := p.EncodePrompt(context.Background(), messages, in.Tools)
	if err != nil {
		t.Fatalf("encode Vulkan warm prompt: %v", err)
	}
	snap, logits, matched, _, _, err := p.scopedTree.LookupSnapshotTieredContext(context.Background(), spec.Scope, encoded.TokenIDs)
	if err != nil || snap == nil || matched != spec.StableTokens || len(logits) == 0 {
		if snap != nil {
			snap.Close()
		}
		t.Fatalf("Vulkan warm readback: snapshot=%t matched=%d stable=%d logits=%d err=%v", snap != nil, matched, spec.StableTokens, len(logits), err)
	}
	snap.Close()
	demand := append(append([]int(nil), encoded.TokenIDs...), synthIDs(cfg.VocabSize, 5, 7719)...)
	result := vulkanWarmDiskChildReceipt{
		Schema: vulkanWarmDiskSchema, Phase: phase, PID: os.Getpid(), ModelIdentity: modelIdentity,
		Backend: observation.Identity.Backend, Device: observation.Identity.Device,
		Driver: observation.Identity.Driver, Runtime: observation.Identity.Runtime,
		DiskOutcome: string(receipt.Disk.Outcome), DiskTier: string(receipt.Disk.Tier),
		WriteBytes: receipt.Disk.WriteBytes, ReadBytes: receipt.Disk.ReadBytes,
		Prefilled: receipt.PrefilledTokens, Stable: spec.StableTokens, LogitsDigest: warmDiskLogitsDigest(logits),
		Transfers: vulkanWarmDiskTransferReceipt{
			ComputeDispatches: observation.Counters.ComputeDispatches, DispatchSubmits: observation.Counters.DispatchSubmits,
			H2DCount: observation.Counters.H2DCount, H2DBytes: observation.Counters.H2DBytes,
			D2HCount: observation.Counters.D2HCount, D2HBytes: observation.Counters.D2HBytes,
		},
	}
	source := vulkanWarmDiskSourceIdentity(t)
	result.SourceRevision, result.SourceArchiveSHA256, result.SourceModified = source.Revision, source.ArchiveSHA256, source.Modified
	result.ExecutableSHA256 = vulkanWarmDiskExecutableSHA256(t)
	_, _, _, result.Matched, _, _, _, _, err = p.generateReusedContextWithBias(
		WithPrefixCacheIdentity(context.Background(), spec.Scope.Tenant, spec.Scope.Agent), demand,
		4, 0, 0, 0, nil, 0, 0, map[int]bool{}, func(id int) bool {
			result.Generated = append(result.Generated, id)
			return false
		})
	if err != nil {
		t.Fatalf("Vulkan restored demand: %v", err)
	}
	if len(result.Generated) != 4 {
		t.Fatalf("Vulkan restored demand generated=%d, want exactly 4", len(result.Generated))
	}
	result.ContinuationLogitsDigest = vulkanWarmDiskContinuationLogitsDigest(t, p, backend, spec, demand, result.Generated)
	payload, err := json.Marshal(result)
	if err != nil {
		t.Fatalf("marshal Vulkan child receipt: %v", err)
	}
	fmt.Printf("%s%s\n", vulkanWarmDiskChildMarker, payload)
}

func requireVulkanWarmDiskEnvironment(t *testing.T) {
	t.Helper()
	if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") != "1" {
		t.Fatal("physical restart witness requires FAK_VULKAN_REQUIRE_DEVICE=1")
	}
	if os.Getenv("FAK_VULKAN_DISPATCH_PROFILE") != "1" {
		t.Fatal("physical restart witness requires FAK_VULKAN_DISPATCH_PROFILE=1 before process startup")
	}
	if strings.TrimSpace(os.Getenv("FAK_VULKAN_EXPECT_DEVICE")) == "" {
		t.Fatal("physical restart witness requires FAK_VULKAN_EXPECT_DEVICE")
	}
}

func vulkanWarmDiskModelIdentity(t *testing.T) string {
	t.Helper()
	wire, err := json.Marshal(struct {
		Kind   string       `json:"kind"`
		Config model.Config `json:"config"`
	}{Kind: "synthetic-vulkan-warm-disk-v1", Config: warmCfg()})
	if err != nil {
		t.Fatalf("marshal Vulkan model identity: %v", err)
	}
	sum := sha256.Sum256(wire)
	return "sha256:" + hex.EncodeToString(sum[:])
}

type vulkanWarmDiskSourceBinding struct {
	Revision      string
	ArchiveSHA256 string
	Modified      bool
}

func vulkanWarmDiskSourceIdentity(t *testing.T) vulkanWarmDiskSourceBinding {
	t.Helper()
	binding := vulkanWarmDiskSourceBinding{
		Revision:      strings.ToLower(strings.TrimSpace(vulkanWarmDiskSourceRevisionStamp)),
		ArchiveSHA256: strings.ToLower(strings.TrimSpace(vulkanWarmDiskSourceArchiveSHA256Stamp)),
	}
	if !vulkanWarmDiskHex(binding.Revision, 40) {
		t.Fatalf("source-bound Vulkan witness requires a 40-hex linker revision stamp, got length=%d", len(binding.Revision))
	}
	if !vulkanWarmDiskHex(binding.ArchiveSHA256, 64) {
		t.Fatalf("source-bound Vulkan witness requires a 64-hex linker archive stamp, got length=%d", len(binding.ArchiveSHA256))
	}
	if info, ok := debug.ReadBuildInfo(); ok {
		for _, setting := range info.Settings {
			switch setting.Key {
			case "vcs.revision":
				if revision := strings.ToLower(strings.TrimSpace(setting.Value)); revision != "" && revision != binding.Revision {
					t.Fatalf("Go build revision disagrees with linker source stamp: build=%q stamp=%q", revision, binding.Revision)
				}
			case "vcs.modified":
				binding.Modified = setting.Value == "true"
			}
		}
	}
	if binding.Modified {
		t.Fatal("source-bound Vulkan witness refuses a vcs.modified=true executable")
	}
	return binding
}

func vulkanWarmDiskHex(value string, length int) bool {
	if len(value) != length {
		return false
	}
	for _, c := range value {
		if (c < '0' || c > '9') && (c < 'a' || c > 'f') {
			return false
		}
	}
	return true
}

func vulkanWarmDiskContinuationLogitsDigest(t *testing.T, p *InKernelPlanner, backend compute.Backend, spec WarmPrefixSpec, prompt, generated []int) string {
	t.Helper()
	forwardedPrefix := make([]int, 0, len(prompt)+len(generated)-1)
	forwardedPrefix = append(forwardedPrefix, prompt...)
	forwardedPrefix = append(forwardedPrefix, generated[:len(generated)-1]...)
	snap, _, matched, _, _, err := p.scopedTree.LookupSnapshotTieredContext(context.Background(), spec.Scope, forwardedPrefix)
	if err != nil || snap == nil || matched != len(forwardedPrefix) {
		if snap != nil {
			snap.Close()
		}
		t.Fatalf("continuation snapshot readback: matched=%d want=%d snapshot=%t err=%v", matched, len(forwardedPrefix), snap != nil, err)
	}
	session := p.m.NewBackendSession(backend)
	defer session.Close()
	p.configureNativeSession(session)
	if err := snap.Restore(session); err != nil {
		snap.Close()
		t.Fatalf("restore continuation snapshot: %v", err)
	}
	snap.Close()
	logits := session.Step(generated[len(generated)-1])
	if len(logits) != warmCfg().VocabSize {
		t.Fatalf("continuation logits=%d, want=%d", len(logits), warmCfg().VocabSize)
	}
	return warmDiskLogitsDigest(logits)
}

func boundedVulkanWarmDiskOutput(output []byte) string {
	const maxBytes = 4096
	if len(output) <= maxBytes {
		return string(output)
	}
	return "...[truncated]..." + string(output[len(output)-maxBytes:])
}

func vulkanWarmDiskExecutableSHA256(t *testing.T) string {
	t.Helper()
	path, err := os.Executable()
	if err != nil {
		t.Fatalf("resolve test executable: %v", err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open test executable: %v", err)
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		t.Fatalf("hash test executable: %v", err)
	}
	return "sha256:" + hex.EncodeToString(h.Sum(nil))
}
