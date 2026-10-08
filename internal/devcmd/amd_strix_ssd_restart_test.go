package devcmd

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/amdgpu"
)

func validCLIStrixSSDRestartReceipt(t *testing.T, opts amdgpu.StrixSSDRestartOpts) *amdgpu.StrixSSDRestartReceipt {
	t.Helper()
	hash := testSHA256([]byte("fixture executable"))
	plainHash := strings.TrimPrefix(hash, "sha256:")
	exit := 0
	child := func(phase, outcome string, pid int) amdgpu.StrixSSDRestartChild {
		c := amdgpu.StrixSSDRestartChild{
			Schema: "fak-vulkan-warm-disk-restart/1", Phase: phase, PID: pid, SourceRevision: opts.GitTip,
			SourceArchiveSHA256: strings.TrimPrefix(opts.SourceArchiveSHA256, "sha256:"), ExecutableSHA256: strings.TrimPrefix(hash, "sha256:"),
			ModelIdentity: hash, Backend: "vulkan", Device: "AMD Radeon 8060S Graphics Strix Halo gfx1151", Driver: "RADV", Runtime: "Vulkan 1.3",
			DiskOutcome: outcome, DiskTier: "local_ssd_l3", Stable: 8, Matched: 8, Generated: []int{1, 2, 3, 4}, LogitsDigest: plainHash, ContinuationLogitsDigest: plainHash,
		}
		if phase == "cold" {
			c.WriteBytes, c.Prefilled = 4096, 8
			c.Transfers = amdgpu.StrixSSDRestartTransfer{ComputeDispatches: 3, DispatchSubmits: 3, D2HCount: 3, D2HBytes: 2048}
		} else {
			c.ReadBytes = 4096
			c.Transfers = amdgpu.StrixSSDRestartTransfer{H2DCount: 2, H2DBytes: 1024, DispatchSubmits: 1}
		}
		return c
	}
	target := amdgpu.StrixTarget{Mode: "ssh", Host: opts.Host, Reachable: true, GPUName: "AMD Radeon 8060S Graphics Strix Halo gfx1151", TargetISA: "gfx1151"}
	r := &amdgpu.StrixSSDRestartReceipt{
		Schema: amdgpu.StrixSSDRestartSchema, Timestamp: time.Now().UTC().Format(time.RFC3339), Verdict: "PASS", Hardware: true, EvidenceRung: "HW_WITNESSED", Verified: true, Target: target,
		Provenance: amdgpu.StrixProvenance{GitTip: opts.GitTip, Command: opts.Command, GeneratedBy: "fak-dev amd-strix-validate", Transport: "ssh", SourceArchiveSHA256: opts.SourceArchiveSHA256, BinarySHA256: hash, ShaderBundleSHA256: hash, BuildCommandSHA256: hash, EngineIdentity: "fak-native/vulkan", CleanupObserved: true},
		Execution:  amdgpu.StrixExecutionEvidence{SourceArchiveSHA256: opts.SourceArchiveSHA256, BinarySHA256: hash, ShaderBundleSHA256: hash, CommandSHA256: hash, DeviceIdentity: target.GPUName + "|" + target.TargetISA, EngineIdentity: "fak-native/vulkan", ArtifactRehashed: true, DeviceTimeoutMS: 60000, LeasePathSHA256: hash, AdmissionWaitMS: 7000, Acquired: true, Released: true, AcquireOrdinal: 1, ReleaseOrdinal: 2, ExitCode: &exit, RawOutputSHA256: hash, RawOutputBytes: 1},
		Device:     &amdgpu.StrixVulkanDeviceObservation{Name: target.GPUName, DeviceType: "PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU", VendorID: "0x1002", DriverID: "DRIVER_ID_MESA_RADV", DriverName: "radv", DriverInfo: "Mesa RADV", ISA: "gfx1151", SummarySHA256: hash, SummaryBytes: 1},
		Filesystem: &amdgpu.StrixSSDFileSystemObservation{Workspace: "/tmp/fak-strix-validation.fixture", MountTarget: "/", Source: "/dev/nvme0n1p1", FSType: "ext4", MajorMinor: "259:1", SourcePath: "/dev/nvme0n1p1", SourceType: "part", ParentDisk: "/dev/nvme0n1", ParentType: "disk", Transport: "nvme", RotationObserved: true, ObservationSHA256: hash},
		Witness: &amdgpu.StrixSSDRestartWitness{
			Schema: "fak-vulkan-warm-disk-restart/1", PID: 100, CacheMode: "isolated-default", SourceRevision: opts.GitTip,
			SourceArchiveSHA256: strings.TrimPrefix(opts.SourceArchiveSHA256, "sha256:"), ExecutableSHA256: plainHash, ModelIdentity: hash,
			ExpectedKVTransferBytes: 1024, ExpectedColdD2HBytes: 2048, ExpectedKVTransferCount: 2,
			Cold: child("cold", "persisted", 101), Warm: child("warm", "restored", 102), TokenParity: true, LogitParity: true, ContinuationLogitParity: true,
		},
	}
	manifest := struct {
		GitTip          string                                `json:"git_tip"`
		Source          string                                `json:"source_archive_sha256"`
		Binary          string                                `json:"binary_sha256"`
		Shader          string                                `json:"shader_bundle_sha256"`
		Build           string                                `json:"build_command_sha256"`
		Command         string                                `json:"command_sha256"`
		Lease           string                                `json:"lease_path_sha256"`
		Device          string                                `json:"device_identity"`
		Acquired        bool                                  `json:"acquired"`
		Released        bool                                  `json:"released"`
		CleanupObserved bool                                  `json:"cleanup_observed"`
		ObservedDevice  *amdgpu.StrixVulkanDeviceObservation  `json:"observed_device"`
		Filesystem      *amdgpu.StrixSSDFileSystemObservation `json:"filesystem"`
		Witness         *amdgpu.StrixSSDRestartWitness        `json:"witness"`
	}{r.Provenance.GitTip, r.Provenance.SourceArchiveSHA256, r.Provenance.BinarySHA256, r.Provenance.ShaderBundleSHA256, r.Provenance.BuildCommandSHA256, r.Execution.CommandSHA256, r.Execution.LeasePathSHA256, r.Execution.DeviceIdentity, r.Execution.Acquired, r.Execution.Released, r.Provenance.CleanupObserved, r.Device, r.Filesystem, r.Witness}
	manifestRaw, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	r.Provenance.ExecutionManifestSHA256 = testSHA256(manifestRaw)
	r.Digest, _ = r.ComputeDigest()
	return r
}

// fak-test:runtime fast est=50ms lane=default
func TestRunAMDStrixValidateNativeSSDRestartBindsCommittedCandidate(t *testing.T) {
	archive := []byte("committed SSD restart candidate")
	_, digest := stubAMDStrixCandidate(t, archive)
	original := runStrixSSDRestartFn
	t.Cleanup(func() { runStrixSSDRestartFn = original })
	var captured amdgpu.StrixSSDRestartOpts
	runStrixSSDRestartFn = func(_ context.Context, opts amdgpu.StrixSSDRestartOpts) (*amdgpu.StrixSSDRestartReceipt, error) {
		captured = opts
		return validCLIStrixSSDRestartReceipt(t, opts), nil
	}
	var stdout, stderr bytes.Buffer
	args := []string{"--native-ssd-restart", "--committed-only", "--host=strix1", "--json", "--timeout=20", "--admission-timeout=7"}
	if code := RunAMDStrixValidate(&stdout, &stderr, args); code != 0 {
		t.Fatalf("native SSD restart exit=%d stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	if !bytes.Equal(captured.CandidateArchive, archive) || captured.SourceArchiveSHA256 != digest || captured.GitTip != testStrixTip || !captured.RequireSourceBinding {
		t.Fatal("native SSD restart runner did not receive the committed source/archive tuple")
	}
	if captured.Host != "strix1" || captured.Timeout != 20*time.Second || captured.AdmissionTimeout != 7*time.Second || !strings.Contains(captured.Command, "--native-ssd-restart") {
		t.Fatalf("native SSD restart runner options changed: %+v", captured)
	}
	var receipt amdgpu.StrixSSDRestartReceipt
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil || receipt.Validate() != nil {
		t.Fatalf("CLI did not emit a valid SSD restart receipt: decode=%v validate=%v", err, receipt.Validate())
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestRunAMDStrixValidateNativeSSDRestartIsExclusive(t *testing.T) {
	original := runStrixSSDRestartFn
	t.Cleanup(func() { runStrixSSDRestartFn = original })
	calls := 0
	runStrixSSDRestartFn = func(context.Context, amdgpu.StrixSSDRestartOpts) (*amdgpu.StrixSSDRestartReceipt, error) {
		calls++
		return nil, nil
	}
	for _, args := range [][]string{
		{"--native-ssd-restart", "--subkernels=argmax", "--ablate=none"},
		{"--native-ssd-restart", "--subkernels=none", "--ablate=target"},
		{"--native-ssd-restart", "--subkernels=all"},
		{"--native-ssd-restart", "--ablate=all"},
	} {
		var stdout, stderr bytes.Buffer
		if code := RunAMDStrixValidate(&stdout, &stderr, args); code != 1 {
			t.Fatalf("combined native SSD restart args=%v exit=%d", args, code)
		}
	}
	if calls != 0 {
		t.Fatalf("exclusive selection refusal reached SSD restart runner %d times", calls)
	}
}
