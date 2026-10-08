package amdgpu

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func validStrixSSDRestartWitnessForTest() *StrixSSDRestartWitness {
	plainHash := strings.TrimPrefix(testHash, "sha256:")
	child := func(phase, outcome string, pid int) StrixSSDRestartChild {
		c := StrixSSDRestartChild{
			Schema: "fak-vulkan-warm-disk-restart/1", Phase: phase, PID: pid,
			SourceRevision: testTip, SourceArchiveSHA256: strings.TrimPrefix(testHash, "sha256:"), ExecutableSHA256: strings.TrimPrefix(testHash, "sha256:"),
			ModelIdentity: testHash, Backend: "vulkan", Device: "AMD Radeon 8060S Graphics Strix Halo gfx1151", Driver: "RADV", Runtime: "Vulkan 1.3",
			DiskOutcome: outcome, DiskTier: "local_ssd_l3", Stable: 8, Matched: 8, Generated: []int{11, 12, 13, 14},
			LogitsDigest: plainHash, ContinuationLogitsDigest: plainHash,
		}
		if phase == "cold" {
			c.WriteBytes, c.Prefilled = 4096, 8
			c.Transfers = StrixSSDRestartTransfer{ComputeDispatches: 4, DispatchSubmits: 4, D2HCount: 3, D2HBytes: 2048}
		} else {
			c.ReadBytes = 4096
			c.Transfers = StrixSSDRestartTransfer{H2DCount: 2, H2DBytes: 1024, DispatchSubmits: 1}
		}
		return c
	}
	return &StrixSSDRestartWitness{
		Schema: "fak-vulkan-warm-disk-restart/1", PID: 100, CacheMode: "isolated-default",
		SourceRevision: testTip, SourceArchiveSHA256: plainHash, ExecutableSHA256: plainHash, ModelIdentity: testHash,
		ExpectedKVTransferBytes: 1024, ExpectedColdD2HBytes: 2048, ExpectedKVTransferCount: 2,
		Cold: child("cold", "persisted", 101), Warm: child("warm", "restored", 102),
		TokenParity: true, LogitParity: true, ContinuationLogitParity: true,
	}
}

func validStrixSSDRestartReceiptForTest(t *testing.T) *StrixSSDRestartReceipt {
	t.Helper()
	exit := 0
	r := &StrixSSDRestartReceipt{
		Schema: StrixSSDRestartSchema, Timestamp: time.Now().UTC().Format(time.RFC3339), Verdict: "PASS", Hardware: true, EvidenceRung: "HW_WITNESSED", Verified: true,
		Target:     StrixTarget{Mode: "ssh", Host: "strix1", Reachable: true, GPUName: "AMD Radeon 8060S Graphics Strix Halo gfx1151", TargetISA: "gfx1151"},
		Provenance: StrixProvenance{GitTip: testTip, Command: "fak-dev amd-strix-validate --native-ssd-restart", GeneratedBy: "fak-dev amd-strix-validate", Transport: "ssh", SourceArchiveSHA256: testHash, BinarySHA256: testHash, ShaderBundleSHA256: testHash, BuildCommandSHA256: testHash, EngineIdentity: "fak-native/vulkan", CleanupObserved: true},
		Execution:  StrixExecutionEvidence{SourceArchiveSHA256: testHash, BinarySHA256: testHash, ShaderBundleSHA256: testHash, CommandSHA256: testHash, DeviceIdentity: "AMD Radeon 8060S Graphics Strix Halo gfx1151|gfx1151", EngineIdentity: "fak-native/vulkan", ArtifactRehashed: true, DeviceTimeoutMS: 60000, LeasePathSHA256: testHash, AdmissionWaitMS: 30000, Acquired: true, Released: true, AcquireOrdinal: 1, ReleaseOrdinal: 2, ExitCode: &exit, RawOutputSHA256: testHash, RawOutputBytes: 1},
		Device:     &StrixVulkanDeviceObservation{Name: "AMD Radeon 8060S Graphics Strix Halo gfx1151", DeviceType: "PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU", VendorID: "0x1002", DriverID: "DRIVER_ID_MESA_RADV", DriverName: "radv", DriverInfo: "Mesa RADV", ISA: "gfx1151", SummarySHA256: testHash, SummaryBytes: 1},
		Filesystem: &StrixSSDFileSystemObservation{Workspace: "/tmp/fak-strix-validation.fixture", MountTarget: "/", Source: "/dev/nvme0n1p1", FSType: "ext4", MajorMinor: "259:1", SourcePath: "/dev/nvme0n1p1", SourceType: "part", ParentDisk: "/dev/nvme0n1", ParentType: "disk", Transport: "nvme", RotationObserved: true, ObservationSHA256: testHash},
		Witness:    validStrixSSDRestartWitnessForTest(),
	}
	r.Provenance.ExecutionManifestSHA256 = strixSSDRestartExecutionManifestDigest(r)
	r.Digest, _ = r.ComputeDigest()
	return r
}

// fak-test:runtime fast est=20ms lane=default
func TestStrixSSDRestartWitnessRequiresCompleteRestartEvidence(t *testing.T) {
	if err := validStrixSSDRestartReceiptForTest(t).Validate(); err != nil {
		t.Fatalf("complete SSD restart receipt rejected: %v", err)
	}
	tests := []struct {
		name   string
		mutate func(*StrixSSDRestartReceipt)
	}{
		{"physical field alone", func(r *StrixSSDRestartReceipt) { r.Witness = nil }},
		{"admission not acquired", func(r *StrixSSDRestartReceipt) { r.Execution.Acquired = false }},
		{"admission not released", func(r *StrixSSDRestartReceipt) { r.Execution.Released = false }},
		{"cleanup absent", func(r *StrixSSDRestartReceipt) { r.Provenance.CleanupObserved = false }},
		{"archive mismatch", func(r *StrixSSDRestartReceipt) { r.Witness.SourceArchiveSHA256 = strings.Repeat("b", 64) }},
		{"binary mismatch", func(r *StrixSSDRestartReceipt) { r.Witness.Warm.ExecutableSHA256 = strings.Repeat("b", 64) }},
		{"same process", func(r *StrixSSDRestartReceipt) { r.Witness.Warm.PID = r.Witness.Cold.PID }},
		{"device absent", func(r *StrixSSDRestartReceipt) { r.Witness.Warm.Device = "" }},
		{"wrong target device", func(r *StrixSSDRestartReceipt) { r.Witness.Warm.Device = "AMD Radeon other" }},
		{"zero cold transfer", func(r *StrixSSDRestartReceipt) { r.Witness.Cold.Transfers.D2HBytes = 0 }},
		{"zero warm transfer", func(r *StrixSSDRestartReceipt) { r.Witness.Warm.Transfers.H2DBytes = 0 }},
		{"warm prefill", func(r *StrixSSDRestartReceipt) { r.Witness.Warm.Prefilled = 1 }},
		{"wrong disk tier", func(r *StrixSSDRestartReceipt) { r.Witness.Warm.DiskTier = "memory" }},
		{"missing logits", func(r *StrixSSDRestartReceipt) { r.Witness.Warm.LogitsDigest = "" }},
		{"prefixed logits digest", func(r *StrixSSDRestartReceipt) { r.Witness.Warm.LogitsDigest = testHash }},
		{"unshaped model identity", func(r *StrixSSDRestartReceipt) { r.Witness.ModelIdentity = "fixture-model" }},
		{"missing generated tokens", func(r *StrixSSDRestartReceipt) { r.Witness.Warm.Generated = nil }},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			r := validStrixSSDRestartReceiptForTest(t)
			tc.mutate(r)
			r.Provenance.ExecutionManifestSHA256 = strixSSDRestartExecutionManifestDigest(r)
			r.Digest, _ = r.ComputeDigest()
			if err := r.Validate(); err == nil {
				t.Fatal("incomplete or contradictory SSD restart evidence validated")
			}
		})
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestStrixSSDRestartMarkerIsSingleStrictJSON(t *testing.T) {
	w := validStrixSSDRestartWitnessForTest()
	raw, err := json.Marshal(w)
	if err != nil {
		t.Fatal(err)
	}
	marker := append([]byte(ssdRestartMarker), raw...)
	parsed, err := parseStrixSSDRestartWitness(append(marker, '\n'))
	if err != nil || parsed.PID != w.PID {
		t.Fatalf("strict marker rejected: parsed=%+v err=%v", parsed, err)
	}
	for _, out := range [][]byte{
		[]byte("ordinary output only\n"),
		append(append(append([]byte{}, marker...), '\n'), append(marker, '\n')...),
		append(append([]byte{}, marker...), []byte(" {}\n")...),
		append([]byte(ssdRestartMarker), bytes.Replace(raw, []byte(`"pid":100`), []byte(`"pid":100,"unknown":true`), 1)...),
		append([]byte(ssdRestartMarker), bytes.Replace(raw, []byte(`"pid":100`), []byte(`"pid":100,"pid":101`), 1)...),
	} {
		if _, err := parseStrixSSDRestartWitness(out); err == nil {
			t.Fatal("non-single or non-strict SSD marker accepted")
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestStrixSSDRestartCommandIsExclusiveDirectExecutable(t *testing.T) {
	target := &StrixTarget{GPUName: "AMD Radeon 8060S Graphics", TargetISA: "gfx1151"}
	binding := SourceBinding{WorkDir: "/tmp/fak-strix-validation.fixture", BinarySHA256: testHash, ShaderBundleSHA256: testHash, AdmissionWait: 7 * time.Second}
	command := buildStrixSSDRestartAdmissionCommand(target, binding)
	for _, required := range []string{"flock -w 7 -x", "FAK_STRIX_ADMISSION_ACQUIRED=1", "FAK_STRIX_ADMISSION_RELEASED=1", "TMPDIR=", "/tmp/fak-strix-validation.fixture/tmp", "build/agent.test", "-test.run=^TestInKernelWarmPrefixDiskAcrossProcessesVulkan$", "-test.count=1", "-fak-vulkan-warm-disk-restart=true"} {
		if !strings.Contains(command, required) {
			t.Fatalf("SSD restart command omitted %q", required)
		}
	}
	for _, forbidden := range []string{"go test", "-subkernels", "-ablate", "TestVulkanMatMul", "TestRADV"} {
		if strings.Contains(command, forbidden) {
			t.Fatalf("SSD restart command combined exclusive witness with %q", forbidden)
		}
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestStrixSSDFileSystemObservationRequiresNVMeBlockAncestry(t *testing.T) {
	valid := StrixSSDFileSystemObservation{
		Workspace: "/tmp/fak-strix-validation.fixture", MountTarget: "/", Source: "/dev/nvme0n1p1", FSType: "ext4", MajorMinor: "259:1",
		SourcePath: "/dev/nvme0n1p1", SourceType: "part", ParentDisk: "/dev/nvme0n1", ParentType: "disk", Transport: "nvme",
		RotationObserved: true, ObservationSHA256: testHash,
	}
	if err := valid.Validate(); err != nil {
		t.Fatalf("measured NVMe filesystem rejected: %v", err)
	}
	tests := []struct {
		name   string
		mutate func(*StrixSSDFileSystemObservation)
	}{
		{"tmpfs", func(g *StrixSSDFileSystemObservation) { g.FSType = "tmpfs" }},
		{"ramfs", func(g *StrixSSDFileSystemObservation) { g.FSType = "ramfs" }},
		{"overlay", func(g *StrixSSDFileSystemObservation) { g.FSType = "overlay" }},
		{"rotational", func(g *StrixSSDFileSystemObservation) { g.Rotational = true }},
		{"rotation unobserved", func(g *StrixSSDFileSystemObservation) { g.RotationObserved = false }},
		{"non NVMe transport", func(g *StrixSSDFileSystemObservation) { g.Transport = "sata" }},
		{"non NVMe parent", func(g *StrixSSDFileSystemObservation) { g.ParentDisk = "/dev/sda" }},
		{"non disk parent", func(g *StrixSSDFileSystemObservation) { g.ParentType = "part" }},
		{"unbound workspace", func(g *StrixSSDFileSystemObservation) { g.MountTarget = "/var/lib/fak" }},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := valid
			tc.mutate(&got)
			if err := got.Validate(); err == nil {
				t.Fatalf("unsafe filesystem observation accepted: %+v", got)
			}
		})
	}
}

func validStrixSSDFileSystemOutputForTest() string {
	return "FAK_STRIX_FS_MOUNT=TARGET=\"/\" SOURCE=\"/dev/nvme0n1p1\" FSTYPE=\"ext4\" MAJ:MIN=\"259:1\"\n" +
		"FAK_STRIX_FS_BLOCK=PATH=\"/dev/nvme0n1p1\" KNAME=\"nvme0n1p1\" TYPE=\"part\" ROTA=\"0\" PKNAME=\"nvme0n1\" TRAN=\"nvme\"\n" +
		"FAK_STRIX_FS_PARENT=PATH=\"/dev/nvme0n1\" KNAME=\"nvme0n1\" TYPE=\"disk\" ROTA=\"0\" PKNAME=\"\" TRAN=\"nvme\"\n" +
		"FAK_STRIX_FS_EXIT=0,0,0\n"
}

// fak-test:runtime fast est=20ms lane=default
func TestStrixSSDFileSystemParserBindsMeasuredNVMeAncestry(t *testing.T) {
	const workspace = "/tmp/fak-strix-validation.fixture"
	out := validStrixSSDFileSystemOutputForTest()
	got, err := ParseStrixSSDFileSystemObservation(workspace, []byte(out))
	if err != nil {
		t.Fatal(err)
	}
	if got.Workspace != workspace || got.MountTarget != "/" || got.Source != "/dev/nvme0n1p1" || got.FSType != "ext4" || got.MajorMinor != "259:1" || got.SourcePath != "/dev/nvme0n1p1" || got.SourceType != "part" || got.ParentDisk != "/dev/nvme0n1" || got.ParentType != "disk" || got.Transport != "nvme" || got.Rotational || !got.RotationObserved || got.ObservationSHA256 == "" {
		t.Fatalf("parsed filesystem observation changed: %+v", got)
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestStrixSSDFileSystemParserRefusesUnsafeOrAmbiguousEvidence(t *testing.T) {
	valid := validStrixSSDFileSystemOutputForTest()
	tests := []struct {
		name string
		out  string
	}{
		{"tmpfs", strings.Replace(valid, `FSTYPE="ext4"`, `FSTYPE="tmpfs"`, 1)},
		{"ramfs", strings.Replace(valid, `FSTYPE="ext4"`, `FSTYPE="ramfs"`, 1)},
		{"overlay", strings.Replace(valid, `FSTYPE="ext4"`, `FSTYPE="overlay"`, 1)},
		{"rotational", strings.Replace(valid, `TYPE="disk" ROTA="0"`, `TYPE="disk" ROTA="1"`, 1)},
		{"non NVMe transport", strings.Replace(valid, `TYPE="disk" ROTA="0" PKNAME="" TRAN="nvme"`, `TYPE="disk" ROTA="0" PKNAME="" TRAN="sata"`, 1)},
		{"non NVMe parent", strings.Replace(valid, `PATH="/dev/nvme0n1" KNAME="nvme0n1" TYPE="disk"`, `PATH="/dev/sda" KNAME="sda" TYPE="disk"`, 1)},
		{"duplicate marker", valid + `FAK_STRIX_FS_PARENT=PATH="/dev/nvme1n1" TYPE="disk" ROTA="0" TRAN="nvme"` + "\n"},
		{"duplicate field", strings.Replace(valid, `TYPE="disk"`, `TYPE="disk" TYPE="disk"`, 1)},
		{"missing parent", strings.Replace(valid, `FAK_STRIX_FS_PARENT=`, `IGNORED_PARENT=`, 1)},
		{"nonzero observation", strings.Replace(valid, `FAK_STRIX_FS_EXIT=0,0,0`, `FAK_STRIX_FS_EXIT=0,1,0`, 1)},
		{"oversized", valid + strings.Repeat("x", 8<<10)},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got, err := ParseStrixSSDFileSystemObservation("/tmp/fak-strix-validation.fixture", []byte(tc.out)); err == nil {
				t.Fatalf("unsafe filesystem payload accepted: %+v", got)
			}
		})
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestStrixSSDVulkanSummaryParsesPhysicalRADVDevice(t *testing.T) {
	summary := []byte("Vulkan Instance Version: 1.3.280\n\nGPU0:\n\tdeviceName = AMD Radeon 8060S Graphics Strix Halo gfx1151\n\tdeviceType = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU\n\tvendorID = 0x1002\n\tdriverID = DRIVER_ID_MESA_RADV\n\tdriverName = radv\n\tdriverInfo = Mesa 26.1 RADV\n")
	got, err := ParseStrixVulkanInfoSummary(summary)
	if err != nil {
		t.Fatal(err)
	}
	if got.Name != "AMD Radeon 8060S Graphics Strix Halo gfx1151" || got.DeviceType != "PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU" || got.VendorID != "0x1002" || got.DriverID != "DRIVER_ID_MESA_RADV" || got.DriverName != "radv" || got.DriverInfo != "Mesa 26.1 RADV" || got.ISA != "gfx1151" || got.SummarySHA256 != digestBytes(summary) || got.SummaryBytes != len(summary) {
		t.Fatalf("parsed Vulkan device observation changed: %+v", got)
	}
	if err := got.Validate(); err != nil {
		t.Fatalf("parsed physical RADV Strix device rejected: %v", err)
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestStrixSSDVulkanSummaryRefusesAmbiguousOrSoftwareDevices(t *testing.T) {
	valid := "GPU0:\n deviceName = AMD Radeon 8060S Strix Halo gfx1151\n deviceType = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU\n vendorID = 0x1002\n driverID = DRIVER_ID_MESA_RADV\n driverName = radv\n driverInfo = Mesa RADV\n"
	tests := []struct {
		name    string
		summary string
	}{
		{"empty", ""},
		{"cpu", strings.Replace(valid, "PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU", "PHYSICAL_DEVICE_TYPE_CPU", 1)},
		{"llvmpipe", strings.Replace(valid, "AMD Radeon 8060S Strix Halo gfx1151", "llvmpipe gfx1151", 1)},
		{"lavapipe", strings.Replace(valid, "Mesa RADV", "Mesa lavapipe", 1)},
		{"software", strings.Replace(valid, "Mesa RADV", "software RADV", 1)},
		{"non AMD vendor", strings.Replace(valid, "0x1002", "0x8086", 1)},
		{"missing driver info", strings.Replace(valid, " driverInfo = Mesa RADV\n", "", 1)},
		{"missing device name", strings.Replace(valid, " deviceName = AMD Radeon 8060S Strix Halo gfx1151\n", "", 1)},
		{"duplicate field", strings.Replace(valid, " deviceName =", " deviceName = AMD duplicate\n deviceName =", 1)},
		{"multiple devices", valid + strings.Replace(valid, "GPU0:", "GPU1:", 1)},
		{"oversized", "GPU0:\n" + strings.Repeat("x", (32<<10)+1)},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got, err := ParseStrixVulkanInfoSummary([]byte(tc.summary)); err == nil {
				t.Fatalf("unsafe Vulkan summary accepted: %+v", got)
			}
		})
	}
}
