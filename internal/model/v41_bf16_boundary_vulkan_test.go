//go:build v41_bf16_boundary_witness && vulkan && (windows || linux) && cgo

package model

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// A reviewed record binds configured artifact files to the candidate and build
// commands under immutable-artifact execution. It is not a runtime attestation
// of module bytes already loaded into Vulkan before this test started.
// The expected record digest and revision come from the external review, not
// from fields a test hashes and then trusts in its own freshly made document.
type v41BF16BoundaryBuildRecord struct {
	Schema         string   `json:"schema"`
	SourceRevision string   `json:"source_revision"`
	NativeDefine   string   `json:"native_define"`
	GoTag          string   `json:"go_tag"`
	Commands       []string `json:"commands"`
	Files          map[string]struct {
		Path   string `json:"path"`
		SHA256 string `json:"sha256"`
	} `json:"files"`
}

func v41BF16BoundaryProvenance(t *testing.T) {
	t.Helper()
	recordPath := os.Getenv("FAK_V41_BF16_BOUNDARY_WITNESS_PROVENANCE")
	expected := os.Getenv("FAK_V41_BF16_BOUNDARY_WITNESS_PROVENANCE_SHA256")
	revision := os.Getenv("FAK_V41_BF16_BOUNDARY_WITNESS_REVISION")
	if len(revision) != 40 || len(expected) != 64 {
		t.Fatal("reviewed full source revision and record digest required")
	}
	if _, err := hex.DecodeString(revision); err != nil {
		t.Fatal(err)
	}
	if _, err := hex.DecodeString(expected); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(recordPath)
	if err != nil {
		t.Fatal(err)
	}
	hash := sha256.Sum256(raw)
	if hex.EncodeToString(hash[:]) != expected {
		t.Fatal("build record differs from reviewed digest")
	}
	var record v41BF16BoundaryBuildRecord
	if err := json.Unmarshal(raw, &record); err != nil {
		t.Fatal(err)
	}
	if record.Schema != "fak.v41-bf16-boundary-witness.v1" || record.SourceRevision != revision || record.NativeDefine != "FAK_V41_BF16_BOUNDARY_WITNESS" || record.GoTag != "v41_bf16_boundary_witness" || len(record.Commands) < 3 {
		t.Fatal("record does not describe this isolated source/build profile")
	}
	for _, command := range record.Commands {
		if command == "" {
			t.Fatal("empty build command")
		}
	}
	binary, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	bundle := os.Getenv("FAK_VULKAN_SPIRV")
	if bundle == "" {
		t.Fatal("explicit reviewed shader bundle required")
	}
	// Reviewed CGO_LDFLAGS may point at a scratch directory. Bind that explicit
	// archive choice, never force generated artifacts into the source checkout.
	archive := os.Getenv("FAK_V41_BF16_BOUNDARY_WITNESS_ARCHIVE")
	if !filepath.IsAbs(archive) || filepath.Clean(record.Files["native_archive"].Path) != filepath.Clean(archive) {
		t.Fatal("record does not name the explicitly reviewed witness archive")
	}
	for _, kind := range []string{"native_archive", "shader_module", "test_binary", "shader_source", "native_source", "native_header", "go_adapter"} {
		file, ok := record.Files[kind]
		if !ok || !filepath.IsAbs(file.Path) || len(file.SHA256) != 64 {
			t.Fatalf("missing reviewed %s", kind)
		}
		if kind == "native_archive" && filepath.Base(file.Path) != "libfakvulkan_v41_bf16_boundary_witness.a" {
			t.Fatal("ordinary native archive is not BF16 witness provenance")
		}
		if kind == "shader_module" && filepath.Clean(file.Path) != filepath.Clean(filepath.Join(bundle, "v41_bf16_boundary.spv")) {
			t.Fatal("record does not name actual loaded module")
		}
		if kind == "test_binary" && filepath.Clean(file.Path) != filepath.Clean(binary) {
			t.Fatal("record does not name actual running test binary")
		}
		f, err := os.Open(file.Path)
		if err != nil {
			t.Fatal(err)
		}
		digest := sha256.New()
		n, readErr := io.Copy(digest, f)
		closeErr := f.Close()
		if readErr != nil || closeErr != nil || n == 0 || hex.EncodeToString(digest.Sum(nil)) != file.SHA256 {
			t.Fatalf("reviewed %s bytes changed", kind)
		}
	}
	t.Logf("reviewed BF16 witness source=%s record_sha256=%s", revision, expected)
}

// Deferred physical witness source. The tag selects a distinct archive whose
// witness-only symbols are absent in production, so wrong-archive linking fails.
// The reviewed record must cover the freshly built test binary, source, shim,
// header and optional module. It does not promote a trusted bundle registry.
// Bound: one raw 26-word dispatch, one 4096-value model-seam dispatch, and a
// synthetic two-token prefill plus one decode. There is no full-checkpoint claim.
// fak-test:runtime integration est=30s lane=optin
func TestV41BF16BoundaryVulkan(t *testing.T) {
	if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") != "1" || os.Getenv("FAK_VULKAN_EXPECT_DEVICE") == "" || os.Getenv("FAK_VULKAN_DISPATCH_PROFILE") != "1" {
		t.Fatal("BF16 witness requires explicit physical identity and profiling before startup")
	}
	v41BF16BoundaryProvenance(t)
	be, ok := compute.Lookup("vulkan")
	if !ok || be == nil || !be.Caps().DeviceMemory {
		t.Fatal("required Vulkan device unavailable")
	}
	physical, ok := be.(interface{ VulkanPhysicalDeviceType() uint32 })
	if !ok || (physical.VulkanPhysicalDeviceType() != 1 && physical.VulkanPhysicalDeviceType() != 2) {
		t.Fatal("physical integrated/discrete GPU required")
	}
	operation, ok := be.(compute.V41BF16BoundaryBackend)
	if !ok || !operation.V41BF16BoundaryAvailable() {
		t.Fatal("witness-tagged module was not admitted")
	}
	snapshot := func() compute.BackendExecutionSnapshot {
		// Free parks rows in the backend reuse pool; drain it so live bytes count owned allocations only.
		if pool, ok := be.(interface{ Trim() }); ok {
			pool.Trim()
		}
		value, available, err := compute.CaptureBackendExecutionSnapshot(be)
		if err != nil || !available || value.Identity.Backend != "vulkan" || value.Identity.Device != os.Getenv("FAK_VULKAN_EXPECT_DEVICE") || !value.TransferCountersObserved || !value.DeviceAllocationObserved {
			t.Fatalf("physical counters/identity unavailable: %+v %v", value, err)
		}
		return value
	}
	// Raw primitive matches host bit semantics even for NaN patterns that wrap.
	words := []uint32{0, 0x80000000, 1, 0x80000001, 0x7fff, 0x8000, 0x8001, 0x18000, 0x818000, 0x3f807fff, 0x3f808000, 0x3f808001, 0x3f818000, 0xbf808000, 0xbf818000, 0x7f7f7fff, 0x7f7f8000, 0xff7f8000, 0x7f800000, 0xff800000, 0x7f800001, 0x7fc00000, 0x7fffffff, 0xff800001, 0xffc01234, 0xffffffff}
	values := make([]float32, len(words))
	for i, word := range words {
		values[i] = math.Float32frombits(word)
	}
	input := be.Upload(compute.NewF32(compute.Default(), []int{len(words)}, values), compute.F32)
	defer be.Free(input)
	output, err := operation.V41BF16Boundary(input)
	if err != nil {
		t.Fatal(err)
	}
	defer be.Free(output)
	got := be.Read(output)
	if len(got) != len(words) || output.Buf() == input.Buf() {
		t.Fatal("raw BF16 output is not a fresh complete row")
	}
	for i, value := range got {
		if math.Float32bits(value) != v41BF16BoundaryOracle(words[i]) {
			t.Fatalf("raw word %08x mismatch: %08x", words[i], math.Float32bits(value))
		}
	}
	// The finite-only model boundary owns the result and checks exact transfers.
	s := &Session{M: &Model{}, Backend: be, halW: map[string]compute.Tensor{}}
	defer s.Close()
	row := make([]float32, 4096)
	for i := range row {
		row[i] = math.Float32frombits(0x3f000000 + uint32(i)*4093)
	}
	row[0], row[1], row[2] = math.Float32frombits(0x80000000), math.Float32frombits(0x00018000), math.Float32frombits(0x7f7f7fff)
	before := snapshot()
	got, err = s.v41BF16BoundaryFunc()(0, "physical copy", row)
	if err != nil {
		t.Fatal(err)
	}
	after := snapshot()
	delta, err := compute.BackendExecutionDelta(before, after)
	if err != nil {
		t.Fatal(err)
	}
	c := delta.Counters
	if c.ComputeDispatches != 1 || c.OtherDispatches != 1 || c.H2DCount != 1 || c.H2DBytes != 4*4096 || c.D2HCount != 1 || c.D2HBytes != 4*4096 || c.D2DCopies != 0 || c.Fallbacks != 0 || after.DeviceAllocationLiveBytes != before.DeviceAllocationLiveBytes {
		t.Fatalf("wrong boundary ownership/transfers: %+v", delta)
	}
	for i, value := range got {
		if math.Float32bits(value) != v41BF16BoundaryOracle(math.Float32bits(row[i])) {
			t.Fatalf("finite copy lane %d mismatch", i)
		}
	}
	// Exercise the real production consumer with every other seam identical.
	m := v41FullStepPatchedNorms(t)
	device, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	defer device.Close()
	control, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	defer control.Close()
	control.v41State().bf16Boundary = nil
	calls := 0
	round := device.v41State().bf16Boundary
	if round == nil {
		t.Fatal("production session did not bind witness boundary")
	}
	device.v41State().bf16Boundary = func(layer int, site string, row []float32) ([]float32, error) {
		calls++
		return round(layer, site, row)
	}
	for _, pair := range [][2][]float32{{device.Prefill([]int{1, 2}), control.Prefill([]int{1, 2})}, {device.Step(3), control.Step(3)}} {
		if len(pair[0]) != len(pair[1]) {
			t.Fatal("logit width mismatch")
		}
		for i := range pair[0] {
			if math.Float32bits(pair[0][i]) != math.Float32bits(pair[1][i]) {
				t.Fatalf("BF16 spine changed final logits at %d", i)
			}
		}
	}
	if calls != 6 {
		t.Fatalf("production prefill/decode rounded %d times, want 6", calls)
	}
	t.Logf("V41_BF16_BOUNDARY_PHYSICAL_COMPLETE source-bound copy counters=%+v identity=%+v", delta.Counters, delta.Identity)
}
