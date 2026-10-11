//go:build v41_logit_scale_witness && vulkan && (windows || linux) && cgo

package model

import (
	"crypto/sha1"
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
type v41LogitScaleBuildRecord struct {
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

func v41LogitScaleProvenance(t *testing.T) {
	t.Helper()
	recordPath := os.Getenv("FAK_V41_LOGIT_SCALE_WITNESS_PROVENANCE")
	expected := os.Getenv("FAK_V41_LOGIT_SCALE_WITNESS_PROVENANCE_SHA256")
	revision := os.Getenv("FAK_V41_LOGIT_SCALE_WITNESS_REVISION")
	if len(revision) != hex.EncodedLen(sha1.Size) || len(expected) != hex.EncodedLen(sha256.Size) {
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
	var record v41LogitScaleBuildRecord
	if err := json.Unmarshal(raw, &record); err != nil {
		t.Fatal(err)
	}
	if record.Schema != "fak.v41-logit-scale-witness.v1" || record.SourceRevision != revision || record.NativeDefine != "FAK_V41_LOGIT_SCALE_WITNESS" || record.GoTag != "v41_logit_scale_witness" || len(record.Commands) < 3 {
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
	archive := os.Getenv("FAK_V41_LOGIT_SCALE_WITNESS_ARCHIVE")
	if !filepath.IsAbs(archive) || filepath.Clean(record.Files["native_archive"].Path) != filepath.Clean(archive) {
		t.Fatal("record does not name the explicitly reviewed witness archive")
	}
	for _, kind := range []string{"native_archive", "shader_module", "test_binary", "shader_source", "native_source", "native_header", "go_adapter"} {
		file, ok := record.Files[kind]
		if !ok || !filepath.IsAbs(file.Path) || len(file.SHA256) != hex.EncodedLen(sha256.Size) {
			t.Fatalf("missing reviewed %s", kind)
		}
		if kind == "native_archive" && filepath.Base(file.Path) != "libfakvulkan_v41_logit_scale_witness.a" {
			t.Fatal("ordinary native archive is not logit-scale witness provenance")
		}
		if kind == "shader_module" && filepath.Clean(file.Path) != filepath.Clean(filepath.Join(bundle, "v41_logit_scale.spv")) {
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
	t.Logf("reviewed logit-scale witness source=%s record_sha256=%s", revision, expected)
}

// Deferred physical witness source, only linked to the isolated scalar archive.
// Nine literal scalar cases, one 4096-value model-copy dispatch, and synthetic
// two-token prefill plus one decode bound this probe. No full-checkpoint or
// trusted production registry qualification is implied by source wiring.
// Compile v41_logit_scale.comp without glslc -O: the optimized module fails native ABI admission.
// fak-test:runtime integration est=30s lane=optin
func TestV41LogitScaleVulkan(t *testing.T) {
	if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") != "1" || os.Getenv("FAK_VULKAN_EXPECT_DEVICE") == "" || os.Getenv("FAK_VULKAN_DISPATCH_PROFILE") != "1" {
		t.Fatal("scalar witness requires explicit physical identity and profiling before startup")
	}
	v41LogitScaleProvenance(t)
	be, ok := compute.Lookup("vulkan")
	if !ok || be == nil || !be.Caps().DeviceMemory {
		t.Fatal("required Vulkan device unavailable")
	}
	physical, ok := be.(interface{ VulkanPhysicalDeviceType() uint32 })
	if !ok || (physical.VulkanPhysicalDeviceType() != 1 && physical.VulkanPhysicalDeviceType() != 2) {
		t.Fatal("physical integrated/discrete GPU required")
	}
	operation, ok := be.(compute.V41LogitScaleBackend)
	if !ok || !operation.V41LogitScaleAvailable() {
		t.Fatal("witness scalar module or actual-device FloatControls unavailable")
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
	for _, tc := range []struct{ input, factor, want uint32 }{
		{0x80000000, 0x3f000000, 0x80000000}, {0, 0xbf000000, 0x80000000},
		{0x80000000, 0xbf000000, 0}, {1, 0x3f000000, 0},
		{0x80000001, 0x3f000000, 0x80000000}, {3, 0x3f000000, 2},
		{0x00800000, 0x3f000000, 0x00400000}, {0x3f800001, 0x3f800001, 0x3f800002},
		{0x7f7fffff, 0x40000000, 0x7f800000},
	} {
		func() {
			input := be.Upload(compute.NewF32(compute.Default(), []int{1}, []float32{math.Float32frombits(tc.input)}), compute.F32)
			defer be.Free(input)
			out, err := operation.V41LogitScale(input, math.Float32frombits(tc.factor))
			if err != nil {
				t.Fatal(err)
			}
			defer be.Free(out)
			got := be.Read(out)
			if out.Backend() != be || out.Buf() == input.Buf() || len(got) != 1 || math.Float32bits(got[0]) != tc.want {
				t.Fatalf("raw scalar %08x * %08x did not publish %08x", tc.input, tc.factor, tc.want)
			}
		}()
	}
	// The actual session-copy seam must own both temporary allocations and
	// publish a host-owned row only after checked completion/readback.
	s := &Session{M: &Model{}, Backend: be, halW: map[string]compute.Tensor{}}
	defer s.Close()
	row := make([]float32, 4096)
	for i := range row {
		row[i] = math.Float32frombits(0x3f000000 + uint32(i)*4093)
	}
	row[0], row[1], row[2] = math.Float32frombits(0x80000000), math.Float32frombits(3), math.Float32frombits(0x00800000)
	before := snapshot()
	got, err := s.v41LogitScaleFunc()(row, 0.5)
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
		t.Fatalf("wrong scalar ownership/transfers: %+v", delta)
	}
	want := append([]float32(nil), row...)
	logitScaleInPlace(want, Config{LogitScale: 0.5})
	v41LogitScaleCheckBits(t, got, want)
	m := v41FullStepPatchedNorms(t)
	defer m.CloseWeights()
	m.Cfg.LogitScale, m.Cfg.LogitSoftcap, m.Cfg.SuppressTokens = 0.0625, 0.75, []int{1}
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
	control.v41State().logitScale = nil
	calls := 0
	scale := device.v41State().logitScale
	if scale == nil {
		t.Fatal("production head did not bind scalar witness")
	}
	device.v41State().logitScale = func(row []float32, factor float32) ([]float32, error) { calls++; return scale(row, factor) }
	for _, pair := range [][2][]float32{{device.Prefill([]int{1, 2}), control.Prefill([]int{1, 2})}, {device.Step(3), control.Step(3)}} {
		v41LogitScaleCheckBits(t, pair[0], pair[1])
	}
	if calls != 3 {
		t.Fatalf("production prefill/decode scaled %d times, want3", calls)
	}
	t.Logf("V41_LOGIT_SCALE_PHYSICAL_COMPLETE source-bound copy counters=%+v identity=%+v", delta.Counters, delta.Identity)
}
