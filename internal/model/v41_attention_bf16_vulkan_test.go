//go:build vulkan && (windows || linux) && cgo && !v41_bf16_boundary_witness && !v41_indexer_witness && !v41_logit_scale_witness && !v41_tail_rope_exact_witness

package model

import (
	"crypto/sha1"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"sort"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/computebuild"
)

// This reviewed execution record binds the test binary to the ordinary archive
// and canonical verifier result for the same source. The canonical production
// binary is distinct from this test binary. Neither this record nor file hashing
// attests to bytes loaded before process startup; the external execution review
// must enforce immutable artifacts and the recorded build/link commands.
type v41AttentionBF16PhysicalRecord struct {
	Schema              string   `json:"schema"`
	Revision            string   `json:"source_revision"`
	GoTags              string   `json:"go_tags"`
	Commands            []string `json:"commands"`
	DirectBoundaryWidth int      `json:"direct_boundary_width"`
	SessionFixture      string   `json:"session_fixture"`
	Files               map[string]struct {
		Path   string `json:"path"`
		SHA256 string `json:"sha256"`
	} `json:"files"`
}

func v41AttentionBF16PhysicalProvenance(t *testing.T) {
	t.Helper()
	expected, revision := os.Getenv("FAK_V41_ATTENTION_BF16_RECORD_SHA256"), os.Getenv("FAK_V41_ATTENTION_BF16_REVISION")
	if len(expected) != hex.EncodedLen(sha256.Size) || len(revision) != hex.EncodedLen(sha1.Size) {
		t.Fatal("externally reviewed record digest and full revision required")
	}
	if _, e := hex.DecodeString(expected); e != nil {
		t.Fatal(e)
	}
	if _, e := hex.DecodeString(revision); e != nil {
		t.Fatal(e)
	}
	raw, e := os.ReadFile(os.Getenv("FAK_V41_ATTENTION_BF16_RECORD"))
	if e != nil {
		t.Fatal(e)
	}
	digest := sha256.Sum256(raw)
	if hex.EncodeToString(digest[:]) != expected {
		t.Fatal("execution record differs from independently reviewed digest")
	}
	var record v41AttentionBF16PhysicalRecord
	if e = json.Unmarshal(raw, &record); e != nil {
		t.Fatal(e)
	}
	if record.Schema != "fak.v41-attention-bf16-physical.v1" || record.Revision != revision || record.GoTags != "vulkan" || record.DirectBoundaryWidth != 5120 || record.SessionFixture != "synthetic-small-model" || len(record.Commands) < 4 {
		t.Fatal("wrong ordinary source/build/execution record")
	}
	for _, c := range record.Commands {
		if c == "" {
			t.Fatal("empty build/verification command")
		}
	}
	binary, e := os.Executable()
	if e != nil {
		t.Fatal(e)
	}
	bundle, archive := os.Getenv("FAK_VULKAN_SPIRV"), os.Getenv("FAK_V41_ATTENTION_BF16_ARCHIVE")
	if !filepath.IsAbs(bundle) || !filepath.IsAbs(archive) || filepath.Base(archive) != "libfakvulkan.a" {
		t.Fatal("explicit ordinary bundle/archive required")
	}
	stems := computebuild.CurrentVulkanShaderRegistryLatest()
	sort.Strings(stems)
	registryBytes, e := json.Marshal(stems)
	if e != nil {
		t.Fatal(e)
	}
	registryHash := sha256.Sum256(registryBytes)
	// Bind the census this source's builder stamps, not a frozen successor that later modules retire.
	if !slices.Contains(stems, "v41_bf16_checked") {
		t.Fatal("current registry policy lacks checked attention BF16 publication")
	}
	required := []string{"canonical_verification", "native_archive", "test_binary", "shader_source", "native_source", "native_header", "go_adapter", "model_owner", "model_forward", "model_forward_weights", "model_incremental", "model_layer_step", "model_attention_helper", "physical_test_source"}
	for _, stem := range stems {
		required = append(required, "module:"+stem)
	}
	for _, kind := range required {
		f, ok := record.Files[kind]
		if !ok || !filepath.IsAbs(f.Path) || len(f.SHA256) != hex.EncodedLen(sha256.Size) {
			t.Fatalf("missing reviewed %s", kind)
		}
		switch kind {
		case "native_archive":
			if filepath.Clean(f.Path) != filepath.Clean(archive) {
				t.Fatal("archive path mismatch")
			}
		case "test_binary":
			if filepath.Clean(f.Path) != filepath.Clean(binary) {
				t.Fatal("running binary mismatch")
			}
		}
		if len(kind) > 7 && kind[:7] == "module:" && filepath.Clean(f.Path) != filepath.Join(filepath.Clean(bundle), kind[7:]+".spv") {
			t.Fatal("record does not bind configured module path")
		}
		in, e := os.Open(f.Path)
		if e != nil {
			t.Fatal(e)
		}
		h := sha256.New()
		n, re := io.Copy(h, in)
		ce := in.Close()
		if re != nil || ce != nil || n == 0 || hex.EncodeToString(h.Sum(nil)) != f.SHA256 {
			t.Fatalf("reviewed %s bytes changed", kind)
		}
	}
	// Read only the stable CLI projection, not a second receipt verifier. The
	// external pinned record must bind the actual canonical verification command.
	verified, e := os.ReadFile(record.Files["canonical_verification"].Path)
	if e != nil {
		t.Fatal(e)
	}
	var v struct {
		Schema        string   `json:"schema"`
		Verdict       string   `json:"verdict"`
		ReceiptSchema string   `json:"receipt_schema"`
		Revision      string   `json:"public_revision"`
		Count         int      `json:"spirv_module_count"`
		Modules       []string `json:"spirv_modules"`
		Registry      struct {
			ID     string `json:"id"`
			SHA256 string `json:"sha256"`
			Count  int    `json:"module_count"`
		} `json:"shader_registry"`
		Archive struct {
			SHA256 string `json:"sha256"`
		} `json:"native_archive"`
	}
	if e = json.Unmarshal(verified, &v); e != nil {
		t.Fatal(e)
	}
	sort.Strings(v.Modules)
	version, versioned := strings.CutPrefix(v.ReceiptSchema, "fak.vulkan-build-receipt.")
	if v.Schema != "fak.vulkan-build-verification.v1" || v.Verdict != "verified" || !versioned || v.Revision != revision || v.Count != len(stems) || !reflect.DeepEqual(v.Modules, stems) || v.Registry.ID != fmt.Sprintf("fak.vulkan-shader-registry.%s.current%d", version, len(stems)) || v.Registry.Count != len(stems) || v.Registry.SHA256 != hex.EncodeToString(registryHash[:]) || v.Archive.SHA256 != record.Files["native_archive"].SHA256 {
		t.Fatal("canonical verification does not bind this ordinary current profile")
	}
	t.Logf("ordinary attention BF16 source=%s execution_record=%s", revision, expected)
}

type v41AttentionBF16PhysicalObserver struct {
	t                *testing.T
	backend          compute.Backend
	identity         compute.BackendRuntimeIdentity
	calls, norms     int
	retained, copies [][]float32
}

func (w *v41AttentionBF16PhysicalObserver) snapshot() compute.BackendExecutionSnapshot {
	w.t.Helper()
	// Free parks rows in the backend reuse pool; drain it so live bytes count owned allocations only.
	if pool, ok := w.backend.(interface{ Trim() }); ok {
		pool.Trim()
	}
	x, ok, e := compute.CaptureBackendExecutionSnapshot(w.backend)
	if e != nil || !ok || x.Identity != w.identity || !x.TransferCountersObserved || !x.DeviceAllocationObserved {
		w.t.Fatal("physical counters/identity unavailable", e)
	}
	return x
}
func (w *v41AttentionBF16PhysicalObserver) publication(run v41BF16BoundaryFunc) v41BF16BoundaryFunc {
	return func(layer int, site string, row []float32) ([]float32, error) {
		original := append([]float32(nil), row...)
		before := w.snapshot()
		result, e := run(layer, site, row)
		if e != nil {
			return result, e
		}
		after := w.snapshot()
		d, e := compute.BackendExecutionDelta(before, after)
		if e != nil {
			w.t.Fatal(e)
		}
		c := d.Counters
		if c.ComputeDispatches != 1 || c.OtherDispatches != 1 || c.H2DCount != 1 || c.H2DBytes != uint64(4*len(row)) || c.D2HCount != 2 || c.D2HBytes != uint64(4*len(row)+16) || c.D2DCopies != 0 || c.Fallbacks != 0 || before.DeviceAllocationLiveBytes != after.DeviceAllocationLiveBytes {
			w.t.Fatalf("publication %s actual counters/ownership=%+v", site, d)
		}
		if len(result) != len(row) {
			w.t.Fatal("publication width")
		}
		for i, x := range result {
			if math.Float32bits(x) != v41BF16BoundaryOracle(math.Float32bits(original[i])) || math.Float32bits(row[i]) != math.Float32bits(original[i]) {
				w.t.Fatal("publication exact bits/input ownership")
			}
		}
		w.calls++
		w.retained = append(w.retained, result)
		w.copies = append(w.copies, append([]float32(nil), result...))
		w.t.Logf("attention publication layer=%d site=%s count=%d counters=%+v", layer, site, len(row), c)
		return result, nil
	}
}
func (w *v41AttentionBF16PhysicalObserver) normalization(s *Session, run v41AttentionInputNormFunc) v41AttentionInputNormFunc {
	return func(layer int, row []float32) ([]float32, error) {
		original := append([]float32(nil), row...)
		gain := append([]float32(nil), s.M.tensor(layerName(layer, "attn_norm.weight"))...)
		if len(gain) != len(row) {
			w.t.Fatal("gain width")
		}
		var sum float64
		for _, x := range row {
			if !finite32(x) || math.Float32bits(x)&0xffff != 0 {
				w.t.Fatal("norm input publication absent")
			}
			sum += float64(x) * float64(x)
		}
		inv := 1 / math.Sqrt(sum/float64(len(row))+s.M.Cfg.RMSNormEps)
		before := w.snapshot()
		result, e := run(layer, row)
		if e != nil {
			return result, e
		}
		after := w.snapshot()
		d, e := compute.BackendExecutionDelta(before, after)
		if e != nil {
			w.t.Fatal(e)
		}
		c := d.Counters
		// Gain residency can survive a same-model restored Session. Count the real
		// one/two uploads without treating an empty session memo as a cold allocation.
		if c.ComputeDispatches != 1 || c.H2DCount < 1 || c.H2DCount > 2 || c.H2DBytes != uint64(4*len(row))*c.H2DCount || c.D2HCount != 1 || c.D2HBytes != uint64(4*len(row)) || c.Fallbacks != 0 || len(result) != len(row) {
			w.t.Fatal("unchanged RMSNorm physical dispatch/transfer contract", c)
		}
		for i, x := range result {
			want := float32(float64(original[i]) * inv * float64(gain[i]))
			if !finite32(x) || !finite32(want) || math.Abs(float64(x-want)) > 2e-4*math.Max(1, math.Abs(float64(want))) || math.Float32bits(row[i]) != math.Float32bits(original[i]) {
				w.t.Fatal("RMSNorm tolerance or input ownership")
			}
		}
		w.norms++
		return result, nil
	}
}
func v41AttentionBF16PhysicalIsolate(st *v41ForwardState) {
	norm, publish := st.attentionNorm, st.attentionBF16
	v41NormRoPEHaloIsolate(st)
	st.queryNorm, st.kvNorm, st.ffnNorm, st.finalNorm, st.tailRoPE = nil, nil, nil, nil, nil
	st.queryPrefix, st.bf16Boundary, st.logitScale = nil, nil, nil
	st.routerTopK, st.routedSum, st.sharedAdd = nil, nil, nil
	st.mhcDevice, st.gatherOps = nil, nil
	st.attentionNorm, st.attentionBF16 = norm, publish
}

// Exact H5120 publication and small real Session routes are separate controls.
// This source requires an actual selected ordinary device run; a skipped or
// compile-only result cannot qualify the operation or remove any inventory row.
// It does not claim checkpoint parity, whole-model device execution or speed.
// fak-test:runtime integration est=30s lane=optin
func TestV41AttentionBF16Vulkan(t *testing.T) {
	if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") != "1" || os.Getenv("FAK_VULKAN_EXPECT_DEVICE") == "" || os.Getenv("FAK_VULKAN_DISPATCH_PROFILE") != "1" {
		t.Fatal("explicit device identity and profiling required before startup")
	}
	v41AttentionBF16PhysicalProvenance(t)
	be, ok := compute.Lookup("vulkan")
	if !ok || be == nil || !be.Caps().DeviceMemory {
		t.Fatal("ordinary Vulkan unavailable")
	}
	physical, ok := be.(interface{ VulkanPhysicalDeviceType() uint32 })
	if !ok || (physical.VulkanPhysicalDeviceType() != 1 && physical.VulkanPhysicalDeviceType() != 2) {
		t.Fatal("physical integrated/discrete device required")
	}
	op, ok := be.(compute.V41AttentionBF16Backend)
	if !ok || !op.SupportsV41AttentionBF16(5120) {
		t.Fatal("ordinary H5120 publication unavailable")
	}
	first, available, e := compute.CaptureBackendExecutionSnapshot(be)
	if e != nil || !available || first.Identity.Backend != "vulkan" || first.Identity.Device != os.Getenv("FAK_VULKAN_EXPECT_DEVICE") {
		t.Fatal("selected physical identity mismatch")
	}
	w := &v41AttentionBF16PhysicalObserver{t: t, backend: be, identity: first.Identity}
	t.Run("H5120-boundaries", func(t *testing.T) {
		w.t = t
		m := v41FullStepPatchedNorms(t)
		t.Cleanup(func() {
			if e := m.CloseWeights(); e != nil {
				t.Error(e)
			}
		})
		m.Cfg.HiddenSize = 5120
		m.Cfg.DeepSeekV41 = nil // Direct boundary contract; no synthetic H5120 checkpoint is inferred.
		s := &Session{M: m, Backend: be, halW: map[string]compute.Tensor{}}
		defer s.Close()
		publish := s.v41AttentionBF16Func()
		if publish == nil {
			t.Fatal("H5120 model boundary unavailable")
		}
		row := make([]float32, 5120)
		for i := range row {
			row[i] = math.Float32frombits(0x3f000000 + uint32(i)*4093)
		}
		row[0] = math.Float32frombits(0x80000000)
		row[1] = math.Float32frombits(0x00018000)
		row[2] = math.Float32frombits(0x7f7f7fff)
		observe := w.publication(publish)
		for _, site := range []string{"attn_norm.weight input", "attn_norm.weight output"} {
			if _, e := observe(0, site, row); e != nil {
				t.Fatal(e)
			}
		}
	})
	t.Run("device-invalid-word", func(t *testing.T) {
		w.t = t
		for _, site := range []uint32{3, 4} {
			for _, bits := range []uint32{0x7fffffff, 0x7f7f8000} {
				in := be.Upload(compute.NewF32(compute.Default(), []int{3}, []float32{1, math.Float32frombits(bits), math.Float32frombits(0xffffffff)}), compute.F32)
				before := w.snapshot()
				out, e := op.V41AttentionBF16(in, site)
				after := w.snapshot()
				be.Free(in)
				var numeric *compute.V41BF16CheckedArithmeticError
				if out.Buf() != nil || !errors.As(e, &numeric) || numeric.Site != site || numeric.Index != 1 {
					t.Fatal("device did not withhold invalid publication with lowest-index status", e)
				}
				d, de := compute.BackendExecutionDelta(before, after)
				if de != nil || d.Counters.ComputeDispatches != 1 || d.Counters.D2HCount != 1 || d.Counters.D2HBytes != 16 {
					t.Fatal("failed output was read or status absent", de, d)
				}
			}
		}
	})
	for _, role := range []bool{false, true} {
		for _, tokenMajor := range []bool{false, true} {
			for _, resident := range []bool{false, true} {
				t.Run(fmt.Sprintf("consumer-role=%t/tokenMajor=%t/resident=%t", role, tokenMajor, resident), func(t *testing.T) {
					w.t = t
					old := v41ForceTokenMajor
					v41ForceTokenMajor = tokenMajor
					defer func() { v41ForceTokenMajor = old }()
					fixture := func() *Model {
						if role {
							return v41IncrementalExpertFixture(t, true, false)
						}
						return v41FullStepPatchedNorms(t)
					}
					m, cm := fixture(), fixture()
					t.Cleanup(func() {
						if e := m.CloseWeights(); e != nil {
							t.Error(e)
						}
						if e := cm.CloseWeights(); e != nil {
							t.Error(e)
						}
					})
					s, e := m.NewBackendSessionChecked(be)
					if e != nil {
						t.Fatal(e)
					}
					defer s.Close()
					control, e := cm.NewBackendSessionChecked(be)
					if e != nil {
						t.Fatal(e)
					}
					defer control.Close()
					bind := func(s *Session, observe bool) {
						st := s.v41State()
						mhc := st.mhcDevice
						v41AttentionBF16PhysicalIsolate(st)
						if resident {
							if mhc == nil || mhc.op == nil || mhc.refusal != nil {
								t.Fatal("physical resident mHC fixture unavailable")
							}
							st.mhcDevice = mhc
						}
						if st.attentionNorm == nil || st.attentionBF16 == nil {
							t.Fatal("actual consumer not bound")
						}
						if observe {
							st.attentionNorm = w.normalization(s, st.attentionNorm)
							st.attentionBF16 = w.publication(st.attentionBF16)
						} else {
							st.attentionBF16 = nil
						}
					}
					bind(s, true)
					bind(control, false)
					calls, norms := w.calls, w.norms
					compare := func(a, b []float32) {
						if len(a) == 0 || len(a) != len(b) {
							t.Fatal("logit shape")
						}
						for i, x := range a {
							if !finite32(x) || !finite32(b[i]) || math.Float32bits(x) != math.Float32bits(b[i]) {
								t.Fatal("device publication changed unchanged-norm control logits")
							}
						}
					}
					compare(s.Prefill([]int{1, 2}), control.Prefill([]int{1, 2}))
					compare(s.Step(3), control.Step(3))
					snap, e := s.PrefixSnapshot()
					if e != nil {
						t.Fatal(e)
					}
					defer snap.Close()
					target, e := m.NewBackendSessionChecked(be)
					if e != nil {
						t.Fatal(e)
					}
					defer target.Close()
					if target.v41Forward != nil {
						t.Fatal("destination not fresh")
					}
					if e = snap.Restore(target); e != nil {
						t.Fatal(e)
					}
					s.Close()
					bind(target, true)
					compare(target.Prefill([]int{4}), control.Prefill([]int{4}))
					if w.calls-calls != 4*m.Cfg.NumLayers || w.norms-norms != 4*m.Cfg.NumLayers || target.v41State().callbackOwner != target {
						t.Fatal("prefill/decode/restore missed output boundaries or republished BF16 inputs")
					}
				})
			}
		}
	}
	for i, row := range w.retained {
		for j, x := range row {
			if math.Float32bits(x) != math.Float32bits(w.copies[i][j]) {
				t.Fatal("published host copy changed after later device reuse")
			}
		}
	}
	t.Logf("V41_ATTENTION_BF16_PHYSICAL_COMPLETE publications=%d norms=%d identity=%+v; synthetic routes and H5120 boundary only", w.calls, w.norms, w.identity)
}
