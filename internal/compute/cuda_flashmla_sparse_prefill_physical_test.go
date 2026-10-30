//go:build cuda && cgo && cudaarchspecific && fak_cuda_flashmla_prefill

package compute

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

type flashPhysicalFile struct {
	Path, SHA256, Dtype, Order string
	Shape                      []int
}
type flashPhysicalFixture struct {
	FAKCommit, FlashMLACommit, DeepInferCommit, ModelCommit, VendorManifestSHA256                 string
	ReferenceExtensionSHA256, GeneratorSHA256, ToleranceApproval, OutputContract, NonfinitePolicy string
	Artifact, ArtifactSHA256, BuildReceipt, BuildReceiptSHA256                                    string
	Queries, Keys, TopK, TableRows, DeviceMajor, DeviceMinor, BinaryVersion                       int
	Scale                                                                                         float32
	AbsTolerance, RelTolerance                                                                    *float64
	Files                                                                                         map[string]flashPhysicalFile
}

// fak-test:runtime integration est=3s lane=optin
// A skipped test is not evidence. Opt-in requires every provenance field and a
// real CUDA target; an absent device cannot silently substitute a CPU backend.
func TestCUDAFlashMLAPhysicalFixture(t *testing.T) {
	if os.Getenv("FAK_FLASHMLA_PHYSICAL_TEST") != "1" {
		t.Skip("explicit physical witness opt-in required")
	}
	root, err := filepath.EvalSymlinks(os.Getenv("FAK_FLASHMLA_FIXTURE_DIR"))
	if err != nil || root == "" {
		t.Fatal("fixture directory required", err)
	}
	manifest, err := os.ReadFile(filepath.Join(root, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	hash := func(b []byte) string { s := sha256.Sum256(b); return hex.EncodeToString(s[:]) }
	if hash(manifest) != os.Getenv("FAK_FLASHMLA_FIXTURE_SHA256") {
		t.Fatal("manifest SHA mismatch")
	}
	var f flashPhysicalFixture
	if err = json.Unmarshal(manifest, &f); err != nil {
		t.Fatal(err)
	}
	validHex := func(s string, n int) bool { b, e := hex.DecodeString(s); return e == nil && len(b) == n }
	if !validHex(f.FAKCommit, 20) || f.FlashMLACommit != "2e5429fc5653bab6e081f09477126f731882a6a9" || f.DeepInferCommit != "774ce244b87cc0eb8d4b94ab9f5de2ab22049f01" || f.ModelCommit != "dba1be0a40aa45a94ad051997016db3960a90277" || f.VendorManifestSHA256 != "23b68f0257c09dc8b48c5ae7bbffda31538f08523034864854220f170ceaeb22" || !validHex(f.ReferenceExtensionSHA256, 32) || !validHex(f.GeneratorSHA256, 32) {
		t.Fatal("full pinned provenance required")
	}
	if f.OutputContract != "fused-inverse-rope-e4m3-ue8m0" || f.NonfinitePolicy != "matching-infinity-no-nan" || f.ToleranceApproval == "" || f.AbsTolerance == nil || f.RelTolerance == nil {
		t.Fatal("explicit fused oracle and approved tolerance required")
	}
	for _, v := range []float64{*f.AbsTolerance, *f.RelTolerance} {
		if v < 0 || math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatal("invalid tolerance")
		}
	}
	head, e := exec.Command("git", "rev-parse", "HEAD").Output()
	if e != nil || strings.TrimSpace(string(head)) != f.FAKCommit {
		t.Fatal("source HEAD mismatch", e)
	}
	dirty, e := exec.Command("git", "status", "--porcelain").Output()
	if e != nil || len(dirty) != 0 {
		t.Fatal("physical witness requires clean source", e)
	}
	digestFile := func(path string) string {
		t.Helper()
		h := sha256.New()
		fd, e := os.Open(path)
		if e != nil {
			t.Fatal(e)
		}
		defer fd.Close()
		if _, e = io.Copy(h, fd); e != nil {
			t.Fatal(e)
		}
		return hex.EncodeToString(h.Sum(nil))
	}
	if !validHex(f.ArtifactSHA256, 32) || digestFile(f.Artifact) != f.ArtifactSHA256 || !validHex(f.BuildReceiptSHA256, 32) || digestFile(f.BuildReceipt) != f.BuildReceiptSHA256 {
		t.Fatal("build bytes mismatch")
	}
	executable, e := os.Executable()
	if e != nil {
		t.Fatal(e)
	}
	exeSHA := digestFile(executable)
	used := int64(0)
	read := func(key, dtype string, shape []int, size int) []byte {
		t.Helper()
		v, ok := f.Files[key]
		if !ok || v.Dtype != dtype || v.Order != "row-major-little-endian" || !reflect.DeepEqual(v.Shape, shape) || !validHex(v.SHA256, 32) {
			t.Fatal("payload contract", key)
		}
		if filepath.IsAbs(v.Path) {
			t.Fatal("absolute fixture payload")
		}
		path, e := filepath.EvalSymlinks(filepath.Join(root, v.Path))
		if e != nil {
			t.Fatal(e)
		}
		rel, e := filepath.Rel(root, path)
		if e != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
			t.Fatal("payload escapes fixture")
		}
		count := int64(size)
		for _, n := range shape {
			if n <= 0 || count > 512*1024*1024/int64(n) {
				t.Fatal("payload bound")
			}
			count *= int64(n)
		}
		used += count
		if used > 512*1024*1024 {
			t.Fatal("fixture total bound")
		}
		stat, e := os.Stat(path)
		if e != nil || !stat.Mode().IsRegular() || stat.Size() != count {
			t.Fatal("payload size", key, e)
		}
		b, e := os.ReadFile(path)
		if e != nil || hash(b) != v.SHA256 {
			t.Fatal("payload hash", key, e)
		}
		return b
	}
	u16 := func(b []byte) []uint16 {
		v := make([]uint16, len(b)/2)
		for i := range v {
			v[i] = binary.LittleEndian.Uint16(b[i*2:])
		}
		return v
	}
	u32 := func(b []byte) []uint32 {
		v := make([]uint32, len(b)/4)
		for i := range v {
			v[i] = binary.LittleEndian.Uint32(b[i*4:])
		}
		return v
	}
	i32 := func(b []byte) []int32 {
		u := u32(b)
		v := make([]int32, len(u))
		for i := range v {
			v[i] = int32(u[i])
		}
		return v
	}
	f32 := func(b []byte) []float32 {
		u := u32(b)
		v := make([]float32, len(u))
		for i := range v {
			v[i] = math.Float32frombits(u[i])
		}
		return v
	}
	x := V41SparsePrefillInput{Queries: f.Queries, Keys: f.Keys, TopK: f.TopK, TableRows: f.TableRows, Scale: f.Scale}
	x.Q = u16(read("q", "bf16", []int{x.Queries, 64, 512}, 2))
	x.KV = u16(read("kv", "bf16", []int{x.Keys, 512}, 2))
	x.Indices = i32(read("indices", "i32", []int{x.Queries, x.TopK}, 4))
	x.Lengths = i32(read("lengths", "i32", []int{x.Queries}, 4))
	x.Positions = u32(read("positions", "u32", []int{x.Queries}, 4))
	x.Sink = f32(read("sink", "f32", []int{64}, 4))
	x.CosSin = f32(read("table", "f32", []int{x.TableRows, 64}, 4))
	want := [][]float32{f32(read("values", "f32", []int{x.Queries, 64, 512}, 4)), f32(read("max_logits", "f32", []int{x.Queries, 64}, 4)), f32(read("lse", "f32", []int{x.Queries, 64}, 4))}
	if err = validateV41SparsePrefill(x); err != nil {
		t.Fatal(err)
	}
	raw, ok := Lookup("cuda")
	if !ok || raw == nil {
		t.Fatal("real CUDA backend required")
	}
	be, ok := raw.(*cudaBackend)
	if !ok {
		t.Fatalf("unexpected backend %T", raw)
	}
	receipt := map[string]any{"qualification": false, "fixture_sha256": hash(manifest), "fak_commit": f.FAKCommit, "executable_sha256": exeSHA, "artifact_sha256": f.ArtifactSHA256, "build_receipt_sha256": f.BuildReceiptSHA256, "reference_extension_sha256": f.ReferenceExtensionSHA256, "generator_sha256": f.GeneratorSHA256, "physical_call_attempted": false, "physical_call_returned": false, "artifact_linkage": "requires independent build receipt review"}
	complete := false
	defer func() {
		terminal, e := cudaFlashMLAPhysicalState()
		receipt["terminal"] = terminal
		if e != nil {
			receipt["terminal_error"] = e.Error()
		}
		receipt["completed"] = complete
		receipt["passed"] = complete && !t.Failed() && e == nil
		b, e := json.Marshal(receipt)
		if e != nil {
			t.Errorf("receipt serialization: %v", e)
			fmt.Println(`{"passed":false,"receipt_serialization_failed":true}`)
		} else {
			fmt.Println(string(b))
		}
	}()
	before, e := cudaFlashMLAPhysicalState()
	receipt["before"] = before
	if e != nil {
		t.Fatal(e)
	}
	if f.DeviceMajor != 10 || (f.DeviceMinor != 0 && f.DeviceMinor != 3) || f.BinaryVersion < 100 || before["major"] != f.DeviceMajor || before["minor"] != f.DeviceMinor || before["binary_version"] != f.BinaryVersion {
		t.Fatal("actual loaded kernel target mismatch")
	}
	if be.SupportsV41SparsePrefill() {
		t.Fatal("ordinary capability must remain false")
	}
	if out, e := be.V41SparsePrefill(x); !errors.Is(e, ErrV41SparsePrefillUnavailable) || len(out.Values) != 0 {
		t.Fatal("ordinary path admitted or returned output")
	}
	bad := x
	bad.TopK++
	if out, e := be.cudaV41SparsePrefill(bad, true); e == nil || len(out.Values) != 0 {
		t.Fatal("invalid selected geometry did not fail closed")
	}
	receipt["physical_call_attempted"] = true
	opened := time.Now()
	out, e := be.cudaV41SparsePrefill(x, true)
	receipt["elapsed_nanoseconds"] = time.Since(opened).Nanoseconds()
	receipt["physical_call_returned"] = true
	if e != nil {
		t.Fatal(e)
	}
	got := [][]float32{out.Values, out.MaxLogits, out.LSE}
	maxAbs, maxRel, zeroAbs := 0.0, 0.0, 0.0
	for part := range want {
		if len(got[part]) != len(want[part]) {
			t.Fatal("result shape")
		}
		for i, w32 := range want[part] {
			g, w := float64(got[part][i]), float64(w32)
			if math.IsNaN(g) || math.IsNaN(w) {
				t.Fatal("NaN output")
			}
			if math.IsInf(g, 0) || math.IsInf(w, 0) {
				if g != w {
					t.Fatal("infinity mismatch")
				}
				continue
			}
			d := math.Abs(g - w)
			maxAbs = math.Max(maxAbs, d)
			if w == 0 {
				zeroAbs = math.Max(zeroAbs, d)
			} else {
				r := d / math.Abs(w)
				if math.IsInf(r, 0) {
					r = math.MaxFloat64
				}
				maxRel = math.Max(maxRel, r)
			}
			if d > *f.AbsTolerance+*f.RelTolerance*math.Abs(w) {
				t.Fatalf("output mismatch part%d index%d", part, i)
			}
		}
	}
	receipt["max_absolute_error"] = maxAbs
	receipt["max_relative_error_nonzero_reference"] = maxRel
	receipt["max_zero_reference_absolute_error"] = zeroAbs
	after, e := cudaFlashMLAPhysicalState()
	receipt["after"] = after
	if e != nil {
		t.Fatal(e)
	}
	if !reflect.DeepEqual(before, after) {
		t.Fatal("device identity/allocation/stream state changed")
	}
	complete = true
}
