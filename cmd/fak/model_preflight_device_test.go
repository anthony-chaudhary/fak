package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

// ggufTestTensor is one tensor directory row of a generated fixture.
type ggufTestTensor struct {
	dims []uint64
	typ  ggufload.TensorType
}

// writeLlamaGGUFTensorFixture writes a minimal single-file llama GGUF carrying only a header
// and tensor directory (no payload bytes), which is all OpenHeaders and preflight read.
func writeLlamaGGUFTensorFixture(t *testing.T, path string, tensors map[string]ggufTestTensor) {
	t.Helper()
	var buf bytes.Buffer
	buf.WriteString("GGUF")
	binary.Write(&buf, binary.LittleEndian, uint32(3))
	names := make([]string, 0, len(tensors))
	for n := range tensors {
		names = append(names, n)
	}
	sort.Strings(names)
	binary.Write(&buf, binary.LittleEndian, uint64(len(names)))
	binary.Write(&buf, binary.LittleEndian, uint64(6)) // kvCount

	writeKey := func(key string) {
		kb := []byte(key)
		binary.Write(&buf, binary.LittleEndian, uint64(len(kb)))
		buf.Write(kb)
	}
	writeStringKV := func(key, val string) {
		writeKey(key)
		binary.Write(&buf, binary.LittleEndian, uint32(8)) // TypeString
		vb := []byte(val)
		binary.Write(&buf, binary.LittleEndian, uint64(len(vb)))
		buf.Write(vb)
	}
	writeUint32KV := func(key string, v uint32) {
		writeKey(key)
		binary.Write(&buf, binary.LittleEndian, uint32(4)) // TypeUint32
		binary.Write(&buf, binary.LittleEndian, v)
	}
	writeFloat32KV := func(key string, v float32) {
		writeKey(key)
		binary.Write(&buf, binary.LittleEndian, uint32(6)) // TypeFloat32
		binary.Write(&buf, binary.LittleEndian, math.Float32bits(v))
	}
	writeStringKV("general.architecture", "llama")
	writeUint32KV("llama.embedding_length", 4096)
	writeUint32KV("llama.block_count", 2)
	writeUint32KV("llama.attention.head_count", 32)
	writeUint32KV("llama.feed_forward_length", 8192)
	writeFloat32KV("llama.attention.layer_norm_rms_epsilon", 1e-5)

	for _, name := range names {
		spec := tensors[name]
		nb := []byte(name)
		binary.Write(&buf, binary.LittleEndian, uint64(len(nb)))
		buf.Write(nb)
		binary.Write(&buf, binary.LittleEndian, uint32(len(spec.dims)))
		for _, d := range spec.dims {
			binary.Write(&buf, binary.LittleEndian, d)
		}
		binary.Write(&buf, binary.LittleEndian, uint32(spec.typ))
		binary.Write(&buf, binary.LittleEndian, uint64(0)) // offset
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}

// TestRunModelPreflightGGUFJSONHasNoFormatOrDeviceKeys pins the byte-compatibility contract:
// without --safetensors or --device the report JSON carries no format or device keys.
func TestRunModelPreflightGGUFJSONHasNoFormatOrDeviceKeys(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "fixture.gguf")
	writeLlamaGGUFTensorFixture(t, path, map[string]ggufTestTensor{
		"token_embd.weight":   {dims: []uint64{32, 4096}, typ: ggufload.TensorF32},
		"blk.0.attn_q.weight": {dims: []uint64{4096, 4096}, typ: ggufload.TensorQ4_K},
		"blk.0.ffn_up.weight": {dims: []uint64{8192, 4096}, typ: ggufload.TensorQ6_K},
		"output_norm.weight":  {dims: []uint64{4096}, typ: ggufload.TensorBF16},
	})
	var out, errOut bytes.Buffer
	code := runModelPreflight(&out, &errOut, []string{"--gguf", path, "--json"})
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %s", code, errOut.String())
	}
	var m map[string]any
	if err := json.Unmarshal(out.Bytes(), &m); err != nil {
		t.Fatalf("parse report: %v\n%s", err, out.String())
	}
	if got, ok := m["format"]; ok {
		t.Errorf("report has format key %v; the GGUF leg must omit it", got)
	}
	if got, ok := m["device"]; ok {
		t.Errorf("report has device key %v; the no-device leg must omit it", got)
	}
	if m["schema"] != "fak.model-manifest-preflight/1" {
		t.Errorf("schema = %v", m["schema"])
	}
	if m["verdict"] != "READY" {
		t.Errorf("verdict = %v, want READY; refusals=%v", m["verdict"], m["refusals"])
	}
}

// safetensorsCLIFixture writes the V4.1-shaped checkpoint the CLI tests share: one I8+F8_E8M0
// routed expert, one F8_E4M3+F8_E8M0 dense weight, one BF16 norm, and one orphan.
func safetensorsCLIFixture(t *testing.T, dir string) {
	t.Helper()
	cfg, err := os.ReadFile("../../internal/model/testdata/deepseek_v41_flash_config.json")
	if err != nil {
		t.Fatalf("read V4.1 config: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "config.json"), cfg, 0o644); err != nil {
		t.Fatal(err)
	}
	shard := map[string]any{
		"layers.0.ffn.experts.0.w1.weight": map[string]any{"dtype": "I8", "shape": []int{2304, 2560}},
		"layers.0.ffn.experts.0.w1.scale":  map[string]any{"dtype": "F8_E8M0", "shape": []int{2304, 160}},
		"layers.0.attn.wq_a.weight":        map[string]any{"dtype": "F8_E4M3", "shape": []int{1280, 5120}},
		"layers.0.attn.wq_a.scale":         map[string]any{"dtype": "F8_E8M0", "shape": []int{40, 160}},
		"model.norm.weight":                map[string]any{"dtype": "BF16", "shape": []int{5120}},
		"layers.0.orphan_tensor.weight":    map[string]any{"dtype": "BF16", "shape": []int{8}},
	}
	hdr, err := json.Marshal(shard)
	if err != nil {
		t.Fatal(err)
	}
	var b []byte
	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(hdr)))
	b = append(b, lenBuf[:]...)
	b = append(b, hdr...)
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), b, 0o644); err != nil {
		t.Fatal(err)
	}
}

// TestRunModelPreflightSafetensorsExit3AndFormatKey pins the CLI contract: the safetensors leg
// exits 3 on the refused V4.1 fixture and reports format "safetensors".
func TestRunModelPreflightSafetensorsExit3AndFormatKey(t *testing.T) {
	dir := t.TempDir()
	safetensorsCLIFixture(t, dir)
	var out, errOut bytes.Buffer
	code := runModelPreflight(&out, &errOut, []string{"--safetensors", dir, "--json"})
	if code != 3 {
		t.Fatalf("exit = %d, want 3; stderr = %s stdout = %s", code, errOut.String(), out.String())
	}
	var m map[string]any
	if err := json.Unmarshal(out.Bytes(), &m); err != nil {
		t.Fatalf("parse report: %v", err)
	}
	if m["format"] != "safetensors" {
		t.Errorf("format = %v, want safetensors", m["format"])
	}
	if m["verdict"] != "REFUSED" {
		t.Errorf("verdict = %v, want REFUSED", m["verdict"])
	}
}

// TestRunModelPreflightSafetensorsDeviceClassesAtSm103 pins the live-table classification at
// sm_103: the packed expert pair lists native, the dense FP8 pair lists expanded:Q8_0, and the
// plain norm stays host.
func TestRunModelPreflightSafetensorsDeviceClassesAtSm103(t *testing.T) {
	dir := t.TempDir()
	safetensorsCLIFixture(t, dir)
	var out, errOut bytes.Buffer
	code := runModelPreflight(&out, &errOut, []string{"--safetensors", dir, "--device", "cuda", "--arch", "sm_103", "--json"})
	if code != 3 { // still refused by admission; the device leg adds its own classes
		t.Fatalf("exit = %d, want 3; stderr = %s", code, errOut.String())
	}
	var rep modelPreflightReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("parse report: %v", err)
	}
	if rep.Device == nil {
		t.Fatal("device block missing")
	}
	if rep.Device.Scope != "weight-ops" {
		t.Errorf("scope = %q, want weight-ops", rep.Device.Scope)
	}
	if rep.Device.Coverage != "covered" {
		t.Errorf("arch_coverage = %q, want covered", rep.Device.Coverage)
	}
	classes := rep.Device.Classes
	if got := classes["layers.0.ffn.experts.0.w1.weight"]; got != "native" {
		t.Errorf("expert class = %q, want native", got)
	}
	if got := classes["layers.0.ffn.experts.0.w1.scale"]; got != "native" {
		t.Errorf("expert scale class = %q, want native", got)
	}
	if got := classes["layers.0.attn.wq_a.weight"]; got != "expanded:Q8_0" {
		t.Errorf("dense weight class = %q, want expanded:Q8_0", got)
	}
	if got := classes["model.norm.weight"]; got != "host" {
		t.Errorf("norm class = %q, want host", got)
	}
	if len(rep.Device.Gaps) != 0 {
		t.Errorf("gaps = %v, want none at sm_103 with the live table", rep.Device.Gaps)
	}
}

// eightDtypeTable filters the live table to the ticket's original 8-dtype CUDA matmul set plus
// the expert rows, so tests can exercise the pre-q3_k envelope through the same classifier.
func eightDtypeTable(t *testing.T) []compute.CUDADeviceOp {
	t.Helper()
	keep := map[string]bool{"f32": true, "f16": true, "q8_0": true, "q4_k": true, "q5_k": true, "q6_k": true, "q2_0": true, "q2_k": true}
	var out []compute.CUDADeviceOp
	for _, row := range compute.CUDADeviceOpTable() {
		if row.Operation != "matmul" || !keep[row.Format] {
			continue
		}
		out = append(out, row)
	}
	return out
}

// TestRunModelPreflightDeviceRemovedExpertRow pins the absent path: with the expert GEMV row
// removed from the declared table, the packed expert pair classifies absent and becomes exactly
// one UNSUPPORTED_DTYPE refusal on the cuda:expert_gemv(I8+F8_E8M0) axis.
func TestRunModelPreflightDeviceRemovedExpertRow(t *testing.T) {
	table := filterTableRows(compute.CUDADeviceOpTable(), func(op, format string) bool {
		return !(op == "routed-expert-gemv" && format == "I8+F8_E8M0")
	})
	plan, err := fakmodel.BuildSafetensorsPreflight(fakmodel.SafetensorsPreflightInput{Dir: v41PlanFixture(t)})
	if err != nil {
		t.Fatalf("BuildSafetensorsPreflight: %v", err)
	}
	block, refs := buildDeviceReport(table, "sm_103", denseKQuantDecision{}.retained, nil, plan.Tensors, plan.Config)
	if got := block.Classes["layers.0.ffn.experts.0.w1.weight"]; got != "absent" {
		t.Errorf("expert class = %q, want absent", got)
	}
	if got := block.Classes["layers.0.ffn.experts.0.w1.scale"]; got != "absent" {
		t.Errorf("expert scale class = %q, want absent", got)
	}
	if len(refs) != 1 {
		t.Fatalf("refusals = %+v, want exactly one", refs)
	}
	if string(refs[0].Reason) != "UNSUPPORTED_DTYPE" || refs[0].Axis != "cuda:expert_gemv(I8+F8_E8M0)" {
		t.Errorf("refusal = %s/%s, want UNSUPPORTED_DTYPE/cuda:expert_gemv(I8+F8_E8M0)", refs[0].Reason, refs[0].Axis)
	}
	var gap modelPreflightDeviceGap
	found := false
	for _, g := range block.Gaps {
		if g.Group == "cuda:expert_gemv(I8+F8_E8M0)" {
			gap = g
			found = true
		}
	}
	if !found || len(gap.Tensors) != 2 {
		t.Errorf("gap = %+v, want both expert tensors under the group", block.Gaps)
	}
}

// TestRunModelPreflightGGUFReadyAtSm103 pins the Q4_K/Q6_K/F32 GGUF fixture as READY at sm_103
// with the device leg: every tensor lands in a non-absent class.
func TestRunModelPreflightGGUFReadyAtSm103(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "fixture.gguf")
	writeLlamaGGUFTensorFixture(t, path, map[string]ggufTestTensor{
		"token_embd.weight":   {dims: []uint64{32, 4096}, typ: ggufload.TensorF32},
		"blk.0.attn_q.weight": {dims: []uint64{4096, 4096}, typ: ggufload.TensorQ4_K},
		"blk.0.ffn_up.weight": {dims: []uint64{8192, 4096}, typ: ggufload.TensorQ6_K},
		"output_norm.weight":  {dims: []uint64{4096}, typ: ggufload.TensorBF16},
	})
	var out, errOut bytes.Buffer
	code := runModelPreflight(&out, &errOut, []string{"--gguf", path, "--device", "cuda", "--arch", "sm_103", "--json"})
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr = %s stdout = %s", code, errOut.String(), out.String())
	}
	var rep modelPreflightReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("parse report: %v", err)
	}
	if rep.Verdict != "READY" {
		t.Errorf("verdict = %s, want READY; refusals=%v", rep.Verdict, rep.Refusals)
	}
	classes := rep.Device.Classes
	if got := classes["blk.0.attn_q.weight"]; got != "native" {
		t.Errorf("q4_k class = %q, want native", got)
	}
	if got := classes["blk.0.ffn_up.weight"]; got != "native" {
		t.Errorf("q6_k class = %q, want native", got)
	}
	if got := classes["output_norm.weight"]; got != "host" {
		t.Errorf("norm class = %q, want host", got)
	}
}

// TestRunModelPreflightGGUFQ3KTableDriven pins the table-injection contract: under the live
// table Q3_K now has a matmul kernel and classifies native; under the ticket's original
// 8-dtype envelope the same tensor has no kernel, is not retained, and expands to Q8_0.
func TestRunModelPreflightGGUFQ3KTableDriven(t *testing.T) {
	cfg := fakmodel.Config{ModelType: "llama", NumLayers: 2, HiddenSize: 4096, NumHeads: 32}
	info := ggufload.TensorInfo{Name: "blk.0.attn_q.weight", Dims: []uint64{4096, 4096}, Type: ggufload.TensorQ3_K}

	live := compute.CUDADeviceOpTable()
	class, group := classifyGGUFTensor(info, cfg, live, "sm_103", cudaStaticKQuantDecision(live, "sm_103").retained)
	if class != "native" || group != "" {
		t.Errorf("live table: q3_k class/group = %q/%q, want native/empty", class, group)
	}

	eight := eightDtypeTable(t)
	class, group = classifyGGUFTensor(info, cfg, eight, "sm_103", cudaStaticKQuantDecision(eight, "sm_103").retained)
	if class != "expanded:Q8_0" || group != "" {
		t.Errorf("8-dtype table: q3_k class/group = %q/%q, want expanded:Q8_0/empty", class, group)
	}
}

// TestRunModelPreflightUncoveredArchRefuses pins the fail-closed arch gate: sm_70 is outside
// the generic full build's embedded arches, so every tensor classifies absent behind one
// HARDWARE_ENVELOPE_INVALID refusal.
func TestRunModelPreflightUncoveredArchRefuses(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "fixture.gguf")
	writeLlamaGGUFTensorFixture(t, path, map[string]ggufTestTensor{
		"blk.0.attn_q.weight": {dims: []uint64{4096, 4096}, typ: ggufload.TensorQ4_K},
	})
	var out, errOut bytes.Buffer
	code := runModelPreflight(&out, &errOut, []string{"--gguf", path, "--device", "cuda", "--arch", "sm_70", "--json"})
	if code != 3 {
		t.Fatalf("exit = %d, want 3; stderr = %s", code, errOut.String())
	}
	var rep modelPreflightReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("parse report: %v", err)
	}
	if rep.Device.Coverage != "uncovered" {
		t.Errorf("arch_coverage = %q, want uncovered", rep.Device.Coverage)
	}
	if got := rep.Device.Classes["blk.0.attn_q.weight"]; got != "absent" {
		t.Errorf("class = %q, want absent", got)
	}
}

// TestRunModelPreflightUsageBothFormatsExit2 pins the mutual-exclusion usage error.
func TestRunModelPreflightUsageBothFormatsExit2(t *testing.T) {
	var out, errOut bytes.Buffer
	code := runModelPreflight(&out, &errOut, []string{"--gguf", "a.gguf", "--safetensors", "dir"})
	if code != 2 {
		t.Fatalf("exit = %d, want 2", code)
	}
	if !strings.Contains(errOut.String(), "usage:") {
		t.Errorf("stderr = %q, want usage text", errOut.String())
	}
}

// filterTableRows returns the declared rows kept by keep.
func filterTableRows(table []compute.CUDADeviceOp, keep func(op, format string) bool) []compute.CUDADeviceOp {
	var out []compute.CUDADeviceOp
	for _, row := range table {
		if keep(row.Operation, row.Format) {
			out = append(out, row)
		}
	}
	return out
}

// v41PlanFixture builds the shared V4.1-shaped checkpoint directory for direct plan tests.
func v41PlanFixture(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	safetensorsCLIFixture(t, dir)
	return dir
}
