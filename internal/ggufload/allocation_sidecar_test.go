package ggufload

import (
	"bytes"
	"crypto/sha256"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Pinned witness for the golden fixture. The full 34,821-byte source manifest
// at
// https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/raw/main/tensor-allocation/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS.rco-allocation.txt
// has sha256 305e14351d8f116f411e0370a9a6f73f0cde2941b7a4650b0fb35be00356a91b
// (verified 2026-10-28). The committed fixture is its header block plus the
// first 20 tensor rows; the structural assertions below are the contract, and
// the fixture's own digest pins it against accidental edits.
const goldenFixtureSHA256 = "510178891014b0c955f3e735eb86335d727cfe28cf52fe34e94d97ba5911f615"

func TestParseAllocationSidecarGoldenExcerpt(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("testdata", "rco_allocation_ista_ds41_iq2xxs_excerpt.txt"))
	if err != nil {
		t.Fatalf("read golden fixture: %v", err)
	}
	if got := fmt.Sprintf("%x", sha256.Sum256(data)); got != goldenFixtureSHA256 {
		t.Fatalf("fixture sha256 = %s, want pinned %s", got, goldenFixtureSHA256)
	}
	sc, err := ParseAllocationSidecar(bytes.NewReader(data))
	if err != nil {
		t.Fatalf("ParseAllocationSidecar(golden): %v", err)
	}
	if sc.Title != "# RCO per-tensor GGUF allocation" {
		t.Errorf("title = %q", sc.Title)
	}
	if sc.Repository != "ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF" {
		t.Errorf("repository = %q", sc.Repository)
	}
	shard := "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00006.gguf (first of 6 shards; this dump covers all of them)"
	if sc.GGUFShard != shard {
		t.Errorf("gguf shard = %q", sc.GGUFShard)
	}
	if len(sc.Sources) != 6 { // five IQ2_XXS shards + the shared Engram shard
		t.Errorf("sources = %d, want 6", len(sc.Sources))
	}
	if sc.TargetBPW != 2.00 || !strings.Contains(sc.TargetScope, "bpw over the backbone") {
		t.Errorf("target = %v scope %q", sc.TargetBPW, sc.TargetScope)
	}
	if sc.MeasuredBPW == nil || *sc.MeasuredBPW != 2.0038 {
		t.Errorf("measured bpw = %v, want 2.0038", sc.MeasuredBPW)
	}
	if sc.GGUFVersion == nil || *sc.GGUFVersion != 3 {
		t.Errorf("gguf version = %v", sc.GGUFVersion)
	}
	if sc.MetadataEntries == nil || *sc.MetadataEntries != 77 {
		t.Errorf("metadata entries = %v", sc.MetadataEntries)
	}
	if sc.TensorCount != 1043 {
		t.Errorf("tensor count = %d, want 1043", sc.TensorCount)
	}
	wantHistogram := map[TensorType]int{
		TensorBF16: 67, TensorF32: 531, TensorIQ1_M: 60,
		TensorIQ2_XS: 18, TensorIQ2_XXS: 42, TensorIQ4_NL: 1,
		TensorQ5_K: 244, TensorQ8_0: 80,
	}
	if fmt.Sprint(sc.Histogram) != fmt.Sprint(wantHistogram) {
		t.Errorf("histogram = %v, want %v", sc.Histogram, wantHistogram)
	}
	if len(sc.Tensors) != 20 {
		t.Fatalf("rows = %d, want 20 (fixture is an excerpt of the 1043-row manifest)", len(sc.Tensors))
	}
	if sc.Tensors[0] != (AllocationTensor{Name: "output.weight", Type: TensorQ5_K}) {
		t.Errorf("first row = %+v", sc.Tensors[0])
	}
	if sc.Tensors[19] != (AllocationTensor{Name: "blk.0.ffn_gate_shexp.weight", Type: TensorQ5_K}) {
		t.Errorf("last row = %+v", sc.Tensors[19])
	}
}

// allocationTestHeader renders a minimal valid manifest header; extra is
// appended verbatim before the body rows.
func allocationTestHeader(extra string) string {
	var b strings.Builder
	b.WriteString("# RCO per-tensor GGUF allocation\n#\n")
	b.WriteString("# Repository: Test/Repo\n# GGUF: test.gguf\n")
	b.WriteString("# Target whole-file bpw: 1.50\n# Tensor count: 2\n")
	b.WriteString("# Quant-type counts: F32=2\n")
	b.WriteString(extra)
	return b.String()
}

func TestParseAllocationSidecarRefusals(t *testing.T) {
	rows := "a.weight: F32\nb.weight: F32\n"
	tests := []struct {
		name    string
		input   string
		wantErr error
	}{
		{"wrong title", "# something else\n" + allocationTestHeader("") + rows, ErrAllocationSidecarMalformed},
		{"row without colon", allocationTestHeader("") + "foo bar\n", ErrAllocationSidecarMalformed},
		{"row with empty name", allocationTestHeader("") + ": F32\n", ErrAllocationSidecarMalformed},
		{"name with space", allocationTestHeader("") + "a b: F32\n", ErrAllocationSidecarMalformed},
		{"unknown type token", allocationTestHeader("") + "a.weight: Z9_9\n", ErrAllocationSidecarUnknownType},
		{"duplicate tensor", allocationTestHeader("") + "a.weight: F32\na.weight: F32\n", ErrAllocationSidecarMalformed},
		{"comment after body", allocationTestHeader("") + rows + "# late comment\n", ErrAllocationSidecarMalformed},
		{"missing target bpw", "# RCO per-tensor GGUF allocation\n#\n# Tensor count: 2\n# Quant-type counts: F32=2\n" + rows, ErrAllocationSidecarMissingField},
		{"missing quant-type counts", "# RCO per-tensor GGUF allocation\n#\n# Target whole-file bpw: 1.50\n# Tensor count: 2\n" + rows, ErrAllocationSidecarMissingField},
		{"missing tensor count", "# RCO per-tensor GGUF allocation\n#\n# Target whole-file bpw: 1.50\n# Quant-type counts: F32=2\n" + rows, ErrAllocationSidecarMissingField},
		{"no rows", allocationTestHeader(""), ErrAllocationSidecarEmpty},
		{"bad tensor count number", "# RCO per-tensor GGUF allocation\n#\n# Target whole-file bpw: 1.50\n# Tensor count: abc\n# Quant-type counts: F32=2\n" + rows, ErrAllocationSidecarMalformed},
		{"multiple target lines", allocationTestHeader("# Target whole-file bpw: 2.00\n") + rows, ErrAllocationSidecarMalformed},
		{"bad histogram pair", "# RCO per-tensor GGUF allocation\n#\n# Target whole-file bpw: 1.50\n# Tensor count: 2\n# Quant-type counts: F32=x\n" + rows, ErrAllocationSidecarMalformed},
		{"unknown histogram type", "# RCO per-tensor GGUF allocation\n#\n# Target whole-file bpw: 1.50\n# Tensor count: 2\n# Quant-type counts: ZED=1\n" + rows, ErrAllocationSidecarUnknownType},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ParseAllocationSidecar(strings.NewReader(tt.input))
			if !errors.Is(err, tt.wantErr) {
				t.Fatalf("err = %v, want errors.Is %v", err, tt.wantErr)
			}
		})
	}
}

func TestAllocationErrorToken(t *testing.T) {
	cases := []struct {
		err  error
		want string
	}{
		{fmt.Errorf("%w: line 3", ErrAllocationSidecarMalformed), "ALLOCATION_SIDECAR_MALFORMED"},
		{ErrAllocationSidecarMissingField, "ALLOCATION_SIDECAR_MISSING_FIELD"},
		{ErrAllocationSidecarEmpty, "ALLOCATION_SIDECAR_EMPTY"},
		{ErrAllocationSidecarUnknownType, "ALLOCATION_SIDECAR_UNKNOWN_TYPE"},
		{ErrAllocationDeclaredCountMismatch, "ALLOCATION_DECLARED_COUNT_MISMATCH"},
		{ErrAllocationCoverageMismatch, "ALLOCATION_COVERAGE_MISMATCH"},
		{ErrAllocationTypeMismatch, "ALLOCATION_TYPE_MISMATCH"},
		{ErrAllocationHistogramDrift, "ALLOCATION_HISTOGRAM_DRIFT"},
		{ErrAllocationBPWDrift, "ALLOCATION_BPW_DRIFT"},
		{ErrAllocationUnsupportedType, "ALLOCATION_UNSUPPORTED_TYPE"},
		{errors.New("something else"), "UNKNOWN"},
	}
	for _, c := range cases {
		if got := AllocationErrorToken(c.err); got != c.want {
			t.Errorf("AllocationErrorToken(%v) = %q, want %q", c.err, got, c.want)
		}
	}
}

type synthAllocTensor struct {
	name string
	dims []uint64
	typ  TensorType
}

// buildSyntheticGGUF renders a minimal v3 GGUF (header + tensor directory +
// zero-filled data section) through this package's own Read, so the validator
// exercises real parse results rather than hand-built slices.
func buildSyntheticGGUF(t *testing.T, rows []synthAllocTensor) []TensorInfo {
	t.Helper()
	var b bytes.Buffer
	writeMinimalHeader(&b, uint64(len(rows)), 0)
	offset := uint64(0)
	for _, r := range rows {
		tr, ok := tensorTypeTraits(r.typ)
		if !ok {
			t.Fatalf("type %v has no traits", r.typ)
		}
		n := uint64(1)
		for _, d := range r.dims {
			n *= d
		}
		payload := ((n + tr.BlockSize - 1) / tr.BlockSize) * tr.TypeSize
		offset = (offset + 31) &^ 31 // 32-byte payload alignment, as in real files
		writeTensorInfoForTest(&b, r.name, r.dims, r.typ, offset)
		offset += payload
	}
	padToAlignment(&b, 32)
	dataStart := b.Len()
	padToLen(&b, dataStart+int(offset))
	f, err := Read(bytes.NewReader(b.Bytes()))
	if err != nil {
		t.Fatalf("Read(synthetic GGUF): %v", err)
	}
	return f.Tensors
}

var synthRows = []synthAllocTensor{
	{"norm.weight", []uint64{4}, TensorF32},
	{"gate.weight", []uint64{4}, TensorBF16},
	{"ffn_down_exps.weight", []uint64{256}, TensorIQ1_M},
	{"attn_q.weight", []uint64{256}, TensorQ5_K},
}

// synthSidecar renders the matching manifest; target and hist are parameterized
// so negative cases can drift one field at a time.
func synthSidecar(target string, hist string, count int, drop, add string) string {
	var b strings.Builder
	b.WriteString("# RCO per-tensor GGUF allocation\n#\n")
	b.WriteString("# Repository: Test/Synthetic\n# GGUF: synthetic.gguf\n")
	fmt.Fprintf(&b, "# Target whole-file bpw: %s\n", target)
	fmt.Fprintf(&b, "# Tensor count: %d\n", count)
	fmt.Fprintf(&b, "# Quant-type counts: %s\n", hist)
	for _, r := range synthRows {
		if r.name == drop {
			continue
		}
		fmt.Fprintf(&b, "%s: %s\n", r.name, r.typ)
	}
	b.WriteString(add)
	return b.String()
}

const synthHist = "F32=1, BF16=1, IQ1_M=1, Q5_K=1"

func TestValidateAllocationSyntheticGGUFPositive(t *testing.T) {
	tensors := buildSyntheticGGUF(t, synthRows)
	if got := len(tensors); got != 4 {
		t.Fatalf("parsed tensors = %d, want 4", got)
	}
	sc, err := ParseAllocationSidecar(strings.NewReader(synthSidecar("3.94", synthHist, 4, "", "")))
	if err != nil {
		t.Fatalf("parse sidecar: %v", err)
	}
	if err := ValidateAllocation(sc, tensors, AllocationValidationOptions{}); err != nil {
		t.Fatalf("ValidateAllocation: %v", err)
	}
	// 8*(16+8+56+176)/520 = 2048/520.
	bpw, err := EffectiveBpw(tensors, nil)
	if err != nil {
		t.Fatalf("EffectiveBpw: %v", err)
	}
	if math.Abs(bpw-2048.0/520.0) > 1e-9 {
		t.Errorf("effective bpw = %v, want %v", bpw, 2048.0/520.0)
	}
}

func TestValidateAllocationRefusals(t *testing.T) {
	tensors := buildSyntheticGGUF(t, synthRows)
	tests := []struct {
		name    string
		sidecar string
		wantErr error
		contain string
	}{
		{
			name:    "declared count mismatch",
			sidecar: synthSidecar("3.94", synthHist, 5, "", ""),
			wantErr: ErrAllocationDeclaredCountMismatch,
			contain: "declared 5, listed 4",
		},
		{
			name:    "tensor only in sidecar",
			sidecar: synthSidecar("3.94", "F32=2, BF16=1, IQ1_M=1, Q5_K=1", 5, "", "ghost.weight: F32\n"),
			wantErr: ErrAllocationCoverageMismatch,
			contain: "ghost.weight",
		},
		{
			name:    "tensor missing from sidecar",
			sidecar: synthSidecar("3.94", "F32=1, IQ1_M=1, Q5_K=1", 3, "gate.weight", ""),
			wantErr: ErrAllocationCoverageMismatch,
			contain: "gate.weight",
		},
		{
			name:    "type mismatch",
			sidecar: sidecarWithGateAsF32(),
			wantErr: ErrAllocationTypeMismatch,
			contain: "gate.weight",
		},
		{
			name:    "histogram drift",
			sidecar: synthSidecar("3.94", "F32=2, BF16=1, IQ1_M=1, Q5_K=1", 4, "", ""),
			wantErr: ErrAllocationHistogramDrift,
			contain: "F32",
		},
		{
			name:    "bpw drift",
			sidecar: synthSidecar("5.00", synthHist, 4, "", ""),
			wantErr: ErrAllocationBPWDrift,
			contain: "tolerance",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sc, err := ParseAllocationSidecar(strings.NewReader(tt.sidecar))
			if err != nil {
				t.Fatalf("parse sidecar: %v", err)
			}
			err = ValidateAllocation(sc, tensors, AllocationValidationOptions{})
			if !errors.Is(err, tt.wantErr) {
				t.Fatalf("err = %v, want errors.Is %v", err, tt.wantErr)
			}
			if !strings.Contains(err.Error(), tt.contain) {
				t.Errorf("err = %q, want it to contain %q", err, tt.contain)
			}
		})
	}
}

func sidecarWithGateAsF32() string {
	s := synthSidecar("3.94", synthHist, 4, "", "")
	return strings.Replace(s, "gate.weight: BF16", "gate.weight: F32", 1)
}

func TestEffectiveBpwScopingAndRefusal(t *testing.T) {
	full := []TensorInfo{{Name: "x", Dims: []uint64{4}, Type: TensorF32}}
	if _, err := EffectiveBpw(full, func(name string) bool { return true }); !errors.Is(err, ErrAllocationUnsupportedType) {
		t.Errorf("exclude-all err = %v, want ErrAllocationUnsupportedType", err)
	}
	bad := []TensorInfo{{Name: "y", Dims: []uint64{4}, Type: TensorType(9999)}}
	if _, err := EffectiveBpw(bad, nil); !errors.Is(err, ErrAllocationUnsupportedType) {
		t.Errorf("unknown type err = %v, want ErrAllocationUnsupportedType", err)
	}
	all := buildSyntheticGGUF(t, synthRows)
	withAll, err := EffectiveBpw(all, nil)
	if err != nil {
		t.Fatalf("EffectiveBpw(all): %v", err)
	}
	// Excluding the two 256-element quant tensors leaves 16+8=24 bytes over 8
	// elements: 8*24/8 = 24 bits per weight.
	withExcluded, err := EffectiveBpw(all, func(name string) bool {
		return name == "ffn_down_exps.weight" || name == "attn_q.weight"
	})
	if err != nil {
		t.Fatalf("EffectiveBpw(excluded): %v", err)
	}
	if math.Abs(withExcluded-24.0) > 1e-9 || withExcluded == withAll {
		t.Errorf("excluded bpw = %v (all=%v), want exactly 24.0, distinct from the full scope", withExcluded, withAll)
	}
}

// TestLoadAllocationSidecarRoundTrip pins LoadAllocationSidecar against a temp
// file and the missing-path refusal.
func TestLoadAllocationSidecarRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "alloc.rco-allocation.txt")
	if err := os.WriteFile(path, []byte(allocationTestHeader("")+"a.weight: F32\nb.weight: F32\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	sc, err := LoadAllocationSidecar(path)
	if err != nil {
		t.Fatalf("LoadAllocationSidecar: %v", err)
	}
	if len(sc.Tensors) != 2 {
		t.Errorf("rows = %d, want 2", len(sc.Tensors))
	}
	if _, err := LoadAllocationSidecar(filepath.Join(dir, "nope")); err == nil {
		t.Error("missing path: want error, got nil")
	}
}
