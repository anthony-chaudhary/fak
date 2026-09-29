package ggufload

import (
	"bytes"
	"compress/gzip"
	"errors"
	"io"
	"os"
	"path/filepath"
	"testing"
)

// openPinnedUDQ2KXLHeaderSource parses the pinned 27B UD-Q2_K_XL header fixture (header only, no
// weight payload) into an estimate-only WeightSource.
func openPinnedUDQ2KXLHeaderSource(t *testing.T) *WeightSource {
	t.Helper()
	compressed, err := os.ReadFile(filepath.Join("testdata", "qwen38_ud_q2kxl_header.gguf.gz"))
	if err != nil {
		t.Fatal(err)
	}
	zr, err := gzip.NewReader(bytes.NewReader(compressed))
	if err != nil {
		t.Fatal(err)
	}
	defer zr.Close()
	raw, err := io.ReadAll(io.LimitReader(zr, 16<<20))
	if err != nil {
		t.Fatal(err)
	}
	gg, err := Read(bytes.NewReader(raw))
	if err != nil {
		t.Fatalf("Read pinned header: %v", err)
	}
	ws, err := NewWeightSource(gg, nil, 0)
	if err != nil {
		t.Fatalf("NewWeightSource: %v", err)
	}
	return ws
}

// The expanded bound must never undercharge what the resident-Q4K loader materializes for an
// expanding quant mix. On the pinned 27B UD-Q2_K_XL header the option-exact estimate prices the
// dense IQ/Q2_K/Q3_K matmul weights at their packed payload, while the loader's dequant->native-Q8
// branch (the WithDenseKQuantResident(false) outcome) holds ~30.87 GiB. The bound equals that
// expanded loader outcome byte-for-byte and dominates both the packed estimate and the on-disk
// payload.
func TestEstimateQ4KLoadExpandedChargesLoaderExpansionOnPinnedUDQ2KXL(t *testing.T) {
	ws := openPinnedUDQ2KXLHeaderSource(t)
	const gib = float64(1 << 30)

	disk, err := ws.EstimateLoadBytes()
	if err != nil {
		t.Fatal(err)
	}
	packed, err := ws.EstimateQ4KLoadMemoryPlan()
	if err != nil {
		t.Fatalf("option-exact estimate: %v", err)
	}
	loaderExpanded, err := ws.EstimateQ4KLoadMemoryPlan(WithDenseKQuantResident(false))
	if err != nil {
		t.Fatalf("loader expansion estimate: %v", err)
	}
	bound, err := ws.EstimateQ4KLoadExpandedMemoryPlan()
	if err != nil {
		t.Fatalf("expanded bound: %v", err)
	}

	if bound.Total() != loaderExpanded.Total() {
		t.Fatalf("expanded bound = %.3f GiB, want the loader's dequant->Q8 outcome %.3f GiB",
			float64(bound.Total())/gib, float64(loaderExpanded.Total())/gib)
	}
	if bound.Total() < disk {
		t.Fatalf("expanded bound %.3f GiB is below the on-disk payload %.3f GiB", float64(bound.Total())/gib, float64(disk)/gib)
	}
	if bound.Total() <= packed.Total() {
		t.Fatalf("expanded bound %.3f GiB must exceed the packed option-exact estimate %.3f GiB for an expanding quant mix",
			float64(bound.Total())/gib, float64(packed.Total())/gib)
	}
	// The witnessed resident for this artifact on a CPU host was 30.87 GiB.
	if got := float64(bound.Total()) / gib; got < 30.5 || got > 31.2 {
		t.Fatalf("expanded bound = %.3f GiB, want ~30.87 GiB (the witnessed expanded resident)", got)
	}
	for _, row := range bound {
		if row.Detail != q4kExpandedUpperBoundDetail {
			t.Fatalf("expanded bound row detail = %q, want %q", row.Detail, q4kExpandedUpperBoundDetail)
		}
	}
	// A caller that asked for packed dense residency still gets the expanded bound: the option
	// is forced off, so a backend-capability guess can never lower the sizing charge.
	withPacked, err := ws.EstimateQ4KLoadExpandedMemoryPlan(WithDenseKQuantResident(true), WithDenseQ2KResident(true), WithDenseQ6KResident(true))
	if err != nil {
		t.Fatal(err)
	}
	if withPacked.Total() != bound.Total() {
		t.Fatalf("expanded bound with packed options = %d, want %d", withPacked.Total(), bound.Total())
	}
}

// When the transformed-storage estimate is unqualified for a DENSE checkpoint, the historical
// fallback charged the on-disk payload -- a lower bound for every tensor the loader expands. The
// expanded bound charges the loader's element-based Q8/F32 rule instead, and never less than the
// raw payload.
func TestEstimateQ4KLoadExpandedUnsupportedDenseChargesExpansion(t *testing.T) {
	f := &File{
		Metadata: map[string]Value{
			"general.architecture":                   {Type: TypeString, Value: "qwen2"},
			"qwen2.context_length":                   {Type: TypeUint64, Value: uint64(4096)},
			"qwen2.embedding_length":                 {Type: TypeUint64, Value: uint64(256)},
			"qwen2.block_count":                      {Type: TypeUint64, Value: uint64(1)},
			"qwen2.feed_forward_length":              {Type: TypeUint64, Value: uint64(512)},
			"qwen2.attention.head_count":             {Type: TypeUint64, Value: uint64(4)},
			"qwen2.attention.head_count_kv":          {Type: TypeUint64, Value: uint64(2)},
			"qwen2.attention.layer_norm_rms_epsilon": {Type: TypeFloat32, Value: float32(1e-5)},
			"qwen2.rope.freq_base":                   {Type: TypeFloat32, Value: float32(10000)},
		},
		Tensors: []TensorInfo{
			// No canonical mapping: the Q4K estimate is unqualified for this checkpoint.
			{Name: "unmapped.sidecar", Dims: []uint64{256}, Type: TensorF32},
			// A dense Q2_K matmul weight the loader expands to native Q8.
			{Name: "blk.0.ffn_down.weight", Dims: []uint64{512, 256}, Type: TensorQ2_K},
		},
	}
	ws, err := NewWeightSource(f, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ws.EstimateQ4KLoadMemoryPlan(); !errors.Is(err, ErrQ4KLoadEstimateUnsupported) {
		t.Fatalf("fixture must be an unqualified Q4K route, got err=%v", err)
	}
	raw, err := ws.EstimateLoadMemoryPlan()
	if err != nil {
		t.Fatal(err)
	}
	q8, err := ws.EstimateQ8LoadMemoryPlan()
	if err != nil {
		t.Fatal(err)
	}
	bound, err := ws.EstimateQ4KLoadExpandedMemoryPlan()
	if err != nil {
		t.Fatal(err)
	}
	if q8.Total() <= raw.Total() {
		t.Fatalf("fixture must expand: q8=%d raw=%d", q8.Total(), raw.Total())
	}
	if bound.Total() != q8.Total() {
		t.Fatalf("unsupported dense expanded bound = %d, want the element-based expansion %d (raw payload %d undercharges)",
			bound.Total(), q8.Total(), raw.Total())
	}
}
