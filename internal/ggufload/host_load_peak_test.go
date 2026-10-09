package ggufload

import (
	"errors"
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func hostPeakRow(t *testing.T, peak HostLoadPeak, typ string) HostLoadTypeRow {
	t.Helper()
	for _, row := range peak.Rows {
		if row.Type == typ {
			return row
		}
	}
	t.Fatalf("missing %s row in %+v", typ, peak.Rows)
	return HostLoadTypeRow{}
}

func hostPeakFixture(t *testing.T) *WeightSource {
	t.Helper()
	f := &File{
		Metadata: map[string]Value{
			"general.architecture":                   {Type: TypeString, Value: "llama"},
			"llama.embedding_length":                 {Type: TypeUint32, Value: uint32(256)},
			"llama.block_count":                      {Type: TypeUint32, Value: uint32(1)},
			"llama.attention.head_count":             {Type: TypeUint32, Value: uint32(1)},
			"llama.feed_forward_length":              {Type: TypeUint32, Value: uint32(256)},
			"llama.attention.layer_norm_rms_epsilon": {Type: TypeFloat32, Value: float32(1e-5)},
		},
		Tensors: []TensorInfo{
			{Name: "token_embd.weight", Dims: []uint64{256, 8}, Type: TensorQ2_K},
			{Name: "output.weight", Dims: []uint64{256, 8}, Type: TensorQ2_K, Offset: 1 << 20},
			{Name: "blk.0.ffn_up.weight", Dims: []uint64{256, 256}, Type: TensorQ4_K},
			{Name: "blk.0.ffn_down.weight", Dims: []uint64{256, 256}, Type: TensorIQ3_XXS},
			{Name: "blk.0.attn_v.weight", Dims: []uint64{256, 256}, Type: TensorQ3_K},
			{Name: "output_norm.weight", Dims: []uint64{256}, Type: TensorF32},
		},
	}
	ws, err := NewWeightSource(f, nil, 0)
	if err != nil {
		t.Fatalf("NewWeightSource: %v", err)
	}
	return ws
}

func hostPeakOptions() []Q4KLoadOption {
	return []Q4KLoadOption{
		WithDenseKQuantResident(false),
		WithDenseQ2KResident(true),
		WithDenseIQResident(NativeIQResidentTypes(compute.IsRawIQ)...),
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestEstimateQ4KHostLoadPeakAccountsStorageAndTransientPhases(t *testing.T) {
	t.Setenv("FAK_GGUF_LOAD_WORKERS", "2")
	ws := hostPeakFixture(t)
	const (
		embedPayload = int64(8 * 84)
		embedF32     = int64(8 * 256 * 4)
		q4Payload    = int64(256 * 144)
		iq3Payload   = int64(256 * 98)
		q3Payload    = int64(256 * 110)
		matrixF32    = int64(256 * 256 * 4)
		q8Bytes      = int64(256*256 + 256*8*4)
		normF32      = int64(256 * 4)
	)

	owned, err := ws.EstimateQ4KHostLoadPeak(false, hostPeakOptions()...)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]HostLoadTypeRow{
		"Q2_K":    {Type: "Q2_K", Tensors: 2, PayloadBytes: 2 * embedPayload, PackedBytes: embedPayload, F32Bytes: embedF32},
		"Q4_K":    {Type: "Q4_K", Tensors: 1, PayloadBytes: q4Payload, PackedBytes: q4Payload},
		"IQ3_XXS": {Type: "IQ3_XXS", Tensors: 1, PayloadBytes: iq3Payload, PackedBytes: iq3Payload},
		"Q3_K":    {Type: "Q3_K", Tensors: 1, PayloadBytes: q3Payload, Q8Bytes: q8Bytes},
		"F32":     {Type: "F32", Tensors: 1, PayloadBytes: normF32, F32Bytes: normF32},
	}
	for typ, expected := range want {
		if got := hostPeakRow(t, owned, typ); got != expected {
			t.Errorf("%s row = %+v, want %+v", typ, got, expected)
		}
	}
	steady := embedF32 + embedPayload + q4Payload + iq3Payload + q8Bytes + normF32
	window := iq3Payload + q3Payload + 2*matrixF32
	arena := embedF32 + normF32
	if owned.SteadyAnonBytes != steady || owned.WorkerWindowBytes != window || owned.ArenaCopyBytes != arena || owned.TransientBytes != window || owned.PeakAnonBytes != steady+window || owned.MappedBytes != 0 || owned.Workers != 2 {
		t.Fatalf("owned peak = %+v, want steady=%d window=%d arena=%d", owned, steady, window, arena)
	}

	mapped, err := ws.EstimateQ4KHostLoadPeak(true, hostPeakOptions()...)
	if err != nil {
		t.Fatal(err)
	}
	if got := hostPeakRow(t, mapped, "Q4_K"); got.MappedBytes != q4Payload || got.PackedBytes != 0 {
		t.Fatalf("mapped Q4_K row = %+v", got)
	}
	if got := hostPeakRow(t, mapped, "IQ3_XXS"); got.MappedBytes != 0 || got.PackedBytes != iq3Payload {
		t.Fatalf("mapped IQ3_XXS row = %+v", got)
	}
	if mapped.SteadyAnonBytes != steady-q4Payload-embedPayload || mapped.MappedBytes != q4Payload+embedPayload {
		t.Fatalf("mapped steady/file bytes = %d/%d, want %d/%d", mapped.SteadyAnonBytes, mapped.MappedBytes, steady-q4Payload-embedPayload, q4Payload+embedPayload)
	}
	for _, peak := range []HostLoadPeak{owned, mapped} {
		var anon, fileBacked int64
		for _, row := range peak.Rows {
			anon += row.AnonBytes()
			fileBacked += row.MappedBytes
		}
		if anon != peak.SteadyAnonBytes || fileBacked != peak.MappedBytes || peak.TransientBytes != max(peak.WorkerWindowBytes, peak.ArenaCopyBytes) || peak.PeakAnonBytes != peak.SteadyAnonBytes+peak.TransientBytes {
			t.Fatalf("peak accounting is inconsistent: %+v", peak)
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestEstimateQ4KHostLoadPeakRefusesUnboundedOrAlternateLoadRoutes(t *testing.T) {
	ws := hostPeakFixture(t)
	for name, opts := range map[string][]Q4KLoadOption{
		"unbounded dense":  {WithStreamedDenseQ4K(true)},
		"streamed experts": {WithStreamedExperts(0)},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := ws.EstimateQ4KHostLoadPeak(false, opts...); !errors.Is(err, ErrQ4KLoadEstimateUnsupported) {
				t.Fatalf("error = %v, want ErrQ4KLoadEstimateUnsupported", err)
			}
		})
	}
	t.Run("expert shard", func(t *testing.T) {
		moe := moeKQuantStreamedTestWeightSource(t, q2kDenseEligibleBytes)
		if _, err := moe.EstimateQ4KHostLoadPeak(false, WithExpertShard(0, 1)); !errors.Is(err, ErrQ4KLoadEstimateUnsupported) {
			t.Fatalf("error = %v, want ErrQ4KLoadEstimateUnsupported", err)
		}
	})
	t.Run("MoE fused expert split", func(t *testing.T) {
		moe := moeKQuantStreamedTestWeightSource(t, q2kDenseEligibleBytes)
		if _, err := moe.EstimateQ4KHostLoadPeak(false); !errors.Is(err, ErrQ4KLoadEstimateUnsupported) {
			t.Fatalf("error = %v, want ErrQ4KLoadEstimateUnsupported", err)
		}
	})
	t.Run("W3 selection", func(t *testing.T) {
		t.Setenv("FAK_W3_MLP", "1")
		if _, err := ws.EstimateQ4KHostLoadPeak(false); !errors.Is(err, ErrQ4KLoadEstimateUnsupported) {
			t.Fatalf("error = %v, want ErrQ4KLoadEstimateUnsupported", err)
		}
	})
}

// fak-test:runtime fast est=2ms lane=default
func TestAdmitHostLoadPeakUsesDefaultBoundaryAndCheckedArithmetic(t *testing.T) {
	const gib = int64(1 << 30)
	peak := HostLoadPeak{PeakAnonBytes: 16 * gib, SteadyAnonBytes: 12 * gib, TransientBytes: 4 * gib, Rows: []HostLoadTypeRow{{Type: "Q2_K", F32Bytes: 5 * gib}}}
	if HostLoadPeakDefaultMarginBytes != 2*gib {
		t.Fatalf("default margin = %d, want %d", HostLoadPeakDefaultMarginBytes, 2*gib)
	}
	if err := AdmitHostLoadPeak(peak, 18*gib, true, HostLoadPeakDefaultMarginBytes); err != nil {
		t.Fatalf("exact boundary refused: %v", err)
	}
	if err := AdmitHostLoadPeak(peak, 0, false, HostLoadPeakDefaultMarginBytes); err != nil {
		t.Fatalf("unknown availability refused: %v", err)
	}
	if err := AdmitHostLoadPeak(peak, compute.FreeUnknown, true, HostLoadPeakDefaultMarginBytes); err != nil {
		t.Fatalf("unprobeable availability refused: %v", err)
	}
	err := AdmitHostLoadPeak(peak, 18*gib-1, true, HostLoadPeakDefaultMarginBytes)
	if !errors.Is(err, ErrHostLoadPeakTooBig) {
		t.Fatalf("over-budget error = %v, want ErrHostLoadPeakTooBig", err)
	}
	var typed *HostLoadPeakError
	if !errors.As(err, &typed) || typed.NeedBytes != peak.PeakAnonBytes || typed.MarginBytes != HostLoadPeakDefaultMarginBytes {
		t.Fatalf("typed refusal = %+v", typed)
	}
	overflow := HostLoadPeak{PeakAnonBytes: math.MaxInt64 - 7}
	if err := AdmitHostLoadPeak(overflow, math.MaxInt64, true, 8); !errors.Is(err, ErrHostLoadPeakTooBig) {
		t.Fatalf("overflowing peak+margin error = %v, want ErrHostLoadPeakTooBig", err)
	}
}
