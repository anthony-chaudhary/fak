package ggufload

import (
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"runtime/metrics"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// Build() coalesces every F32 builder chunk into one arena while the chunks are still live, so
// the streamed estimate must charge that copy as a transient phase, not only the steady F32
// (fak#13798 item 1).
// fak-test:runtime fast est=10ms lane=default
func TestStreamedExpertHostLoadPeakChargesArenaCopy(t *testing.T) {
	ws, err := OpenWeights(writeDeepSeek41Q2KDenseFile(t))
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	defer ws.Close()
	peak, err := ws.EstimateStreamedExpertHostLoadPeak(WithDenseQ2KResident(true), WithStreamedExperts(0), WithStreamedDenseQ4KWorkingSet(1<<20))
	if err != nil {
		t.Fatalf("estimate: %v", err)
	}
	var f32 int64
	for _, r := range peak.Rows {
		f32 += r.F32Bytes
	}
	if f32 == 0 {
		t.Fatal("fixture carries no F32 tensor; the witness is vacuous")
	}
	if peak.ArenaCopyBytes != f32 {
		t.Fatalf("ArenaCopyBytes = %d, want every steady F32 byte %d", peak.ArenaCopyBytes, f32)
	}
	if peak.TransientBytes < peak.ArenaCopyBytes {
		t.Fatalf("TransientBytes = %d below the arena copy %d", peak.TransientBytes, peak.ArenaCopyBytes)
	}
	if peak.PeakAnonBytes != peak.SteadyAnonBytes+peak.TransientBytes {
		t.Fatalf("peak %d != steady %d + transient %d", peak.PeakAnonBytes, peak.SteadyAnonBytes, peak.TransientBytes)
	}
}

// A tied embedding stays F32 for the row gather and is also quantized into a Q8 LM head
// (quantizeDecodedFloatTensorIntoWithRetention), so both copies are steady (fak#13798 item 2).
// fak-test:runtime fast est=1ms lane=default
func TestStreamedTensorHostChargeTiedHeadQ8(t *testing.T) {
	info := TensorInfo{Name: "token_embd.weight", Dims: []uint64{64, 16}, Type: TensorF32}
	for _, tied := range []bool{false, true} {
		cfg := model.Config{ModelType: "llama", TieWordEmbeddings: tied}
		ch, err := streamedTensorHostCharge(cfg, q4kLoadOptions{}, info)
		if err != nil {
			t.Fatalf("tied=%v: %v", tied, err)
		}
		const f32 = 64 * 16 * 4
		if ch.f32 != f32 {
			t.Fatalf("tied=%v: f32 = %d, want %d", tied, ch.f32, f32)
		}
		wantQ8 := int64(0)
		if tied {
			wantQ8 = 64 * 16 / 32 * 36
		}
		if ch.q8 != wantQ8 {
			t.Fatalf("tied=%v: q8 = %d, want %d", tied, ch.q8, wantQ8)
		}
	}
}

// An F32 tensor whose canonical transform reallocates (the rotary unpermute) holds the dequant and
// the normalized copy at once; a pass-through tensor holds only the dequant (fak#13798 item 4).
// fak-test:runtime fast est=1ms lane=default
func TestStreamedTensorHostChargeCanonicalCopy(t *testing.T) {
	cfg := model.Config{ModelType: "llama"}
	const in, out = 48, 8 // in % 32 != 0 keeps q_proj on the F32 path
	const payload, f32 = in * out * 4, in * out * 4
	for _, tc := range []struct {
		name string
		want int64
	}{
		{"blk.0.attn_q.weight", payload + 2*f32},
		{"blk.0.attn_norm.weight", payload + f32},
	} {
		ch, err := streamedTensorHostCharge(cfg, q4kLoadOptions{}, TensorInfo{Name: tc.name, Dims: []uint64{in, out}, Type: TensorF32})
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		if ch.window != tc.want {
			t.Fatalf("%s: window = %d, want %d", tc.name, ch.window, tc.want)
		}
	}
}

// An MLA KV-b half is dequantized, copied, and buffered until its partner arrives; the pair's
// merge then holds both halves, the merged tensor and its normalized copy (fak#13798 item 3).
// fak-test:runtime fast est=1ms lane=default
func TestStreamedTensorHostChargeKVBBuffering(t *testing.T) {
	cfg := model.Config{ModelType: "glm_moe_dsa"}
	half := func(name string) streamedTensorCharge {
		t.Helper()
		ch, err := streamedTensorHostCharge(cfg, q4kLoadOptions{}, TensorInfo{Name: name, Dims: []uint64{16, 32, 4}, Type: TensorF32})
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		return ch
	}
	const f32 = 16 * 32 * 4 * 4
	k0, v0, k1 := half("blk.0.attn_k_b.weight"), half("blk.0.attn_v_b.weight"), half("blk.1.attn_k_b.weight")
	if k0.kvbLayer != 0 || k0.kvbF32 != f32 || k0.window != 3*f32 {
		t.Fatalf("k0 = %+v, want layer 0, %d buffered bytes, window raw+dequant+copy", k0, f32)
	}
	sim := kvbBufferSim{buffered: map[int]int64{}}
	w0, _ := sim.admit(k0)
	w1, _ := sim.admit(k1)
	w2, _ := sim.admit(v0)
	norm, _ := sim.admit(streamedTensorCharge{kvbLayer: -1, window: 7})
	if w0 != 3*f32+f32 {
		t.Fatalf("first half window = %d, want its window + its buffered copy", w0)
	}
	if w1 != 3*f32+2*f32 {
		t.Fatalf("second layer's first half window = %d, want both buffered copies", w1)
	}
	if w2 != 3*f32+f32+3*2*f32 {
		t.Fatalf("pair-completing window = %d, want own + other layer's buffered + 3x the pair", w2)
	}
	if norm != 7+f32 {
		t.Fatalf("unrelated tensor window = %d, want its own + the still-buffered layer-1 half", norm)
	}
}

// The RSS-sampled qualification witness: the real streamed-expert loader, run on a synthetic V4.1
// GGUF whose dense base mixes lazy Q2_K, an F32 embedding and a Q8 LM head, never holds more live
// Go heap than EstimateStreamedExpertHostLoadPeak predicts. GOGC=1 keeps the heap within ~1% of
// the live set, and a sampler records the heap high-water mark during the load.
// fak-test:runtime fast est=2s lane=default
func TestStreamedExpertLoadHeapPeakWithinEstimate(t *testing.T) {
	path := writeDeepSeek41WideDenseFile(t, 16384)
	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	t.Cleanup(func() { _ = ws.Close() })
	opts := []Q4KLoadOption{WithDenseQ2KResident(true), WithStreamedExperts(0), WithStreamedDenseQ4KWorkingSet(1 << 20)}
	peak, err := ws.EstimateStreamedExpertHostLoadPeak(opts...)
	if err != nil {
		t.Fatalf("estimate: %v", err)
	}
	var m *model.Model
	measured := sampleHeapHighWater(t, func() {
		m, err = ws.QuantModelQ4KProfileOptionsContext(t.Context(), nil, opts...)
	})
	if err != nil {
		t.Fatalf("streamed-expert load: %v", err)
	}
	runtime.KeepAlive(m)
	t.Logf("measured heap high-water %d B; estimate peak %d B (steady %d + transient %d; window %d, arena %d)",
		measured, peak.PeakAnonBytes, peak.SteadyAnonBytes, peak.TransientBytes, peak.WorkerWindowBytes, peak.ArenaCopyBytes)
	if peak.PeakAnonBytes < 16<<20 {
		t.Fatalf("estimate %d B is too small to dominate runtime noise; the witness is vacuous", peak.PeakAnonBytes)
	}
	if measured > peak.PeakAnonBytes {
		t.Fatalf("live heap high-water %d B exceeds the admitted streamed-expert estimate %d B", measured, peak.PeakAnonBytes)
	}
}

// sampleHeapHighWater runs fn under GOGC=1 and returns the largest heap-object growth over the
// pre-run baseline that a 20 µs sampler observed.
func sampleHeapHighWater(t *testing.T, fn func()) int64 {
	t.Helper()
	sample := []metrics.Sample{{Name: "/memory/classes/heap/objects:bytes"}}
	read := func() int64 {
		metrics.Read(sample)
		return int64(sample[0].Value.Uint64())
	}
	runtime.GC()
	base := read()
	defer debug.SetGCPercent(debug.SetGCPercent(1))
	var high atomic.Int64
	stop := make(chan struct{})
	done := make(chan struct{})
	go func() {
		defer close(done)
		s := []metrics.Sample{{Name: sample[0].Name}}
		for {
			metrics.Read(s)
			if v := int64(s[0].Value.Uint64()); v > high.Load() {
				high.Store(v)
			}
			select {
			case <-stop:
				return
			default:
				time.Sleep(20 * time.Microsecond)
			}
		}
	}()
	fn()
	close(stop)
	<-done
	return max(high.Load()-base, 0)
}

// writeDeepSeek41WideDenseFile is the Q2_K-dense V4.1 fixture with a vocab-wide F32 embedding and
// LM head, so the F32 arena copy and the Q8 head conversion dominate the host load.
func writeDeepSeek41WideDenseFile(t *testing.T, vocab uint64) string {
	t.Helper()
	src, err := OpenWeights(writeDeepSeek41Q2KDenseFile(t))
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	defer src.Close()
	var tensors []splitTensor
	for _, info := range src.File.Tensors {
		if info.Name == "token_embd.weight" || info.Name == "output.weight" {
			data := make([]float32, ds41HALHidden*vocab)
			for i := range data {
				data[i] = float32(i%97) * 1e-3
			}
			tensors = append(tensors, splitTensor{name: info.Name, dims: []uint64{ds41HALHidden, vocab}, typ: TensorF32, data: f32Payload(data...)})
			continue
		}
		raw, _, err := src.TensorBytes(info.Name)
		if err != nil {
			t.Fatalf("TensorBytes(%s): %v", info.Name, err)
		}
		tensors = append(tensors, splitTensor{name: info.Name, dims: info.Dims, typ: info.Type, data: append([]byte(nil), raw...)})
	}
	path := filepath.Join(t.TempDir(), "deepseek41-wide-dense.gguf")
	if err := os.WriteFile(path, writeDeepSeek41ExpertHALGGUF(t, tensors), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	return path
}
