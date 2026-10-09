package ggufload

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func writeStreamedExpertDenseFixture(t *testing.T) (path string, dense []splitTensor, experts []splitTensor) {
	t.Helper()
	const E, I, H = 4, 256, 256
	q4 := func(name string, dims ...uint64) splitTensor {
		return splitTensor{name: name, dims: dims, typ: TensorQ4_K, data: make([]byte, qwen3MoEPayloadBytes(TensorQ4_K, dims))}
	}
	dense = []splitTensor{
		q4("blk.0.ffn_gate_shexp.weight", H, I),
		q4("blk.0.ffn_up_shexp.weight", H, I),
		q4("blk.0.ffn_down_shexp.weight", I, H),
	}
	experts = []splitTensor{
		q4("blk.0.ffn_gate_exps.weight", H, I, E),
		q4("blk.0.ffn_up_exps.weight", H, I, E),
		q4("blk.0.ffn_down_exps.weight", I, H, E),
	}
	tensors := append([]splitTensor{{name: "blk.0.attn_norm.weight", dims: []uint64{4}, typ: TensorF32, data: f32Payload(1, 2, 3, 4)}}, dense...)
	tensors = append(tensors, experts...)
	path = filepath.Join(t.TempDir(), "ds41-streamed-dense-00001-of-00001.gguf")
	if err := os.WriteFile(path, writeDeepSeek41SplitShard(t, 1, 1, uint32(len(tensors)), true, tensors), 0o600); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	return path, dense, experts
}

// The device-ring route must not stage its dense base in host RAM: eligible dense tensors are
// charged at the declared working set, routed slabs not at all, and the transient at one tensor.
// fak-test:runtime fast est=5ms lane=default
func TestStreamedExpertHostLoadPeakBoundsDenseStaging(t *testing.T) {
	path, dense, experts := writeStreamedExpertDenseFixture(t)
	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	defer ws.Close()

	var denseBytes, largest, expertBytes int64
	for _, d := range dense {
		denseBytes += int64(len(d.data))
		largest = max(largest, int64(len(d.data)))
	}
	for _, e := range experts {
		expertBytes += int64(len(e.data))
	}

	if _, err := ws.EstimateStreamedExpertHostLoadPeak(WithStreamedExperts(0)); !errors.Is(err, ErrQ4KLoadEstimateUnsupported) {
		t.Fatalf("unbounded dense estimate error = %v, want ErrQ4KLoadEstimateUnsupported", err)
	}

	peak := func(bound int64) HostLoadPeak {
		t.Helper()
		p, err := ws.EstimateStreamedExpertHostLoadPeak(WithStreamedExperts(0), WithStreamedExpertDeviceRing(1<<30), WithStreamedDenseQ4KWorkingSet(bound))
		if err != nil {
			t.Fatalf("estimate(bound=%d): %v", bound, err)
		}
		return p
	}
	zero := peak(0)
	if zero.SteadyAnonBytes >= denseBytes {
		t.Fatalf("stream-through steady %d charges the dense base (%d bytes)", zero.SteadyAnonBytes, denseBytes)
	}
	if zero.PeakAnonBytes >= denseBytes+expertBytes {
		t.Fatalf("peak %d charges the dense base and routed slabs (%d bytes)", zero.PeakAnonBytes, denseBytes+expertBytes)
	}
	if zero.TransientBytes < largest || zero.TransientBytes >= denseBytes {
		t.Fatalf("transient %d, want one dense tensor's staging in [%d, %d)", zero.TransientBytes, largest, denseBytes)
	}
	var streamedRow int64
	for _, r := range zero.Rows {
		streamedRow += r.StreamBytes
	}
	if streamedRow != denseBytes {
		t.Fatalf("streamed dense bytes = %d, want every eligible dense byte %d", streamedRow, denseBytes)
	}
	bounded := peak(denseBytes)
	if err := AdmitHostLoadPeak(bounded, bounded.PeakAnonBytes, true, 0); err != nil {
		t.Fatalf("peak within budget refused: %v", err)
	}
	if err := AdmitHostLoadPeak(bounded, bounded.PeakAnonBytes-1, true, 0); !errors.Is(err, ErrHostLoadPeakTooBig) {
		t.Fatalf("peak over budget: err = %v, want ErrHostLoadPeakTooBig", err)
	}
	for _, bound := range []int64{largest, denseBytes, 4 * denseBytes} {
		got := peak(bound).SteadyAnonBytes - zero.SteadyAnonBytes
		if want := min(bound, denseBytes); got != want {
			t.Fatalf("bound %d charged %d dense working-set bytes, want %d", bound, got, want)
		}
	}
}

// The loader half: on the streamed-expert route with a bounded dense working set, every dense
// tensor the estimate charges as a range is a lazy checkpoint range after the real load, so the
// load stages no dense payload on the host.
// fak-test:runtime fast est=300ms lane=default
func TestStreamedExpertLoadKeepsDenseBaseLazy(t *testing.T) {
	ws, err := OpenWeights(writeDeepSeek41Q2KDenseFile(t))
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	t.Cleanup(func() { _ = ws.Close() })
	opts := []Q4KLoadOption{WithDenseQ2KResident(true), WithStreamedExperts(0), WithStreamedDenseQ4KWorkingSet(1 << 20)}
	m, err := ws.QuantModelQ4KProfileOptionsContext(t.Context(), nil, opts...)
	if err != nil {
		t.Fatalf("streamed-expert load: %v", err)
	}
	cfg, err := ws.File.Config()
	if err != nil {
		t.Fatalf("Config: %v", err)
	}
	o, err := resolveQ4KLoadOptions(cfg, opts)
	if err != nil {
		t.Fatalf("resolve options: %v", err)
	}
	lazy := 0
	for _, info := range ws.File.Tensors {
		canon, ok := CanonicalTensorNameArch(info.Name, cfg.ModelType)
		if !ok {
			continue
		}
		if canon, ok = model.QuantSourceTensorName(cfg, canon); !ok || !streamedDenseLazy(cfg, o, info.Type, canon) {
			continue
		}
		lazy++
		if !m.Q4KLazy(canon) && !m.KQuantLazy(canon) {
			t.Errorf("%s (%s) was materialized on the host, want a lazy checkpoint range", info.Name, canon)
		}
	}
	if lazy == 0 {
		t.Fatal("fixture carries no streamed-dense eligible tensor; the witness is vacuous")
	}
	peak, err := ws.EstimateStreamedExpertHostLoadPeak(opts...)
	if err != nil {
		t.Fatalf("estimate: %v", err)
	}
	if peak.PeakAnonBytes <= 0 {
		t.Fatalf("peak = %+v, want a positive bounded charge", peak)
	}
}

// writeDeepSeek41Q2KDenseFile is the mixed-quant V4.1 HAL fixture with its shared-expert
// projections re-encoded as Q2_K, so the dense base carries streamed-dense eligible tensors.
func writeDeepSeek41Q2KDenseFile(t *testing.T) string {
	t.Helper()
	src, err := OpenWeights(writeDeepSeek41ExpertHALFile(t))
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	defer src.Close()
	var tensors []splitTensor
	for _, info := range src.File.Tensors {
		if strings.HasSuffix(info.Name, "_shexp.weight") {
			tensors = append(tensors, splitTensor{name: info.Name, dims: info.Dims, typ: TensorQ2_K,
				data: ds41HALExpertSlab(TensorQ2_K, int(info.Dims[1]), int(info.Dims[0]), 1)})
			continue
		}
		raw, _, err := src.TensorBytes(info.Name)
		if err != nil {
			t.Fatalf("TensorBytes(%s): %v", info.Name, err)
		}
		tensors = append(tensors, splitTensor{name: info.Name, dims: info.Dims, typ: info.Type, data: append([]byte(nil), raw...)})
	}
	path := filepath.Join(t.TempDir(), "deepseek41-q2k-dense.gguf")
	if err := os.WriteFile(path, writeDeepSeek41ExpertHALGGUF(t, tensors), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	return path
}

// A working set sized from the headroom after the fixed peak admits exactly when the fixed peak
// fits, and refuses typed when it does not.
// fak-test:runtime fast est=1ms lane=default
func TestStreamedDenseWorkingSetFromHeadroomAdmitsOnlyFittingFixedPeak(t *testing.T) {
	const gib = int64(1 << 30)
	withBound := func(fixed HostLoadPeak, bound int64) HostLoadPeak {
		fixed.SteadyAnonBytes += bound
		fixed.PeakAnonBytes += bound
		return fixed
	}
	fits := HostLoadPeak{SteadyAnonBytes: 3 * gib, TransientBytes: gib, PeakAnonBytes: 4 * gib}
	bound := StreamedDenseWorkingSetFromHeadroom(fits, 28*gib, 2*gib, 0.10)
	if bound <= 0 {
		t.Fatalf("bound = %d, want a positive working set", bound)
	}
	if err := AdmitHostLoadPeak(withBound(fits, bound), 28*gib, true, 2*gib); err != nil {
		t.Fatalf("fixed peak inside the budget refused: %v", err)
	}
	over := HostLoadPeak{SteadyAnonBytes: 63 * gib, TransientBytes: gib, PeakAnonBytes: 64 * gib}
	bound = StreamedDenseWorkingSetFromHeadroom(over, 28*gib, 2*gib, 0.10)
	if bound != 1 {
		t.Fatalf("over-budget bound = %d, want the 1-byte floor", bound)
	}
	if err := AdmitHostLoadPeak(withBound(over, bound), 28*gib, true, 2*gib); !errors.Is(err, ErrHostLoadPeakTooBig) {
		t.Fatalf("fixed peak over the budget: err = %v, want ErrHostLoadPeakTooBig", err)
	}
}
