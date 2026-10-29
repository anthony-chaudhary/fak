package agent

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// qwen38_27BPrefillConfig is the Qwen3.8-27B hybrid geometry served on raider-fak
// (fak#14097): 64 layers, every fourth full attention.
func qwen38_27BPrefillConfig() model.Config {
	types := make([]string, 64)
	for i := range types {
		types[i] = "linear_attention"
		if i%4 == 3 {
			types[i] = "full_attention"
		}
	}
	return model.Config{
		ModelType: "qwen3_5_text", NumLayers: 64, LayerTypes: types,
		NumHeads: 24, NumKVHeads: 4, HeadDim: 256,
		HiddenSize: 5120, IntermediateSize: 17408, VocabSize: 256,
		LinearConvKernelDim: 4,
		LinearNumKeyHeads:   16, LinearKeyHeadDim: 128,
		LinearNumValueHeads: 48, LinearValueHeadDim: 128,
	}
}

func TestSequencePrefillPeakBytesPricesSplitScratch(t *testing.T) {
	cfg := qwen38_27BPrefillConfig()
	peak := func(rows int, arm compute.CUDAQ4KPrefillArm) int64 {
		t.Helper()
		got, ok := qwenSequencePrefillPeakBytes(cfg, rows, arm)
		if !ok {
			t.Fatalf("peak(%d, %s) not priced", rows, arm)
		}
		return got
	}
	// The compensated arm's fp16 hi/lo weight halves of the widest FFN projection
	// are P-independent and appear only at P >= 128 (quant_matmul.cuh).
	weightSplit := int64(4) * int64(cfg.IntermediateSize) * int64(cfg.HiddenSize)
	if jump := peak(128, compute.CUDAQ4KPrefillCompensated) - peak(127, compute.CUDAQ4KPrefillCompensated); jump < weightSplit {
		t.Fatalf("compensated peak jump at 128 rows = %d, want >= weight split %d", jump, weightSplit)
	}
	if mmq, comp := peak(512, compute.CUDAQ4KPrefillMMQ), peak(512, compute.CUDAQ4KPrefillCompensated); mmq >= comp {
		t.Fatalf("mmq peak %d >= compensated peak %d at 512 rows", mmq, comp)
	}
	for _, arm := range []compute.CUDAQ4KPrefillArm{compute.CUDAQ4KPrefillCompensated, compute.CUDAQ4KPrefillMMQ, compute.CUDAQ4KPrefillFP8LT} {
		for rows := 2; rows <= 1024; rows *= 2 {
			if peak(rows, arm) <= peak(rows/2, arm) {
				t.Fatalf("peak(%d, %s) not above peak(%d)", rows, arm, rows/2)
			}
		}
	}
	if _, ok := qwenSequencePrefillPeakBytes(cfg, 0, compute.CUDAQ4KPrefillCompensated); ok {
		t.Fatal("zero rows priced")
	}
	bad := cfg
	bad.HeadDim = 0
	if _, ok := qwenSequencePrefillPeakBytes(bad, 128, compute.CUDAQ4KPrefillCompensated); ok {
		t.Fatal("invalid geometry priced")
	}
}

type headroomPrefillBackend struct {
	panelCapPrefillBackend
	free  int64
	known bool
}

func (b headroomPrefillBackend) Caps() compute.Caps {
	return compute.Caps{DeviceMemory: true, CapacityProbe: true}
}

func (b headroomPrefillBackend) DeviceMemory() (int64, int64, bool) {
	return 24463 << 20, b.free, b.known
}

func TestInKernelQwenQ4KPrefillChunkHeadroomDefault(t *testing.T) {
	cfg := qwen38_27BPrefillConfig()
	arm, err := compute.CUDAQ4KPrefillArmInUse()
	if err != nil {
		arm = compute.CUDAQ4KPrefillCompensated
	}
	newPlanner := func(free int64, known bool) *InKernelPlanner {
		backend := headroomPrefillBackend{panelCapPrefillBackend: panelCapPrefillBackend{Backend: compute.Default()}, free: free, known: known}
		return NewInKernelPlannerWithConfig(&model.Model{Cfg: cfg}, nil, "qwen-headroom", true, backend, false, InKernelPlannerConfig{})
	}
	checkFits := func(t *testing.T, free int64, chunk int) {
		t.Helper()
		budget := ApplyByteHeadroom(free, inKernelRequestDeviceHeadroom)
		peak, ok := qwenSequencePrefillPeakBytes(cfg, chunk, arm)
		if !ok || peak > budget {
			t.Fatalf("peak(%d) = %d, want <= headroom-margin %d", chunk, peak, budget)
		}
		if chunk < inKernelQwenQ4KPrefillChunkTokens {
			if next, _ := qwenSequencePrefillPeakBytes(cfg, chunk+1, arm); next <= budget {
				t.Fatalf("chunk %d not maximal: peak(%d) = %d fits %d", chunk, chunk+1, next, budget)
			}
		}
	}

	t.Run("unknown headroom keeps 512", func(t *testing.T) {
		for _, p := range []*InKernelPlanner{newPlanner(0, false), newPlanner(compute.FreeUnknown, true)} {
			if got := p.effectiveQwenQ4KPrefillChunkTokens(); got != inKernelQwenQ4KPrefillChunkTokens {
				t.Fatalf("effective chunk = %d, want %d", got, inKernelQwenQ4KPrefillChunkTokens)
			}
		}
	})
	t.Run("ample headroom keeps 512", func(t *testing.T) {
		if got := newPlanner(8<<30, true).effectiveQwenQ4KPrefillChunkTokens(); got != inKernelQwenQ4KPrefillChunkTokens {
			t.Fatalf("effective chunk = %d, want %d", got, inKernelQwenQ4KPrefillChunkTokens)
		}
	})
	t.Run("raider idle 1218 MiB fits its priced peak", func(t *testing.T) {
		const free = int64(1218) << 20
		checkFits(t, free, newPlanner(free, true).effectiveQwenQ4KPrefillChunkTokens())
	})
	t.Run("headroom below split scratch drops under 128", func(t *testing.T) {
		const free = int64(306) << 20 // first live free reading of the witnessed 500-token request
		got := newPlanner(free, true).effectiveQwenQ4KPrefillChunkTokens()
		if arm == compute.CUDAQ4KPrefillCompensated && got >= qwenQ4KSplitScratchRows {
			t.Fatalf("effective chunk = %d, want < %d to skip the compensated split", got, qwenQ4KSplitScratchRows)
		}
		if got < qwenQ4KPrefillHeadroomMinRows {
			t.Fatalf("effective chunk = %d, below floor %d", got, qwenQ4KPrefillHeadroomMinRows)
		}
		checkFits(t, free, got)
	})
	t.Run("nothing fits returns the floor", func(t *testing.T) {
		if got := newPlanner(1<<20, true).effectiveQwenQ4KPrefillChunkTokens(); got != qwenQ4KPrefillHeadroomMinRows {
			t.Fatalf("effective chunk = %d, want floor %d", got, qwenQ4KPrefillHeadroomMinRows)
		}
	})
	t.Run("explicit chunk is also bounded by headroom", func(t *testing.T) {
		backend := headroomPrefillBackend{panelCapPrefillBackend: panelCapPrefillBackend{Backend: compute.Default()}, free: 306 << 20, known: true}
		p := NewInKernelPlannerWithConfig(&model.Model{Cfg: cfg}, nil, "qwen-headroom", true, backend, false, InKernelPlannerConfig{QwenQ4KPrefillChunkTokens: 128})
		if got := p.effectiveQwenQ4KPrefillChunkTokens(); arm == compute.CUDAQ4KPrefillCompensated && got >= 128 {
			t.Fatalf("explicit 128 under 306 MiB headroom = %d, want < 128", got)
		}
	})
}
