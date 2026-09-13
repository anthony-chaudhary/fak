package ggufload

import (
	"bytes"
	"io"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

type mtpEstimateReadCounter struct {
	io.ReaderAt
	reads atomic.Int64
}

func (r *mtpEstimateReadCounter) ReadAt(p []byte, off int64) (int, error) {
	r.reads.Add(1)
	return r.ReaderAt.ReadAt(p, off)
}

func TestEstimateQ4KRetainedMTPMatchesLoadedWeights(t *testing.T) {
	globalBefore := model.RetainMTP
	t.Cleanup(func() {
		if model.RetainMTP != globalBefore {
			t.Errorf("estimate/load changed RetainMTP from %v to %v", globalBefore, model.RetainMTP)
		}
	})
	// Independent storage arithmetic for qwen38MTPFixture's h=vocab=256:
	// Q8 native scales occupy four bytes, unlike GGUF's two-byte wire scales.
	const targetQ6 = 256 * 256 / 256 * 210
	const mtpQ6 = (128 + 256) * 256 / 256 * 210
	const mtpQ4 = (512 + 128 + 3*256) * 256 / 256 * 144
	const mtpQ8 = 256 * 512 / 32 * (32 + 4)
	const mtpF32 = (5*256 + 2*64) * 4
	const retained = targetQ6 + mtpQ6 + mtpQ4 + mtpQ8 + mtpF32
	var target []byte
	for _, retain := range []bool{true, false, true} {
		ws := qwen38MTPFixture(t, "", nil)
		counter := &mtpEstimateReadCounter{ReaderAt: ws.r}
		ws.r = counter
		opts := []Q4KLoadOption{WithMTPRetention(retain)}
		plan, estimateErr := ws.EstimateQ4KLoadMemoryPlan(opts...)
		if n := counter.reads.Load(); n != 0 {
			t.Errorf("retain=%v estimate read payload %d times", retain, n)
		}
		m, err := ws.QuantModelQ4KProfileOptions(nil, opts...)
		if err != nil {
			t.Fatalf("retain=%v actual load: %v", retain, err)
		}
		t.Cleanup(func() { _ = m.CloseWeights() })
		if counter.reads.Load() == 0 {
			t.Fatal("actual loader did not read fixture payload")
		}
		want := int64(targetQ6)
		r := m.ResidentReport()
		if retain {
			want = retained
			if r.Q4KBytes != mtpQ4 || r.Q8Bytes != mtpQ8 || r.KQuantBytes != targetQ6+mtpQ6 || r.F32Bytes != mtpF32 {
				t.Errorf("retained format buckets=%+v, want Q4=%d Q8=%d K=%d F32=%d", r, mtpQ4, mtpQ8, targetQ6+mtpQ6, mtpF32)
			}
		} else if r.Q4KBytes != 0 || r.Q8Bytes != 0 || r.F32Bytes != 0 || r.KQuantBytes != targetQ6 {
			t.Errorf("dropped format buckets=%+v, want only target Q6=%d", r, targetQ6)
		}
		if r.TotalResidentBytes != want {
			t.Errorf("retain=%v actual stored=%d, independent count=%d", retain, r.TotalResidentBytes, want)
		}
		if estimateErr != nil {
			t.Errorf("retain=%v estimate failed for supported actual load: %v", retain, estimateErr)
		} else if plan.Total() != want || plan.Total() != r.TotalResidentBytes {
			t.Errorf("retain=%v estimated=%d, actual stored=%d, independent count=%d", retain, plan.Total(), r.TotalResidentBytes, want)
		}
		_, layoutErr := m.Qwen38MTPTensorLayout()
		if (layoutErr == nil) != retain {
			t.Errorf("retain=%v complete layout error=%v", retain, layoutErr)
		}
		raw, ok := m.KQuantRaw("lm_head.weight")
		if !ok {
			t.Fatal("actual target head missing")
		}
		if target == nil {
			target = append([]byte(nil), raw...)
		} else if !bytes.Equal(raw, target) {
			t.Error("retention policy changed target bytes")
		}
		if model.RetainMTP != globalBefore {
			t.Error("retention policy mutated the global")
		}
	}
	for _, defect := range []string{"missing role", "wrong FC type", "wrong norm shape"} {
		t.Run(defect, func(t *testing.T) {
			for _, retain := range []bool{true, false} {
				omit := ""
				var override map[string]TensorType
				if defect == "missing role" {
					omit = "blk.1.ffn_down.weight"
				} else if defect == "wrong FC type" {
					override = map[string]TensorType{"blk.1.nextn.eh_proj.weight": TensorQ4_K}
				}
				ws := qwen38MTPFixture(t, omit, override)
				if defect == "wrong norm shape" {
					for i := range ws.File.Tensors {
						if ws.File.Tensors[i].Name == "blk.1.nextn.enorm.weight" {
							ws.File.Tensors[i].Dims = []uint64{128}
						}
					}
				}
				counter := &mtpEstimateReadCounter{ReaderAt: ws.r}
				ws.r = counter
				_, estimateErr := ws.EstimateQ4KLoadMemoryPlan(WithMTPRetention(retain))
				if counter.reads.Load() != 0 {
					t.Error("malformed fixture estimate read payload")
				}
				m, loadErr := ws.QuantModelQ4KProfileOptions(nil, WithMTPRetention(retain))
				if m != nil {
					_ = m.CloseWeights()
				}
				if (estimateErr != nil) != retain || (loadErr != nil) != retain {
					t.Errorf("retain=%v estimate error=%v, actual load error=%v; both must reject only retained malformed head", retain, estimateErr, loadErr)
				}
			}
		})
	}
}
