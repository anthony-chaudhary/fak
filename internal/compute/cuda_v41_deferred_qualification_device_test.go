//go:build cuda && cgo && fak_cuda_v41_routed_mmid

package compute

import (
	"fmt"
	"math"
	"testing"
)

// fak-test:runtime integration est=1s lane=cuda
// One real-interface pipeline, not full numerical/model qualification. Requires
// the recovered ordinary native producers and real CUDA; no fallback backend.
func TestCUDAV41DeferredQualificationDevicePipeline(t *testing.T) {
	c := q3kCUDAOrFail(t)
	q := &cudaV41DeferredQualification{owner: c}
	if !q.SupportsV41DeferredStatus(4) {
		t.Fatal("private ordinary deferred modules/admission unavailable")
	}
	var deferred V41DeferredStatusBackend = q
	run := func() (err error) {
		var statuses []V41DeferredStatus
		var owned []Tensor
		defer func() {
			if recovered := recover(); recovered != nil {
				err = fmt.Errorf("qualification pipeline panic: %v", recovered)
			}
			if err != nil {
				// Preserve the primary error and settle records. On failure the
				// witness retains outputs rather than guessing completion.
				err = q.cancel(statuses, err)
				return
			}
			for _, tensor := range owned {
				c.Free(tensor)
			}
		}()
		g := []float32{0, 1, -1, 2}
		u := []float32{1, 1, 1, 1}
		gate := c.Upload(NewF32(Default(), []int{4}, g), F32)
		owned = append(owned, gate)
		up := c.Upload(NewF32(Default(), []int{4}, u), F32)
		owned = append(owned, up)
		mid, sw, err := deferred.V41RoutedSwiGLUDeferred(gate, up, 0, 1)
		if err != nil {
			return err
		}
		owned = append(owned, mid)
		statuses = append(statuses, sw)
		out, pub, err := deferred.V41BF16PublishDeferred(mid, V41BF16PublishRoutedDown)
		if err != nil {
			return err
		}
		owned = append(owned, out)
		statuses = append(statuses, pub)
		verdicts, rows, err := deferred.V41ReadDeferred(statuses, []Tensor{out})
		if err != nil {
			return err
		}
		if len(verdicts) != 2 || len(rows) != 1 || len(rows[0]) != 4 {
			return fmt.Errorf("wrong interface result extents: verdicts=%d rows=%d", len(verdicts), len(rows))
		}
		for _, verdict := range verdicts {
			if verdict != nil {
				return verdict
			}
		}
		for i, value := range rows[0] {
			want := float64(g[i]) / (1 + math.Exp(-float64(g[i])))
			if math.IsNaN(float64(value)) || math.Abs(float64(value)-want) > 0.02 || math.Float32bits(value)&0xffff != 0 {
				return fmt.Errorf("lane %d got %g want BF16 near %g", i, value, want)
			}
		}
		return nil
	}
	if err := run(); err != nil {
		t.Fatal(err)
	}
}
