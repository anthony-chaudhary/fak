//go:build cuda && cgo

package compute

import (
	"bytes"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"
)

// TestCUDAKernelProfileDecodeRows is the CUDA-host rung of the kernel timestamp profile
// (docs/cuda-dev.md section 6). Run without FAK_CUDA_TIMESTAMP_PROFILE, it checks that no
// reader is registered and re-runs itself as a child with the profile on and a stderr report
// every token boundary. The child runs 4 synthetic decode steps and requires named rows whose
// GPU time is positive, sums to GPUNS and stays within the decode window's wall time, with
// nothing truncated or untimed. Skips without a CUDA device.
//
// fak-test:runtime slow est=20s
func TestCUDAKernelProfileDecodeRows(t *testing.T) {
	cb := cudaOrSkip(t)
	if !KernelProfileReaderRegistered() {
		cmd := exec.Command(os.Args[0], "-test.run", "^TestCUDAKernelProfileDecodeRows$", "-test.count=1", "-test.v")
		cmd.Env = append(os.Environ(), cudaTimestampProfileEnv+"=1", "FAK_CUDA_TIMESTAMP_PROFILE_EVERY=1")
		var stderr bytes.Buffer
		cmd.Stdout, cmd.Stderr = &stderr, &stderr
		if err := cmd.Run(); err != nil {
			t.Fatalf("profiled child failed: %v\n%s", err, stderr.String())
		}
		out := stderr.String()
		for _, want := range []string{"fak-cuda ts-profile mode=per_launch_event_pairs", "timing=event_pairs graphs=disabled", "fak-cuda ts-stage kernel="} {
			if !strings.Contains(out, want) {
				t.Fatalf("profiled child stderr lacks %q:\n%s", want, out)
			}
		}
		return
	}

	if cb.GraphBegin() {
		cb.GraphAbort()
		t.Fatal("GraphBegin opened a capture while the kernel profile is on")
	}
	cfg, host := asyncWitnessCfg()
	m := newSynth(cb, cfg, host)
	next := argmaxF32(m.step(5)) // warm: pool, weight cache and the event ring
	cb.Recycle()
	before := KernelProfileSnapshot()
	start := time.Now()
	const steps = 4
	for i := 0; i < steps; i++ {
		next = argmaxF32(m.step(next))
		cb.Recycle()
	}
	wall := time.Since(start)
	after := KernelProfileSnapshot()
	d := KernelProfileDelta(before, after)
	if before == nil || after == nil || d == nil || d.Backend != "cuda" {
		t.Fatalf("profile window: before=%v after=%v delta=%+v", before != nil, after != nil, d)
	}
	if d.Batches != steps || d.Ops == 0 || len(d.Rows) == 0 || d.Truncated != 0 || d.UntimedCompute != 0 || d.BarriersElided != 0 {
		t.Fatalf("decode delta = %+v", d)
	}
	var sum float64
	for _, r := range d.Rows {
		if r.Kernel == "" || r.Kernel == "unknown" || r.Calls == 0 || !(r.NS > 0) {
			t.Fatalf("row %+v", r)
		}
		sum += r.NS
	}
	if diff := sum - d.GPUNS; diff > 1 || diff < -1 {
		t.Fatalf("rows sum %.0f ns, GPUNS %.0f ns", sum, d.GPUNS)
	}
	if d.GPUNS > float64(wall.Nanoseconds()) {
		t.Fatalf("GPU time %.0f ns exceeds the %v decode window", d.GPUNS, wall)
	}
	t.Logf("%d steps: %d timed launches, gpu %.3f ms, wall %v, untimed transfers %d, top row %s %dx%d %.3f ms",
		steps, d.Ops, d.GPUNS/1e6, wall, d.UntimedTransfer, d.Rows[0].Kernel, d.Rows[0].GX, d.Rows[0].GY, d.Rows[0].NS/1e6)
}
