package model

import (
	"net"
	"sync"
	"testing"
)

// pipeline_worker_test.go — the band-running worker serve loop gate (#30).
//
// pipeline_test.go's TestTCPTransportMatchesLocal proves the TRANSPORT is
// interchangeable with the in-process path, but its peer is EchoFrames: the driver
// (RunPipelineWith) still runs EVERY band in its own process, so PP is a proven
// substrate, not a running multi-node engine. These tests close that gap. The remote
// bands run ONLY inside ServeBand worker loops on the far end of a real loopback TCP
// socket; the driver (RunPipelineAcrossWorkers) runs only the first stage. A
// >=2-stage run is therefore a genuine cross-process pipeline, and its logits are
// asserted bit-identical (max|Δ|=0) to the monolithic Forward — the shipped
// correctness contract, now carried across the wire rather than across a slice.
//
// "Cross-process" here is loopback goroutines + real OS sockets, the same standard
// the shipped TCPTransport test uses; separate OS processes are the deployment form
// of the identical socket path.

// runAcrossWorkers wires stages[1:] as ServeBand workers chained over loopback TCP
// and drives stages[0] locally through RunPipelineAcrossWorkers, returning the last
// stage's logits. Each worker runs its band ONLY in its own goroutine; the driver
// never re-runs a remote band. Worker errors (other than a clean EOF on shutdown)
// are surfaced through t after the run completes.
func runAcrossWorkers(t *testing.T, ids []int, stages []PipelineStage) [][]float32 {
	t.Helper()
	if len(stages) < 2 {
		t.Fatalf("runAcrossWorkers needs >=2 stages, got %d", len(stages))
	}
	workers := stages[1:]

	// One listener per remote worker, all created before any goroutine dials, so the
	// cascade of accept-then-dial down the chain cannot miss a not-yet-listening peer.
	lns := make([]net.Listener, len(workers))
	for i := range workers {
		ln, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			t.Fatalf("listen worker %d: %v", i, err)
		}
		defer ln.Close()
		lns[i] = ln
	}

	var wg sync.WaitGroup
	werr := make(chan error, len(workers))
	for i := range workers {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			connUp, aerr := lns[i].Accept()
			if aerr != nil {
				werr <- aerr
				return
			}
			defer connUp.Close()
			var downstream StageTransport
			if i < len(workers)-1 {
				connDown, derr := net.Dial("tcp", lns[i+1].Addr().String())
				if derr != nil {
					werr <- derr
					return
				}
				defer connDown.Close()
				downstream = NewTCPTransport(connDown)
			}
			werr <- ServeBand(connUp, workers[i], downstream)
		}(i)
	}

	conn, err := net.Dial("tcp", lns[0].Addr().String())
	if err != nil {
		t.Fatalf("dial first worker: %v", err)
	}
	logits, err := RunPipelineAcrossWorkers(ids, stages[0], NewTCPTransport(conn))
	if err != nil {
		conn.Close()
		wg.Wait()
		t.Fatalf("RunPipelineAcrossWorkers: %v", err)
	}
	// Closing the head connection cascades EOF down the chain, so every ServeBand loop
	// returns cleanly (nil); anything else is a real worker failure.
	conn.Close()
	wg.Wait()
	close(werr)
	for e := range werr {
		if e != nil {
			t.Errorf("worker serve loop error: %v", e)
		}
	}
	return logits
}

// assertLogitsBitExact fails unless got and want are bit-exact (max|Δ|=0), the
// shipped pipeline correctness contract reused here for the across-the-wire run.
func assertLogitsBitExact(t *testing.T, got, want [][]float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("logits seq = %d, monolithic = %d", len(got), len(want))
	}
	var maxAbs float32
	for r := range got {
		if len(got[r]) != len(want[r]) {
			t.Fatalf("logits[%d] len = %d, monolithic = %d", r, len(got[r]), len(want[r]))
		}
		for i := range got[r] {
			d := got[r][i] - want[r][i]
			if d < 0 {
				d = -d
			}
			if d > maxAbs {
				maxAbs = d
			}
		}
	}
	if maxAbs != 0 {
		t.Fatalf("across-worker pipeline logits differ from monolithic: max|delta|=%.3e (want bit-exact 0)", maxAbs)
	}
}

// TestServeBandWorkerPipelineMatchesMonolithic is the headline #30 gate: a 2-stage
// pipeline where stage B's band runs ONLY inside a ServeBand worker on the far end of
// a loopback TCP socket produces logits bit-identical to the monolithic Forward. The
// residency checks prove stage B genuinely ran from its own narrowed checkpoint, so a
// bit-exact pass cannot be an artifact of the driver accidentally running a full model.
func TestServeBandWorkerPipelineMatchesMonolithic(t *testing.T) {
	dir, cfg := writeTinyGLMDsaShardedSafetensorsDirN(
		t, "BF16", 3, []string{"full", "shared", "full"}, false, true, true, true)

	monoAct := mono(t, dir, cfg).Forward([]int{3, 1, 4, 1, 5})

	stageA, err := LoadSafetensorsQuantDir(dir, cfg, WithLayerWindow(0, 2))
	if err != nil {
		t.Fatalf("stage A load: %v", err)
	}
	stageB, err := LoadSafetensorsQuantDir(dir, cfg, WithLayerWindow(2, 3))
	if err != nil {
		t.Fatalf("stage B load: %v", err)
	}
	// The worker (stage B) holds ONLY its band; the driver (stage A) holds ONLY its.
	assertNoLayerTensors(t, "driver stage A", stageA, 2)
	assertNoLayerTensors(t, "worker stage B", stageB, 0, 1)
	if !hasAnyLayerTensor(stageB, 2) {
		t.Fatalf("worker stage B [2,3) is missing layer 2 weights; nothing to run")
	}

	logits := runAcrossWorkers(t, []int{3, 1, 4, 1, 5}, []PipelineStage{
		{Spec: StageSpec{Lo: 0, Hi: 2, First: true}, Model: stageA},
		{Spec: StageSpec{Lo: 2, Hi: 3, Last: true}, Model: stageB},
	})
	assertLogitsBitExact(t, logits, monoAct.Logits)
}

// TestServeBandThreeStageWorkerPipelineMatchesMonolithic exercises the interior-worker
// relay path (a non-first, non-last ServeBand that forwards downstream and relays the
// reply): a 3-stage pipeline over a 5-layer full,shared,full,shared,full checkpoint,
// cut at the two full-indexer layers (2 and 4) so no IndexShare group crosses the wire,
// is still bit-identical to the monolith — proving the chain composes beyond two stages.
func TestServeBandThreeStageWorkerPipelineMatchesMonolithic(t *testing.T) {
	dir, cfg := writeTinyGLMDsaShardedSafetensorsDirN(
		t, "BF16", 5, []string{"full", "shared", "full", "shared", "full"}, false, true, true, true)
	if cfg.NumLayers != 5 {
		t.Fatalf("fixture NumLayers = %d, test assumes 5", cfg.NumLayers)
	}

	ids := []int{3, 1, 4, 1, 5}
	monoAct := mono(t, dir, cfg).Forward(ids)

	s0, err := LoadSafetensorsQuantDir(dir, cfg, WithLayerWindow(0, 2))
	if err != nil {
		t.Fatalf("stage 0 load: %v", err)
	}
	s1, err := LoadSafetensorsQuantDir(dir, cfg, WithLayerWindow(2, 4))
	if err != nil {
		t.Fatalf("stage 1 load: %v", err)
	}
	s2, err := LoadSafetensorsQuantDir(dir, cfg, WithLayerWindow(4, 5))
	if err != nil {
		t.Fatalf("stage 2 load: %v", err)
	}
	// The interior worker (stage 1) holds only its band — it neither embeds nor heads.
	assertNoLayerTensors(t, "interior stage 1", s1, 0, 1, 4)
	if !hasAnyLayerTensor(s1, 2) || !hasAnyLayerTensor(s1, 3) {
		t.Fatalf("interior stage 1 [2,4) is missing its band weights; nothing to run")
	}

	logits := runAcrossWorkers(t, ids, []PipelineStage{
		{Spec: StageSpec{Lo: 0, Hi: 2, First: true}, Model: s0},
		{Spec: StageSpec{Lo: 2, Hi: 4}, Model: s1},
		{Spec: StageSpec{Lo: 4, Hi: 5, Last: true}, Model: s2},
	})
	assertLogitsBitExact(t, logits, monoAct.Logits)
}

// TestServeBandRejectsMisroutedFrame proves the worker's boundary-integrity check
// fails closed: a frame whose resume-layer does not match the worker's band start is
// rejected BEFORE ForwardBand runs, so a misrouted hidden state never runs a band
// against the wrong layer. This is the worker-side mirror of handoff's gotLo check.
func TestServeBandRejectsMisroutedFrame(t *testing.T) {
	dir, cfg := writeTinyGLMDsaShardedSafetensorsDirN(
		t, "BF16", 3, []string{"full", "shared", "full"}, false, true, true, true)
	stageB, err := LoadSafetensorsQuantDir(dir, cfg, WithLayerWindow(2, 3))
	if err != nil {
		t.Fatalf("stage B load: %v", err)
	}

	srv, cli := net.Pipe()
	done := make(chan error, 1)
	go func() {
		done <- ServeBand(srv, PipelineStage{Spec: StageSpec{Lo: 2, Hi: 3, Last: true}, Model: stageB}, nil)
	}()

	// A well-formed hidden frame that resumes at layer 0, sent to a worker owning [2,3).
	bad, err := MarshalHidden([][]float32{make([]float32, cfg.HiddenSize)}, 0)
	if err != nil {
		t.Fatalf("MarshalHidden: %v", err)
	}
	if err := writeFrame(cli, bad); err != nil {
		t.Fatalf("writeFrame: %v", err)
	}
	cli.Close()

	werr := <-done
	if werr == nil {
		t.Fatal("ServeBand accepted a frame resuming at the wrong layer; want a fail-closed rejection")
	}
}
