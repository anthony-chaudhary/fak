package model

import (
	"bytes"
	"encoding/json"
	"net"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"
	"time"
)

// TestIssue85NetworkStageTransportProcessGenerateMatchesSession is the #85
// acceptance witness: stage 0 runs in this process, stage 1 runs in a separate
// OS process reached through TCPTransport, and greedy generation matches the
// monolithic Session.Generate token-for-token.
func TestIssue85NetworkStageTransportProcessGenerateMatchesSession(t *testing.T) {
	dir, cfg := writeTinyGLMDsaShardedSafetensorsDirN(
		t, "BF16", 3, []string{"full", "shared", "full"}, false, true, true, true)

	mono, err := LoadSafetensorsQuantDir(dir, cfg)
	if err != nil {
		t.Fatalf("monolithic load: %v", err)
	}
	stage0, err := LoadSafetensorsQuantDir(dir, cfg, WithLayerWindow(0, 2))
	if err != nil {
		t.Fatalf("stage 0 load: %v", err)
	}
	assertNoLayerTensors(t, "stage 0", stage0, 2)

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen worker: %v", err)
	}
	defer ln.Close()
	if tl, ok := ln.(*net.TCPListener); ok {
		if err := tl.SetDeadline(time.Now().Add(10 * time.Second)); err != nil {
			t.Fatalf("set accept deadline: %v", err)
		}
	}

	cfgJSON, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal cfg: %v", err)
	}
	exe, err := os.Executable()
	if err != nil {
		t.Fatalf("test executable: %v", err)
	}
	cmd := exec.Command(exe, "-test.run=^TestIssue85NetworkStageTransportWorkerProcess$", "-test.v")
	cmd.Env = append(os.Environ(),
		"FAK_ISSUE85_PP_WORKER=1",
		"FAK_ISSUE85_PP_ADDR="+ln.Addr().String(),
		"FAK_ISSUE85_PP_DIR="+dir,
		"FAK_ISSUE85_PP_CFG="+string(cfgJSON),
		"FAK_ISSUE85_PP_LO=2",
		"FAK_ISSUE85_PP_HI=3",
		"FAK_ISSUE85_PP_ABSENT_LAYERS=0,1",
	)
	var childLog bytes.Buffer
	cmd.Stdout = &childLog
	cmd.Stderr = &childLog
	if err := cmd.Start(); err != nil {
		t.Fatalf("start worker process: %v", err)
	}

	conn, err := ln.Accept()
	if err != nil {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		t.Fatalf("accept worker process: %v\n%s", err, childLog.String())
	}

	prompt := []int{3, 1, 4, 1}
	const nGen = 6
	want := mono.NewSession().Generate(prompt, nGen)
	got := make([]int, 0, nGen)
	ids := append([]int(nil), prompt...)
	transport := NewTCPTransport(conn)
	for len(got) < nGen {
		logits, err := RunPipelineAcrossWorkers(ids, PipelineStage{
			Spec:  StageSpec{Lo: 0, Hi: 2, First: true},
			Model: stage0,
		}, transport)
		if err != nil {
			conn.Close()
			_ = cmd.Wait()
			t.Fatalf("network pipeline: %v\n%s", err, childLog.String())
		}
		if len(logits) == 0 || len(logits[len(logits)-1]) == 0 {
			conn.Close()
			_ = cmd.Wait()
			t.Fatalf("network pipeline produced empty logits\n%s", childLog.String())
		}
		next := argmaxF32(logits[len(logits)-1])
		got = append(got, next)
		if cfg.IsEOS(next) {
			break
		}
		ids = append(ids, next)
	}
	conn.Close()
	if err := cmd.Wait(); err != nil {
		t.Fatalf("worker process failed: %v\n%s", err, childLog.String())
	}
	if len(got) != len(want) {
		t.Fatalf("network generation length = %d, monolithic = %d: got=%v want=%v", len(got), len(want), got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("network generation diverged at token %d: got=%v want=%v", i, got, want)
		}
	}
}

// TestIssue85NetworkStageTransportWorkerProcess is launched by the acceptance
// test above as the stage-1 worker process. It loads only its assigned layer band
// and serves it through the same TCPTransport framing used by the driver.
func TestIssue85NetworkStageTransportWorkerProcess(t *testing.T) {
	if os.Getenv("FAK_ISSUE85_PP_WORKER") != "1" {
		t.Skip("helper worker process")
	}
	var cfg Config
	if err := json.Unmarshal([]byte(os.Getenv("FAK_ISSUE85_PP_CFG")), &cfg); err != nil {
		t.Fatalf("unmarshal cfg: %v", err)
	}
	lo := issue85AtoiEnv(t, "FAK_ISSUE85_PP_LO")
	hi := issue85AtoiEnv(t, "FAK_ISSUE85_PP_HI")
	stage, err := LoadSafetensorsQuantDir(os.Getenv("FAK_ISSUE85_PP_DIR"), cfg, WithLayerWindow(lo, hi))
	if err != nil {
		t.Fatalf("worker load [%d,%d): %v", lo, hi, err)
	}
	for _, layer := range issue85CSVIntsEnv(t, "FAK_ISSUE85_PP_ABSENT_LAYERS") {
		if hasAnyLayerTensor(stage, layer) {
			t.Fatalf("worker [%d,%d) holds out-of-band layer %d weights", lo, hi, layer)
		}
	}
	if !hasAnyLayerTensor(stage, lo) {
		t.Fatalf("worker [%d,%d) is missing layer %d weights", lo, hi, lo)
	}
	conn, err := net.Dial("tcp", os.Getenv("FAK_ISSUE85_PP_ADDR"))
	if err != nil {
		t.Fatalf("worker dial: %v", err)
	}
	defer conn.Close()
	if err := ServeBand(conn, PipelineStage{
		Spec:  StageSpec{Lo: lo, Hi: hi, Last: true},
		Model: stage,
	}, nil); err != nil {
		t.Fatalf("ServeBand: %v", err)
	}
}

func issue85AtoiEnv(t *testing.T, name string) int {
	t.Helper()
	v, err := strconv.Atoi(os.Getenv(name))
	if err != nil {
		t.Fatalf("%s parse: %v", name, err)
	}
	return v
}

func issue85CSVIntsEnv(t *testing.T, name string) []int {
	t.Helper()
	raw := os.Getenv(name)
	if raw == "" {
		return nil
	}
	parts := strings.Split(raw, ",")
	out := make([]int, 0, len(parts))
	for _, p := range parts {
		v, err := strconv.Atoi(p)
		if err != nil {
			t.Fatalf("%s parse %q: %v", name, p, err)
		}
		out = append(out, v)
	}
	return out
}
