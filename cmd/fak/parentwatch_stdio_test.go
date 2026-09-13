package main

import (
	"bytes"
	"context"
	"io"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/parentwatch"
)

// TestLSPExitsOnStdinEOF is the cmd-level contract: a closed stdin (EOF) must
// make runLSP return 0. The parentwatch.Watch wrapping added to runLSP must not
// change EOF shutdown, so a normal client hangup still exits cleanly.
func TestLSPExitsOnStdinEOF(t *testing.T) {
	r, w := io.Pipe()
	var out, errb bytes.Buffer
	done := make(chan int, 1)
	go func() { done <- runLSP(r, &out, &errb, nil) }()
	_ = w.Close()
	select {
	case code := <-done:
		if code != 0 {
			t.Fatalf("runLSP exited %d on stdin EOF, stderr=%s", code, errb.String())
		}
	case <-time.After(3 * time.Second):
		t.Fatal("runLSP did not return within 3s after stdin EOF")
	}
}

// TestParentWatchCancelsOnChildExit is an integration smoke over the real
// parentwatch package the wiring depends on: a watched child that dies must
// cancel the derived ctx promptly.
func TestParentWatchCancelsOnChildExit(t *testing.T) {
	if _, err := exec.LookPath("sleep"); err != nil {
		t.Skip("sleep not available")
	}
	cmd := exec.Command("sleep", "30")
	if err := cmd.Start(); err != nil {
		t.Fatalf("start child: %v", err)
	}
	pid := cmd.Process.Pid
	ctx, stop := parentwatch.Watch(context.Background(), pid)
	defer stop()
	if !parentwatch.ParentAlive(pid) {
		t.Fatal("child should be alive immediately after start")
	}
	_ = cmd.Process.Kill()
	_, _ = cmd.Process.Wait()
	select {
	case <-ctx.Done():
	case <-time.After(3 * time.Second):
		t.Fatal("Watch did not cancel within 3s of child exit")
	}
}

// TestServeStdioUsesParentWatch is a deterministic source-wiring assertion: the
// stdio branch of serve_stages.go must pass the parentwatch-wrapped ctx into
// ServeStdio and keep the exit observations/return contract.
func TestServeStdioUsesParentWatch(t *testing.T) {
	src, err := os.ReadFile("serve_stages.go")
	if err != nil {
		t.Fatalf("read serve_stages.go: %v", err)
	}
	s := string(src)
	for _, want := range []string{
		`parentwatch.Watch(ctx, os.Getppid())`,
		`rt.srv.ServeStdio(wctx, os.Stdin, os.Stdout)`,
		`persistServeExitObservations("stdio")`,
	} {
		if !strings.Contains(s, want) {
			t.Fatalf("serve_stages.go missing stdio wiring %q", want)
		}
	}
}
