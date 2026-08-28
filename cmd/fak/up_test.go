package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestUpHelpUsesServeSurface(t *testing.T) {
	fs, _ := newServeFlagSet()
	for _, name := range []string{"addr", "gguf", "base-url", "policy", "require-key-env", "metrics-snapshot", "session-state"} {
		if fs.Lookup(name) == nil {
			t.Fatalf("serve/up flag %q is missing", name)
		}
	}
}

func TestUpBootsUnifiedAgentRuntime(t *testing.T) {
	if testing.Short() {
		t.Skip("process witness")
	}
	root := filepath.Clean(filepath.Join("..", ".."))
	cacheRoot := t.TempDir()
	bin := filepath.Join(cacheRoot, "fak")
	if runtime.GOOS == "windows" {
		bin += ".exe"
	}
	goCache := filepath.Join(cacheRoot, "gocache")
	goTmp := filepath.Join(cacheRoot, "gotmp")
	if err := os.MkdirAll(goTmp, 0o755); err != nil {
		t.Fatal(err)
	}
	build := exec.Command("go", "build", "-o", bin, "./cmd/fak")
	build.Dir = root
	build.Env = append(os.Environ(), "GOCACHE="+goCache, "GOTMPDIR="+goTmp)
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build fak: %v\n%s", err, out)
	}

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := ln.Addr().String()
	_ = ln.Close()

	cmd := exec.Command(bin, "up", "--addr", addr, "--engine", "mock", "--native")
	cmd.Dir = root
	var output bytes.Buffer
	cmd.Stdout, cmd.Stderr = &output, &output
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	waited := false
	defer func() {
		if !waited && (cmd.ProcessState == nil || !cmd.ProcessState.Exited()) {
			_ = cmd.Process.Kill()
			_ = cmd.Wait()
		}
	}()

	base := "http://" + addr
	deadline := time.Now().Add(20 * time.Second)
	for {
		resp, getErr := http.Get(base + "/readyz")
		if getErr == nil {
			_, _ = io.Copy(io.Discard, resp.Body)
			_ = resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				break
			}
		}
		if time.Now().After(deadline) {
			t.Fatalf("ready timeout: %v\n%s", getErr, output.String())
		}
		time.Sleep(40 * time.Millisecond)
	}

	body := strings.NewReader(`{"goal":"book the task","max_turns":4}`)
	resp, err := http.Post(base+"/v1/fak/agent/sessions", "application/json", body)
	if err != nil {
		t.Fatalf("post session: %v\n%s", err, output.String())
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		t.Fatalf("session status=%d body=%s", resp.StatusCode, raw)
	}
	seenEnd := false
	scan := bufio.NewScanner(resp.Body)
	for scan.Scan() {
		var event struct {
			Event string `json:"event"`
		}
		if err := json.Unmarshal(scan.Bytes(), &event); err != nil {
			t.Fatalf("invalid NDJSON %q: %v", scan.Text(), err)
		}
		if event.Event == "session.end" {
			seenEnd = true
		}
	}
	if err := scan.Err(); err != nil {
		t.Fatal(err)
	}
	if !seenEnd {
		t.Fatalf("session.end not observed; process output:\n%s", output.String())
	}

	if err := cmd.Process.Signal(os.Interrupt); err != nil {
		if runtime.GOOS == "windows" {
			_ = cmd.Process.Kill()
		} else {
			t.Fatalf("interrupt: %v", err)
		}
	}
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	select {
	case err := <-done:
		waited = true
		if err != nil && runtime.GOOS != "windows" {
			t.Fatalf("up did not terminate cleanly: %v\n%s", err, output.String())
		}
	case <-time.After(8 * time.Second):
		t.Fatalf("up did not stop after interrupt\n%s", output.String())
	}
}
