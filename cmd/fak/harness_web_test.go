package main

import (
	"bytes"
	"context"
	"net"
	"net/http"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

func TestHarnessWebSelfcheckThroughFakDispatch(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := runHarnessCommand(&out, &errOut, []string{"web", "--selfcheck"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	if !bytes.Contains(out.Bytes(), []byte("HARNESS_WEB_SELFCHECK ok")) {
		t.Fatalf("stdout=%s", out.String())
	}
}

func TestHarnessWebHelpListsShippedFlags(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := runHarnessCommand(&out, &errOut, []string{"web", "--help"}); code != 2 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	for _, flag := range []string{"-addr", "-state", "-fak-url", "-workspace", "-selfcheck"} {
		if !bytes.Contains(errOut.Bytes(), []byte(flag)) {
			t.Fatalf("help missing %s: %s", flag, errOut.String())
		}
	}
}

func TestHarnessWebRejectsNonLoopbackBind(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := runHarnessCommand(&out, &errOut, []string{"web", "--addr", "0.0.0.0:0"}); code != 2 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
}

func TestHarnessWebServesFromFakDispatch(t *testing.T) {
	probe, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := probe.Addr().String()
	_ = probe.Close()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var out, errOut bytes.Buffer
	done := make(chan int, 1)
	go func() {
		done <- runHarnessWebWithCancel(ctx, &out, &errOut, []string{"--addr", addr, "--state", t.TempDir() + "/state.json"})
	}()
	var resp *http.Response
	for i := 0; i < 50; i++ {
		resp, err = http.Get("http://" + addr + "/")
		if err == nil {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if err != nil {
		t.Fatalf("GET: %v stderr=%s", err, errOut.String())
	}
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status=%d", resp.StatusCode)
	}
	cancel()
	select {
	case code := <-done:
		if code != 0 {
			t.Fatalf("code=%d stderr=%s", code, errOut.String())
		}
	case <-time.After(3 * time.Second):
		t.Fatal("server did not stop")
	}
}

func TestHarnessWebTempBuiltFakBinary(t *testing.T) {
	if testing.Short() {
		t.Skip("builds the fak binary")
	}
	binary := filepath.Join(t.TempDir(), "fak")
	if runtime.GOOS == "windows" {
		binary += ".exe"
	}
	build := exec.Command("go", "build", "-o", binary, ".")
	if output, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build fak: %v\n%s", err, output)
	}

	selfcheck := exec.Command(binary, "harness", "web", "--selfcheck")
	output, err := selfcheck.CombinedOutput()
	if err != nil || !bytes.Contains(output, []byte("HARNESS_WEB_SELFCHECK ok")) {
		t.Fatalf("selfcheck: %v\n%s", err, output)
	}

	probe, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := probe.Addr().String()
	_ = probe.Close()
	serve := exec.Command(binary, "harness", "web", "--addr", addr, "--state", filepath.Join(t.TempDir(), "state.json"))
	var serveOutput bytes.Buffer
	serve.Stdout = &serveOutput
	serve.Stderr = &serveOutput
	if err := serve.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if serve.Process != nil {
			_ = serve.Process.Kill()
			_, _ = serve.Process.Wait()
		}
	})
	for i := 0; i < 100; i++ {
		resp, getErr := http.Get("http://" + addr + "/")
		if getErr == nil {
			_ = resp.Body.Close()
			if resp.StatusCode != http.StatusOK {
				t.Fatalf("status=%d", resp.StatusCode)
			}
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("temp-built fak did not serve HTTP: %s", serveOutput.String())
}
