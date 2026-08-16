package main

import (
	"bytes"
	"context"
	"net"
	"net/http"
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
