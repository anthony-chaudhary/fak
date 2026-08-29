package main

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gpulease"
)

func TestLoadLocalLauncherModelWithMetalLeaseRefusesBeforeLoadAndReleasesAfterServe(t *testing.T) {
	const holderEnv = "FAK_LOCAL_LAUNCHER_METAL_LEASE_HOLDER_TEST"
	if path := os.Getenv(holderEnv); path != "" {
		lease, err := gpulease.Acquire(gpulease.Options{Path: path, NoWait: true})
		if err != nil {
			fmt.Fprintln(os.Stderr, "child lease acquire:", err)
			os.Exit(3)
		}
		fmt.Fprintf(os.Stdout, "READY %d\n", os.Getpid())
		_ = os.Stdout.Sync()
		_, _ = io.Copy(io.Discard, os.Stdin)
		runtime.KeepAlive(lease)
		os.Exit(0) // The OS, not an in-process Release call, drops the child flock.
	}

	path := filepath.Join(t.TempDir(), "gpu.lease")
	t.Setenv("FAK_GPU_LEASE", path)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	t.Cleanup(cancel)
	child := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestLoadLocalLauncherModelWithMetalLeaseRefusesBeforeLoadAndReleasesAfterServe$")
	child.Env = append(os.Environ(), holderEnv+"="+path)
	childIn, err := child.StdinPipe()
	if err != nil {
		t.Fatalf("child stdin: %v", err)
	}
	childOut, err := child.StdoutPipe()
	if err != nil {
		t.Fatalf("child stdout: %v", err)
	}
	var childStderr strings.Builder
	child.Stderr = &childStderr
	if err := child.Start(); err != nil {
		t.Fatalf("start unrelated lease holder: %v", err)
	}
	waited := false
	t.Cleanup(func() {
		_ = childIn.Close()
		if !waited && child.Process != nil {
			_ = child.Process.Kill()
			_ = child.Wait()
		}
	})
	ready, err := bufio.NewReader(childOut).ReadString('\n')
	if err != nil {
		t.Fatalf("wait for child lease holder: %v; stderr=%s", err, childStderr.String())
	}
	wantReady := "READY " + strconv.Itoa(child.Process.Pid)
	if strings.TrimSpace(ready) != wantReady {
		t.Fatalf("child readiness = %q, want %q; stderr=%s", strings.TrimSpace(ready), wantReady, childStderr.String())
	}

	loads := 0
	release, err := loadLocalLauncherModelWithMetalLease(true, "qwen3.8-27b-q4_k_m.gguf", gpulease.Options{}, func() {
		loads++
	})
	if err == nil {
		t.Fatal("Metal serve admission succeeded while modelbench-compatible lease was held")
	}
	if !errors.Is(err, gpulease.ErrBusy) {
		t.Fatalf("busy admission error = %v, want errors.Is(ErrBusy)", err)
	}
	for _, want := range []string{path, "pid " + strconv.Itoa(child.Process.Pid), "before model load", "stop the holder process"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("busy admission error %q does not contain %q", err, want)
		}
	}
	if loads != 0 {
		t.Fatalf("load callback calls while lease held = %d, want 0", loads)
	}
	release()

	if err := childIn.Close(); err != nil {
		t.Fatalf("signal holder exit: %v", err)
	}
	if err := child.Wait(); err != nil {
		t.Fatalf("holder exit: %v; stderr=%s", err, childStderr.String())
	}
	waited = true
	release, err = loadLocalLauncherModelWithMetalLease(true, "qwen3.8-27b-q4_k_m.gguf", gpulease.Options{}, func() {
		loads++
	})
	if err != nil {
		t.Fatalf("admit after holder release: %v", err)
	}
	if loads != 1 {
		t.Fatalf("load callback calls after admission = %d, want 1", loads)
	}
	if _, err := gpulease.Acquire(gpulease.Options{NoWait: true}); !errors.Is(err, gpulease.ErrBusy) {
		t.Fatalf("serve lease was not retained after load callback: got %v, want ErrBusy", err)
	}

	release()
	reacquired, err := gpulease.Acquire(gpulease.Options{NoWait: true})
	if err != nil {
		t.Fatalf("reacquire after serve cleanup: %v", err)
	}
	reacquired.Release()
}

func TestLoadLocalLauncherModelWithMetalLeaseLeavesCPUAndEmptyModelUnserialized(t *testing.T) {
	tests := []struct {
		name     string
		metal    bool
		ggufPath string
	}{
		{name: "CPU model", metal: false, ggufPath: "model.gguf"},
		{name: "Metal proxy without local model", metal: true, ggufPath: ""},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "gpu.lease")
			held, err := gpulease.Acquire(gpulease.Options{Path: path})
			if err != nil {
				t.Fatalf("hold unrelated lease: %v", err)
			}
			defer held.Release()

			loads := 0
			release, err := loadLocalLauncherModelWithMetalLease(tc.metal, tc.ggufPath, gpulease.Options{Path: path}, func() { loads++ })
			if err != nil {
				t.Fatalf("unserialized path: %v", err)
			}
			defer release()
			if loads != 1 {
				t.Fatalf("load callback calls = %d, want 1", loads)
			}
		})
	}
}
