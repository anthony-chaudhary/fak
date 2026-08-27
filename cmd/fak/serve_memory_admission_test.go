package main

import (
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gpulease"
)

func TestLoadServeModelWithMetalLeaseRefusesBeforeLoadAndReleasesAfterServe(t *testing.T) {
	path := filepath.Join(t.TempDir(), "gpu.lease")
	t.Setenv("FAK_GPU_LEASE", path)
	held, err := gpulease.Acquire(gpulease.Options{})
	if err != nil {
		t.Fatalf("hold modelbench-compatible lease: %v", err)
	}

	loads := 0
	release, err := loadServeModelWithMetalLease(true, "qwen3.8-27b-q4_k_m.gguf", gpulease.Options{}, func() {
		loads++
	})
	if err == nil {
		t.Fatal("Metal serve admission succeeded while modelbench-compatible lease was held")
	}
	if !errors.Is(err, gpulease.ErrBusy) {
		t.Fatalf("busy admission error = %v, want errors.Is(ErrBusy)", err)
	}
	for _, want := range []string{path, "pid " + strconv.Itoa(os.Getpid()), "before model load", "stop the holder process"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("busy admission error %q does not contain %q", err, want)
		}
	}
	if loads != 0 {
		t.Fatalf("load callback calls while lease held = %d, want 0", loads)
	}
	release()

	held.Release()
	release, err = loadServeModelWithMetalLease(true, "qwen3.8-27b-q4_k_m.gguf", gpulease.Options{}, func() {
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

func TestLoadServeModelWithMetalLeaseLeavesCPUAndEmptyModelUnserialized(t *testing.T) {
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
			release, err := loadServeModelWithMetalLease(tc.metal, tc.ggufPath, gpulease.Options{Path: path}, func() { loads++ })
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
