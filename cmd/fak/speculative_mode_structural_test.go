package main

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

type speculativeTestBackend struct {
	compute.Backend
	name string
}

func (b speculativeTestBackend) Name() string { return b.name }

type speculativeReplayBackend struct {
	speculativeTestBackend
	replayPath string
}

func (b speculativeReplayBackend) Qwen35SequencePrefixReplayPath() string { return b.replayPath }

func writeQwen35ConfigDir(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"model_type":"qwen35"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

// fak-test:runtime fast est=10ms lane=default
func TestSpeculativeGuardWiresRetainMTP(t *testing.T) {
	prior := fakmodel.RetainMTP
	defer fakmodel.SetRetainMTP(prior)

	qwenDir := writeQwen35ConfigDir(t)
	vulkanReplay := speculativeReplayBackend{
		speculativeTestBackend: speculativeTestBackend{name: "vulkan"},
		replayPath:             compute.Qwen35SequencePrefixReplayPath,
	}

	var warn bytes.Buffer
	mode, retained, err := configureGuardSpeculativeMTP("mtp", false, vulkanReplay, qwenDir, noSpeculativeEnv, &warn)
	if err != nil {
		t.Fatalf("configureGuardSpeculativeMTP(mtp, vulkan): %v", err)
	}
	if mode != serveSpeculativeMTP || !retained || !fakmodel.RetainMTP {
		t.Fatalf("mode=%q retained=%v RetainMTP=%v, want mtp/true/true", mode, retained, fakmodel.RetainMTP)
	}
	if warn.Len() != 0 {
		t.Fatalf("capable vulkan backend emitted unexpected warning: %q", warn.String())
	}

	mode, retained, err = configureGuardSpeculativeMTP("off", false, vulkanReplay, qwenDir, noSpeculativeEnv, &warn)
	if err != nil {
		t.Fatalf("configureGuardSpeculativeMTP(off): %v", err)
	}
	if mode != "" || retained || fakmodel.RetainMTP {
		t.Fatalf("off mode=%q retained=%v RetainMTP=%v, want empty/false/false", mode, retained, fakmodel.RetainMTP)
	}

	mode, retained, err = configureGuardSpeculativeMTP("mtp", true, nil, qwenDir, noSpeculativeEnv, &warn)
	if err != nil {
		t.Fatalf("configureGuardSpeculativeMTP(mtp, metal): %v", err)
	}
	if mode != serveSpeculativeMTP || !retained || !fakmodel.RetainMTP {
		t.Fatalf("metal mode=%q retained=%v RetainMTP=%v, want mtp/true/true", mode, retained, fakmodel.RetainMTP)
	}
	if warn.Len() != 0 {
		t.Fatalf("metal backend emitted unexpected warning: %q", warn.String())
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestSpeculativeMTPBackendReplayCapabilityWarning(t *testing.T) {
	prior := fakmodel.RetainMTP
	defer fakmodel.SetRetainMTP(prior)

	qwenDir := writeQwen35ConfigDir(t)
	cudaNoReplay := speculativeTestBackend{name: "cuda"}
	cudaWithReplay := speculativeReplayBackend{
		speculativeTestBackend: speculativeTestBackend{name: "cuda"},
		replayPath:             compute.Qwen35SequencePrefixReplayPath,
	}

	ok, reason := serveMTPBackendReplayStatus(serveSpeculativeMTP, false, cudaNoReplay)
	if ok || !strings.Contains(reason, compute.Qwen35SequencePrefixReplayPath) {
		t.Fatalf("cuda without replay: ok=%v reason=%q, want false mentioning %s", ok, reason, compute.Qwen35SequencePrefixReplayPath)
	}

	ok, reason = serveMTPBackendReplayStatus(serveSpeculativeMTP, false, cudaWithReplay)
	if !ok || reason != "" {
		t.Fatalf("cuda with replay: ok=%v reason=%q, want true/empty", ok, reason)
	}

	ok, reason = serveMTPBackendReplayStatus("", false, cudaNoReplay)
	if !ok || reason != "" {
		t.Fatalf("disabled mode: ok=%v reason=%q, want true/empty", ok, reason)
	}

	var warn bytes.Buffer
	mode, retained, err := configureGuardSpeculativeMTP("mtp", false, cudaNoReplay, qwenDir, noSpeculativeEnv, &warn)
	if err != nil {
		t.Fatalf("configureGuardSpeculativeMTP(cuda no replay): %v", err)
	}
	if mode != serveSpeculativeMTP || !retained || !fakmodel.RetainMTP {
		t.Fatalf("cuda mtp mode=%q retained=%v RetainMTP=%v, want mtp/true/true", mode, retained, fakmodel.RetainMTP)
	}
	if !strings.Contains(warn.String(), compute.Qwen35SequencePrefixReplayPath) {
		t.Fatalf("expected honest capability warning mentioning %s, got %q", compute.Qwen35SequencePrefixReplayPath, warn.String())
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestSpeculativeGuardRejectsUnknownMode(t *testing.T) {
	_, _, err := configureGuardSpeculativeMTP("bogus", false, nil, "", noSpeculativeEnv, nil)
	if !errors.Is(err, ErrSpeculativeModeUnknown) {
		t.Fatalf("unknown mode err=%v, want ErrSpeculativeModeUnknown", err)
	}
}
