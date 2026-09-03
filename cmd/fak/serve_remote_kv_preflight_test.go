package main

import (
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/l3kv"
)

func newTestServeFlags(mode, remoteURL, token string) *serveFlags {
	backend := "l3kv-blobhttp"
	timeout := 50 * time.Millisecond
	return &serveFlags{
		remoteKVMode:    &mode,
		remoteKVBackend: &backend,
		remoteKVURL:     &remoteURL,
		remoteKVToken:   &token,
		remoteKVTimeout: &timeout,
	}
}

func TestServeRemoteKVCheckOrdering(t *testing.T) {
	t.Run("execution ordering check before model load", func(t *testing.T) {
		var sequence []string
		probe := func(ctx context.Context, cfg l3kv.RemoteKVConfig) error {
			sequence = append(sequence, "remote_kv_check")
			return nil
		}
		loadModel := func() {
			sequence = append(sequence, "model_load")
		}

		sf := newTestServeFlags(string(l3kv.RemoteKVModeOptional), "http://127.0.0.1:8080/blobs", "")
		receipt, err := loadServeModelWithRemoteKVCheck(context.Background(), sf, probe, loadModel)
		if err != nil {
			t.Fatalf("unexpected check error: %v", err)
		}
		if receipt.Outcome != l3kv.RemoteKVOutcomeReady {
			t.Fatalf("receipt.Outcome = %q, want %q", receipt.Outcome, l3kv.RemoteKVOutcomeReady)
		}

		if len(sequence) != 2 {
			t.Fatalf("expected 2 execution steps, got %d: %v", len(sequence), sequence)
		}
		if sequence[0] != "remote_kv_check" {
			t.Fatalf("step 0 was %q, want 'remote_kv_check'", sequence[0])
		}
		if sequence[1] != "model_load" {
			t.Fatalf("step 1 was %q, want 'model_load'", sequence[1])
		}
	})

	t.Run("production serve order in serve.go", func(t *testing.T) {
		root := repoRootFromTest(t)
		serveBytes, err := os.ReadFile(filepath.Join(root, "cmd", "fak", "serve.go"))
		if err != nil {
			t.Fatalf("read serve.go: %v", err)
		}
		src := string(serveBytes)

		checkIdx := strings.Index(src, "checkServeRemoteKV(")
		if checkIdx < 0 {
			t.Fatal("serve.go does not call checkServeRemoteKV")
		}

		loadModelIdx := strings.Index(src, "rt.loadModel(sf)")
		if loadModelIdx < 0 {
			t.Fatal("serve.go does not call rt.loadModel(sf)")
		}

		if checkIdx >= loadModelIdx {
			t.Fatalf("checkServeRemoteKV at byte %d must precede rt.loadModel(sf) at byte %d",
				checkIdx, loadModelIdx)
		}
	})
}

func TestServeRemoteKVCheckBlocksModelLoadOnRequiredFailure(t *testing.T) {
	t.Run("probe connection failure", func(t *testing.T) {
		loadCalls := 0
		loadModel := func() {
			loadCalls++
		}
		probeErr := errors.New("dial tcp 127.0.0.1:8080: connect: connection refused")
		probe := func(ctx context.Context, cfg l3kv.RemoteKVConfig) error {
			return probeErr
		}

		sf := newTestServeFlags(string(l3kv.RemoteKVModeRequired), "http://127.0.0.1:8080/blobs", "")
		receipt, err := loadServeModelWithRemoteKVCheck(context.Background(), sf, probe, loadModel)
		if err == nil {
			t.Fatal("expected error for required mode failure, got nil")
		}
		if !strings.Contains(err.Error(), "probe unavailable") {
			t.Errorf("error %q should mention probe unavailable", err.Error())
		}
		if loadCalls != 0 {
			t.Fatalf("expected model load callback to be called 0 times, got %d", loadCalls)
		}
		if receipt.Outcome != l3kv.RemoteKVOutcomeUnavailable {
			t.Fatalf("receipt.Outcome = %q, want %q", receipt.Outcome, l3kv.RemoteKVOutcomeUnavailable)
		}
		if receipt.Error != probeErr.Error() {
			t.Fatalf("receipt.Error = %q, want %q", receipt.Error, probeErr.Error())
		}
	})

	t.Run("probe timeout failure", func(t *testing.T) {
		loadCalls := 0
		loadModel := func() {
			loadCalls++
		}
		timeoutProbe := func(ctx context.Context, cfg l3kv.RemoteKVConfig) error {
			return context.DeadlineExceeded
		}

		sf := newTestServeFlags(string(l3kv.RemoteKVModeRequired), "http://127.0.0.1:8080/blobs", "")
		receipt, err := loadServeModelWithRemoteKVCheck(context.Background(), sf, timeoutProbe, loadModel)
		if err == nil {
			t.Fatal("expected error for required mode timeout, got nil")
		}
		if !strings.Contains(err.Error(), "probe timed out") {
			t.Errorf("error %q should mention probe timed out", err.Error())
		}
		if loadCalls != 0 {
			t.Fatalf("expected model load callback to be called 0 times, got %d", loadCalls)
		}
		if receipt.Outcome != l3kv.RemoteKVOutcomeTimeout {
			t.Fatalf("receipt.Outcome = %q, want %q", receipt.Outcome, l3kv.RemoteKVOutcomeTimeout)
		}
	})

	t.Run("incomplete configuration failure", func(t *testing.T) {
		loadCalls := 0
		loadModel := func() {
			loadCalls++
		}

		sf := newTestServeFlags(string(l3kv.RemoteKVModeRequired), "", "")
		receipt, err := loadServeModelWithRemoteKVCheck(context.Background(), sf, nil, loadModel)
		if err == nil {
			t.Fatal("expected error for required mode without URL, got nil")
		}
		if loadCalls != 0 {
			t.Fatalf("expected model load callback to be called 0 times, got %d", loadCalls)
		}
		if receipt.Outcome != l3kv.RemoteKVOutcomeIncompleteConfig {
			t.Fatalf("receipt.Outcome = %q, want %q", receipt.Outcome, l3kv.RemoteKVOutcomeIncompleteConfig)
		}
	})
}

func TestServeRemoteKVCheckAllowsModelLoadOnOptionalFailure(t *testing.T) {
	t.Run("probe timeout falls back to local residency", func(t *testing.T) {
		loadCalls := 0
		loadModel := func() {
			loadCalls++
		}
		timeoutProbe := func(ctx context.Context, cfg l3kv.RemoteKVConfig) error {
			return context.DeadlineExceeded
		}

		sf := newTestServeFlags(string(l3kv.RemoteKVModeOptional), "http://127.0.0.1:8080/blobs", "")
		receipt, err := loadServeModelWithRemoteKVCheck(context.Background(), sf, timeoutProbe, loadModel)
		if err != nil {
			t.Fatalf("optional mode timeout must return nil error, got: %v", err)
		}
		if loadCalls != 1 {
			t.Fatalf("expected model load callback to be executed exactly once, got %d", loadCalls)
		}
		if receipt.Outcome != l3kv.RemoteKVOutcomeTimeout {
			t.Fatalf("receipt.Outcome = %q, want %q", receipt.Outcome, l3kv.RemoteKVOutcomeTimeout)
		}
		if receipt.FallbackReason == "" {
			t.Fatal("expected receipt.FallbackReason to note fallback, got empty string")
		}
		if !strings.Contains(receipt.FallbackReason, "falling back to local residency") {
			t.Fatalf("fallback reason %q missing expected phrase", receipt.FallbackReason)
		}
	})

	t.Run("probe connection refusal falls back to local residency", func(t *testing.T) {
		loadCalls := 0
		loadModel := func() {
			loadCalls++
		}
		failingProbe := func(ctx context.Context, cfg l3kv.RemoteKVConfig) error {
			return errors.New("connection refused")
		}

		sf := newTestServeFlags(string(l3kv.RemoteKVModeOptional), "http://127.0.0.1:8080/blobs", "")
		receipt, err := loadServeModelWithRemoteKVCheck(context.Background(), sf, failingProbe, loadModel)
		if err != nil {
			t.Fatalf("optional mode failure must return nil error, got: %v", err)
		}
		if loadCalls != 1 {
			t.Fatalf("expected model load callback to be executed exactly once, got %d", loadCalls)
		}
		if receipt.Outcome != l3kv.RemoteKVOutcomeUnavailable {
			t.Fatalf("receipt.Outcome = %q, want %q", receipt.Outcome, l3kv.RemoteKVOutcomeUnavailable)
		}
		if receipt.FallbackReason == "" {
			t.Fatal("expected receipt.FallbackReason to note fallback, got empty string")
		}
		if !strings.Contains(receipt.FallbackReason, "falling back to local residency") {
			t.Fatalf("fallback reason %q missing expected phrase", receipt.FallbackReason)
		}
	})
}

func TestServeRemoteKVCheckDefaultModeIsOptional(t *testing.T) {
	// An empty sf without explicit mode should default to optional.
	var logBuf bytes.Buffer
	remoteKVLogWriter = &logBuf
	defer func() { remoteKVLogWriter = nil }()

	url := "http://127.0.0.1:8080/blobs"
	sf := &serveFlags{
		remoteKVURL: &url,
	}

	receipt, err := checkServeRemoteKV(context.Background(), sf, func(ctx context.Context, cfg l3kv.RemoteKVConfig) error {
		if cfg.Mode != l3kv.RemoteKVModeOptional {
			t.Errorf("cfg.Mode = %q, want %q", cfg.Mode, l3kv.RemoteKVModeOptional)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if receipt.Mode != l3kv.RemoteKVModeOptional {
		t.Errorf("receipt.Mode = %q, want %q", receipt.Mode, l3kv.RemoteKVModeOptional)
	}
	if receipt.Outcome != l3kv.RemoteKVOutcomeReady {
		t.Errorf("receipt.Outcome = %q, want %q", receipt.Outcome, l3kv.RemoteKVOutcomeReady)
	}

	logged := logBuf.String()
	if !strings.Contains(logged, "fak serve: remote kv preflight receipt:") {
		t.Errorf("log output missing receipt header: %s", logged)
	}
	if !strings.Contains(logged, "remote KV store ready:") {
		t.Errorf("log output missing readiness message: %s", logged)
	}
}

func TestServeRemoteKVCheckReadsEnvironment(t *testing.T) {
	t.Setenv("FAK_REMOTE_KV_MODE", "optional")
	t.Setenv("FAK_BLOB_HTTP_URL", "http://127.0.0.1:9090/v1/blobs")
	t.Setenv("FAK_BLOB_HTTP_TOKEN", "secret-token-123")
	t.Setenv("FAK_REMOTE_KV_TIMEOUT", "150ms")

	var logBuf bytes.Buffer
	remoteKVLogWriter = &logBuf
	defer func() { remoteKVLogWriter = nil }()

	probeChecked := false
	receipt, err := checkServeRemoteKV(context.Background(), nil, func(ctx context.Context, cfg l3kv.RemoteKVConfig) error {
		probeChecked = true
		if cfg.RemoteURL != "http://127.0.0.1:9090/v1/blobs" {
			t.Errorf("cfg.RemoteURL = %q, want http://127.0.0.1:9090/v1/blobs", cfg.RemoteURL)
		}
		if cfg.Token != "secret-token-123" {
			t.Errorf("cfg.Token = %q, want secret-token-123", cfg.Token)
		}
		if cfg.Timeout != 150*time.Millisecond {
			t.Errorf("cfg.Timeout = %v, want 150ms", cfg.Timeout)
		}
		if cfg.Mode != l3kv.RemoteKVModeOptional {
			t.Errorf("cfg.Mode = %q, want optional", cfg.Mode)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !probeChecked {
		t.Fatal("probe was not called")
	}
	if receipt.Outcome != l3kv.RemoteKVOutcomeReady {
		t.Fatalf("receipt.Outcome = %q, want %q", receipt.Outcome, l3kv.RemoteKVOutcomeReady)
	}

	// Verify credentials are sanitized in receipt and logs.
	if strings.Contains(receipt.SanitizedEndpoint, "secret-token-123") {
		t.Errorf("sanitized endpoint leaked secret token: %s", receipt.SanitizedEndpoint)
	}
	if strings.Contains(logBuf.String(), "secret-token-123") {
		t.Errorf("log output leaked secret token: %s", logBuf.String())
	}
}
