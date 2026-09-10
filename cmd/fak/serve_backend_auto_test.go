package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/gpulease"
)

const backendAutoChildEnv = "FAK_TEST_SERVE_BACKEND_AUTO_CHILD"

type serveBackendAutoSentinel struct {
	compute.Backend
	name string
}

func (b *serveBackendAutoSentinel) Name() string { return b.name }

func TestServeBackendAutoSelection(t *testing.T) {
	autoWant := "cpu"
	if runtime.GOOS == "linux" || runtime.GOOS == "windows" {
		autoWant = "vulkan"
	}
	tests := []struct {
		name       string
		requested  string
		envBackend string
		registered string
		want       string
		wantErr    string
	}{
		{name: "explicit Vulkan wins environment", requested: " VULKAN ", envBackend: "unknown-env", registered: "vulkan", want: "vulkan"},
		{name: "explicit backend wins environment", requested: "cuda", envBackend: "vulkan", registered: "cuda,vulkan", want: "cuda"},
		{name: "explicit auto wins environment", requested: "auto", envBackend: "cuda", registered: "cuda,vulkan", want: autoWant},
		{name: "environment Vulkan when omitted", envBackend: "vulkan", registered: "vulkan", want: "vulkan"},
		{name: "environment auto when omitted", envBackend: "auto", registered: "vulkan", want: autoWant},
		{name: "omitted uses platform automatic selection", registered: "vulkan", want: autoWant},
		{name: "omitted keeps CPU without Vulkan", want: "cpu"},
		{name: "auto never selects arbitrary accelerator", registered: "cuda", want: "cpu"},
		{name: "explicit CPU override", requested: "cpu", envBackend: "vulkan", registered: "vulkan", want: "cpu"},
		{name: "environment CPU override", envBackend: "cpu", registered: "vulkan", want: "cpu"},
		{name: "unknown explicit fails visibly", requested: "definitely-not-a-backend", envBackend: "vulkan", registered: "vulkan", wantErr: "--backend"},
		{name: "unknown environment fails visibly", envBackend: "definitely-not-a-backend", registered: "vulkan", wantErr: "FAK_BACKEND"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestServeBackendAutoChild$")
			cmd.Env = backendAutoChildEnvironment(map[string]string{
				backendAutoChildEnv:                 "1",
				"FAK_TEST_SERVE_BACKEND_REQUESTED":  tt.requested,
				"FAK_TEST_SERVE_BACKEND_REGISTERED": tt.registered,
				"FAK_TEST_SERVE_BACKEND_WANT":       tt.want,
				"FAK_TEST_SERVE_BACKEND_WANT_ERROR": tt.wantErr,
				"FAK_BACKEND":                       tt.envBackend,
				"FAK_VULKAN_SPIRV":                  "",
			})
			if out, err := cmd.CombinedOutput(); err != nil {
				if ctx.Err() != nil {
					t.Fatalf("child resolver witness timed out: %v", ctx.Err())
				}
				t.Fatalf("child resolver witness failed: %v\n%s", err, out)
			}
		})
	}
}

func TestServeBackendAutoMetalDoesNotSuppressEnvironmentError(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestServeBackendAutoChild$")
	cmd.Env = backendAutoChildEnvironment(map[string]string{
		backendAutoChildEnv:                 "1",
		"FAK_TEST_SERVE_BACKEND_ACTION":     "metal",
		"FAK_TEST_SERVE_BACKEND_WANT_ERROR": "FAK_BACKEND",
		"FAK_BACKEND":                       "definitely-not-a-backend",
		"FAK_VULKAN_SPIRV":                  "",
	})
	if out, err := cmd.CombinedOutput(); err != nil {
		if ctx.Err() != nil {
			t.Fatalf("child Metal resolver witness timed out: %v", ctx.Err())
		}
		t.Fatalf("child Metal resolver witness failed: %v\n%s", err, out)
	}
}

func TestServeBackendAutoMetalRejectsExplicitAuto(t *testing.T) {
	t.Setenv("FAK_BACKEND", "")
	if _, err := resolveServeMetal(true, false, "auto"); err == nil || !strings.Contains(err.Error(), "--backend") {
		t.Fatalf("resolveServeMetal explicit auto error = %v, want --backend conflict", err)
	}
}

func TestServeBackendAutoVulkanLeaseEligibility(t *testing.T) {
	autoWant := "false"
	if runtime.GOOS == "linux" || runtime.GOOS == "windows" {
		autoWant = "true"
	}
	tests := []struct {
		name       string
		requested  string
		envBackend string
		registered string
		baseURL    string
		want       string
	}{
		{name: "explicit Vulkan", requested: "vulkan", registered: "vulkan", want: "true"},
		{name: "environment Vulkan", envBackend: "vulkan", registered: "vulkan", want: "true"},
		{name: "explicit auto", requested: "auto", envBackend: "cpu", registered: "vulkan", want: autoWant},
		{name: "environment auto", envBackend: "auto", registered: "vulkan", want: autoWant},
		{name: "CPU override", requested: "cpu", registered: "vulkan", want: "false"},
		{name: "unknown environment", envBackend: "unknown", registered: "vulkan", want: "false"},
		{name: "no registered Vulkan", want: "false"},
		{name: "proxy remains lease free", requested: "vulkan", registered: "vulkan", baseURL: "http://127.0.0.1:8080/v1", want: "false"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestServeBackendAutoChild$")
			cmd.Env = backendAutoChildEnvironment(map[string]string{
				backendAutoChildEnv:                 "1",
				"FAK_TEST_SERVE_BACKEND_ACTION":     "lease",
				"FAK_TEST_SERVE_BACKEND_REQUESTED":  tt.requested,
				"FAK_TEST_SERVE_BACKEND_REGISTERED": tt.registered,
				"FAK_TEST_SERVE_BACKEND_BASE_URL":   tt.baseURL,
				"FAK_TEST_SERVE_BACKEND_WANT":       tt.want,
				"FAK_BACKEND":                       tt.envBackend,
				"FAK_VULKAN_SPIRV":                  "",
			})
			if out, err := cmd.CombinedOutput(); err != nil {
				if ctx.Err() != nil {
					t.Fatalf("child lease witness timed out: %v", ctx.Err())
				}
				t.Fatalf("child lease witness failed: %v\n%s", err, out)
			}
		})
	}
}

func TestServeBackendAutoChild(t *testing.T) {
	if os.Getenv(backendAutoChildEnv) != "1" {
		return
	}
	for _, name := range strings.Split(os.Getenv("FAK_TEST_SERVE_BACKEND_REGISTERED"), ",") {
		name = strings.TrimSpace(name)
		if name == "" {
			continue
		}
		// Registration is isolated to this short-lived child process. The embedded
		// CPU backend is an opaque resolver sentinel and makes no device claim.
		compute.Register(&serveBackendAutoSentinel{Backend: compute.Default(), name: name})
	}
	if os.Getenv("FAK_TEST_SERVE_BACKEND_ACTION") == "lease-load" {
		requested := os.Getenv("FAK_TEST_SERVE_BACKEND_REQUESTED")
		baseURL := os.Getenv("FAK_TEST_SERVE_BACKEND_BASE_URL")
		gguf := os.Getenv("FAK_TEST_SERVE_BACKEND_GGUF")
		model := "mock"
		path := os.Getenv("FAK_TEST_SERVE_BACKEND_LEASE_PATH")
		wantLease := os.Getenv("FAK_TEST_SERVE_BACKEND_WANT") == "true"
		sf := &serveFlags{backendName: &requested, baseURL: &baseURL, ggufPath: &gguf, model: &model}
		var held *gpulease.Lease
		if !wantLease {
			var err error
			held, err = gpulease.Acquire(gpulease.Options{Path: path, NoWait: true})
			if err != nil {
				t.Fatalf("hold unrelated lease: %v", err)
			}
			defer held.Release()
		}
		loads := 0
		release, err := loadServeModelWithVulkanLease(sf, gpulease.Options{Path: path}, func() { loads++ })
		if err != nil {
			t.Fatalf("serve admission: %v", err)
		}
		if loads != 1 {
			t.Fatalf("load callback calls = %d, want 1", loads)
		}
		if wantLease {
			if competing, err := gpulease.Acquire(gpulease.Options{Path: path, NoWait: true}); !errors.Is(err, gpulease.ErrBusy) {
				if err == nil {
					competing.Release()
				}
				t.Fatalf("selected Vulkan GGUF did not retain lease: %v", err)
			}
		}
		release()
		if wantLease {
			after, err := gpulease.Acquire(gpulease.Options{Path: path, NoWait: true})
			if err != nil {
				t.Fatalf("lease remained held after release: %v", err)
			}
			after.Release()
		}
		return
	}
	if os.Getenv("FAK_TEST_SERVE_BACKEND_ACTION") == "lease" {
		requested := os.Getenv("FAK_TEST_SERVE_BACKEND_REQUESTED")
		baseURL := os.Getenv("FAK_TEST_SERVE_BACKEND_BASE_URL")
		sf := &serveFlags{backendName: &requested, baseURL: &baseURL}
		got := isServeVulkan(sf)
		want := os.Getenv("FAK_TEST_SERVE_BACKEND_WANT") == "true"
		if got != want {
			t.Fatalf("isServeVulkan = %t, want %t (requested=%q FAK_BACKEND=%q baseURL=%q)", got, want, requested, os.Getenv("FAK_BACKEND"), baseURL)
		}
		return
	}
	if os.Getenv("FAK_TEST_SERVE_BACKEND_ACTION") == "metal" {
		_, err := resolveServeMetal(false, false, "")
		wantErr := os.Getenv("FAK_TEST_SERVE_BACKEND_WANT_ERROR")
		if err == nil || !strings.Contains(err.Error(), wantErr) {
			t.Fatalf("Metal resolver error = %v, want text %q", err, wantErr)
		}
		return
	}

	backend, err := resolveServeChatBackend(os.Getenv("FAK_TEST_SERVE_BACKEND_REQUESTED"))
	if wantErr := os.Getenv("FAK_TEST_SERVE_BACKEND_WANT_ERROR"); wantErr != "" {
		if err == nil || !strings.Contains(err.Error(), wantErr) {
			t.Fatalf("error = %v, want text %q", err, wantErr)
		}
		if backend != nil {
			t.Fatalf("backend = %q after error, want nil", backend.Name())
		}
		return
	}
	if err != nil {
		t.Fatalf("resolve: %v", err)
	}
	want := os.Getenv("FAK_TEST_SERVE_BACKEND_WANT")
	if want == "cpu" {
		if backend != nil {
			t.Fatalf("backend = %q, want CPU floor", backend.Name())
		}
		return
	}
	if backend == nil || backend.Name() != want {
		got := "cpu"
		if backend != nil {
			got = backend.Name()
		}
		t.Fatalf("backend = %q, want %q", got, want)
	}
}

func backendAutoChildEnvironment(overrides map[string]string) []string {
	env := make([]string, 0, len(os.Environ())+len(overrides))
	for _, entry := range os.Environ() {
		key, _, ok := strings.Cut(entry, "=")
		if !ok {
			continue
		}
		replaced := strings.HasPrefix(strings.ToUpper(key), "FAK_TEST_SERVE_BACKEND_")
		for override := range overrides {
			if strings.EqualFold(key, override) {
				replaced = true
				break
			}
		}
		if replaced {
			continue
		}
		env = append(env, entry)
	}
	for key, value := range overrides {
		env = append(env, fmt.Sprintf("%s=%s", key, value))
	}
	return env
}
