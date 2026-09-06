package dockerprocess

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
)

var (
	benchCmdSink  *exec.Cmd
	benchBoolSink bool
)

// TestBenchmarkSanity ensures that benchmarked code paths execute cleanly and produce valid outputs.
func TestBenchmarkSanity(t *testing.T) {
	ctx := context.Background()
	cmd := composeCommand(ctx, "/workspace", []string{"KEY=VAL"}, "ps")
	if cmd == nil || cmd.Path == "" {
		t.Fatal("composeCommand returned nil or empty command")
	}

	// Available must return a boolean without panic.
	_ = Available()
}

// BenchmarkComposeCommand measures constructing a standard production Compose command.
func BenchmarkComposeCommand(b *testing.B) {
	ctx := context.Background()
	env := []string{
		"COMPOSE_PROJECT_NAME=fak_dashboard",
		"DOCKER_BUILDKIT=1",
		"PROMETHEUS_CONFIG=/etc/prometheus/prometheus.yml",
	}
	args := []string{"-f", "deploy/compose.yml", "--profile", "local-prometheus", "up", "-d"}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchCmdSink = composeCommand(ctx, "/workspace/project", env, args...)
	}
}

// BenchmarkComposeCommandMinimal measures constructing a bare Compose command with no extra environment.
func BenchmarkComposeCommandMinimal(b *testing.B) {
	ctx := context.Background()
	args := []string{"ps"}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchCmdSink = composeCommand(ctx, "/workspace", nil, args...)
	}
}

// BenchmarkComposeCommandHeavy measures constructing a Compose command with many environment variables and flags.
func BenchmarkComposeCommandHeavy(b *testing.B) {
	ctx := context.Background()
	env := make([]string, 32)
	for i := range env {
		env[i] = fmt.Sprintf("FAK_CONFIG_VAR_%d=production_value_%d", i, i)
	}
	args := []string{
		"-f", "docker-compose.yml",
		"-f", "docker-compose.override.yml",
		"-f", "docker-compose.monitoring.yml",
		"--profile", "metrics",
		"--profile", "tracing",
		"up", "-d",
		"--build",
		"--force-recreate",
		"--remove-orphans",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchCmdSink = composeCommand(ctx, "/workspace/large-deployment", env, args...)
	}
}

// BenchmarkComposeCommandParallel measures concurrent construction of Compose commands across goroutines.
func BenchmarkComposeCommandParallel(b *testing.B) {
	ctx := context.Background()
	env := []string{"COMPOSE_PROJECT_NAME=fak_dashboard", "DOCKER_BUILDKIT=1"}
	args := []string{"-f", "compose.yml", "up", "-d"}

	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			benchCmdSink = composeCommand(ctx, "/workspace", env, args...)
		}
	})
}

// BenchmarkAvailable measures resolving Docker CLI availability in the current environment.
func BenchmarkAvailable(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchBoolSink = Available()
	}
}

// BenchmarkAvailablePathMiss measures Available() performance when Docker is absent from PATH.
func BenchmarkAvailablePathMiss(b *testing.B) {
	b.Setenv("PATH", b.TempDir())
	if runtime.GOOS == "windows" {
		b.Setenv("PATHEXT", ".EXE")
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchBoolSink = Available()
	}
}

// BenchmarkAvailablePathHit measures Available() performance when a Docker binary is present on PATH.
func BenchmarkAvailablePathHit(b *testing.B) {
	dir := b.TempDir()
	binName := "docker"
	if runtime.GOOS == "windows" {
		binName = "docker.exe"
	}
	binPath := filepath.Join(dir, binName)
	if err := os.WriteFile(binPath, []byte("#!/bin/sh\nexit 0\n"), 0755); err != nil {
		b.Fatalf("failed to write dummy docker binary: %v", err)
	}

	b.Setenv("PATH", dir)
	if runtime.GOOS == "windows" {
		b.Setenv("PATHEXT", ".EXE")
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchBoolSink = Available()
	}
}
