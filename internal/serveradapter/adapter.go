// Package serveradapter renders and probes supported external inference servers.
package serveradapter

import (
	"context"
	"crypto/sha256"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
)

const (
	AdapterLlamaServer = "llama-server"
	LocalBindIP        = "127.0.0.1"

	MinTokenWindow = 128
	MaxTokenWindow = 1 << 20
	MaxThreads     = 1024
	MaxGPULayers   = 1024
)

// ExecutableIdentity is the observed identity of a supported server binary.
type ExecutableIdentity struct {
	Adapter       string `json:"adapter"`
	Path          string `json:"path"`
	Version       string `json:"version"`
	VersionDigest string `json:"version_digest"`
}

// InvocationSpec is the bounded authored input to a llama-server invocation.
type InvocationSpec struct {
	ModelPath   string `json:"model_path"`
	ModelAlias  string `json:"model_alias"`
	Port        int    `json:"port"`
	TokenWindow int    `json:"token_window"`
	Threads     int    `json:"threads"`
	GPULayers   int    `json:"gpu_layers"`
}

// Invocation is a direct-process command contract. Executable and Args remain
// separate so consumers do not need a shell to recover argument boundaries.
type Invocation struct {
	Executable string   `json:"executable"`
	Args       []string `json:"args"`
	BaseURL    string   `json:"base_url"`
	ModelAlias string   `json:"model_alias"`
}

// Argv returns a defensive copy of the complete direct-process argument vector.
func (i Invocation) Argv() []string {
	argv := make([]string, 0, len(i.Args)+1)
	argv = append(argv, i.Executable)
	return append(argv, i.Args...)
}

// InspectExecutable resolves a llama-server binary and observes its version
// through a direct --version invocation.
func InspectExecutable(ctx context.Context, path string) (ExecutableIdentity, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return ExecutableIdentity{}, fmt.Errorf("executable path is empty")
	}
	abs, err := filepath.Abs(path)
	if err != nil {
		return ExecutableIdentity{}, fmt.Errorf("resolve executable path: %w", err)
	}
	abs = filepath.Clean(abs)
	info, err := os.Stat(abs)
	if err != nil {
		return ExecutableIdentity{}, fmt.Errorf("stat executable: %w", err)
	}
	if !info.Mode().IsRegular() {
		return ExecutableIdentity{}, fmt.Errorf("executable path is not a regular file: %s", abs)
	}
	if runtime.GOOS != "windows" && info.Mode().Perm()&0o111 == 0 {
		return ExecutableIdentity{}, fmt.Errorf("executable path is not executable: %s", abs)
	}
	name := strings.TrimSuffix(strings.ToLower(filepath.Base(abs)), ".exe")
	if name != AdapterLlamaServer {
		return ExecutableIdentity{}, fmt.Errorf("executable name %q is not %s", filepath.Base(abs), AdapterLlamaServer)
	}

	output, err := exec.CommandContext(ctx, abs, "--version").CombinedOutput()
	if err != nil {
		return ExecutableIdentity{}, fmt.Errorf("inspect %s version: %w", AdapterLlamaServer, err)
	}
	normalized := strings.TrimSpace(strings.ReplaceAll(string(output), "\r\n", "\n"))
	version := firstNonEmptyLine(normalized)
	if version == "" || !looksLikeVersionOutput(normalized) {
		return ExecutableIdentity{}, fmt.Errorf("%s --version returned unrecognized output", AdapterLlamaServer)
	}
	sum := sha256.Sum256([]byte(normalized))
	return ExecutableIdentity{
		Adapter:       AdapterLlamaServer,
		Path:          abs,
		Version:       version,
		VersionDigest: "sha256:" + fmt.Sprintf("%x", sum),
	}, nil
}

// NewLlamaInvocation validates the adapter identity and authored options, then
// returns deterministic arguments for the one supported local-process adapter.
func NewLlamaInvocation(identity ExecutableIdentity, spec InvocationSpec) (Invocation, error) {
	if identity.Adapter != AdapterLlamaServer || strings.TrimSpace(identity.Path) == "" ||
		strings.TrimSpace(identity.Version) == "" || !validSHA256(identity.VersionDigest) {
		return Invocation{}, fmt.Errorf("executable identity is not a complete %s observation", AdapterLlamaServer)
	}
	if !filepath.IsAbs(spec.ModelPath) || filepath.Clean(spec.ModelPath) != spec.ModelPath {
		return Invocation{}, fmt.Errorf("model path must be absolute and clean")
	}
	if !strings.EqualFold(filepath.Ext(spec.ModelPath), ".gguf") {
		return Invocation{}, fmt.Errorf("model path must name a .gguf artifact")
	}
	if !validModelAlias(spec.ModelAlias) {
		return Invocation{}, fmt.Errorf("model alias must be 1-128 safe identifier characters")
	}
	if spec.Port < 1 || spec.Port > 65535 {
		return Invocation{}, fmt.Errorf("port must be between 1 and 65535")
	}
	if spec.TokenWindow < MinTokenWindow || spec.TokenWindow > MaxTokenWindow {
		return Invocation{}, fmt.Errorf("context size must be between %d and %d", MinTokenWindow, MaxTokenWindow)
	}
	if spec.Threads < 1 || spec.Threads > MaxThreads {
		return Invocation{}, fmt.Errorf("threads must be between 1 and %d", MaxThreads)
	}
	if spec.GPULayers < 0 || spec.GPULayers > MaxGPULayers {
		return Invocation{}, fmt.Errorf("GPU layers must be between 0 and %d", MaxGPULayers)
	}

	args := []string{
		"--model", spec.ModelPath,
		"--alias", spec.ModelAlias,
		"--host", LocalBindIP,
		"--port", strconv.Itoa(spec.Port),
		"--ctx-size", strconv.Itoa(spec.TokenWindow),
		"--threads", strconv.Itoa(spec.Threads),
		"--n-gpu-layers", strconv.Itoa(spec.GPULayers),
	}
	return Invocation{
		Executable: identity.Path,
		Args:       args,
		BaseURL:    "http://" + LocalBindIP + ":" + strconv.Itoa(spec.Port),
		ModelAlias: spec.ModelAlias,
	}, nil
}

func firstNonEmptyLine(s string) string {
	for _, line := range strings.Split(s, "\n") {
		if line = strings.TrimSpace(line); line != "" {
			return line
		}
	}
	return ""
}

func looksLikeVersionOutput(s string) bool {
	lower := strings.ToLower(s)
	return strings.Contains(lower, "version") || strings.Contains(lower, "build") || strings.Contains(lower, "llama")
}

func validSHA256(s string) bool {
	if !strings.HasPrefix(s, "sha256:") || len(s) != len("sha256:")+64 {
		return false
	}
	for _, r := range s[len("sha256:"):] {
		if !strings.ContainsRune("0123456789abcdef", r) {
			return false
		}
	}
	return true
}

func validModelAlias(s string) bool {
	if len(s) < 1 || len(s) > 128 {
		return false
	}
	for i, r := range s {
		if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' {
			continue
		}
		if i > 0 && strings.ContainsRune("._-/:", r) {
			continue
		}
		return false
	}
	return true
}
