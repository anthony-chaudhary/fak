package tb4bench

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

// LlamaServerConfig defines the parameters for starting the llama.cpp reference server.
type LlamaServerConfig struct {
	BinaryPath  string        `json:"binary_path"`
	ModelPath   string        `json:"model_path"`
	ModelSha256 string        `json:"model_sha256"`
	Host        string        `json:"host"`
	Port        int           `json:"port"`
	ContextSize int           `json:"context_size"`
	BatchSize   int           `json:"batch_size"`
	UBatchSize  int           `json:"ubatch_size"`
	Temperature float64       `json:"temperature"`
	Seed        int64         `json:"seed"`
	TopP        float64       `json:"top_p"`
	TopK        int           `json:"top_k"`
	Parallel    int           `json:"parallel"`
	Threads     int           `json:"threads"`
	GPULayers   int           `json:"gpu_layers"`
	Timeout     time.Duration `json:"timeout"`
}

// DefaultLlamaServerConfig returns the pinned determinism configuration for TB4.
func DefaultLlamaServerConfig(modelPath, modelSha256 string, port int) LlamaServerConfig {
	if port <= 0 {
		freePort, err := FindFreePort()
		if err == nil {
			port = freePort
		} else {
			port = 8080
		}
	}
	return LlamaServerConfig{
		BinaryPath:  FindLlamaServerBinary(),
		ModelPath:   modelPath,
		ModelSha256: modelSha256,
		Host:        "127.0.0.1",
		Port:        port,
		ContextSize: DefaultMaxContextTokens,
		BatchSize:   2048,
		UBatchSize:  512,
		Temperature: DefaultTemperature,
		Seed:        DefaultSeed,
		TopP:        DefaultTopP,
		TopK:        DefaultTopK,
		Parallel:    1,
		Threads:     4,
		GPULayers:   0,
		Timeout:     120 * time.Second,
	}
}

// FindLlamaServerBinary discovers the llama-server binary via env or PATH.
func FindLlamaServerBinary() string {
	if env := os.Getenv("LLAMA_SERVER_PATH"); env != "" {
		if _, err := os.Stat(env); err == nil {
			return env
		}
	}
	if p, err := exec.LookPath("llama-server"); err == nil {
		return p
	}
	return "llama-server"
}

// FindFreePort finds an available ephemeral loopback TCP port.
func FindFreePort() (int, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port, nil
}

// BuildLlamaServerArgs constructs the deterministic CLI argument list.
func BuildLlamaServerArgs(cfg LlamaServerConfig) []string {
	args := []string{
		"--model", cfg.ModelPath,
		"--host", cfg.Host,
		"--port", strconv.Itoa(cfg.Port),
		"--ctx-size", strconv.Itoa(cfg.ContextSize),
		"--batch-size", strconv.Itoa(cfg.BatchSize),
		"--ubatch-size", strconv.Itoa(cfg.UBatchSize),
		"--temp", fmt.Sprintf("%.1f", cfg.Temperature),
		"--seed", strconv.FormatInt(cfg.Seed, 10),
		"--top-p", fmt.Sprintf("%.1f", cfg.TopP),
		"--top-k", strconv.Itoa(cfg.TopK),
		"--parallel", strconv.Itoa(cfg.Parallel),
	}
	if cfg.Threads > 0 {
		args = append(args, "--threads", strconv.Itoa(cfg.Threads))
	}
	if cfg.GPULayers > 0 {
		args = append(args, "--n-gpu-layers", strconv.Itoa(cfg.GPULayers))
	}
	return args
}

// LlamaServerSupervisor manages the lifecycle of a llama-server background instance.
type LlamaServerSupervisor struct {
	mu         sync.Mutex
	config     LlamaServerConfig
	cmd        *exec.Cmd
	client     *http.Client
	mockServer *http.Server
	running    bool
}

// NewLlamaServerSupervisor creates a supervisor for a given config.
func NewLlamaServerSupervisor(cfg LlamaServerConfig) *LlamaServerSupervisor {
	return &LlamaServerSupervisor{
		config: cfg,
		client: &http.Client{Timeout: 5 * time.Second},
	}
}

// Config returns a copy of the active configuration.
func (s *LlamaServerSupervisor) Config() LlamaServerConfig {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.config
}

// BaseURL returns the base URL of the supervised server.
func (s *LlamaServerSupervisor) BaseURL() string {
	return fmt.Sprintf("http://%s:%d", s.config.Host, s.config.Port)
}

// VerifyModelHash verifies that the configured model file matches the required SHA-256 digest.
func (s *LlamaServerSupervisor) VerifyModelHash() error {
	if s.config.ModelSha256 == "" {
		return errors.New("model sha256 hash not specified in config")
	}
	data, err := os.ReadFile(s.config.ModelPath)
	if err != nil {
		return fmt.Errorf("failed to read model file %s: %w", s.config.ModelPath, err)
	}
	h := sha256.Sum256(data)
	got := hex.EncodeToString(h[:])
	expected := strings.TrimPrefix(s.config.ModelSha256, "sha256:")
	if got != expected {
		return fmt.Errorf("model hash mismatch: got %s, want %s", got, expected)
	}
	return nil
}

// Start launches the llama-server subprocess or a mock server.
func (s *LlamaServerSupervisor) Start(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.running {
		return errors.New("server already running")
	}

	// Verify model path if specified and exists
	if s.config.ModelPath != "" && s.config.ModelSha256 != "" {
		if _, err := os.Stat(s.config.ModelPath); err == nil {
			_ = s.VerifyModelHash()
		}
	}

	args := BuildLlamaServerArgs(s.config)
	cmd := exec.CommandContext(ctx, s.config.BinaryPath, args...)
	cmd.Dir = filepath.Dir(s.config.ModelPath)
	s.cmd = cmd

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start llama-server: %w", err)
	}

	s.running = true
	return nil
}

// StartMock starts an in-process mock HTTP server mimicking llama-server's /health and /v1 endpoints.
func (s *LlamaServerSupervisor) StartMock() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		resp := map[string]interface{}{
			"id":      "chatcmpl-mock",
			"object":  "chat.completion",
			"created": time.Now().Unix(),
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"message": map[string]interface{}{
						"role":    "assistant",
						"content": "TASK_COMPLETED",
					},
					"finish_reason": "stop",
				},
			},
			"usage": map[string]int{
				"prompt_tokens":     100,
				"completion_tokens": 20,
				"total_tokens":      120,
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	})

	listener, err := net.Listen("tcp", fmt.Sprintf("%s:%d", s.config.Host, s.config.Port))
	if err != nil {
		return err
	}
	s.config.Port = listener.Addr().(*net.TCPAddr).Port

	s.mockServer = &http.Server{Handler: mux}
	go func() {
		_ = s.mockServer.Serve(listener)
	}()

	s.running = true
	return nil
}

// WaitReady polls the server's /health endpoint until it reports {"status":"ok"}.
func (s *LlamaServerSupervisor) WaitReady(ctx context.Context, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	healthURL := fmt.Sprintf("%s/health", s.BaseURL())

	for time.Now().Before(deadline) {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		req, err := http.NewRequestWithContext(ctx, "GET", healthURL, nil)
		if err == nil {
			resp, err := s.client.Do(req)
			if err == nil {
				defer resp.Body.Close()
				if resp.StatusCode == http.StatusOK {
					var body map[string]interface{}
					if err := json.NewDecoder(resp.Body).Decode(&body); err == nil {
						if status, ok := body["status"].(string); ok && status == "ok" {
							return nil
						}
					}
				}
			}
		}

		time.Sleep(100 * time.Millisecond)
	}

	return fmt.Errorf("llama-server failed to become ready within %v", timeout)
}

// Stop gracefully shuts down the server.
func (s *LlamaServerSupervisor) Stop() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.running {
		return nil
	}

	var err error
	if s.mockServer != nil {
		err = s.mockServer.Close()
		s.mockServer = nil
	}
	if s.cmd != nil && s.cmd.Process != nil {
		_ = s.cmd.Process.Kill()
		_ = s.cmd.Wait()
		s.cmd = nil
	}

	s.running = false
	return err
}
