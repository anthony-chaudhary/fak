package tb4bench

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"sync"
	"time"
)

const (
	NetworkModeNone     = "none"
	NetworkModeBridge   = "bridge"
	DefaultSocketPath   = "/var/run/docker.sock"
	WindowsPipePath     = `//./pipe/docker_engine`
	ContainerNamePrefix = "tb4-sandbox-"
)

var digestPattern = regexp.MustCompile(`@[a-zA-Z0-9_-]+:([a-fA-F0-9]{64})$`)

// ValidateImageDigest ensures the container image specifies an immutable SHA-256 digest.
func ValidateImageDigest(image string) error {
	if strings.Contains(image, ":latest") {
		return fmt.Errorf("mutable image tag ':latest' forbidden in TB4: %s", image)
	}
	if !digestPattern.MatchString(image) {
		return fmt.Errorf("image %q does not contain a valid pinned digest (@sha256:<hex64>)", image)
	}
	return nil
}

// ContainerConfig defines resource bounds, network fencing, and execution properties.
type ContainerConfig struct {
	ImageDigest string            `json:"image_digest"`
	Name        string            `json:"name"`
	NetworkMode string            `json:"network_mode"`
	NanoCPUs    int64             `json:"nano_cpus"`
	MemoryBytes int64             `json:"memory_bytes"`
	PidsLimit   int64             `json:"pids_limit"`
	WorkingDir  string            `json:"working_dir"`
	Env         []string          `json:"env"`
	Labels      map[string]string `json:"labels"`
}

// ContainerInstance holds metadata for an active or managed container.
type ContainerInstance struct {
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Config    ContainerConfig `json:"config"`
	CreatedAt time.Time       `json:"created_at"`
}

// ContainerState captures runtime status of a container.
type ContainerState struct {
	Running   bool   `json:"running"`
	ExitCode  int    `json:"exit_code"`
	OOMKilled bool   `json:"oom_killed"`
	Error     string `json:"error,omitempty"`
}

// ExecConfig specifies a command to run inside a container.
type ExecConfig struct {
	Cmd        []string      `json:"cmd"`
	WorkingDir string        `json:"working_dir,omitempty"`
	Env        []string      `json:"env,omitempty"`
	User       string        `json:"user,omitempty"`
	Timeout    time.Duration `json:"timeout,omitempty"`
}

// ExecResult captures standard streams and exit status of an executed command.
type ExecResult struct {
	ExitCode   int    `json:"exit_code"`
	Stdout     []byte `json:"stdout"`
	Stderr     []byte `json:"stderr"`
	DurationMs int64  `json:"duration_ms"`
}

// ContainerEngine defines the container orchestration contract for TB4.
type ContainerEngine interface {
	CreateContainer(ctx context.Context, config ContainerConfig) (*ContainerInstance, error)
	StartContainer(ctx context.Context, id string) error
	ExecCommand(ctx context.Context, id string, cmd ExecConfig) (*ExecResult, error)
	StopContainer(ctx context.Context, id string, timeout time.Duration) error
	RemoveContainer(ctx context.Context, id string, force bool) error
	InspectContainer(ctx context.Context, id string) (*ContainerState, error)
	ReapOrphaned(ctx context.Context, prefix string) (int, error)
	IsAvailable(ctx context.Context) bool
}

// DockerEngine communicates with the Docker/Podman daemon over Unix domain socket or Windows named pipe.
type DockerEngine struct {
	client     *http.Client
	socketPath string
}

// NewDockerEngine initializes a Docker engine client using default sockets.
func NewDockerEngine(customSocket string) *DockerEngine {
	socket := customSocket
	if socket == "" {
		if runtime.GOOS == "windows" {
			socket = WindowsPipePath
		} else {
			socket = DefaultSocketPath
		}
	}

	transport := &http.Transport{
		DialContext: func(ctx context.Context, proto, addr string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, "unix", socket)
		},
		DisableKeepAlives: true,
	}

	return &DockerEngine{
		client: &http.Client{
			Transport: transport,
			Timeout:   120 * time.Second,
		},
		socketPath: socket,
	}
}

func (d *DockerEngine) IsAvailable(ctx context.Context) bool {
	if _, err := os.Stat(d.socketPath); err != nil && runtime.GOOS != "windows" {
		return false
	}
	req, err := http.NewRequestWithContext(ctx, "GET", "http://localhost/v1.41/_ping", nil)
	if err != nil {
		return false
	}
	resp, err := d.client.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

func (d *DockerEngine) CreateContainer(ctx context.Context, config ContainerConfig) (*ContainerInstance, error) {
	if err := ValidateImageDigest(config.ImageDigest); err != nil {
		return nil, err
	}

	reqBody := map[string]interface{}{
		"Image":        config.ImageDigest,
		"WorkingDir":   config.WorkingDir,
		"Env":          config.Env,
		"Labels":       config.Labels,
		"AttachStdout": true,
		"AttachStderr": true,
		"HostConfig": map[string]interface{}{
			"NetworkMode": config.NetworkMode,
			"NanoCPUs":    config.NanoCPUs,
			"Memory":      config.MemoryBytes,
			"PidsLimit":   config.PidsLimit,
		},
	}

	data, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	url := fmt.Sprintf("http://localhost/v1.41/containers/create?name=%s", config.Name)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := d.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to create container: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusCreated {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("docker create returned %d: %s", resp.StatusCode, string(body))
	}

	var res struct {
		ID string `json:"Id"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return nil, err
	}

	return &ContainerInstance{
		ID:        res.ID,
		Name:      config.Name,
		Config:    config,
		CreatedAt: time.Now().UTC(),
	}, nil
}

func (d *DockerEngine) StartContainer(ctx context.Context, id string) error {
	url := fmt.Sprintf("http://localhost/v1.41/containers/%s/start", id)
	req, err := http.NewRequestWithContext(ctx, "POST", url, nil)
	if err != nil {
		return err
	}
	resp, err := d.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNoContent && resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("docker start returned %d: %s", resp.StatusCode, string(body))
	}
	return nil
}

func (d *DockerEngine) ExecCommand(ctx context.Context, id string, cmd ExecConfig) (*ExecResult, error) {
	start := time.Now()
	timeout := cmd.Timeout
	if timeout <= 0 {
		timeout = 60 * time.Second
	}
	execCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	// 1. Create exec
	reqBody := map[string]interface{}{
		"AttachStdout": true,
		"AttachStderr": true,
		"Cmd":          cmd.Cmd,
		"WorkingDir":   cmd.WorkingDir,
		"Env":          cmd.Env,
		"User":         cmd.User,
	}
	data, _ := json.Marshal(reqBody)
	createURL := fmt.Sprintf("http://localhost/v1.41/containers/%s/exec", id)
	req, err := http.NewRequestWithContext(execCtx, "POST", createURL, bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := d.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var execCreateRes struct {
		ID string `json:"Id"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&execCreateRes); err != nil {
		return nil, err
	}

	// 2. Start exec
	startURL := fmt.Sprintf("http://localhost/v1.41/exec/%s/start", execCreateRes.ID)
	startBody := map[string]interface{}{
		"Detach": false,
		"Tty":    false,
	}
	startData, _ := json.Marshal(startBody)
	reqStart, err := http.NewRequestWithContext(execCtx, "POST", startURL, bytes.NewReader(startData))
	if err != nil {
		return nil, err
	}
	reqStart.Header.Set("Content-Type", "application/json")
	respStart, err := d.client.Do(reqStart)
	if err != nil {
		return nil, err
	}
	defer respStart.Body.Close()

	rawOut, _ := io.ReadAll(respStart.Body)

	// 3. Inspect exec to get exit code
	inspectURL := fmt.Sprintf("http://localhost/v1.41/exec/%s/json", execCreateRes.ID)
	reqInspect, err := http.NewRequestWithContext(execCtx, "GET", inspectURL, nil)
	if err != nil {
		return nil, err
	}
	respInspect, err := d.client.Do(reqInspect)
	if err != nil {
		return nil, err
	}
	defer respInspect.Body.Close()

	var inspectRes struct {
		ExitCode int `json:"ExitCode"`
	}
	if err := json.NewDecoder(respInspect.Body).Decode(&inspectRes); err != nil {
		return nil, err
	}

	return &ExecResult{
		ExitCode:   inspectRes.ExitCode,
		Stdout:     rawOut,
		Stderr:     nil,
		DurationMs: time.Since(start).Milliseconds(),
	}, nil
}

func (d *DockerEngine) StopContainer(ctx context.Context, id string, timeout time.Duration) error {
	seconds := int(timeout.Seconds())
	url := fmt.Sprintf("http://localhost/v1.41/containers/%s/stop?t=%d", id, seconds)
	req, err := http.NewRequestWithContext(ctx, "POST", url, nil)
	if err != nil {
		return err
	}
	resp, err := d.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	return nil
}

func (d *DockerEngine) RemoveContainer(ctx context.Context, id string, force bool) error {
	url := fmt.Sprintf("http://localhost/v1.41/containers/%s?v=true&force=%t", id, force)
	req, err := http.NewRequestWithContext(ctx, "DELETE", url, nil)
	if err != nil {
		return err
	}
	resp, err := d.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	return nil
}

func (d *DockerEngine) InspectContainer(ctx context.Context, id string) (*ContainerState, error) {
	url := fmt.Sprintf("http://localhost/v1.41/containers/%s/json", id)
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, err
	}
	resp, err := d.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var data struct {
		State struct {
			Running   bool   `json:"Running"`
			ExitCode  int    `json:"ExitCode"`
			OOMKilled bool   `json:"OOMKilled"`
			Error     string `json:"Error"`
		} `json:"State"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return nil, err
	}
	return &ContainerState{
		Running:   data.State.Running,
		ExitCode:  data.State.ExitCode,
		OOMKilled: data.State.OOMKilled,
		Error:     data.State.Error,
	}, nil
}

func (d *DockerEngine) ReapOrphaned(ctx context.Context, prefix string) (int, error) {
	url := "http://localhost/v1.41/containers/json?all=true"
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return 0, err
	}
	resp, err := d.client.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()

	var containers []struct {
		ID    string   `json:"Id"`
		Names []string `json:"Names"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&containers); err != nil {
		return 0, err
	}

	reaped := 0
	for _, c := range containers {
		for _, name := range c.Names {
			trimmed := strings.TrimPrefix(name, "/")
			if strings.HasPrefix(trimmed, prefix) {
				_ = d.RemoveContainer(ctx, c.ID, true)
				reaped++
				break
			}
		}
	}
	return reaped, nil
}

// MockContainerEngine provides an in-memory, deterministic container simulator for offline testing.
type MockContainerEngine struct {
	mu         sync.Mutex
	containers map[string]*ContainerInstance
	workspaces map[string]string
	tempDir    string
}

// NewMockContainerEngine creates a mock container engine.
func NewMockContainerEngine() *MockContainerEngine {
	temp, _ := os.MkdirTemp("", "tb4-mock-engine-*")
	return &MockContainerEngine{
		containers: make(map[string]*ContainerInstance),
		workspaces: make(map[string]string),
		tempDir:    temp,
	}
}

func (m *MockContainerEngine) Close() {
	if m.tempDir != "" {
		_ = os.RemoveAll(m.tempDir)
	}
}

func (m *MockContainerEngine) IsAvailable(ctx context.Context) bool {
	return true
}

func (m *MockContainerEngine) CreateContainer(ctx context.Context, config ContainerConfig) (*ContainerInstance, error) {
	if err := ValidateImageDigest(config.ImageDigest); err != nil {
		return nil, err
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	id := fmt.Sprintf("mock-c-%d", time.Now().UnixNano())
	wsDir := filepath.Join(m.tempDir, id)
	_ = os.MkdirAll(wsDir, 0755)

	inst := &ContainerInstance{
		ID:        id,
		Name:      config.Name,
		Config:    config,
		CreatedAt: time.Now().UTC(),
	}
	m.containers[id] = inst
	m.workspaces[id] = wsDir
	return inst, nil
}

func (m *MockContainerEngine) StartContainer(ctx context.Context, id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, ok := m.containers[id]; !ok {
		return fmt.Errorf("container %s not found", id)
	}
	return nil
}

func (m *MockContainerEngine) ExecCommand(ctx context.Context, id string, cmd ExecConfig) (*ExecResult, error) {
	m.mu.Lock()
	inst, ok := m.containers[id]
	wsDir := m.workspaces[id]
	m.mu.Unlock()

	if !ok {
		return nil, fmt.Errorf("container %s not found", id)
	}

	start := time.Now()

	// Network isolation enforcement check
	if inst.Config.NetworkMode == NetworkModeNone {
		for _, arg := range cmd.Cmd {
			lower := strings.ToLower(arg)
			if strings.Contains(lower, "ping ") || strings.Contains(lower, "curl ") ||
				strings.Contains(lower, "wget ") || strings.HasPrefix(lower, "http://") ||
				strings.HasPrefix(lower, "https://") {
				return &ExecResult{
					ExitCode:   127,
					Stdout:     nil,
					Stderr:     []byte("Network unreachable: container is running in network_mode 'none' (airgapped)"),
					DurationMs: time.Since(start).Milliseconds(),
				}, nil
			}
		}
	}

	if len(cmd.Cmd) == 0 {
		return &ExecResult{ExitCode: 0, DurationMs: time.Since(start).Milliseconds()}, nil
	}

	// Execute locally inside the mock workspace directory
	workDir := wsDir
	if cmd.WorkingDir != "" && cmd.WorkingDir != "/workspace" {
		workDir = filepath.Join(wsDir, strings.TrimPrefix(cmd.WorkingDir, "/workspace/"))
	}
	_ = os.MkdirAll(workDir, 0755)

	var execCmd *exec.Cmd
	if len(cmd.Cmd) == 1 {
		execCmd = exec.CommandContext(ctx, "sh", "-c", cmd.Cmd[0])
	} else {
		execCmd = exec.CommandContext(ctx, cmd.Cmd[0], cmd.Cmd[1:]...)
	}
	execCmd.Dir = workDir
	execCmd.Env = append(os.Environ(), cmd.Env...)

	var stdout, stderr bytes.Buffer
	execCmd.Stdout = &stdout
	execCmd.Stderr = &stderr

	err := execCmd.Run()
	exitCode := 0
	if err != nil {
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) {
			exitCode = exitErr.ExitCode()
		} else {
			exitCode = 1
			stderr.WriteString(err.Error())
		}
	}

	return &ExecResult{
		ExitCode:   exitCode,
		Stdout:     stdout.Bytes(),
		Stderr:     stderr.Bytes(),
		DurationMs: time.Since(start).Milliseconds(),
	}, nil
}

func (m *MockContainerEngine) StopContainer(ctx context.Context, id string, timeout time.Duration) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.containers, id)
	return nil
}

func (m *MockContainerEngine) RemoveContainer(ctx context.Context, id string, force bool) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.containers, id)
	if ws, ok := m.workspaces[id]; ok {
		_ = os.RemoveAll(ws)
		delete(m.workspaces, id)
	}
	return nil
}

func (m *MockContainerEngine) InspectContainer(ctx context.Context, id string) (*ContainerState, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, ok := m.containers[id]; !ok {
		return nil, fmt.Errorf("container %s not found", id)
	}
	return &ContainerState{Running: true, ExitCode: 0}, nil
}

func (m *MockContainerEngine) ReapOrphaned(ctx context.Context, prefix string) (int, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	reaped := 0
	for id, inst := range m.containers {
		if strings.HasPrefix(inst.Name, prefix) {
			delete(m.containers, id)
			if ws, ok := m.workspaces[id]; ok {
				_ = os.RemoveAll(ws)
				delete(m.workspaces, id)
			}
			reaped++
		}
	}
	return reaped, nil
}
