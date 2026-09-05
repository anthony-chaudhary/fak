package tb4bench

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"strings"
	"testing"
	"time"
)

func TestDockerReadinessUnavailableSocket(t *testing.T) {
	ctx := context.Background()

	// 1. Non-existent Unix/named socket engine must be unavailable
	engine := NewDockerEngine("/tmp/nonexistent-docker-socket-tb4.sock")
	if engine.IsAvailable(ctx) {
		t.Fatalf("expected engine on nonexistent socket to be unavailable")
	}

	// 2. CheckContainerDaemonReadiness must return ErrContainerDaemonUnavailable
	err := CheckContainerDaemonReadiness(ctx, engine)
	if err == nil {
		t.Fatalf("expected error from CheckContainerDaemonReadiness on unavailable engine, got nil")
	}
	if !errors.Is(err, ErrContainerDaemonUnavailable) {
		t.Errorf("expected errors.Is(err, ErrContainerDaemonUnavailable), got %v", err)
	}
	if !strings.Contains(err.Error(), CONTAINER_DAEMON_UNAVAILABLE) {
		t.Errorf("expected error message to contain %q, got %q", CONTAINER_DAEMON_UNAVAILABLE, err.Error())
	}

	// 3. EnsureSandboxReady in non-mock mode must fail closed with ErrContainerDaemonUnavailable
	readyErr := EnsureSandboxReady(ctx, engine, false)
	if readyErr == nil {
		t.Fatalf("expected EnsureSandboxReady(mock=false) to fail closed on unavailable engine")
	}
	if !errors.Is(readyErr, ErrContainerDaemonUnavailable) {
		t.Errorf("expected ErrContainerDaemonUnavailable from EnsureSandboxReady, got %v", readyErr)
	}

	// 4. nil engine must also fail readiness check
	nilErr := CheckContainerDaemonReadiness(ctx, nil)
	if nilErr == nil || !errors.Is(nilErr, ErrContainerDaemonUnavailable) {
		t.Errorf("expected ErrContainerDaemonUnavailable for nil engine, got %v", nilErr)
	}
}

func TestDockerReadinessMockEngine(t *testing.T) {
	ctx := context.Background()
	mockEngine := NewMockContainerEngine()
	defer mockEngine.Close()

	if !mockEngine.IsAvailable(ctx) {
		t.Fatalf("expected mock engine to report available")
	}

	if err := CheckContainerDaemonReadiness(ctx, mockEngine); err != nil {
		t.Errorf("expected mock engine to pass daemon readiness, got: %v", err)
	}

	if err := EnsureSandboxReady(ctx, mockEngine, false); err != nil {
		t.Errorf("expected EnsureSandboxReady(mock=false) to pass for mock engine, got: %v", err)
	}

	if err := EnsureSandboxReady(ctx, mockEngine, true); err != nil {
		t.Errorf("expected EnsureSandboxReady(mock=true) to pass, got: %v", err)
	}
}

func TestSandboxAirgappedNetworkCommands(t *testing.T) {
	mockEngine := NewMockContainerEngine()
	defer mockEngine.Close()

	ctx := context.Background()
	config := ContainerConfig{
		ImageDigest: "ghcr.io/fak/tb4-sandbox@sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
		Name:        "tb4-test-airgap-cmds",
		NetworkMode: NetworkModeNone,
	}

	inst, err := mockEngine.CreateContainer(ctx, config)
	if err != nil {
		t.Fatalf("failed to create container: %v", err)
	}
	if err := mockEngine.StartContainer(ctx, inst.ID); err != nil {
		t.Fatalf("failed to start container: %v", err)
	}

	// Blocked commands under airgapped network mode
	blockedCommands := [][]string{
		{"curl", "https://api.fak.dev"},
		{"curl", "-s", "http://169.254.169.254/latest/meta-data"},
		{"ping", "1.1.1.1"},
		{"ping", "-c", "1", "8.8.8.8"},
		{"wget", "http://example.com/payload.sh"},
		{"wget", "-q", "https://fak.local/file"},
	}

	for _, cmd := range blockedCommands {
		res, err := mockEngine.ExecCommand(ctx, inst.ID, ExecConfig{Cmd: cmd})
		if err != nil {
			t.Fatalf("unexpected ExecCommand error for %v: %v", cmd, err)
		}
		if res.ExitCode == 0 {
			t.Errorf("expected network command %v to fail under NetworkModeNone, got exit 0", cmd)
		}
		if !strings.Contains(string(res.Stderr), "airgapped") && !strings.Contains(string(res.Stderr), "Network unreachable") {
			t.Errorf("expected airgapped stderr refusal for %v, got: %s", cmd, string(res.Stderr))
		}
	}

	// Non-network commands must pass
	allowedRes, err := mockEngine.ExecCommand(ctx, inst.ID, ExecConfig{
		Cmd: []string{"echo", "airgap-verified"},
	})
	if err != nil {
		t.Fatalf("unexpected error for allowed command: %v", err)
	}
	if allowedRes.ExitCode != 0 {
		t.Errorf("expected echo to exit 0 in airgapped container, got %d", allowedRes.ExitCode)
	}
}

func TestDockerContainerLifecycle(t *testing.T) {
	mockEngine := NewMockContainerEngine()
	defer mockEngine.Close()

	ctx := context.Background()

	// 1. Readiness check
	if err := EnsureSandboxReady(ctx, mockEngine, false); err != nil {
		t.Fatalf("readiness gate failed: %v", err)
	}

	// 2. Container creation with empty NetworkMode defaults to airgapped NetworkModeNone
	config := ContainerConfig{
		ImageDigest: "ghcr.io/fak/tb4-sandbox@sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
		Name:        "tb4-test-lifecycle",
	}
	inst, err := mockEngine.CreateContainer(ctx, config)
	if err != nil {
		t.Fatalf("failed to create container: %v", err)
	}
	if inst.Config.NetworkMode != NetworkModeNone {
		t.Errorf("expected default NetworkMode to be %q, got %q", NetworkModeNone, inst.Config.NetworkMode)
	}

	// 3. Start container
	if err := mockEngine.StartContainer(ctx, inst.ID); err != nil {
		t.Fatalf("failed to start container: %v", err)
	}

	// 4. Inspect container -> running
	state, err := mockEngine.InspectContainer(ctx, inst.ID)
	if err != nil {
		t.Fatalf("failed to inspect container: %v", err)
	}
	if !state.Running {
		t.Errorf("expected container to be running")
	}

	// 5. Exec command inside container
	execRes, err := mockEngine.ExecCommand(ctx, inst.ID, ExecConfig{
		Cmd: []string{"echo", "lifecycle-step"},
	})
	if err != nil {
		t.Fatalf("failed to exec command: %v", err)
	}
	if execRes.ExitCode != 0 {
		t.Errorf("expected exit code 0, got %d", execRes.ExitCode)
	}

	// 6. Stop container
	if err := mockEngine.StopContainer(ctx, inst.ID, 5*time.Second); err != nil {
		t.Fatalf("failed to stop container: %v", err)
	}
	stateAfterStop, err := mockEngine.InspectContainer(ctx, inst.ID)
	if err != nil {
		t.Fatalf("failed to inspect stopped container: %v", err)
	}
	if stateAfterStop.Running {
		t.Errorf("expected container to not be running after stop")
	}

	// 7. Remove container
	if err := mockEngine.RemoveContainer(ctx, inst.ID, true); err != nil {
		t.Fatalf("failed to remove container: %v", err)
	}

	// Inspect after remove must return error
	_, err = mockEngine.InspectContainer(ctx, inst.ID)
	if err == nil {
		t.Errorf("expected inspect after remove to fail, got nil")
	}
}

func TestDockerStreamDemux(t *testing.T) {
	// Frame 1: stdout "hello stdout\n"
	stdoutMsg := []byte("hello stdout\n")
	frame1 := make([]byte, 8+len(stdoutMsg))
	frame1[0] = 1 // stdout
	binary.BigEndian.PutUint32(frame1[4:8], uint32(len(stdoutMsg)))
	copy(frame1[8:], stdoutMsg)

	// Frame 2: stderr "warning stderr\n"
	stderrMsg := []byte("warning stderr\n")
	frame2 := make([]byte, 8+len(stderrMsg))
	frame2[0] = 2 // stderr
	binary.BigEndian.PutUint32(frame2[4:8], uint32(len(stderrMsg)))
	copy(frame2[8:], stderrMsg)

	combined := append(frame1, frame2...)
	stdout, stderr, err := demuxDockerStream(bytes.NewReader(combined))
	if err != nil {
		t.Fatalf("demux failed: %v", err)
	}

	if string(stdout) != string(stdoutMsg) {
		t.Errorf("expected stdout %q, got %q", string(stdoutMsg), string(stdout))
	}
	if string(stderr) != string(stderrMsg) {
		t.Errorf("expected stderr %q, got %q", string(stderrMsg), string(stderr))
	}

	// Test non-multiplexed raw fallback
	rawMsg := []byte("plain non-docker output")
	rawStdout, rawStderr, err := demuxDockerStream(bytes.NewReader(rawMsg))
	if err != nil {
		t.Fatalf("raw demux failed: %v", err)
	}
	if string(rawStdout) != string(rawMsg) {
		t.Errorf("expected raw stdout %q, got %q", string(rawMsg), string(rawStdout))
	}
	if len(rawStderr) != 0 {
		t.Errorf("expected empty stderr for raw stream, got %q", string(rawStderr))
	}
}
