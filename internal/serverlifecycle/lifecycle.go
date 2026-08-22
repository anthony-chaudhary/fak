// Package serverlifecycle owns one local server instance from configuration
// through readiness and identity-checked teardown.
package serverlifecycle

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
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/processstart"
	"github.com/anthony-chaudhary/fak/internal/serveradapter"
	"github.com/anthony-chaudhary/fak/internal/serverartifact"
	"github.com/anthony-chaudhary/fak/internal/serverproduct"
)

const (
	ResultSchema = "fak.server-lifecycle-result/v1"
	configSchema = "fak.server-lifecycle-config/v1"
	stateSchema  = "fak.server-lifecycle-state/v1"

	SpecFilename    = "server-spec.json"
	ConfigFilename  = "server-runtime.json"
	StateFilename   = "server-state.json"
	ReceiptFilename = "server-receipt.json"
	LogFilename     = "server.log"
	LockFilename    = ".server-lifecycle.lock"

	StateConfigured State = "configured"
	StateStarting   State = "starting"
	StateReady      State = "ready"
	StateStale      State = "stale"
	StateFailed     State = "failed"
	StateStopped    State = "stopped"

	ReasonInstanceLocked           = "INSTANCE_LOCKED"
	ReasonProcessIdentityMismatch  = "PROCESS_IDENTITY_MISMATCH"
	ReasonReceiptOwnershipMismatch = "RECEIPT_OWNERSHIP_MISMATCH"
	ReasonStopTimeout              = "STOP_TIMEOUT"
)

const (
	defaultReadinessTimeout = 5 * time.Minute
	defaultStopTimeout      = 10 * time.Second
	defaultProbeInterval    = 100 * time.Millisecond
	defaultProbeTimeout     = 2 * time.Second
)

// State is the closed operator-visible lifecycle vocabulary.
type State string

// InitOptions is the authored local-process configuration persisted by Init.
type InitOptions struct {
	InstanceDirectory string
	ServerName        string
	ModelPath         string
	ArtifactSHA256    string
	AdapterExecutable string
	ModelAlias        string
	Port              uint16
	TokenWindow       int
	Threads           int
	GPULayers         int
	VersionConstraint string
	ProtocolRevision  string
}

// Options bounds process readiness and teardown waits.
type Options struct {
	ReadinessTimeout time.Duration
	StopTimeout      time.Duration
	ProbeInterval    time.Duration
}

// Evidence reports the observations used to derive Result.State.
type Evidence struct {
	SpecValid            bool `json:"spec_valid"`
	LockHeld             bool `json:"lock_held"`
	ProcessIdentityMatch bool `json:"process_identity_match"`
	ReceiptValid         bool `json:"receipt_valid"`
	ProtocolReady        bool `json:"protocol_ready"`
}

// Result is the stable machine-readable response for every lifecycle command.
type Result struct {
	Schema            string   `json:"schema"`
	Operation         string   `json:"operation"`
	State             State    `json:"state"`
	InstanceDirectory string   `json:"instance_directory"`
	SpecPath          string   `json:"spec_path"`
	ReceiptPath       string   `json:"receipt_path"`
	InstanceID        string   `json:"instance_id,omitempty"`
	Generation        uint64   `json:"generation,omitempty"`
	ProcessID         int      `json:"process_id,omitempty"`
	BaseURL           string   `json:"base_url,omitempty"`
	Refused           bool     `json:"refused"`
	Reason            string   `json:"reason,omitempty"`
	Detail            string   `json:"detail,omitempty"`
	ObservedAt        string   `json:"observed_at"`
	Evidence          Evidence `json:"evidence"`
}

type runtimeConfig struct {
	Schema            string `json:"schema"`
	AdapterExecutable string `json:"adapter_executable"`
	ModelAlias        string `json:"model_alias"`
	TokenWindow       int    `json:"token_window"`
	Threads           int    `json:"threads"`
	GPULayers         int    `json:"gpu_layers"`
}

type stateRecord struct {
	Schema               string `json:"schema"`
	State                State  `json:"state"`
	InstanceID           string `json:"instance_id"`
	Generation           uint64 `json:"generation"`
	ProcessID            int    `json:"process_id,omitempty"`
	ProcessStartIdentity string `json:"process_start_identity,omitempty"`
	BaseURL              string `json:"base_url,omitempty"`
	Error                string `json:"error,omitempty"`
	UpdatedAt            string `json:"updated_at"`
	ReadinessDeadline    string `json:"readiness_deadline,omitempty"`
}

type lockRecord struct {
	ProcessID            int    `json:"process_id"`
	ProcessStartIdentity string `json:"process_start_identity"`
	AcquiredAt           string `json:"acquired_at"`
}

type instance struct {
	dir     string
	spec    serverproduct.ServerSpec
	runtime runtimeConfig
}

// RefusalError is returned only for a typed fail-closed lifecycle refusal.
type RefusalError struct {
	Reason string
	Detail string
}

func (e *RefusalError) Error() string { return e.Reason + ": " + e.Detail }

// Init writes a validated spec and runtime configuration without resolving the
// artifact, inspecting the executable, or launching a process.
func Init(_ context.Context, opts InitOptions) (Result, error) {
	dir, err := cleanAbsolute(opts.InstanceDirectory)
	if err != nil {
		return resultFor("init", opts.InstanceDirectory), err
	}
	modelPath, err := cleanAbsolute(opts.ModelPath)
	if err != nil {
		return resultFor("init", dir), fmt.Errorf("model path: %w", err)
	}
	executable, err := cleanAbsolute(opts.AdapterExecutable)
	if err != nil {
		return resultFor("init", dir), fmt.Errorf("adapter executable: %w", err)
	}
	if opts.ModelAlias == "" {
		opts.ModelAlias = opts.ServerName
	}
	if opts.TokenWindow == 0 {
		opts.TokenWindow = 4096
	}
	if opts.Threads == 0 {
		opts.Threads = 1
	}
	if opts.VersionConstraint == "" {
		opts.VersionConstraint = "installed"
	}
	if opts.ProtocolRevision == "" {
		opts.ProtocolRevision = "2026-01"
	}
	digest := strings.TrimPrefix(strings.ToLower(strings.TrimSpace(opts.ArtifactSHA256)), "sha256:")
	specPath := filepath.Join(dir, SpecFilename)
	spec := serverproduct.ServerSpec{
		Schema:            serverproduct.SchemaV1,
		ServerName:        opts.ServerName,
		InstanceDirectory: dir,
		Artifact: serverproduct.ArtifactSpec{
			Reference: modelPath,
			Digest:    "sha256:" + digest,
		},
		Adapter: serverproduct.AdapterSpec{Name: serveradapter.AdapterLlamaServer, VersionConstraint: opts.VersionConstraint},
		Protocol: serverproduct.ProtocolSpec{
			Family:               serverproduct.ProtocolOpenAIHTTP,
			Revision:             opts.ProtocolRevision,
			RequiredCapabilities: []string{"chat.completions", "models.list"},
		},
		Endpoint:   serverproduct.EndpointSpec{BindHost: serveradapter.LocalBindIP, RequestedPort: opts.Port},
		Auth:       serverproduct.AuthReference{Mode: serverproduct.AuthNone},
		Lifecycle:  serverproduct.LifecycleLocalProcess,
		Provenance: serverproduct.Provenance{Kind: serverproduct.ProvenanceAuthored, Source: "server-spec:" + filepath.ToSlash(specPath)},
	}
	specJSON, err := serverproduct.EncodeSpec(spec)
	if err != nil {
		return resultFor("init", dir), err
	}
	runtime := runtimeConfig{
		Schema:            configSchema,
		AdapterExecutable: executable,
		ModelAlias:        opts.ModelAlias,
		TokenWindow:       opts.TokenWindow,
		Threads:           opts.Threads,
		GPULayers:         opts.GPULayers,
	}
	if err := validateRuntime(runtime, modelPath); err != nil {
		return resultFor("init", dir), err
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return resultFor("init", dir), fmt.Errorf("create instance directory: %w", err)
	}
	for _, path := range []string{specPath, filepath.Join(dir, ConfigFilename)} {
		if _, err := os.Stat(path); err == nil {
			return resultFor("init", dir), fmt.Errorf("instance is already initialized: %s", path)
		} else if !errors.Is(err, os.ErrNotExist) {
			return resultFor("init", dir), err
		}
	}
	runtimeJSON, err := marshalJSON(runtime)
	if err != nil {
		return resultFor("init", dir), err
	}
	if err := atomicWrite(filepath.Join(dir, ConfigFilename), runtimeJSON); err != nil {
		return resultFor("init", dir), err
	}
	if err := atomicWrite(specPath, specJSON); err != nil {
		return resultFor("init", dir), err
	}
	result := resultFor("init", dir)
	result.State = StateConfigured
	result.Evidence.SpecValid = true
	return result, nil
}

// Up starts the configured adapter and publishes a receipt only after the full
// protocol readiness probe succeeds.
func Up(ctx context.Context, dir string, opts Options) (result Result, retErr error) {
	inst, err := loadInstance(dir)
	if err != nil {
		return resultFor("up", dir), err
	}
	result = resultFor("up", inst.dir)
	result.Evidence.SpecValid = true
	lock, err := acquireLock(inst.dir)
	if err != nil {
		return refuse(result, ReasonInstanceLocked, err.Error())
	}
	defer func() {
		if err := lock.release(); retErr == nil && err != nil {
			retErr = err
		}
	}()
	opts = normalizeOptions(opts)
	previous, _ := readState(inst.dir)
	if processMatches(previous.ProcessID, previous.ProcessStartIdentity) {
		return refuse(result, ReasonInstanceLocked, fmt.Sprintf("generation %d is still active", previous.Generation))
	}
	generation := previous.Generation + 1
	if generation == 0 {
		generation = 1
	}
	digest, err := serverproduct.DigestSpec(inst.spec)
	if err != nil {
		return result, err
	}
	instanceID := inst.spec.ServerName + "-" + strings.TrimPrefix(digest, "sha256:")[:12]
	deadline := time.Now().UTC().Add(opts.ReadinessTimeout)
	state := stateRecord{
		Schema:            stateSchema,
		State:             StateStarting,
		InstanceID:        instanceID,
		Generation:        generation,
		UpdatedAt:         nowString(),
		ReadinessDeadline: deadline.Format(time.RFC3339Nano),
	}
	if err := writeState(inst.dir, state); err != nil {
		return result, err
	}
	result.InstanceID, result.Generation = instanceID, generation

	resolved, err := serverartifact.Resolve(ctx, serverartifact.Reference{
		Path:   inst.spec.Artifact.Reference,
		SHA256: strings.TrimPrefix(inst.spec.Artifact.Digest, "sha256:"),
	})
	if err != nil {
		return failUp(inst.dir, result, state, err)
	}
	defer resolved.Close()
	identity, err := serveradapter.InspectExecutable(ctx, inst.runtime.AdapterExecutable)
	if err != nil {
		return failUp(inst.dir, result, state, err)
	}
	port := int(inst.spec.Endpoint.RequestedPort)
	if port == 0 {
		port, err = reservePort(inst.spec.Endpoint.BindHost)
		if err != nil {
			return failUp(inst.dir, result, state, err)
		}
	}
	invocation, err := serveradapter.NewLlamaInvocation(identity, serveradapter.InvocationSpec{
		ModelPath:   resolved.Identity().CanonicalPath,
		ModelAlias:  inst.runtime.ModelAlias,
		Port:        port,
		TokenWindow: inst.runtime.TokenWindow,
		Threads:     inst.runtime.Threads,
		GPULayers:   inst.runtime.GPULayers,
	})
	if err != nil {
		return failUp(inst.dir, result, state, err)
	}
	if err := resolved.VerifyUnchanged(ctx); err != nil {
		return failUp(inst.dir, result, state, err)
	}
	cmd := exec.Command(invocation.Executable, invocation.Args...)
	logFile, err := os.OpenFile(filepath.Join(inst.dir, LogFilename), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return failUp(inst.dir, result, state, err)
	}
	cmd.Stdout, cmd.Stderr = logFile, logFile
	if err := cmd.Start(); err != nil {
		_ = logFile.Close()
		return failUp(inst.dir, result, state, err)
	}
	_ = logFile.Close()
	processID := cmd.Process.Pid
	startIdentity, err := waitForProcessIdentity(processID, time.Second)
	if err != nil {
		_ = cmd.Process.Kill()
		_, _ = cmd.Process.Wait()
		return failUp(inst.dir, result, state, err)
	}
	state.ProcessID = processID
	state.ProcessStartIdentity = startIdentity
	state.BaseURL = invocation.BaseURL
	state.UpdatedAt = nowString()
	if err := writeState(inst.dir, state); err != nil {
		stopStartedProcess(cmd, startIdentity, opts.StopTimeout)
		return failUp(inst.dir, result, state, err)
	}
	result.ProcessID, result.BaseURL = processID, invocation.BaseURL

	probe, observedAt, err := waitUntilReady(ctx, invocation, deadline, opts.ProbeInterval)
	if err != nil {
		stopStartedProcess(cmd, startIdentity, opts.StopTimeout)
		return failUp(inst.dir, result, state, err)
	}
	receipt := makeReceipt(inst, resolved.Identity(), identity, probe, observedAt, instanceID, generation, processID, startIdentity, digest)
	ready, err := serverproduct.NewReadyReceipt(inst.spec, receipt)
	if err != nil {
		stopStartedProcess(cmd, startIdentity, opts.StopTimeout)
		return failUp(inst.dir, result, state, err)
	}
	if err := serverproduct.WriteReadyReceipt(filepath.Join(inst.dir, ReceiptFilename), ready); err != nil {
		stopStartedProcess(cmd, startIdentity, opts.StopTimeout)
		return failUp(inst.dir, result, state, err)
	}
	state.State = StateReady
	state.UpdatedAt = nowString()
	state.ReadinessDeadline = ""
	if err := writeState(inst.dir, state); err != nil {
		_ = os.Remove(filepath.Join(inst.dir, ReceiptFilename))
		stopStartedProcess(cmd, startIdentity, opts.StopTimeout)
		return failUp(inst.dir, result, state, err)
	}
	_ = cmd.Process.Release()
	result.State = StateReady
	result.Evidence = Evidence{SpecValid: true, ProcessIdentityMatch: true, ReceiptValid: true, ProtocolReady: true}
	return result, nil
}

// Status derives lifecycle state from the persisted record plus fresh process,
// receipt, and protocol observations.
func Status(ctx context.Context, dir string, opts Options) (Result, error) {
	inst, err := loadInstance(dir)
	if err != nil {
		return resultFor("status", dir), err
	}
	result := resultFor("status", inst.dir)
	result.Evidence.SpecValid = true
	_, lockErr := os.Stat(filepath.Join(inst.dir, LockFilename))
	result.Evidence.LockHeld = lockErr == nil
	state, err := readState(inst.dir)
	if errors.Is(err, os.ErrNotExist) {
		result.State = StateConfigured
		return result, nil
	}
	if err != nil {
		return result, err
	}
	result.InstanceID, result.Generation, result.ProcessID, result.BaseURL = state.InstanceID, state.Generation, state.ProcessID, state.BaseURL
	switch state.State {
	case StateConfigured, StateFailed, StateStopped:
		result.State = state.State
		result.Detail = state.Error
		return result, nil
	case StateStarting:
		if state.ProcessID == 0 {
			if deadline, parseErr := time.Parse(time.RFC3339Nano, state.ReadinessDeadline); parseErr == nil && time.Now().Before(deadline) && result.Evidence.LockHeld {
				result.State = StateStarting
				return result, nil
			}
			result.State, result.Detail = StateStale, "starting state has no live process identity"
			return result, nil
		}
		result.Evidence.ProcessIdentityMatch = processMatches(state.ProcessID, state.ProcessStartIdentity)
		if !result.Evidence.ProcessIdentityMatch {
			result.State, result.Detail = StateStale, "recorded process identity is no longer live"
			return result, nil
		}
		result.State = StateStarting
		return result, nil
	case StateReady:
		result.Evidence.ProcessIdentityMatch = processMatches(state.ProcessID, state.ProcessStartIdentity)
		if !result.Evidence.ProcessIdentityMatch {
			result.State, result.Detail = StateStale, "recorded process identity is no longer live"
			return result, nil
		}
		receipt, err := loadReadyReceipt(inst, state)
		if err != nil {
			result.State, result.Detail = StateStale, err.Error()
			return result, nil
		}
		result.Evidence.ReceiptValid = true
		opts = normalizeOptions(opts)
		probeCtx, cancel := context.WithTimeout(ctx, minDuration(opts.ReadinessTimeout, defaultProbeTimeout))
		_, err = serveradapter.ProbeLlamaServer(probeCtx, http.DefaultClient, serveradapter.ProbeTarget{BaseURL: receipt.Endpoint.BaseURL, ModelAlias: receipt.ModelAlias})
		cancel()
		if err != nil {
			result.State, result.Detail = StateStale, "protocol readiness lost: "+err.Error()
			return result, nil
		}
		result.Evidence.ProtocolReady = true
		result.State = StateReady
		return result, nil
	default:
		return result, fmt.Errorf("unknown lifecycle state %q", state.State)
	}
}

// Down stops only the process whose instance, generation, PID, and OS start
// identity agree across state, receipt, and current observation.
func Down(_ context.Context, dir string, opts Options) (result Result, retErr error) {
	inst, err := loadInstance(dir)
	if err != nil {
		return resultFor("down", dir), err
	}
	result = resultFor("down", inst.dir)
	result.Evidence.SpecValid = true
	lock, err := acquireLock(inst.dir)
	if err != nil {
		return refuse(result, ReasonInstanceLocked, err.Error())
	}
	defer func() {
		if err := lock.release(); retErr == nil && err != nil {
			retErr = err
		}
	}()
	state, err := readState(inst.dir)
	if errors.Is(err, os.ErrNotExist) {
		state = stateRecord{Schema: stateSchema, State: StateStopped, UpdatedAt: nowString()}
		if err := writeState(inst.dir, state); err != nil {
			return result, err
		}
		result.State = StateStopped
		return result, nil
	}
	if err != nil {
		return result, err
	}
	result.InstanceID, result.Generation, result.ProcessID, result.BaseURL = state.InstanceID, state.Generation, state.ProcessID, state.BaseURL
	if state.State == StateStopped || state.ProcessID == 0 {
		state.State, state.Error, state.UpdatedAt = StateStopped, "", nowString()
		if err := writeState(inst.dir, state); err != nil {
			return result, err
		}
		result.State = StateStopped
		return result, nil
	}
	if state.State == StateReady {
		if _, err := loadReadyReceipt(inst, state); err != nil {
			result.State = StateStale
			return refuse(result, ReasonReceiptOwnershipMismatch, err.Error())
		}
		result.Evidence.ReceiptValid = true
	}
	current, live := processIdentity(state.ProcessID)
	if !live {
		state.State, state.Error, state.UpdatedAt = StateStopped, "", nowString()
		if err := writeState(inst.dir, state); err != nil {
			return result, err
		}
		result.State = StateStopped
		return result, nil
	}
	if current != state.ProcessStartIdentity {
		result.State = StateStale
		return refuse(result, ReasonProcessIdentityMismatch, "recorded PID belongs to a different process start identity")
	}
	result.Evidence.ProcessIdentityMatch = true
	if err := signalProcess(state.ProcessID); err != nil {
		return result, fmt.Errorf("signal owned process: %w", err)
	}
	opts = normalizeOptions(opts)
	if !waitForOriginalProcess(state.ProcessID, state.ProcessStartIdentity, opts.StopTimeout) {
		return refuse(result, ReasonStopTimeout, fmt.Sprintf("process %d remained live past %s", state.ProcessID, opts.StopTimeout))
	}
	state.State, state.Error, state.UpdatedAt = StateStopped, "", nowString()
	if err := writeState(inst.dir, state); err != nil {
		return result, err
	}
	result.State = StateStopped
	return result, nil
}

func makeReceipt(inst instance, artifact serverartifact.Identity, adapter serveradapter.ExecutableIdentity, probe serveradapter.ProbeResult, observedAt time.Time, instanceID string, generation uint64, pid int, startIdentity, specDigest string) serverproduct.ServerReceipt {
	capabilities := make([]string, 0, len(probe.Capabilities))
	for _, capability := range probe.Capabilities {
		switch capability {
		case serveradapter.FeatureHealth:
			capabilities = append(capabilities, "health")
		case serveradapter.FeatureModelList:
			capabilities = append(capabilities, "models.list")
		case serveradapter.FeatureChat:
			capabilities = append(capabilities, "chat.completions")
		}
	}
	return serverproduct.ServerReceipt{
		Schema:     serverproduct.SchemaV1,
		State:      serverproduct.ReceiptStateReady,
		Identity:   serverproduct.ServerIdentity{ServerName: inst.spec.ServerName, InstanceID: instanceID},
		SpecDigest: specDigest,
		Generation: generation,
		CreatedAt:  nowString(),
		Artifact:   serverproduct.ArtifactIdentity{Reference: artifact.CanonicalPath, Digest: "sha256:" + artifact.Digest},
		Adapter:    serverproduct.AdapterIdentity{Name: adapter.Adapter, Version: adapter.Version, ExecutableDigest: adapter.VersionDigest},
		Endpoint:   serverproduct.LoopbackEndpoint{BaseURL: probe.BaseURL},
		ModelAlias: probe.ModelAlias,
		Auth:       inst.spec.Auth,
		Protocol:   serverproduct.ProtocolObservation{Family: inst.spec.Protocol.Family, Revision: inst.spec.Protocol.Revision, Capabilities: capabilities},
		Readiness:  serverproduct.ReadinessEvidence{Probe: serveradapter.ProbeSchema, ProbeDigest: probe.ProbeDigest, ObservedAt: observedAt.UTC().Format(time.RFC3339Nano)},
		Ownership:  serverproduct.OwnershipReference{InstanceID: instanceID, ProcessID: pid, ProcessStartIdentity: startIdentity},
		Provenance: serverproduct.ReceiptProvenance{
			Spec:      inst.spec.Provenance,
			Artifact:  serverproduct.Provenance{Kind: serverproduct.ProvenanceObserved, Source: "sha256-check"},
			Adapter:   serverproduct.Provenance{Kind: serverproduct.ProvenanceObserved, Source: "process-inspection"},
			Endpoint:  serverproduct.Provenance{Kind: serverproduct.ProvenanceObserved, Source: "listener-probe"},
			Readiness: serverproduct.Provenance{Kind: serverproduct.ProvenanceObserved, Source: "http-probe"},
			Ownership: serverproduct.Provenance{Kind: serverproduct.ProvenanceObserved, Source: "process-start"},
		},
	}
}

func loadReadyReceipt(inst instance, state stateRecord) (serverproduct.ServerReceipt, error) {
	raw, err := os.ReadFile(filepath.Join(inst.dir, ReceiptFilename))
	if err != nil {
		return serverproduct.ServerReceipt{}, fmt.Errorf("read ready receipt: %w", err)
	}
	ready, err := serverproduct.DecodeReadyReceipt(inst.spec, raw)
	if err != nil {
		return serverproduct.ServerReceipt{}, err
	}
	receipt := ready.Receipt()
	if receipt.Identity.InstanceID != state.InstanceID || receipt.Generation != state.Generation ||
		receipt.Ownership.ProcessID != state.ProcessID || receipt.Ownership.ProcessStartIdentity != state.ProcessStartIdentity {
		return serverproduct.ServerReceipt{}, errors.New("receipt ownership does not match lifecycle state")
	}
	return receipt, nil
}

func waitUntilReady(ctx context.Context, invocation serveradapter.Invocation, deadline time.Time, interval time.Duration) (serveradapter.ProbeResult, time.Time, error) {
	var lastErr error
	for {
		if err := ctx.Err(); err != nil {
			return serveradapter.ProbeResult{}, time.Time{}, err
		}
		remaining := time.Until(deadline)
		if remaining <= 0 {
			return serveradapter.ProbeResult{}, time.Time{}, fmt.Errorf("readiness deadline exceeded: %w", lastErr)
		}
		attemptScope, cancel := context.WithTimeout(ctx, minDuration(remaining, defaultProbeTimeout))
		probe, err := serveradapter.ProbeLlamaServer(attemptScope, http.DefaultClient, serveradapter.ProbeTarget{BaseURL: invocation.BaseURL, ModelAlias: invocation.ModelAlias})
		cancel()
		if err == nil {
			return probe, time.Now().UTC(), nil
		}
		lastErr = err
		var probeErr *serveradapter.ProbeError
		if errors.As(err, &probeErr) && probeErr.Kind != serveradapter.FailureNotListening && probeErr.Kind != serveradapter.FailureNotReady {
			return serveradapter.ProbeResult{}, time.Time{}, err
		}
		timer := time.NewTimer(minDuration(interval, remaining))
		select {
		case <-ctx.Done():
			timer.Stop()
			return serveradapter.ProbeResult{}, time.Time{}, ctx.Err()
		case <-timer.C:
		}
	}
}

func failUp(dir string, result Result, state stateRecord, cause error) (Result, error) {
	state.State, state.Error, state.UpdatedAt, state.ReadinessDeadline = StateFailed, cause.Error(), nowString(), ""
	if err := writeState(dir, state); err != nil {
		return result, fmt.Errorf("%v; write failed state: %w", cause, err)
	}
	result.State, result.Detail = StateFailed, cause.Error()
	return result, cause
}

func loadInstance(dir string) (instance, error) {
	dir, err := cleanAbsolute(dir)
	if err != nil {
		return instance{}, err
	}
	specRaw, err := os.ReadFile(filepath.Join(dir, SpecFilename))
	if err != nil {
		return instance{}, fmt.Errorf("read server spec: %w", err)
	}
	spec, err := serverproduct.DecodeSpec(specRaw)
	if err != nil {
		return instance{}, err
	}
	if spec.InstanceDirectory != dir {
		return instance{}, errors.New("server spec instance_directory does not match requested directory")
	}
	var runtime runtimeConfig
	if err := readStrict(filepath.Join(dir, ConfigFilename), &runtime); err != nil {
		return instance{}, fmt.Errorf("read runtime config: %w", err)
	}
	if err := validateRuntime(runtime, spec.Artifact.Reference); err != nil {
		return instance{}, err
	}
	return instance{dir: dir, spec: spec, runtime: runtime}, nil
}

func validateRuntime(config runtimeConfig, modelPath string) error {
	if config.Schema != configSchema {
		return fmt.Errorf("runtime schema must be %q", configSchema)
	}
	if _, err := cleanAbsolute(config.AdapterExecutable); err != nil {
		return fmt.Errorf("adapter executable: %w", err)
	}
	identity := serveradapter.ExecutableIdentity{
		Adapter: serveradapter.AdapterLlamaServer, Path: config.AdapterExecutable,
		Version: "configured", VersionDigest: "sha256:" + strings.Repeat("0", 64),
	}
	_, err := serveradapter.NewLlamaInvocation(identity, serveradapter.InvocationSpec{
		ModelPath:  modelPath,
		ModelAlias: config.ModelAlias, Port: 1, TokenWindow: config.TokenWindow, Threads: config.Threads, GPULayers: config.GPULayers,
	})
	return err
}

func cleanAbsolute(path string) (string, error) {
	if strings.TrimSpace(path) == "" {
		return "", errors.New("path is required")
	}
	abs, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	return filepath.Clean(abs), nil
}

func reservePort(host string) (int, error) {
	listener, err := net.Listen("tcp", net.JoinHostPort(host, "0"))
	if err != nil {
		return 0, fmt.Errorf("reserve loopback port: %w", err)
	}
	defer listener.Close()
	return listener.Addr().(*net.TCPAddr).Port, nil
}

func processIdentity(pid int) (string, bool) {
	started, ok := processstart.Start(pid)
	if !ok {
		return "", false
	}
	return started.UTC().Format(time.RFC3339Nano), true
}

func processMatches(pid int, expected string) bool {
	current, ok := processIdentity(pid)
	return ok && expected != "" && current == expected
}

func waitForProcessIdentity(pid int, timeout time.Duration) (string, error) {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if identity, ok := processIdentity(pid); ok {
			return identity, nil
		}
		time.Sleep(10 * time.Millisecond)
	}
	return "", fmt.Errorf("observe process %d start identity", pid)
}

func terminateIfOwned(pid int, identity string) error {
	if !processMatches(pid, identity) {
		return nil
	}
	return signalProcess(pid)
}

func stopStartedProcess(cmd *exec.Cmd, identity string, timeout time.Duration) {
	if cmd == nil || cmd.Process == nil {
		return
	}
	_ = terminateIfOwned(cmd.Process.Pid, identity)
	done := make(chan struct{})
	go func() {
		_, _ = cmd.Process.Wait()
		close(done)
	}()
	if timeout <= 0 {
		timeout = defaultStopTimeout
	}
	select {
	case <-done:
		return
	case <-time.After(timeout):
	}
	if processMatches(cmd.Process.Pid, identity) {
		_ = cmd.Process.Kill()
	}
	select {
	case <-done:
	case <-time.After(time.Second):
	}
}

func waitForOriginalProcess(pid int, identity string, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if !processMatches(pid, identity) {
			return true
		}
		time.Sleep(25 * time.Millisecond)
	}
	return !processMatches(pid, identity)
}

type instanceLock struct{ path string }

func acquireLock(dir string) (instanceLock, error) {
	path := filepath.Join(dir, LockFilename)
	identity, ok := processIdentity(os.Getpid())
	if !ok {
		return instanceLock{}, errors.New("cannot observe lifecycle owner process identity")
	}
	record, err := marshalJSON(lockRecord{ProcessID: os.Getpid(), ProcessStartIdentity: identity, AcquiredAt: nowString()})
	if err != nil {
		return instanceLock{}, err
	}
	file, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
	if err != nil {
		return instanceLock{}, err
	}
	accepted := false
	defer func() {
		_ = file.Close()
		if !accepted {
			_ = os.Remove(path)
		}
	}()
	if _, err := file.Write(record); err != nil {
		return instanceLock{}, err
	}
	if err := file.Sync(); err != nil {
		return instanceLock{}, err
	}
	accepted = true
	return instanceLock{path: path}, nil
}

func (lock instanceLock) release() error {
	if lock.path == "" {
		return nil
	}
	return os.Remove(lock.path)
}

func readState(dir string) (stateRecord, error) {
	var state stateRecord
	if err := readStrict(filepath.Join(dir, StateFilename), &state); err != nil {
		return stateRecord{}, err
	}
	if state.Schema != stateSchema {
		return stateRecord{}, fmt.Errorf("state schema must be %q", stateSchema)
	}
	return state, nil
}

func writeState(dir string, state stateRecord) error {
	data, err := marshalJSON(state)
	if err != nil {
		return err
	}
	return atomicWrite(filepath.Join(dir, StateFilename), data)
}

func readStrict(path string, target any) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return err
	}
	var extra any
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		if err == nil {
			return errors.New("multiple JSON values are not allowed")
		}
		return err
	}
	return nil
}

func marshalJSON(value any) ([]byte, error) {
	var data bytes.Buffer
	encoder := json.NewEncoder(&data)
	encoder.SetEscapeHTML(false)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(value); err != nil {
		return nil, err
	}
	return data.Bytes(), nil
}

func atomicWrite(path string, data []byte) error {
	dir := filepath.Dir(path)
	tmp, err := os.CreateTemp(dir, ".server-lifecycle-*.tmp")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	closed := false
	defer func() {
		if !closed {
			_ = tmp.Close()
		}
		_ = os.Remove(tmpPath)
	}()
	if err := tmp.Chmod(0o600); err != nil {
		return err
	}
	if _, err := tmp.Write(data); err != nil {
		return err
	}
	if err := tmp.Sync(); err != nil {
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	closed = true
	return os.Rename(tmpPath, path)
}

func resultFor(operation, dir string) Result {
	abs, _ := filepath.Abs(dir)
	abs = filepath.Clean(abs)
	return Result{
		Schema:            ResultSchema,
		Operation:         operation,
		InstanceDirectory: abs,
		SpecPath:          filepath.Join(abs, SpecFilename),
		ReceiptPath:       filepath.Join(abs, ReceiptFilename),
		ObservedAt:        nowString(),
	}
}

func refuse(result Result, reason, detail string) (Result, error) {
	result.Refused, result.Reason, result.Detail = true, reason, detail
	return result, &RefusalError{Reason: reason, Detail: detail}
}

func normalizeOptions(opts Options) Options {
	if opts.ReadinessTimeout <= 0 {
		opts.ReadinessTimeout = defaultReadinessTimeout
	}
	if opts.StopTimeout <= 0 {
		opts.StopTimeout = defaultStopTimeout
	}
	if opts.ProbeInterval <= 0 {
		opts.ProbeInterval = defaultProbeInterval
	}
	return opts
}

func minDuration(a, b time.Duration) time.Duration {
	if a < b {
		return a
	}
	return b
}

func nowString() string { return time.Now().UTC().Format(time.RFC3339Nano) }
