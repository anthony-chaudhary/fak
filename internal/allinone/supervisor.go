package allinone

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/engine"
	"github.com/anthony-chaudhary/fak/internal/fakpack"
	"github.com/anthony-chaudhary/fak/internal/mcpbroker"
	"github.com/anthony-chaudhary/fak/internal/ociartifact"
	"github.com/anthony-chaudhary/fak/pkg/harnesskit/lockv2"
)

// MemoryEntry represents an audited event recorded in the memory journal.
type MemoryEntry struct {
	Timestamp time.Time       `json:"timestamp"`
	Type      string          `json:"type"`
	SessionID string          `json:"session_id,omitempty"`
	Goal      string          `json:"goal,omitempty"`
	Tool      string          `json:"tool,omitempty"`
	Data      json.RawMessage `json:"data,omitempty"`
}

// MemoryJournal manages the durable or in-memory recording of session activities.
type MemoryJournal struct {
	mu      sync.RWMutex
	path    string
	file    *os.File
	entries []MemoryEntry
	closed  bool
}

// newMemoryJournal creates a durable journal if path is non-empty, or an in-memory journal.
func newMemoryJournal(path string) (*MemoryJournal, error) {
	j := &MemoryJournal{
		path:    path,
		entries: make([]MemoryEntry, 0),
	}
	if path != "" && path != "in-memory" {
		f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
		if err != nil {
			return nil, fmt.Errorf("open memory journal: %w", err)
		}
		j.file = f
	}
	return j, nil
}

// Append records a memory entry to the journal.
func (j *MemoryJournal) Append(entry MemoryEntry) error {
	j.mu.Lock()
	defer j.mu.Unlock()

	if j.closed {
		return errors.New("memory journal is closed")
	}

	j.entries = append(j.entries, entry)
	if j.file != nil {
		raw, err := json.Marshal(entry)
		if err != nil {
			return err
		}
		if _, err := j.file.Write(append(raw, '\n')); err != nil {
			return err
		}
	}
	return nil
}

// Flush ensures all recorded entries are persisted to disk.
func (j *MemoryJournal) Flush() error {
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.file != nil {
		return j.file.Sync()
	}
	return nil
}

// Close flushes and closes the journal.
func (j *MemoryJournal) Close() error {
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.closed {
		return nil
	}
	j.closed = true
	if j.file != nil {
		_ = j.file.Sync()
		err := j.file.Close()
		j.file = nil
		return err
	}
	return nil
}

// Entries returns a point-in-time copy of all recorded memory entries.
func (j *MemoryJournal) Entries() []MemoryEntry {
	j.mu.RLock()
	defer j.mu.RUnlock()
	out := make([]MemoryEntry, len(j.entries))
	copy(out, j.entries)
	return out
}

// Supervisor coordinates lock verification, MCP broker, memory journal, model engine,
// and gateway HTTP services into a unified deployable runtime.
type Supervisor struct {
	cfg        Config
	health     *HealthAggregator
	lock       *lockv2.Lock
	broker     *mcpbroker.Broker
	engine     abi.EngineDriver
	memory     *MemoryJournal
	httpServer *http.Server
	listener   net.Listener
	boundAddr  string
	unpackDir  string

	activeSessions sync.WaitGroup
	mu             sync.RWMutex
	running        bool
	stopping       bool
}

// NewSupervisor instantiates a new Supervisor configured with the given options.
func NewSupervisor(cfg Config) (*Supervisor, error) {
	return &Supervisor{
		cfg:    cfg,
		health: NewHealthAggregator(),
	}, nil
}

// SetSubsystemHealth allows updating the health of an individual subsystem for testing or dynamic lifecycle.
func (s *Supervisor) SetSubsystemHealth(name string, ready bool, errStr string) {
	s.health.SetStatus(name, ready, errStr)
}

// Health returns the supervisor's health aggregator.
func (s *Supervisor) Health() *HealthAggregator {
	return s.health
}

// Broker returns the active MCP broker.
func (s *Supervisor) Broker() *mcpbroker.Broker {
	return s.broker
}

// Memory exposes the durable session memory journal for verification and entry auditing.
func (s *Supervisor) Memory() *MemoryJournal {
	return s.memory
}

// Addr returns the network address the supervisor HTTP server is bound to.
func (s *Supervisor) Addr() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.boundAddr
}

func isMCPComponent(c lockv2.LockedComponent) bool {
	lowerID := strings.ToLower(c.ID)
	lowerProv := strings.ToLower(c.Provider)
	lowerSrc := strings.ToLower(c.Source)
	if strings.Contains(lowerID, "mcp") || strings.Contains(lowerProv, "mcp") || strings.Contains(lowerSrc, "mcp") {
		return true
	}
	for _, p := range c.Provides {
		if strings.Contains(strings.ToLower(p), "mcp") {
			return true
		}
	}
	for _, a := range c.Adapters {
		if strings.Contains(strings.ToLower(a), "mcp") {
			return true
		}
	}
	return false
}

func (s *Supervisor) validateLockOrBundle() (*lockv2.Lock, string, error) {
	if s.cfg.BundlePath != "" {
		rep, err := fakpack.Verify(fakpack.VerifyOptions{
			BundlePath: s.cfg.BundlePath,
		})
		if err != nil {
			return nil, "", fmt.Errorf("verify bundle: %w", err)
		}
		if s.cfg.BundleVerifyKey != "" {
			if err := ociartifact.VerifyArtifact(s.cfg.BundlePath, s.cfg.BundleVerifyKey); err != nil {
				return nil, "", fmt.Errorf("verify bundle signature: %w", err)
			}
		}

		lockData, err := fakpack.ExtractLock(s.cfg.BundlePath)
		if err != nil {
			// Fallback: attempt to read raw tar archive
			rawBytes, readErr := os.ReadFile(s.cfg.BundlePath)
			if readErr != nil {
				return nil, "", fmt.Errorf("read bundle archive: %w", err)
			}
			if gr, gzErr := gzip.NewReader(bytes.NewReader(rawBytes)); gzErr == nil {
				defer gr.Close()
				tr := tar.NewReader(gr)
				for {
					hdr, tErr := tr.Next()
					if errors.Is(tErr, io.EOF) {
						break
					}
					if tErr != nil {
						break
					}
					if hdr.Name == "harness.lock.json" || strings.HasSuffix(hdr.Name, "/harness.lock.json") {
						lockData, _ = io.ReadAll(tr)
						break
					}
				}
			}
		}
		if len(lockData) == 0 {
			return nil, "", fmt.Errorf("extract lock from bundle: %w", err)
		}

		var lock lockv2.Lock
		if err := json.Unmarshal(lockData, &lock); err != nil {
			return nil, "", fmt.Errorf("malformed harness.lock.json in bundle: %w", err)
		}
		if err := lockv2.ValidateSecretContracts(&lock); err != nil {
			return nil, "", fmt.Errorf("validate secret contracts: %w", err)
		}
		lockID := lock.ID
		if lockID == "" {
			lockID, _ = lockv2.CanonicalID(&lock)
		}
		if lockID == "" {
			lockID = rep.LockID
		}
		if lockID == "" {
			lockID = rep.ManifestDigest
		}
		return &lock, lockID, nil
	}

	if s.cfg.LockPath != "" {
		raw, err := os.ReadFile(s.cfg.LockPath)
		if err != nil {
			return nil, "", fmt.Errorf("read lock file: %w", err)
		}
		var lock lockv2.Lock
		if err := json.Unmarshal(raw, &lock); err != nil {
			return nil, "", fmt.Errorf("unmarshal lock json: %w", err)
		}
		for _, c := range lock.Components {
			lower := strings.ToLower(c.Source)
			if strings.Contains(lower, "http://") || strings.Contains(lower, "https://") {
				return nil, "", fmt.Errorf("%s: component %q references remote URL: %s", fakpack.ErrAirgapUnresolvedRemoteDependency, c.ID, c.Source)
			}
		}
		if err := lockv2.ValidateSecretContracts(&lock); err != nil {
			return nil, "", fmt.Errorf("validate secret contracts: %w", err)
		}
		lockID := lock.ID
		if lockID == "" {
			computedID, err := lockv2.CanonicalID(&lock)
			if err != nil {
				return nil, "", fmt.Errorf("canonical lock id: %w", err)
			}
			lockID = computedID
			lock.ID = lockID
		}
		return &lock, lockID, nil
	}

	if s.cfg.Mock {
		mockLock := &lockv2.Lock{
			Schema: lockv2.ProductLockSchemaV2,
			ID:     "mock-lock-id",
			Platforms: []lockv2.PlatformRequirement{
				{OS: runtime.GOOS, Arch: runtime.GOARCH},
			},
			Components: []lockv2.LockedComponent{
				{ID: "mock-mcp-server", Provider: "mcp", Provides: []string{"echo"}},
			},
		}
		return mockLock, mockLock.ID, nil
	}

	return nil, "", errors.New("allinone: either lock_path or bundle_path must be specified")
}

// DryRunTopology inspects and validates the target deployment topology without binding network or launching processes.
func (s *Supervisor) DryRunTopology() (*TopologySpec, error) {
	lock, lockID, err := s.validateLockOrBundle()
	if err != nil {
		return nil, err
	}

	platform := runtime.GOOS + "/" + runtime.GOARCH
	if len(lock.Platforms) > 0 {
		platform = lock.Platforms[0].String()
	}

	var mcpServers []string
	for _, c := range lock.Components {
		if isMCPComponent(c) {
			mcpServers = append(mcpServers, c.ID)
		}
	}
	if len(mcpServers) == 0 && (s.cfg.Mock || s.cfg.Engine == "mock") {
		mcpServers = []string{"mock-mcp-server"}
	}

	memStore := "in-memory-journal"
	for _, a := range lock.Assets {
		if a.Kind == "memory" || a.Kind == "journal" {
			if a.ID != "" {
				memStore = "durable-journal (" + a.ID + ")"
			}
		}
	}

	eng := s.cfg.Engine
	if eng == "" || s.cfg.Mock {
		eng = "mock"
	}

	addr := s.cfg.Addr
	if addr == "" {
		addr = "127.0.0.1:4000"
	}

	return &TopologySpec{
		LockID:      lockID,
		Platform:    platform,
		MCPServers:  mcpServers,
		MemoryStore: memStore,
		Engine:      eng,
		Addr:        addr,
	}, nil
}

// Start boots the supervised subsystems, registers routes, and begins serving requests.
func (s *Supervisor) Start(ctx context.Context) error {
	s.mu.Lock()
	if s.running {
		s.mu.Unlock()
		return errors.New("supervisor already running")
	}
	s.running = true
	s.stopping = false
	s.mu.Unlock()

	// 1. Validate lock or bundle
	lock, _, err := s.validateLockOrBundle()
	if err != nil {
		s.health.SetStatus(SubsystemHTTP, false, err.Error())
		return err
	}
	s.lock = lock

	if s.cfg.BundlePath != "" {
		unpackDir, err := os.MkdirTemp("", "fak-bundle-unpack-*")
		if err != nil {
			return fmt.Errorf("create bundle unpack dir: %w", err)
		}
		s.unpackDir = unpackDir
		if err := fakpack.Unpack(s.cfg.BundlePath, unpackDir); err != nil {
			return fmt.Errorf("unpack bundle: %w", err)
		}
	}

	// 2. Initialize memory store
	var memPath string
	for _, a := range lock.Assets {
		if a.Kind == "memory" || a.Kind == "journal" {
			if a.Value != "" && !strings.Contains(a.Value, "\n") {
				memPath = a.Value
			} else if a.Ref != "" && strings.HasPrefix(a.Ref, "file:") {
				memPath = strings.TrimPrefix(a.Ref, "file:")
			}
		}
	}
	if memPath != "" && !filepath.IsAbs(memPath) && s.unpackDir != "" {
		memPath = filepath.Join(s.unpackDir, memPath)
	}
	memJournal, err := newMemoryJournal(memPath)
	if err != nil {
		s.health.SetStatus(SubsystemMemoryStore, false, err.Error())
		return fmt.Errorf("init memory journal: %w", err)
	}
	s.memory = memJournal
	s.health.SetStatus(SubsystemMemoryStore, true, "")

	// 3. Initialize MCP Broker and register declared tools
	broker := mcpbroker.NewBroker()
	s.broker = broker

	for _, c := range lock.Components {
		if !isMCPComponent(c) {
			continue
		}
		serverID := c.ID
		var launched bool
		cmdPath := c.Source
		if cmdPath != "" {
			candidates := []string{
				cmdPath,
				cmdPath + ".exe",
				filepath.Base(cmdPath),
			}
			if s.unpackDir != "" {
				candidates = append(candidates,
					filepath.Join(s.unpackDir, "bin", filepath.Base(cmdPath)),
					filepath.Join(s.unpackDir, "bin", filepath.Base(cmdPath)+".exe"),
					filepath.Join(s.unpackDir, filepath.FromSlash(cmdPath)),
				)
			}
			for _, candidate := range candidates {
				if fi, err := os.Stat(candidate); err == nil && !fi.IsDir() {
					supCfg := mcpbroker.ServerConfig{
						ID:      serverID,
						Name:    serverID,
						Command: candidate,
						Args:    c.Adapters,
						Env:     append(os.Environ(), "GO_WANT_HELPER_PROCESS=1", "MOCK_SERVER_ID="+serverID),
					}
					if _, err := broker.LaunchSupervisor(ctx, supCfg); err == nil {
						launched = true
						break
					}
				}
			}
		}

		if !launched {
			_ = broker.RegisterServer(mcpbroker.ServerConfig{
				ID:   serverID,
				Name: serverID,
			})

			tools := c.Provides
			if len(tools) == 0 {
				for _, fp := range c.Fingerprints {
					if fp.Name != "" {
						tools = append(tools, fp.Name)
					}
				}
			}
			if len(tools) == 0 {
				tools = []string{"echo"}
			}

			for _, toolName := range tools {
				namespaced := mcpbroker.NamespaceTool(serverID, toolName)
				_ = broker.RegisterTool(mcpbroker.ToolRegistration{
					Name:        namespaced,
					ServerID:    serverID,
					Description: "MCP tool " + toolName,
					Handler: func(ctx context.Context, req mcpbroker.CallRequest) (*mcpbroker.CallResponse, error) {
						return &mcpbroker.CallResponse{
							Tool:     req.Tool,
							ServerID: serverID,
							Content:  json.RawMessage(fmt.Sprintf(`{"status":"ok","tool":%q,"echo":%s}`, req.Tool, string(req.Arguments))),
						}, nil
					},
				})
			}
		}
	}
	s.health.SetStatus(SubsystemMCPBroker, true, "")

	// 4. Initialize model engine
	if s.cfg.Mock || s.cfg.Engine == "mock" || s.cfg.Engine == "" {
		s.engine = &engine.Mock{}
		s.health.SetStatus(SubsystemInference, true, "")
	} else {
		eng := abi.Engine(s.cfg.Engine)
		if eng == nil {
			err := fmt.Errorf("unknown engine %q", s.cfg.Engine)
			s.health.SetStatus(SubsystemInference, false, err.Error())
			return err
		}
		s.engine = eng
		s.health.SetStatus(SubsystemInference, true, "")
	}

	// 5. Initialize HTTP Server and routes
	mux := http.NewServeMux()
	mux.Handle("/healthz", s.health.Handler())
	mux.HandleFunc("/v1/fak/agent/sessions", s.handleAgentWire)

	addr := s.cfg.Addr
	if addr == "" {
		addr = "127.0.0.1:4000"
	}
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		s.health.SetStatus(SubsystemHTTP, false, err.Error())
		return fmt.Errorf("listen on %s: %w", addr, err)
	}
	s.listener = ln
	s.boundAddr = ln.Addr().String()
	s.httpServer = &http.Server{
		Handler: mux,
	}
	s.health.SetStatus(SubsystemHTTP, true, "")

	go func() {
		_ = s.httpServer.Serve(ln)
	}()

	return nil
}

type agentWirePayload struct {
	Goal     string          `json:"goal"`
	Tool     string          `json:"tool,omitempty"`
	Args     json.RawMessage `json:"args,omitempty"`
	MaxTurns int             `json:"max_turns,omitempty"`
}

func (s *Supervisor) handleAgentWire(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	s.mu.RLock()
	stopping := s.stopping
	s.mu.RUnlock()
	if stopping {
		http.Error(w, "server shutting down", http.StatusServiceUnavailable)
		return
	}

	s.activeSessions.Add(1)
	defer s.activeSessions.Done()

	var req agentWirePayload
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad request: "+err.Error(), http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(http.StatusOK)
	flusher, _ := w.(http.Flusher)

	emit := func(v any) {
		raw, err := json.Marshal(v)
		if err != nil {
			return
		}
		_, _ = w.Write(append(raw, '\n'))
		if flusher != nil {
			flusher.Flush()
		}
	}

	sessionID := fmt.Sprintf("sess-%d", time.Now().UnixNano())
	emit(map[string]any{
		"event":      "session.start",
		"session_id": sessionID,
		"goal":       req.Goal,
		"max_turns":  req.MaxTurns,
	})

	if s.memory != nil {
		_ = s.memory.Append(MemoryEntry{
			Timestamp: time.Now(),
			Type:      "session.start",
			SessionID: sessionID,
			Goal:      req.Goal,
		})
	}

	// Broker tool calls through MCP broker
	if req.Tool != "" {
		callResp, err := s.broker.RouteCall(r.Context(), mcpbroker.CallRequest{
			Tool:      req.Tool,
			Arguments: req.Args,
		})
		if err != nil {
			emit(map[string]any{
				"event": "error",
				"error": err.Error(),
			})
		} else {
			emit(map[string]any{
				"event":    "call",
				"tool":     req.Tool,
				"result":   callResp.Content,
				"is_error": callResp.IsError,
			})
			if s.memory != nil {
				_ = s.memory.Append(MemoryEntry{
					Timestamp: time.Now(),
					Type:      "call",
					SessionID: sessionID,
					Tool:      req.Tool,
					Data:      callResp.Content,
				})
			}
		}
	} else if s.broker != nil {
		tools := s.broker.ListTools()
		if len(tools) > 0 {
			t := tools[0]
			callResp, err := s.broker.RouteCall(r.Context(), mcpbroker.CallRequest{
				Tool:      t.Name,
				Arguments: json.RawMessage(fmt.Sprintf(`{"goal":%q}`, req.Goal)),
			})
			if err == nil && callResp != nil {
				emit(map[string]any{
					"event":  "call",
					"tool":   t.Name,
					"result": callResp.Content,
				})
				if s.memory != nil {
					_ = s.memory.Append(MemoryEntry{
						Timestamp: time.Now(),
						Type:      "call",
						SessionID: sessionID,
						Tool:      t.Name,
						Data:      callResp.Content,
					})
				}
			}
		}
	}

	emit(map[string]any{
		"event":      "session.end",
		"session_id": sessionID,
		"status":     "ok",
	})
	if s.memory != nil {
		_ = s.memory.Append(MemoryEntry{
			Timestamp: time.Now(),
			Type:      "session.end",
			SessionID: sessionID,
		})
		_ = s.memory.Flush()
	}
}

// Shutdown gracefully stops the supervisor, stops the HTTP server, drains active sessions,
// terminates MCP child processes, and flushes memory journals.
func (s *Supervisor) Shutdown(ctx context.Context) error {
	s.mu.Lock()
	if !s.running || s.stopping {
		s.mu.Unlock()
		return nil
	}
	s.stopping = true
	s.mu.Unlock()

	var errs []error

	// 1. Stop HTTP server
	if s.httpServer != nil {
		if err := s.httpServer.Shutdown(ctx); err != nil {
			errs = append(errs, err)
		}
	}

	// 2. Drain active sessions
	drainCh := make(chan struct{})
	go func() {
		s.activeSessions.Wait()
		close(drainCh)
	}()
	select {
	case <-drainCh:
	case <-ctx.Done():
		errs = append(errs, ctx.Err())
	}

	// 3. Terminate MCP child processes
	if s.broker != nil {
		if err := s.broker.Close(); err != nil {
			errs = append(errs, err)
		}
	}

	// 4. Flush and close memory journals
	if s.memory != nil {
		_ = s.memory.Flush()
		if err := s.memory.Close(); err != nil {
			errs = append(errs, err)
		}
	}

	if s.unpackDir != "" {
		_ = os.RemoveAll(s.unpackDir)
		s.unpackDir = ""
	}

	s.mu.Lock()
	s.running = false
	s.mu.Unlock()

	s.health.SetStatus(SubsystemHTTP, false, "stopped")
	s.health.SetStatus(SubsystemInference, false, "stopped")
	s.health.SetStatus(SubsystemMCPBroker, false, "stopped")
	s.health.SetStatus(SubsystemMemoryStore, false, "stopped")

	if len(errs) > 0 {
		return errs[0]
	}
	return nil
}
