package gateway

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

// ScalarConfig defines the Tier 0 hot-swappable scalar parameters for inference serving
// and gateway operational limits. All fields are accessed lock-free via atomic.Pointer[VersionedScalarConfig].
type ScalarConfig struct {
	CompletionDeadlineMs    uint32 `json:"completion_deadline_ms"`
	StreamProgressTimeoutMs uint32 `json:"stream_progress_timeout_ms"`
	MaxWaitingSeqs          uint32 `json:"max_waiting_seqs"`
	CompactHistoryBudget    int    `json:"compact_history_budget"`
	CompactAnchorHead       int    `json:"compact_anchor_head"`
	LogLevel                string `json:"log_level"`
}

// VersionedScalarConfig bundles the monotonic configuration epoch with the immutable
// ScalarConfig snapshot, enabling atomic single-pointer loads without torn reads.
type VersionedScalarConfig struct {
	Epoch  uint64
	Config ScalarConfig
}

// ScalarConfigPatch represents a partial update to Tier 0 scalar configuration knobs.
// Nil fields indicate unchanged parameters.
type ScalarConfigPatch struct {
	CompletionDeadlineMs    *uint32 `json:"completion_deadline_ms,omitempty"`
	StreamProgressTimeoutMs *uint32 `json:"stream_progress_timeout_ms,omitempty"`
	MaxWaitingSeqs          *uint32 `json:"max_waiting_seqs,omitempty"`
	CompactHistoryBudget    *int    `json:"compact_history_budget,omitempty"`
	CompactAnchorHead       *int    `json:"compact_anchor_head,omitempty"`
	LogLevel                *string `json:"log_level,omitempty"`
}

// ControlConfigResponse is returned by GET /v1/control/config and PATCH /v1/control/config.
type ControlConfigResponse struct {
	Status      string       `json:"status,omitempty"`
	ConfigEpoch uint64       `json:"config_epoch"`
	Config      ScalarConfig `json:"config"`
}

// Validate ensures all fields of ScalarConfig satisfy operational boundaries.
func (c *ScalarConfig) Validate() error {
	if c.CompletionDeadlineMs > 3600000 {
		return errors.New("completion_deadline_ms exceeds ceiling of 3600000ms (1h)")
	}
	if c.StreamProgressTimeoutMs != 0 && (c.StreamProgressTimeoutMs < 5000 || c.StreamProgressTimeoutMs > 600000) {
		return errors.New("stream_progress_timeout_ms must be 0 (default) or within [5000, 600000]ms (5s-600s)")
	}
	if c.MaxWaitingSeqs > 1000000 {
		return errors.New("max_waiting_seqs exceeds ceiling of 1000000")
	}
	if c.CompactHistoryBudget < 0 || c.CompactHistoryBudget > 10000000 {
		return errors.New("compact_history_budget must be between 0 and 10000000")
	}
	if c.CompactAnchorHead < 0 || c.CompactAnchorHead > 1 {
		return errors.New("compact_anchor_head must be 0 or 1")
	}
	if c.LogLevel != "" {
		lvl := strings.ToLower(strings.TrimSpace(c.LogLevel))
		switch lvl {
		case "debug", "info", "warn", "warning", "error":
		default:
			return fmt.Errorf("invalid log_level %q: must be debug, info, warn, or error", c.LogLevel)
		}
	}
	return nil
}

// VersionedConfig returns the active epoch and configuration snapshot atomically.
// Lock-free via atomic.Pointer dereference (<1 ns CPU overhead).
func (s *Server) VersionedConfig() VersionedScalarConfig {
	if s == nil {
		return VersionedScalarConfig{Epoch: 0, Config: ScalarConfig{LogLevel: "info"}}
	}
	if vc := s.versionedConfig.Load(); vc != nil {
		return *vc
	}
	var anchor int
	if s.compactAnchorHead {
		anchor = 1
	}
	return VersionedScalarConfig{
		Epoch: 1,
		Config: ScalarConfig{
			CompactHistoryBudget: s.compactHistoryBudget,
			CompactAnchorHead:    anchor,
			MaxWaitingSeqs:       1024,
			LogLevel:             "info",
		},
	}
}

// ScalarConfig returns the active Tier 0 scalar configuration snapshot.
func (s *Server) ScalarConfig() *ScalarConfig {
	vc := s.VersionedConfig()
	return &vc.Config
}

// ConfigEpoch returns the monotonic configuration epoch counter.
func (s *Server) ConfigEpoch() uint64 {
	return s.VersionedConfig().Epoch
}

// PatchScalarConfig applies partial updates to the active scalar config atomically.
// It validates bounds, increments ConfigEpoch, and stores the new immutable snapshot.
func (s *Server) PatchScalarConfig(patch ScalarConfigPatch) (*ScalarConfig, uint64, error) {
	if s == nil {
		return nil, 0, errors.New("nil server")
	}
	s.controlConfigMu.Lock()
	defer s.controlConfigMu.Unlock()

	cur := s.VersionedConfig()
	next := cur.Config

	if patch.CompletionDeadlineMs != nil {
		next.CompletionDeadlineMs = *patch.CompletionDeadlineMs
	}
	if patch.StreamProgressTimeoutMs != nil {
		next.StreamProgressTimeoutMs = *patch.StreamProgressTimeoutMs
	}
	if patch.MaxWaitingSeqs != nil {
		next.MaxWaitingSeqs = *patch.MaxWaitingSeqs
	}
	if patch.CompactHistoryBudget != nil {
		next.CompactHistoryBudget = *patch.CompactHistoryBudget
	}
	if patch.CompactAnchorHead != nil {
		next.CompactAnchorHead = *patch.CompactAnchorHead
	}
	if patch.LogLevel != nil {
		next.LogLevel = strings.ToLower(strings.TrimSpace(*patch.LogLevel))
	}

	if err := next.Validate(); err != nil {
		return nil, 0, err
	}

	newEpoch := cur.Epoch + 1
	nextVersioned := &VersionedScalarConfig{
		Epoch:  newEpoch,
		Config: next,
	}
	s.versionedConfig.Store(nextVersioned)

	// Propagate dynamic updates to admission controller under lock
	s.admissionMu.RLock()
	ctl := s.admissionCtl
	s.admissionMu.RUnlock()
	if ctl != nil {
		ctl.SetMaxWaiting(int(next.MaxWaitingSeqs))
	}

	return &next, newEpoch, nil
}

// SetScalarConfig replaces the active scalar config atomically.
func (s *Server) SetScalarConfig(sc ScalarConfig) (uint64, error) {
	if err := sc.Validate(); err != nil {
		return 0, err
	}
	s.controlConfigMu.Lock()
	defer s.controlConfigMu.Unlock()

	cur := s.VersionedConfig()
	newEpoch := cur.Epoch + 1
	s.versionedConfig.Store(&VersionedScalarConfig{
		Epoch:  newEpoch,
		Config: sc,
	})

	s.admissionMu.RLock()
	ctl := s.admissionCtl
	s.admissionMu.RUnlock()
	if ctl != nil {
		ctl.SetMaxWaiting(int(sc.MaxWaitingSeqs))
	}

	return newEpoch, nil
}

// ControlSocketServer serves control requests over a local Unix domain socket.
// It supports both HTTP-over-UDS (for standard REST clients) and raw line-delimited
// JSON IPC (for local daemons and lightweight scripts).
type ControlSocketServer struct {
	listener net.Listener
	hs       *http.Server
	chanLn   *controlChanListener
	path     string
	srv      *Server
	done     chan struct{}
	once     sync.Once
}

type controlChanListener struct {
	addr  net.Addr
	conns chan net.Conn
	done  chan struct{}
}

func (c *controlChanListener) Accept() (net.Conn, error) {
	select {
	case conn, ok := <-c.conns:
		if !ok {
			return nil, net.ErrClosed
		}
		return conn, nil
	case <-c.done:
		return nil, net.ErrClosed
	}
}

func (c *controlChanListener) Close() error {
	return nil
}

func (c *controlChanListener) Addr() net.Addr {
	return c.addr
}

type controlPrefixedConn struct {
	net.Conn
	r io.Reader
}

func (p *controlPrefixedConn) Read(b []byte) (int, error) {
	return p.r.Read(b)
}

// StartControlSocket binds a Unix domain socket at socketPath and starts serving.
func (s *Server) StartControlSocket(socketPath string) (*ControlSocketServer, error) {
	if s == nil {
		return nil, errors.New("nil server")
	}
	socketPath = strings.TrimSpace(socketPath)
	if socketPath == "" {
		return nil, errors.New("empty socket path")
	}

	_ = os.Remove(socketPath)
	ln, err := net.Listen("unix", socketPath)
	if err != nil {
		return nil, fmt.Errorf("listen unix %s: %w", socketPath, err)
	}
	_ = os.Chmod(socketPath, 0600)

	chanLn := &controlChanListener{
		addr:  ln.Addr(),
		conns: make(chan net.Conn, 64),
		done:  make(chan struct{}),
	}

	cs := &ControlSocketServer{
		listener: ln,
		chanLn:   chanLn,
		path:     socketPath,
		srv:      s,
		done:     make(chan struct{}),
	}
	cs.hs = &http.Server{
		Handler: s.Handler(),
	}

	go func() {
		_ = cs.hs.Serve(chanLn)
	}()
	go cs.serve()

	s.controlSocketMu.Lock()
	s.controlSocket = cs
	s.controlSocketMu.Unlock()

	return cs, nil
}

func (cs *ControlSocketServer) serve() {
	defer close(cs.done)
	for {
		conn, err := cs.listener.Accept()
		if err != nil {
			return
		}
		go cs.handle(conn)
	}
}

type controlIPCRequest struct {
	Operation string             `json:"operation,omitempty"`
	Op        string             `json:"op,omitempty"`
	Patch     *ScalarConfigPatch `json:"patch,omitempty"`
	Config    *ScalarConfig      `json:"config,omitempty"`
}

type controlIPCResponse struct {
	Status      string        `json:"status"`
	ConfigEpoch uint64        `json:"config_epoch,omitempty"`
	Config      *ScalarConfig `json:"config,omitempty"`
	Error       string        `json:"error,omitempty"`
}

func (cs *ControlSocketServer) handle(conn net.Conn) {
	_ = conn.SetDeadline(time.Now().Add(5 * time.Second))
	lr := io.LimitReader(conn, 64*1024)
	br := bufio.NewReader(lr)
	b, err := br.Peek(1)
	if err != nil {
		_ = conn.Close()
		return
	}

	// If connection starts with '{', handle as line-delimited JSON IPC.
	if b[0] == '{' {
		defer conn.Close()
		line, err := br.ReadBytes('\n')
		if err != nil && len(line) == 0 {
			return
		}
		var req controlIPCRequest
		if err := json.Unmarshal(bytes.TrimSpace(line), &req); err != nil {
			_ = json.NewEncoder(conn).Encode(controlIPCResponse{Status: "error", Error: "invalid json: " + err.Error()})
			return
		}

		op := strings.ToLower(strings.TrimSpace(req.Operation))
		if op == "" {
			op = strings.ToLower(strings.TrimSpace(req.Op))
		}

		switch op {
		case "get_config", "get":
			vc := cs.srv.VersionedConfig()
			_ = json.NewEncoder(conn).Encode(controlIPCResponse{
				Status:      "ok",
				ConfigEpoch: vc.Epoch,
				Config:      &vc.Config,
			})
		case "patch_config", "patch":
			if req.Patch == nil {
				_ = json.NewEncoder(conn).Encode(controlIPCResponse{Status: "error", Error: "missing patch object"})
				return
			}
			updated, epoch, err := cs.srv.PatchScalarConfig(*req.Patch)
			if err != nil {
				_ = json.NewEncoder(conn).Encode(controlIPCResponse{Status: "error", Error: err.Error()})
				return
			}
			_ = json.NewEncoder(conn).Encode(controlIPCResponse{
				Status:      "applied",
				ConfigEpoch: epoch,
				Config:      updated,
			})
		default:
			_ = json.NewEncoder(conn).Encode(controlIPCResponse{Status: "error", Error: "unknown operation: " + op})
		}
		return
	}

	// Otherwise, clear deadline and forward buffered connection to HTTP server.
	_ = conn.SetDeadline(time.Time{})
	prefixed := &controlPrefixedConn{
		Conn: conn,
		r:    io.MultiReader(br, conn),
	}
	select {
	case cs.chanLn.conns <- prefixed:
	case <-cs.done:
		_ = conn.Close()
	}
}

// Path returns the socket path.
func (cs *ControlSocketServer) Path() string {
	if cs == nil {
		return ""
	}
	return cs.path
}

// Close gracefully terminates the socket server and removes the socket file.
func (cs *ControlSocketServer) Close() error {
	if cs == nil {
		return nil
	}
	var err error
	cs.once.Do(func() {
		close(cs.chanLn.done)
		_ = cs.hs.Close()
		err = cs.listener.Close()
		for {
			select {
			case c := <-cs.chanLn.conns:
				_ = c.Close()
			default:
				goto drained
			}
		}
	drained:
		<-cs.done
		_ = os.Remove(cs.path)
	})
	return err
}
