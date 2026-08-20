package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/slackoutbox"
)

const (
	guardLifecycleSocketEnv = "FAK_GUARD_LIFECYCLE_SOCKET"
	guardLifecycleTokenEnv  = "FAK_GUARD_LIFECYCLE_TOKEN"
)

type guardLifecycleRequest struct {
	Token             string `json:"token"`
	Operation         string `json:"operation,omitempty"`
	Provider          string `json:"provider,omitempty"`
	Source            string `json:"source,omitempty"`
	ProviderSessionID string `json:"provider_session_id,omitempty"`
}

type guardLifecycleResponse struct {
	Signals  gateway.LifecycleSignals     `json:"signals"`
	Boundary *guardProviderBoundaryResult `json:"boundary,omitempty"`
	Error    string                       `json:"error,omitempty"`
}

const guardLifecycleOperationProviderSessionStart = "provider_session_start"

type guardProviderBoundaryResult struct {
	Applied       bool   `json:"applied"`
	PreviousTrace string `json:"previous_trace"`
	NewTrace      string `json:"new_trace"`
	Provider      string `json:"provider"`
	Source        string `json:"source"`
}

type guardProviderBoundaryFunc func(previousTrace, provider, source, providerSessionID string) (guardProviderBoundaryResult, error)

type guardLifecycleServer struct {
	listener              net.Listener
	path                  string
	token                 string
	srv                   *gateway.Server
	done                  chan struct{}
	once                  sync.Once
	boundaryMu            sync.Mutex
	beginProviderBoundary guardProviderBoundaryFunc
}

func startGuardLifecycleServer(srv *gateway.Server, boundaryFuncs ...guardProviderBoundaryFunc) (*guardLifecycleServer, error) {
	if srv == nil {
		return nil, errors.New("nil gateway server")
	}
	dir, err := guardSessionTempDir("lifecycle")
	if err != nil {
		return nil, err
	}
	path := filepath.Join(dir, "signals.sock")
	ln, err := net.Listen("unix", path)
	if err != nil {
		_ = os.RemoveAll(dir)
		return nil, err
	}
	begin := guardProviderBoundaryFunc(beginGuardProviderBoundary)
	if len(boundaryFuncs) > 0 {
		begin = boundaryFuncs[0]
	}
	g := &guardLifecycleServer{
		listener: ln, path: path, token: slackoutbox.NewNonce(), srv: srv,
		done: make(chan struct{}), beginProviderBoundary: begin,
	}
	go g.serve()
	return g, nil
}

func (g *guardLifecycleServer) Env() [][2]string {
	if g == nil {
		return nil
	}
	return [][2]string{{guardLifecycleSocketEnv, g.path}, {guardLifecycleTokenEnv, g.token}}
}

func (g *guardLifecycleServer) Close() {
	if g == nil {
		return
	}
	g.once.Do(func() {
		_ = g.listener.Close()
		<-g.done
		_ = os.RemoveAll(filepath.Dir(g.path))
	})
}

func (g *guardLifecycleServer) serve() {
	defer close(g.done)
	for {
		conn, err := g.listener.Accept()
		if err != nil {
			return
		}
		go g.handle(conn)
	}
}

func (g *guardLifecycleServer) handle(conn net.Conn) {
	defer conn.Close()
	_ = conn.SetDeadline(time.Now().Add(time.Second))
	var req guardLifecycleRequest
	if json.NewDecoder(io.LimitReader(conn, 4096)).Decode(&req) != nil || req.Token != g.token {
		_ = json.NewEncoder(conn).Encode(guardLifecycleResponse{Error: "unauthorized"})
		return
	}
	if req.Operation == guardLifecycleOperationProviderSessionStart {
		result, err := g.applyProviderBoundary(req)
		if err != nil {
			_ = json.NewEncoder(conn).Encode(guardLifecycleResponse{Error: err.Error()})
			return
		}
		_ = json.NewEncoder(conn).Encode(guardLifecycleResponse{Boundary: &result})
		return
	}
	if req.Operation != "" && req.Operation != "snapshot" {
		_ = json.NewEncoder(conn).Encode(guardLifecycleResponse{Error: "unknown lifecycle operation"})
		return
	}
	_ = json.NewEncoder(conn).Encode(guardLifecycleResponse{Signals: g.srv.LifecycleSignalsSnapshot()})
}

func (g *guardLifecycleServer) applyProviderBoundary(req guardLifecycleRequest) (guardProviderBoundaryResult, error) {
	if g.beginProviderBoundary == nil {
		return guardProviderBoundaryResult{}, errors.New("provider session boundaries are not configured")
	}
	if strings.TrimSpace(req.ProviderSessionID) == "" {
		return guardProviderBoundaryResult{}, errors.New("provider session id is required")
	}
	g.boundaryMu.Lock()
	defer g.boundaryMu.Unlock()
	result, err := g.beginProviderBoundary(g.srv.DefaultTraceID(), req.Provider, req.Source, req.ProviderSessionID)
	if err != nil {
		return guardProviderBoundaryResult{}, err
	}
	if strings.TrimSpace(result.NewTrace) != "" {
		g.srv.SetDefaultTraceID(result.NewTrace)
	}
	return result, nil
}

func fetchGuardLifecycleSignals(socketPath, token string, timeout time.Duration) (gateway.LifecycleSignals, error) {
	if socketPath == "" || token == "" {
		return gateway.LifecycleSignals{}, errors.New("lifecycle IPC not configured")
	}
	if timeout <= 0 {
		timeout = 500 * time.Millisecond
	}
	conn, err := net.DialTimeout("unix", socketPath, timeout)
	if err != nil {
		return gateway.LifecycleSignals{}, err
	}
	defer conn.Close()
	_ = conn.SetDeadline(time.Now().Add(timeout))
	if err := json.NewEncoder(conn).Encode(guardLifecycleRequest{Token: token, Operation: "snapshot"}); err != nil {
		return gateway.LifecycleSignals{}, err
	}
	var resp guardLifecycleResponse
	if err := json.NewDecoder(bufio.NewReader(conn)).Decode(&resp); err != nil {
		return gateway.LifecycleSignals{}, err
	}
	if resp.Error != "" {
		return gateway.LifecycleSignals{}, fmt.Errorf("lifecycle IPC: %s", resp.Error)
	}
	return resp.Signals, nil
}

func notifyGuardProviderSessionStart(socketPath, token, provider, source, providerSessionID string, timeout time.Duration) (guardProviderBoundaryResult, error) {
	if socketPath == "" || token == "" {
		return guardProviderBoundaryResult{}, errors.New("lifecycle IPC not configured")
	}
	if timeout <= 0 {
		timeout = 500 * time.Millisecond
	}
	conn, err := net.DialTimeout("unix", socketPath, timeout)
	if err != nil {
		return guardProviderBoundaryResult{}, err
	}
	defer conn.Close()
	_ = conn.SetDeadline(time.Now().Add(timeout))
	req := guardLifecycleRequest{
		Token: token, Operation: guardLifecycleOperationProviderSessionStart,
		Provider: provider, Source: source, ProviderSessionID: providerSessionID,
	}
	if err := json.NewEncoder(conn).Encode(req); err != nil {
		return guardProviderBoundaryResult{}, err
	}
	var resp guardLifecycleResponse
	if err := json.NewDecoder(bufio.NewReader(conn)).Decode(&resp); err != nil {
		return guardProviderBoundaryResult{}, err
	}
	if resp.Error != "" {
		return guardProviderBoundaryResult{}, fmt.Errorf("lifecycle IPC: %s", resp.Error)
	}
	if resp.Boundary == nil {
		return guardProviderBoundaryResult{}, errors.New("lifecycle IPC returned no provider boundary")
	}
	return *resp.Boundary, nil
}
