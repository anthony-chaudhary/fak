package gym

import (
	"fmt"
	"net/http/httptest"
	"os"
	"strconv"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	_ "github.com/anthony-chaudhary/fak/internal/registrations"
)

// EphemeralGatewayOptions supplies configuration parameters for spinning up an isolated ephemeral gateway.
type EphemeralGatewayOptions struct {
	CASDir              string
	DeferColdTools      bool
	MaxSubturnToolCalls int
	MaxSubturnTokens    int
	CustomPlanner       agent.Planner
}

// EphemeralGateway encapsulates an in-process gateway.Server hosted on an ephemeral httptest.Server.
type EphemeralGateway struct {
	opts              EphemeralGatewayOptions
	casDir            string
	server            *gateway.Server
	httpServer        *httptest.Server
	createdTempCASDir string
	prevCASDir        string
	prevCASDirSet     bool
	prevMaxCalls      string
	prevMaxCallsSet   bool
	prevMaxTokens     string
	prevMaxTokensSet  bool
	closed            bool
	mu                sync.Mutex
}

// NewEphemeralGateway provisions isolated directories and launches an in-process gateway.Server.
func NewEphemeralGateway(opts EphemeralGatewayOptions) (*EphemeralGateway, error) {
	eg := &EphemeralGateway{
		opts: opts,
	}

	// 1. Provision isolated CAS and session scratch directories
	casDir := opts.CASDir
	if casDir == "" {
		td, err := os.MkdirTemp("", "fak-gym-cas-*")
		if err != nil {
			return nil, fmt.Errorf("ephemeral gateway: failed to create temporary CAS directory: %w", err)
		}
		eg.createdTempCASDir = td
		casDir = td
	} else {
		if err := os.MkdirAll(casDir, 0755); err != nil {
			return nil, fmt.Errorf("ephemeral gateway: failed to create CAS directory %q: %w", casDir, err)
		}
	}
	eg.casDir = casDir

	if v, ok := os.LookupEnv("FAK_CTXRESTORE_CAS_DIR"); ok {
		eg.prevCASDir = v
		eg.prevCASDirSet = true
	}
	_ = os.Setenv("FAK_CTXRESTORE_CAS_DIR", casDir)

	// 2. Configure subturn threshold valves if requested
	if opts.MaxSubturnToolCalls > 0 {
		if v, ok := os.LookupEnv("FAK_RESPONSES_MAX_SUBTURN_TOOL_CALLS"); ok {
			eg.prevMaxCalls = v
			eg.prevMaxCallsSet = true
		}
		_ = os.Setenv("FAK_RESPONSES_MAX_SUBTURN_TOOL_CALLS", strconv.Itoa(opts.MaxSubturnToolCalls))
	}

	if opts.MaxSubturnTokens > 0 {
		if v, ok := os.LookupEnv("FAK_RESPONSES_MAX_SUBTURN_TOKENS"); ok {
			eg.prevMaxTokens = v
			eg.prevMaxTokensSet = true
		}
		_ = os.Setenv("FAK_RESPONSES_MAX_SUBTURN_TOKENS", strconv.Itoa(opts.MaxSubturnTokens))
	}

	// 3. Build gateway server
	gwCfg := gateway.Config{
		EngineID:       "mock",
		Model:          "gym-simulation-model",
		VDSO:           true,
		DeferColdTools: opts.DeferColdTools,
	}

	srv, err := gateway.New(gwCfg)
	if err != nil {
		_ = eg.Close()
		return nil, fmt.Errorf("ephemeral gateway: failed to initialize gateway server: %w", err)
	}

	if opts.CustomPlanner != nil {
		srv.SetPlanner(opts.CustomPlanner)
	}

	eg.server = srv
	eg.httpServer = httptest.NewServer(srv.Handler())
	return eg, nil
}

// URL returns the loopback base URL for the ephemeral HTTP test server.
func (g *EphemeralGateway) URL() string {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.httpServer == nil {
		return ""
	}
	return g.httpServer.URL
}

// Server returns the underlying in-process gateway.Server instance.
func (g *EphemeralGateway) Server() *gateway.Server {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.server
}

// CASDir returns the path to the isolated CAS storage directory.
func (g *EphemeralGateway) CASDir() string {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.casDir
}

// Close gracefully terminates the HTTP server and gateway, restores environment variables,
// and reaps provisioned temporary directories.
func (g *EphemeralGateway) Close() error {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.closed {
		return nil
	}
	g.closed = true

	var firstErr error

	if g.httpServer != nil {
		g.httpServer.Close()
	}

	if g.server != nil {
		g.server.Close()
	}

	// Restore environment variables
	if g.prevCASDirSet {
		_ = os.Setenv("FAK_CTXRESTORE_CAS_DIR", g.prevCASDir)
	} else {
		_ = os.Unsetenv("FAK_CTXRESTORE_CAS_DIR")
	}

	if g.prevMaxCallsSet {
		_ = os.Setenv("FAK_RESPONSES_MAX_SUBTURN_TOOL_CALLS", g.prevMaxCalls)
	} else if g.opts.MaxSubturnToolCalls > 0 {
		_ = os.Unsetenv("FAK_RESPONSES_MAX_SUBTURN_TOOL_CALLS")
	}

	if g.prevMaxTokensSet {
		_ = os.Setenv("FAK_RESPONSES_MAX_SUBTURN_TOKENS", g.prevMaxTokens)
	} else if g.opts.MaxSubturnTokens > 0 {
		_ = os.Unsetenv("FAK_RESPONSES_MAX_SUBTURN_TOKENS")
	}

	if g.createdTempCASDir != "" {
		if err := os.RemoveAll(g.createdTempCASDir); err != nil && firstErr == nil {
			firstErr = err
		}
	}

	return firstErr
}
