package allinone

import (
	"errors"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// ErrEngineUnconfigured is the typed failure when real mode has no engine configured.
var ErrEngineUnconfigured = errors.New("allinone: engine unconfigured: set cfg.Engine, cfg.EngineDriver, or explicit mock")

// Config defines the execution parameters for the all-in-one bootstrap orchestrator.
type Config struct {
	LockPath        string           `json:"lock_path"`
	BundlePath      string           `json:"bundle_path"`
	BundleVerifyKey string           `json:"bundle_verify_key"`
	Addr            string           `json:"addr"`
	PolicyPath      string           `json:"policy_path"`
	Engine          string           `json:"engine"`
	EngineDriver    abi.EngineDriver `json:"-"`
	ComponentEnv    []string         `json:"-"` // extra env for launched MCP component processes (e.g. test helpers)
	DryRun          bool             `json:"dry_run"`
	Mock            bool             `json:"mock"`
}

// SubsystemStatus captures the operational health and readiness of an individual subsystem.
type SubsystemStatus struct {
	Name  string `json:"name"`
	Ready bool   `json:"ready"`
	Error string `json:"error,omitempty"`
}

// HealthResponse represents the aggregated health status of the orchestrator and all managed subsystems.
type HealthResponse struct {
	Status     string                     `json:"status"` // "ok" | "degraded" | "unavailable"
	Subsystems map[string]SubsystemStatus `json:"subsystems"`
}

// TopologySpec describes the static deployment topology resolved from lock, bundle, or config.
type TopologySpec struct {
	LockID      string   `json:"lock_id"`
	Platform    string   `json:"platform"`
	MCPServers  []string `json:"mcp_servers"`
	MemoryStore string   `json:"memory_store"`
	Engine      string   `json:"engine"`
	Addr        string   `json:"addr"`
}

// ChildProcessInfo captures the runtime state and tracking metadata of a supervised child process.
type ChildProcessInfo struct {
	ID       string `json:"id"`
	PID      int    `json:"pid"`
	Running  bool   `json:"running"`
	Crashed  bool   `json:"crashed"`
	Restarts int    `json:"restarts"`
}

// DefaultEngine is the real-model default: the fak-native in-kernel backend.
// The zero value ("") is NOT defaulted - it fails fast with
// ErrEngineUnconfigured so a missing engine can never silently become a mock.
const DefaultEngine = "inkernel"

// MockEngineID is the only engine id that selects the offline mock driver.
// It is honored only as an explicit opt-in (cfg.Mock or Engine == "mock").
const MockEngineID = "mock"

// IsMock reports whether the config explicitly opts into mock mode.
// Mock mode is NEVER a default: it requires cfg.Mock or Engine == "mock".
func (c Config) IsMock() bool {
	return c.Mock || c.Engine == MockEngineID
}

// ResolvedEngine returns the effective engine label for topology reporting:
// a configured engine id, "custom" for an injected driver, "mock" for explicit
// mock mode, or "unconfigured" for real mode with nothing configured (Start
// fails fast with ErrEngineUnconfigured; it never falls back to a mock).
func (c Config) ResolvedEngine() string {
	if c.Engine != "" {
		return c.Engine
	}
	if c.EngineDriver != nil {
		return "custom"
	}
	if c.Mock {
		return MockEngineID
	}
	return "unconfigured"
}

// Validate enforces the default-real capability contract: real mode requires a
// lock or bundle, an explicitly configured engine (id or driver), and no mock
// selection; mock mode requires the explicit opt-in and never activates by
// default. Returns nil when Start or DryRunTopology may proceed.
func (c Config) Validate() error {
	if !c.IsMock() {
		if c.LockPath == "" && c.BundlePath == "" {
			return errors.New("allinone: either lock_path or bundle_path must be specified (or explicit mock opt-in)")
		}
		if c.Engine == "" && c.EngineDriver == nil {
			return ErrEngineUnconfigured
		}
		return nil
	}
	// Explicit mock mode: an engine id other than "mock" combined with the
	// mock flag is contradictory - refuse rather than guess which wins.
	if c.Mock && c.Engine != "" && c.Engine != MockEngineID {
		return errors.New("allinone: cfg.Mock with explicit Engine " + c.Engine + ": use Engine \"mock\" or a real engine, not both")
	}
	return nil
}
