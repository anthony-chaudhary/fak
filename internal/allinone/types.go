package allinone

// Config defines the execution parameters for the all-in-one bootstrap orchestrator.
type Config struct {
	LockPath        string `json:"lock_path"`
	BundlePath      string `json:"bundle_path"`
	BundleVerifyKey string `json:"bundle_verify_key"`
	Addr            string `json:"addr"`
	PolicyPath      string `json:"policy_path"`
	Engine          string `json:"engine"`
	DryRun          bool   `json:"dry_run"`
	Mock            bool   `json:"mock"`
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
