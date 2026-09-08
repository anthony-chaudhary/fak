package goalrunner

import "time"

// LaunchOptions parameterizes LaunchDetachedWorker.
type LaunchOptions struct {
	Workspace           string        `json:"workspace"`
	PointerFile         string        `json:"pointer_file"`
	PointerContent      string        `json:"pointer_content"`
	LogDir              string        `json:"log_dir"`
	Tag                 string        `json:"tag"`
	FakExe              string        `json:"fak_exe"`
	ClaudeExe           string        `json:"claude_exe"`
	Product             string        `json:"product"`
	WorkKind            string        `json:"work_kind"`
	Tier                string        `json:"tier"`
	Account             string        `json:"account"`
	AccountTag          string        `json:"account_tag"`
	ConfigDir           string        `json:"config_dir"`
	OAuthToken          string        `json:"oauth_token"`
	Model               string        `json:"model"`
	ContextBudgetTokens int           `json:"context_budget_tokens"`
	RestartLimit        int           `json:"restart_limit"`
	MaxDuration         string        `json:"max_duration"`
	Guarded             bool          `json:"guarded"`
	RawSpawn            bool          `json:"raw_spawn"`
	ExposeProfile       string        `json:"expose_profile"`
	Environment         []string      `json:"environment"`
	SkipPreflight       bool          `json:"skip_preflight"`
	PreflightMaxWorkers int           `json:"preflight_max_workers"`
	PlanOnly            bool          `json:"plan_only"`
	AllowTierFallback   bool          `json:"allow_tier_fallback"`
	Timeout             time.Duration `json:"timeout"`
}

// LaunchResult captures the outcome of LaunchDetachedWorker.
type LaunchResult struct {
	PID           int      `json:"pid"`
	Tag           string   `json:"tag"`
	RunID         string   `json:"run_id"`
	LaunchWitness string   `json:"launch_witness"`
	PromptFile    string   `json:"prompt_file"`
	OutLog        string   `json:"out_log"`
	ErrLog        string   `json:"err_log"`
	PIDFile       string   `json:"pid_file"`
	SeedDir       string   `json:"seed_dir"`
	Command       string   `json:"command"`
	Args          []string `json:"args"`
	PlanOnly      bool     `json:"plan_only"`
}

// GoalContract represents a single goal task in a fleet.
type GoalContract struct {
	N        int    `json:"n"`
	Lane     string `json:"lane"`
	Pointer  string `json:"pointer"`
	HTest    string `json:"htest"`
	Host     string `json:"host"`
	Priority int    `json:"priority,omitempty"`
}

// FleetPlan orchestrates serial or ordered execution of goal contracts.
type FleetPlan struct {
	Name                string         `json:"name"`
	Workspace           string         `json:"workspace"`
	ContractsDir        string         `json:"contracts_dir"`
	RunRoot             string         `json:"run_root"`
	Contracts           []GoalContract `json:"contracts"`
	PerWorkerTimeout    time.Duration  `json:"per_worker_timeout"`
	PerWorkerTimeoutMin int            `json:"per_worker_timeout_min"`
	DryRun              bool           `json:"dry_run"`
	RollupPath          string         `json:"rollup_path"`
	StatusPath          string         `json:"status_path"`
}

// WitnessResult captures post-execution audit and verification evidence.
type WitnessResult struct {
	Issue           int    `json:"issue"`
	Outcome         string `json:"outcome"`
	ShippedSHA      string `json:"shipped_sha,omitempty"`
	ShippedVerdict  string `json:"shipped_verdict,omitempty"`
	ShippedWitness  string `json:"shipped_witness,omitempty"`
	BestSeenSHA     string `json:"best_seen_sha,omitempty"`
	BestSeenVerdict string `json:"best_seen_verdict,omitempty"`
	IssueState      string `json:"issue_state,omitempty"`
	SweepReview     string `json:"sweep_review,omitempty"`
	WorkerLog       string `json:"worker_log,omitempty"`
	TimedOut        bool   `json:"timed_out"`
	PID             int    `json:"pid,omitempty"`
}
