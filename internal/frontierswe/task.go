// Package frontierswe is the dataset spine for FrontierSWE — Proximal Labs'
// time-to-solution benchmark of 17 long-horizon engineering tasks (a C-to-Zig
// port, an ffmpeg swscale rewrite, an RL post-training run, …). It is the
// FrontierSWE-shaped mirror of internal/swebench's Instance type: a typed,
// in-binary model of one task's agent-facing contract so the rest of the epic
// (#1706) can be exercised offline, with no network, no Docker, and no model.
//
// A FrontierSWE task is a directory under tasks/<name>/ holding task.toml,
// job.yaml, oracle.yaml, instruction.md, and environment/ / solution/ / tests/
// subdirs. The agent-facing contract lives in task.toml:
//
//	version = "1.0"
//	[metadata]    difficulty = "hard"; category = "..."; tags = [...]
//	[agent]       timeout_sec = 72000.0      # the 20-hour wall-clock budget
//	[verifier]    timeout_sec = 86400.0
//	[environment] docker_image = "ghcr.io/proximal-labs/frontier-swe/<name>:v6"
//	              build_timeout_sec; cpus; memory_mb; storage_mb; gpus
//	              allow_internet = false; mcp_servers = []
//
// This file is the typed model: the Task struct (field-for-field with task.toml,
// plus the optional job.yaml / oracle.yaml fold), and the Category enum
// (implementation / performance / ml_research) that classifies each of the 17
// tasks per scripts/score_from_reward.py. load.go parses the files; catalog.go
// holds the canonical task->category list.
package frontierswe

// Category is the scoring family a FrontierSWE task belongs to, mirroring the
// IMPLEMENTATION / PERFORMANCE / ML_RESEARCH lists in the upstream
// scripts/score_from_reward.py. It is NOT the free-form [metadata] category
// string in task.toml (which is descriptive prose like "c-to-zig" or
// "rl-post-training"); the scoring family is fixed by which task name appears in
// which list, and is supplied by the Catalog — see catalog.go.
type Category string

const (
	// CategoryImplementation is the IMPLEMENTATION family: build/port a system
	// (e.g. git-to-zig, lua-native-compiler), graded on whether it works.
	CategoryImplementation Category = "implementation"
	// CategoryPerformance is the PERFORMANCE family: make an existing system
	// faster (e.g. ffmpeg-swscale-rewrite), graded on a speedup reward.
	CategoryPerformance Category = "performance"
	// CategoryMLResearch is the ML_RESEARCH family: an open-ended ML research
	// task (e.g. optimizer-design, frogsgame-rl), graded on a metric reward.
	CategoryMLResearch Category = "ml_research"
)

// Valid reports whether c is one of the three known scoring families.
func (c Category) Valid() bool {
	switch c {
	case CategoryImplementation, CategoryPerformance, CategoryMLResearch:
		return true
	default:
		return false
	}
}

// String returns the category token ("implementation" / "performance" /
// "ml_research"), or "unknown" for the zero value.
func (c Category) String() string {
	if c == "" {
		return "unknown"
	}
	return string(c)
}

// Metadata mirrors task.toml's [metadata] table. difficulty and category here
// are the upstream author's free-form descriptive strings (difficulty buckets
// like "hard" / "very_hard" / "frontier"; a category blurb like "c-to-zig"),
// distinct from the scoring Category enum which is resolved from the task name.
type Metadata struct {
	Difficulty string   `toml:"difficulty"`
	Category   string   `toml:"category"`
	Tags       []string `toml:"tags"`
}

// Timeout mirrors a [agent] / [verifier] table: a single wall-clock budget in
// seconds, stored as float64 because task.toml writes it as a float
// (e.g. 72000.0 — the 20-hour agent budget that dominates a run's cost).
type Timeout struct {
	TimeoutSec float64 `toml:"timeout_sec"`
}

// Environment mirrors task.toml's [environment] table: the full Docker /
// resource envelope the task runs inside.
type Environment struct {
	DockerImage     string   `toml:"docker_image"`
	BuildTimeoutSec float64  `toml:"build_timeout_sec"`
	CPUs            int      `toml:"cpus"`
	MemoryMB        int      `toml:"memory_mb"`
	StorageMB       int      `toml:"storage_mb"`
	GPUs            int      `toml:"gpus"`
	AllowInternet   bool     `toml:"allow_internet"`
	MCPServers      []string `toml:"mcp_servers"`
}

// Job mirrors the fields of job.yaml the spine cares about: which agents run,
// how many attempts, how many concurrent trials, and the artifacts to collect.
// It is folded onto a Task by LoadTask when job.yaml is present; absent fields
// stay at their zero values.
type Job struct {
	Agents           []string `yaml:"agents"`
	NAttempts        int      `yaml:"n_attempts"`
	NConcurrentTrial int      `yaml:"n_concurrent_trials"`
	Artifacts        []string `yaml:"artifacts"`
}

// Oracle mirrors the fields of oracle.yaml the spine cares about. The upstream
// oracle config is small and varies by task; the spine keeps the verification
// command and the reward-key name where present.
type Oracle struct {
	Command   string `yaml:"command"`
	RewardKey string `yaml:"reward_key"`
}

// Task is one FrontierSWE task — the typed model of tasks/<name>/. The toml
// tags map field-for-field onto task.toml so a task.toml unmarshals straight
// into this struct via LoadTask. Name and ScoringCategory are not task.toml
// fields: Name is the directory name and ScoringCategory is overlaid from the
// Catalog (see catalog.go). Job and Oracle are folded from job.yaml / oracle.yaml
// when those files are present, and stay zero-valued when they are not.
type Task struct {
	// Name is the task directory name (e.g. "git-to-zig"). It is the key the
	// Catalog uses to resolve ScoringCategory; it is not stored in task.toml.
	Name string `toml:"-"`

	// ScoringCategory is the implementation / performance / ml_research family
	// from scripts/score_from_reward.py, resolved by Name via the Catalog. Empty
	// when Name is not one of the 17 known tasks.
	ScoringCategory Category `toml:"-"`

	Version     string      `toml:"version"`
	Metadata    Metadata    `toml:"metadata"`
	Agent       Timeout     `toml:"agent"`
	Verifier    Timeout     `toml:"verifier"`
	Environment Environment `toml:"environment"`

	// Job and Oracle are the optional job.yaml / oracle.yaml fold. They are
	// zero-valued when the file is absent.
	Job    Job    `toml:"-"`
	Oracle Oracle `toml:"-"`
}

// AgentTimeoutSec is the agent's wall-clock budget in seconds — the 20-hour
// (72000s) number that dominates a FrontierSWE run's cost. A convenience over
// reaching through Agent.TimeoutSec.
func (t *Task) AgentTimeoutSec() float64 { return t.Agent.TimeoutSec }

// VerifierTimeoutSec is the verifier's wall-clock budget in seconds.
func (t *Task) VerifierTimeoutSec() float64 { return t.Verifier.TimeoutSec }
