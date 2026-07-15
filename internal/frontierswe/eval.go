package frontierswe

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// This file is the C13 grade seam for FrontierSWE (epic #1706, issue #1719): the
// FrontierSWE analogue of `fak swebench eval`. It turns a trial's SUBMISSION into
// the leaderboard number the benchmark is scored on, in two honest halves:
//
//   - SCORE (RUNNABLE NOW). Given a reward.json — the verifier's raw output, whether
//     produced by a prior run or handed in — RunEval parses it and runs the C3 scorer
//     (score.go: ExtractScore -> GatedScoreAntiCheat) to produce the correctness,
//     speedup, and gated leaderboard score. This is pure Go: no Docker, no model, no
//     network. It is the real grade, and the raw reward.json + a verifier logs dir are
//     captured for traceability.
//
//   - PRODUCE (honestly gated). A reward.json comes from standing the task's verifier
//     up in its environment (the Modal sandbox / [environment] docker_image) under the
//     [verifier] timeout_sec budget — a heavyweight, long-horizon Linux job, so eval
//     never spawns it as a silent side effect of a grade call: --run-verifier is the
//     explicit consent. With it, on a Docker-capable host, eval stands the verifier up,
//     captures the reward.json + /logs/verifier artifacts it produces, and scores them
//     (eval_verify.go — the #1719 acceptance path); an incapable host still refuses
//     honestly. Without it, RunEval returns an honest available:false result carrying
//     the EXACT remote command to produce one, and never a fabricated score — the
//     operator runs that on a Docker/Modal box, then re-runs eval with --reward to
//     score the reward.json it produces.
//
// The gate reason and the remote command come straight from the C7 env-adapter plan,
// so eval's honest "can't grade here" is the same witness as the run driver's.

// EvalSchema is the versioned schema id stamped on the fak.frontierswe.eval.v1
// result emitted by `fak frontierswe eval`, so a produced grade is inspectable and
// machine-joinable by the cross-run compare (C12).
const EvalSchema = "fak.frontierswe.eval.v1"

// DefaultVerifyCommand is the FrontierSWE verifier invocation run inside the task
// environment when a task's oracle.yaml carries no explicit command. It mirrors the
// harbor shape of DefaultRunCommand ("python -m harbor run job.yaml"); the submission
// tree and the verifier artifact dir are conveyed by volume mounts + env, not by
// invented CLI flags, so the command stays faithful to whatever the task's oracle is.
const DefaultVerifyCommand = "python -m harbor verify job.yaml"

// EvalGatedReason is the honest-gate reason token stamped when this host cannot
// stand the verifier environment up. It is a witnessed refusal, never a silent 0.
const EvalGatedReason = "FRONTIERSWE_VERIFIER_GATED"

// EvalConfig is the operator-facing shape of one FrontierSWE grade.
type EvalConfig struct {
	Task             *Task   // the loaded task (docker_image, [verifier] timeout_sec, oracle command)
	SubmissionDir    string  // the submission tree to grade (the target the verifier reads)
	RewardPath       string  // explicit reward.json to score (default: <submission>/reward.json)
	OutDir           string  // capture the raw reward.json + verifier logs here (optional)
	AntiCheatFlagged bool    // a trial flagged in scoring/anticheat.json scores 0 (C3 anti-cheat)
	SSIMThreshold    float64 // revideo-perf-opt SSIM gate (0 => the library default 0.99)

	// RunVerifier consents to standing the verifier up HERE when this host is
	// capable (Docker present + the no-internet boundary holds): the heavyweight
	// [verifier] timeout_sec job never runs as a silent side effect of a grade
	// call. On an incapable host the consented call still refuses honestly.
	RunVerifier bool

	// runner overrides how the verifier invocation is executed (nil = the real
	// docker exec) — the seam that makes the available-path wiring testable
	// offline. capability, when non-nil, overrides the detected local capability
	// so tests can pin the host shape deterministically on any box.
	runner     verifierRunner
	capability *EnvAdapterCapability
}

// EvalResult is the fak.frontierswe.eval.v1 grade payload. Available is true only
// when a reward.json was found or produced and scored; a gated grade sets
// Available=false, Score=0, and carries the exact remote command — never a fabricated
// number. Correctness/Speedup/Score are the C3 outputs; the capture + gate fields make
// the grade auditable and one copy-paste from a live run.
type EvalResult struct {
	Schema    string `json:"schema"`
	Task      string `json:"task"`
	GateClass string `json:"gate_class"` // the scoring family: implementation/performance/ml_research

	Available     bool     `json:"available"`         // a reward.json was found/produced AND scored
	Source        string   `json:"source"`            // "existing-reward" | "verifier-run" | "none"
	Reason        string   `json:"reason,omitempty"`  // why unavailable, when Available is false
	Correctness   float64  `json:"correctness"`       // C3 correctness in [0,1] (raw metric for ml_research)
	Speedup       *float64 `json:"speedup,omitempty"` // C3 speedup, when the task/path yields one
	Score         float64  `json:"leaderboard_score"` // the C3 gated leaderboard number
	AntiCheatFlag bool     `json:"anti_cheat_flag,omitempty"`

	// Traceability: where the graded reward.json came from and where its raw bytes +
	// the verifier artifact dir were captured.
	RewardPath      string `json:"reward_path,omitempty"`
	VerifierLogsDir string `json:"verifier_logs_dir,omitempty"`

	// Honest local gate: whether the verifier environment could be stood up here,
	// and the exact command to grade on a Docker/Modal-capable box otherwise.
	DockerPresent      bool   `json:"docker_present"`
	IntegrityOK        bool   `json:"integrity_ok"`
	VerifierTimeoutSec int64  `json:"verifier_timeout_sec"`
	RemoteCommand      string `json:"remote_command"`
}

// ParseReward decodes a FrontierSWE reward.json into the typed Reward the C3 scorer
// reads. FrontierSWE reward files are heterogeneous across tasks; Reward's fields are
// the union the reference scorer reads (see score.go), all optional.
func ParseReward(b []byte) (*Reward, error) {
	var r Reward
	if err := json.Unmarshal(b, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

// RunEval grades one FrontierSWE submission into the C3 leaderboard number. It scores
// an existing/handed-in reward.json offline (RUNNABLE NOW); absent one it stands the
// verifier up here only under explicit RunVerifier consent on a capable host, and
// otherwise returns an honest available:false result carrying the exact remote command
// to produce one on a capable box. It never fabricates a score, and never spawns the
// verifier container as a silent side effect. The error return is reserved for local
// I/O / verifier-run / malformed-reward faults; an honest gate is a valid result, not
// an error.
func RunEval(cfg EvalConfig) (EvalResult, error) {
	task := cfg.Task
	if task == nil {
		task = &Task{Name: "unknown"}
	}
	cat, _ := CategoryOf(task.Name)
	res := EvalResult{
		Schema:             EvalSchema,
		Task:               task.Name,
		GateClass:          cat.String(),
		VerifierTimeoutSec: int64(task.VerifierTimeoutSec()),
	}

	// The capability gate + remote command come from the C7 env-adapter plan so
	// eval's honest gate is the same witness as run/env-adapter. BuildEnvAdapterPlan
	// starts nothing — it only reports whether Docker is present and the no-internet
	// boundary holds. Tests pin the capability so both gate branches are checkable
	// offline regardless of what the box running them has installed.
	plan := BuildEnvAdapterPlan(EnvAdapterConfig{Task: task})
	if cfg.capability != nil {
		plan.Capability = *cfg.capability
	}
	res.DockerPresent = plan.Capability.DockerPresent
	res.IntegrityOK = plan.Integrity.OK
	verifyCmd := verifyCommand(task)
	res.RemoteCommand = verifierDockerCommand(task, cfg.SubmissionDir, verifyCmd)

	// 1) Score an existing reward.json offline — the real grade, no Docker needed.
	rewardPath := cfg.RewardPath
	if rewardPath == "" && cfg.SubmissionDir != "" {
		rewardPath = filepath.Join(cfg.SubmissionDir, "reward.json")
	}
	if rewardPath != "" {
		if b, err := os.ReadFile(rewardPath); err == nil {
			reward, perr := ParseReward(b)
			if perr != nil {
				return res, fmt.Errorf("parse reward.json %s: %w", rewardPath, perr)
			}
			res.Source = "existing-reward"
			res.RewardPath = rewardPath
			if err := res.scoreAndCapture(cfg, task, reward, b); err != nil {
				return res, err
			}
			return res, nil
		}
	}

	// 2) No reward.json present → the verifier must produce one. That is a heavyweight
	// C7 Linux job (a docker_image pull + up to [verifier] timeout_sec of work), so
	// eval never spawns it as a silent side effect of a grade call: RunVerifier is the
	// explicit consent. With it, the verifier is stood up here when this host is
	// capable and the reward.json it produces is scored; an incapable host still
	// refuses honestly (eval_verify.go). Without it, eval refuses with the exact
	// remote command; the operator runs that on a Docker/Modal box (or re-runs with
	// --run-verifier on one), then scores the produced reward.json with --reward. A
	// missing verifier environment is folded into the reason so the refusal is
	// specific about why this host can't produce the reward.
	if cfg.RunVerifier {
		if err := runVerifierAndScore(&res, cfg, task, plan.Capability, verifyCmd); err != nil {
			return res, err
		}
		return res, nil
	}
	if plan.Capability.Runnable {
		res.gate("no reward.json to score; re-run with --run-verifier to stand the verifier up here, or run the verifier command and re-run with --reward")
	} else {
		res.gate(plan.Capability.Reason)
	}
	return res, nil
}

// gate marks res honestly ungraded: Available=false, Source=none, Score left 0,
// and a reason prefixed with the EvalGatedReason token so every refusal — no
// reward, incapable host, empty-handed verifier — is one grep-able shape.
func (res *EvalResult) gate(cause string) {
	res.Available = false
	res.Source = "none"
	res.Reason = fmt.Sprintf("%s: %s", EvalGatedReason, cause)
}

// scoreAndCapture runs the C3 scorer over reward and folds the correctness, speedup,
// and gated leaderboard number into res, then captures the raw reward.json + a
// verifier logs dir under cfg.OutDir for traceability (when OutDir is set).
func (res *EvalResult) scoreAndCapture(cfg EvalConfig, task *Task, reward *Reward, raw []byte) error {
	var correctness float64
	var speedup *float64
	if cfg.SSIMThreshold > 0 {
		correctness, speedup = ExtractScoreSSIM(reward, task.Name, cfg.SSIMThreshold)
	} else {
		correctness, speedup = ExtractScore(reward, task.Name)
	}
	res.Correctness = correctness
	res.Speedup = speedup
	res.Score = GatedScoreAntiCheat(correctness, speedup, task.Name, cfg.AntiCheatFlagged)
	res.AntiCheatFlag = cfg.AntiCheatFlagged
	res.Available = true

	if strings.TrimSpace(cfg.OutDir) == "" {
		return nil
	}
	if err := os.MkdirAll(cfg.OutDir, 0o755); err != nil {
		return fmt.Errorf("create out dir %s: %w", cfg.OutDir, err)
	}
	rp := filepath.Join(cfg.OutDir, "reward.json")
	if err := os.WriteFile(rp, raw, 0o644); err != nil {
		return fmt.Errorf("capture reward.json: %w", err)
	}
	res.RewardPath = rp
	logsDir := filepath.Join(cfg.OutDir, "logs", "verifier")
	if err := os.MkdirAll(logsDir, 0o755); err != nil {
		return fmt.Errorf("create verifier logs dir: %w", err)
	}
	res.VerifierLogsDir = logsDir
	return nil
}

// verifyCommand resolves the verifier invocation for a task: the task's oracle.yaml
// command when present (the committed authority), else the harbor-shaped default.
func verifyCommand(task *Task) string {
	if task != nil && strings.TrimSpace(task.Oracle.Command) != "" {
		return strings.TrimSpace(task.Oracle.Command)
	}
	return DefaultVerifyCommand
}

// verifierDockerCommand renders the copy-pasteable form of the verifier invocation
// (verifierDockerArgs, shell-quoted): the exact command eval prints when gated and
// execs when --run-verifier consents on a capable host — one witness. The rendered
// form keeps the relative ./logs/verifier artifact mount so it is portable to
// whatever box the operator pastes it on.
func verifierDockerCommand(task *Task, submissionDir, verifyCmd string) string {
	args := verifierDockerArgs(task, submissionDir, "", verifyCmd)
	for i, a := range args {
		args[i] = shWord(a)
	}
	return strings.Join(args, " ")
}
