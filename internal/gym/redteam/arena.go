package redteam

import (
	"context"
	"errors"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/egressfloor"
	"github.com/anthony-chaudhary/fak/internal/gym"
	"github.com/anthony-chaudhary/fak/internal/sandbox"
)

// ErrEgressBlocked is returned when an outbound egress attempt to a restricted endpoint or SSRF target is caught.
var ErrEgressBlocked = errors.New("EGRESS_BLOCKED")

// Config specifies the runtime configuration for a RedTeamArena.
type Config struct {
	BaseDir          string   `json:"base_dir"`
	HoneypotsEnabled bool     `json:"honeypots_enabled"`
	StrictEgress     bool     `json:"strict_egress"`
	ExtraDenyHosts   []string `json:"extra_deny_hosts,omitempty"`
}

// AttackPayload represents an adversarial command or programmatic operation to execute in the red-team arena.
type AttackPayload struct {
	Name      string                       `json:"name"`
	Command   string                       `json:"command"`
	Argv      []string                     `json:"argv,omitempty"`
	Env       []string                     `json:"env,omitempty"`
	Stdin     []byte                       `json:"stdin,omitempty"`
	TimeoutMS int64                        `json:"timeout_ms,omitempty"`
	Action    func(arena *gym.Arena) error `json:"-"`
}

// AttackReport records the detection, containment, and rollback results of an attack turn.
type AttackReport struct {
	PayloadName         string                  `json:"payload_name"`
	Contained           bool                    `json:"contained"`
	EgressBlocked       bool                    `json:"egress_blocked"`
	CanaryTripped       bool                    `json:"canary_tripped"`
	ExecutionResult     sandbox.ExecutionResult `json:"execution_result"`
	ResidualFilesOnHost int                     `json:"residual_files_on_host"`
	Error               string                  `json:"error,omitempty"`
}

// RedTeamArena orchestrates adversarial testing in an isolated gym arena
// with honeypot tripwires, egress containment, and sub-10ms rollback.
type RedTeamArena struct {
	mu    sync.Mutex
	cfg   Config
	arena *gym.Arena
	hp    *Honeypot
}

// NewArena creates an isolated gym.Arena equipped with honeypot tripwires and egress monitoring.
func NewArena(cfg Config) (*RedTeamArena, error) {
	if strings.TrimSpace(cfg.BaseDir) == "" {
		return nil, errors.New("base directory is required")
	}

	gymCfg := gym.Config{
		BaseDir:       cfg.BaseDir,
		WorkspaceName: "redteam-arena",
		PinnedPTY:     true,
	}

	a, err := gym.Create(context.Background(), gymCfg)
	if err != nil {
		return nil, fmt.Errorf("failed creating gym arena: %w", err)
	}

	var hp *Honeypot
	if cfg.HoneypotsEnabled {
		hp = GenerateHoneypot("redteam")
		if err := InjectIntoArena(a, hp); err != nil {
			_ = a.Destroy()
			return nil, fmt.Errorf("failed injecting honeypot into arena: %w", err)
		}
	}

	return &RedTeamArena{
		cfg:   cfg,
		arena: a,
		hp:    hp,
	}, nil
}

// Arena returns the underlying gym.Arena instance.
func (rta *RedTeamArena) Arena() *gym.Arena {
	return rta.arena
}

// Honeypot returns the active Honeypot instance, or nil if disabled.
func (rta *RedTeamArena) Honeypot() *Honeypot {
	return rta.hp
}

// Destroy tears down the arena and cleans up temporary overlay files.
func (rta *RedTeamArena) Destroy() error {
	rta.mu.Lock()
	defer rta.mu.Unlock()
	if rta.arena != nil {
		return rta.arena.Destroy()
	}
	return nil
}

// Reset rolls back the arena mutations in <10ms.
func (rta *RedTeamArena) Reset(ctx context.Context) error {
	rta.mu.Lock()
	defer rta.mu.Unlock()
	if rta.arena != nil {
		if err := rta.arena.Reset(ctx); err != nil {
			return err
		}
		if rta.cfg.HoneypotsEnabled && rta.hp != nil {
			_ = InjectIntoArena(rta.arena, rta.hp)
		}
	}
	return nil
}

// ExecuteAttack executes an adversarial payload turn:
// 1. Checks command/args for cloud metadata SSRF or private subnets via egressfloor.Classify (refuses EGRESS_BLOCKED).
// 2. Checks command/args for honeypot canary leakage (refuses SECRET_EXFILTRATION_ATTEMPT).
// 3. Executes command in gym.Arena.
// 4. Traps destructive file operations safely within the CoW layer.
// 5. Resets the arena immediately after attack turn (<10ms).
// 6. Returns AttackReport with containment status and residual file count.
func (rta *RedTeamArena) ExecuteAttack(ctx context.Context, attack AttackPayload) (*AttackReport, error) {
	rta.mu.Lock()
	defer rta.mu.Unlock()

	// 1. Check egress containment (cloud metadata SSRF / private subnets)
	if blocked, _ := rta.classifyEgress(attack); blocked {
		report := &AttackReport{
			PayloadName:         attack.Name,
			Contained:           true,
			EgressBlocked:       true,
			CanaryTripped:       false,
			ResidualFilesOnHost: 0,
			Error:               ErrEgressBlocked.Error(),
		}
		return report, ErrEgressBlocked
	}

	// 2. Check honeypot canary leakage
	if rta.hp != nil {
		if tripped, _ := rta.checkCanaryTrip(attack); tripped {
			report := &AttackReport{
				PayloadName:         attack.Name,
				Contained:           true,
				EgressBlocked:       false,
				CanaryTripped:       true,
				ResidualFilesOnHost: 0,
				Error:               ErrSecretExfiltrationAttempt.Error(),
			}
			return report, ErrSecretExfiltrationAttempt
		}
	}

	// 3. Snapshot base directory prior to execution
	beforeSnap, snapErr := snapshotDir(rta.cfg.BaseDir)
	if snapErr != nil {
		return nil, fmt.Errorf("failed to snapshot base workspace: %w", snapErr)
	}

	// 4. Execute command in gym.Arena
	var res sandbox.ExecutionResult
	var execErr error

	if attack.Action != nil {
		execErr = attack.Action(rta.arena)
	} else if attack.Command != "" {
		req := sandbox.ExecutionRequest{
			Command:    attack.Command,
			Argv:       attack.Argv,
			Env:        attack.Env,
			Stdin:      attack.Stdin,
			TimeoutMS:  attack.TimeoutMS,
			WorkingDir: rta.arena.Path(),
		}
		res, execErr = rta.arena.Execute(ctx, req)
	}

	// Check if execution output leaked canaries
	canaryTrippedOutput := false
	if rta.hp != nil {
		outText := string(res.Stdout) + " " + string(res.Stderr)
		if tripped, _ := CheckExfiltration(outText, rta.hp); tripped {
			canaryTrippedOutput = true
		}
	}

	// 5. Reset the arena immediately after attack turn (<10ms)
	resetErr := rta.arena.Reset(ctx)
	if rta.cfg.HoneypotsEnabled && rta.hp != nil {
		_ = InjectIntoArena(rta.arena, rta.hp)
	}

	// 6. Inspect base directory for residual mutations
	afterSnap, _ := snapshotDir(rta.cfg.BaseDir)
	residuals := countResiduals(beforeSnap, afterSnap)

	contained := (residuals == 0)
	report := &AttackReport{
		PayloadName:         attack.Name,
		Contained:           contained,
		EgressBlocked:       false,
		CanaryTripped:       canaryTrippedOutput,
		ExecutionResult:     res,
		ResidualFilesOnHost: residuals,
	}

	if canaryTrippedOutput {
		report.Error = ErrSecretExfiltrationAttempt.Error()
		return report, ErrSecretExfiltrationAttempt
	}
	if execErr != nil {
		report.Error = execErr.Error()
	} else if resetErr != nil {
		report.Error = resetErr.Error()
	}

	return report, execErr
}

func (rta *RedTeamArena) classifyEgress(attack AttackPayload) (bool, string) {
	fullCmd := strings.TrimSpace(attack.Command + " " + strings.Join(attack.Argv, " "))
	if len(attack.Stdin) > 0 {
		fullCmd += " " + string(attack.Stdin)
	}

	tool := "bash"
	if attack.Command != "" {
		tool = attack.Command
	}

	args := map[string]any{
		"command": fullCmd,
	}

	host, _ := egressfloor.Classify(tool, args, rta.cfg.ExtraDenyHosts...)
	if host != "" {
		return true, host
	}

	// Extra scan over command tokens for raw metadata hostnames or addresses
	for _, tok := range strings.Fields(fullCmd) {
		clean := strings.Trim(tok, `"'=;,()[]<>`)
		if b, _ := egressfloor.ClassifyHost(clean); b {
			return true, clean
		}
	}

	if rta.cfg.StrictEgress {
		for _, d := range egressfloor.Destinations(tool, args) {
			ip := net.ParseIP(d)
			if ip != nil && (ip.IsPrivate() || ip.IsLoopback()) {
				return true, d
			}
		}
	}

	return false, ""
}

func (rta *RedTeamArena) checkCanaryTrip(attack AttackPayload) (bool, string) {
	fullText := attack.Command + " " + strings.Join(attack.Argv, " ")
	if len(attack.Stdin) > 0 {
		fullText += " " + string(attack.Stdin)
	}
	for _, e := range attack.Env {
		fullText += " " + e
	}
	return CheckExfiltration(fullText, rta.hp)
}

type fileMeta struct {
	size    int64
	mode    os.FileMode
	modTime time.Time
}

func snapshotDir(dir string) (map[string]fileMeta, error) {
	m := make(map[string]fileMeta)
	err := filepath.Walk(dir, func(p string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(dir, p)
		if err != nil || rel == "." {
			return nil
		}
		m[filepath.ToSlash(rel)] = fileMeta{
			size:    info.Size(),
			mode:    info.Mode(),
			modTime: info.ModTime(),
		}
		return nil
	})
	return m, err
}

func countResiduals(before, after map[string]fileMeta) int {
	diff := 0
	for p, post := range after {
		pre, ok := before[p]
		if !ok {
			diff++
		} else if pre.size != post.size {
			diff++
		}
	}
	for p := range before {
		if _, ok := after[p]; !ok {
			diff++
		}
	}
	return diff
}
