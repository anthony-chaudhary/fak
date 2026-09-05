package trajctlhook

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/trajctl"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// Action is the closed vocabulary of trajectory intervention actions.
type Action string

const (
	// ActionSteer injects targeted guidance into the trajectory.
	ActionSteer Action = "STEER"
	// ActionPause suspends execution to await operator or automated review.
	ActionPause Action = "PAUSE"
	// ActionRollback reverts uncommitted provisional scratch files while preserving committed progress.
	ActionRollback Action = "ROLLBACK"
	// ActionRetry re-prompts the current step with negative constraints.
	ActionRetry Action = "RETRY"
)

// String returns the string representation of Action.
func (a Action) String() string {
	return string(a)
}

// IsValid reports whether the Action is one of the closed intervention types.
func (a Action) IsValid() bool {
	switch a {
	case ActionSteer, ActionPause, ActionRollback, ActionRetry:
		return true
	default:
		return false
	}
}

// ParseAction parses a string into an Action, case-insensitively.
func ParseAction(s string) (Action, error) {
	switch strings.ToUpper(strings.TrimSpace(s)) {
	case "STEER", "ACTIONSTEER":
		return ActionSteer, nil
	case "PAUSE", "ACTIONPAUSE":
		return ActionPause, nil
	case "ROLLBACK", "ACTIONROLLBACK":
		return ActionRollback, nil
	case "RETRY", "ACTIONRETRY":
		return ActionRetry, nil
	default:
		return "", fmt.Errorf("trajctlhook: unknown intervention action %q", s)
	}
}

// StepClassification is the structured verdict emitted by a step classifier.
type StepClassification struct {
	Action              Action   `json:"action"`
	Confidence          float64  `json:"confidence"`
	Reason              string   `json:"reason"`
	Guidance            string   `json:"guidance,omitempty"`
	NegativeConstraints []string `json:"negative_constraints,omitempty"`
}

// Intervention is an alias for StepClassification for intervention-centric call sites.
type Intervention = StepClassification

// ClassifierVerdict is an alias for StepClassification.
type ClassifierVerdict = StepClassification

// InterventionResult captures the executed outcome of an intervention.
type InterventionResult struct {
	Action      Action   `json:"action"`
	Executed    bool     `json:"executed"`
	Details     string   `json:"details"`
	Reverted    []string `json:"reverted,omitempty"`
	Guidance    string   `json:"guidance,omitempty"`
	Constraints []string `json:"constraints,omitempty"`
	State       string   `json:"state,omitempty"`
	Err         error    `json:"-"`
}

// RollbackHandler is a function type that implements scratch file rollback.
type RollbackHandler func(ctx context.Context, workDir string, scratchDirs []string) ([]string, error)

// InterventionExecutor links classifier verdicts to trajectory control state.
type InterventionExecutor struct {
	WorkDir      string
	ScratchDirs  []string
	LedgerPath   string
	SessionID    string
	ObjectiveID  string
	RunID        string
	State        *trajctl.State
	SteerDeliver trajctl.SteerFunc
	RollbackFn   RollbackHandler
}

// NewInterventionExecutor creates a new InterventionExecutor for the given working directory.
func NewInterventionExecutor(workDir string, scratchDirs ...string) *InterventionExecutor {
	return &InterventionExecutor{
		WorkDir:     workDir,
		ScratchDirs: scratchDirs,
	}
}

// Rollback invokes the configured or default rollback execution handler.
func (e *InterventionExecutor) Rollback(ctx context.Context) ([]string, error) {
	if e.RollbackFn != nil {
		return e.RollbackFn(ctx, e.WorkDir, e.ScratchDirs)
	}
	return RollbackProvisionalScratch(ctx, e.WorkDir, e.ScratchDirs...)
}

// Execute links the classifier verdict to trajectory control state and executes the action.
func (e *InterventionExecutor) Execute(ctx context.Context, c StepClassification) (InterventionResult, error) {
	if !c.Action.IsValid() {
		return InterventionResult{Action: c.Action}, fmt.Errorf("trajctlhook: invalid intervention action %q", c.Action)
	}

	res := InterventionResult{
		Action:   c.Action,
		Executed: true,
	}

	switch c.Action {
	case ActionSteer:
		res.Guidance = c.Guidance
		res.Details = c.Guidance
		if res.Details == "" {
			res.Details = c.Reason
		}
		res.State = "steered"

		var deliverErr error
		delivered := false
		if e.SteerDeliver != nil && e.SessionID != "" && c.Guidance != "" {
			deliverErr = e.SteerDeliver(e.SessionID, c.Guidance)
			delivered = (deliverErr == nil)
		}

		d := trajctl.SteerDecision{
			ObjectiveID: e.ObjectiveID,
			Action:      trajctl.ActionNudge,
			Signal:      trajctl.SignalDrift,
			Reason:      c.Reason,
			Packet:      c.Guidance,
			Delivered:   delivered,
			SessionID:   e.SessionID,
			RunID:       e.RunID,
		}
		if deliverErr != nil {
			d.DeliverErr = deliverErr.Error()
		}

		if e.State != nil {
			e.State.Steers = append(e.State.Steers, d)
		}
		if e.LedgerPath != "" {
			_ = trajctl.Append(e.LedgerPath, trajctl.SteerRecord(d))
		}

	case ActionPause:
		res.State = string(trajctl.StatusPaused)
		res.Details = c.Reason
		if res.Details == "" {
			res.Details = "execution suspended by step classifier"
		}

		var stmt string
		if e.State != nil && e.ObjectiveID != "" {
			if obj, ok := e.State.Objectives[e.ObjectiveID]; ok {
				obj.Status = trajctl.StatusPaused
				e.State.Objectives[e.ObjectiveID] = obj
				stmt = obj.Statement
			}
		}
		if stmt == "" && e.LedgerPath != "" {
			rows := trajctl.ReadLedgerFile(e.LedgerPath)
			for _, r := range rows {
				if r.Kind == trajctl.KindObjective && r.Objective != nil && r.Objective.ID == e.ObjectiveID {
					stmt = r.Objective.Statement
				}
			}
		}
		if stmt == "" {
			stmt = "objective " + e.ObjectiveID
		}

		if e.LedgerPath != "" && e.ObjectiveID != "" {
			_ = trajctl.Append(e.LedgerPath, trajctl.ObjectiveRecord(trajctl.Objective{
				ID:        e.ObjectiveID,
				Statement: stmt,
				Status:    trajctl.StatusPaused,
			}))
		}

	case ActionRollback:
		res.State = "rolled_back"
		reverted, err := e.Rollback(ctx)
		if err != nil {
			res.Err = err
			return res, err
		}
		res.Reverted = reverted
		res.Details = fmt.Sprintf("reverted %d uncommitted provisional scratch files: %s", len(reverted), c.Reason)

	case ActionRetry:
		res.State = "retrying"
		res.Guidance = c.Guidance
		res.Constraints = append([]string(nil), c.NegativeConstraints...)
		res.Details = formatRetryDetails(c.Guidance, c.NegativeConstraints)

		if e.SteerDeliver != nil && e.SessionID != "" {
			_ = e.SteerDeliver(e.SessionID, res.Details)
		}

		d := trajctl.SteerDecision{
			ObjectiveID: e.ObjectiveID,
			Action:      trajctl.ActionNudge,
			Signal:      trajctl.SignalStall,
			Reason:      c.Reason,
			Packet:      res.Details,
			Delivered:   true,
			SessionID:   e.SessionID,
			RunID:       e.RunID,
		}
		if e.State != nil {
			e.State.Steers = append(e.State.Steers, d)
		}
		if e.LedgerPath != "" {
			_ = trajctl.Append(e.LedgerPath, trajctl.SteerRecord(d))
		}
	}

	return res, nil
}

func formatRetryDetails(guidance string, negativeConstraints []string) string {
	var b strings.Builder
	b.WriteString("RETRY STEP with constraints:")
	if len(negativeConstraints) > 0 {
		b.WriteString("\nNegative constraints (DO NOT):")
		for _, nc := range negativeConstraints {
			b.WriteString("\n- ")
			b.WriteString(nc)
		}
	}
	if guidance != "" {
		b.WriteString("\nGuidance: ")
		b.WriteString(guidance)
	}
	return b.String()
}

func isProvisionalExt(name string) bool {
	ext := strings.ToLower(filepath.Ext(name))
	return ext == ".tmp" || ext == ".scratch" || ext == ".provisional"
}

func isGitWorkTree(ctx context.Context, dir string) bool {
	cmd := exec.CommandContext(ctx, "git", "-C", dir, "rev-parse", "--is-inside-work-tree")
	windowgate.ConfigureBackgroundCommand(cmd)
	return cmd.Run() == nil
}

// RollbackProvisionalScratch safely reverts uncommitted provisional scratch files
// while preserving committed progress.
func RollbackProvisionalScratch(ctx context.Context, workDir string, scratchDirs ...string) ([]string, error) {
	if workDir == "" {
		workDir = "."
	}
	absWorkDir, err := filepath.Abs(workDir)
	if err != nil {
		return nil, err
	}

	if len(scratchDirs) == 0 {
		scratchDirs = []string{"_scratch"}
	}

	isGit := isGitWorkTree(ctx, absWorkDir)
	var reverted []string
	seen := make(map[string]bool)

	for _, sdir := range scratchDirs {
		targetPath := filepath.Join(absWorkDir, sdir)
		fi, err := os.Stat(targetPath)
		if err != nil {
			continue
		}

		if !fi.IsDir() {
			rel, _ := filepath.Rel(absWorkDir, targetPath)
			rel = filepath.ToSlash(rel)
			if revertedFile, ok := processProvisionalPath(ctx, absWorkDir, targetPath, rel, isGit); ok {
				if !seen[revertedFile] {
					seen[revertedFile] = true
					reverted = append(reverted, revertedFile)
				}
			}
			continue
		}

		var dirsToClean []string
		err = filepath.Walk(targetPath, func(path string, info os.FileInfo, walkErr error) error {
			if walkErr != nil {
				return nil
			}
			if info.IsDir() {
				if path != targetPath {
					dirsToClean = append(dirsToClean, path)
				}
				return nil
			}

			rel, relErr := filepath.Rel(absWorkDir, path)
			if relErr != nil {
				rel = path
			}
			rel = filepath.ToSlash(rel)

			if revertedFile, ok := processProvisionalPath(ctx, absWorkDir, path, rel, isGit); ok {
				if !seen[revertedFile] {
					seen[revertedFile] = true
					reverted = append(reverted, revertedFile)
				}
			}
			return nil
		})
		if err != nil {
			return reverted, err
		}

		for i := len(dirsToClean) - 1; i >= 0; i-- {
			_ = os.Remove(dirsToClean[i])
		}
	}

	// Also clean any uncommitted provisional extension files directly in workDir
	entries, err := os.ReadDir(absWorkDir)
	if err == nil {
		for _, e := range entries {
			if e.IsDir() {
				continue
			}
			if isProvisionalExt(e.Name()) {
				path := filepath.Join(absWorkDir, e.Name())
				rel := filepath.ToSlash(e.Name())
				if revertedFile, ok := processProvisionalPath(ctx, absWorkDir, path, rel, isGit); ok {
					if !seen[revertedFile] {
						seen[revertedFile] = true
						reverted = append(reverted, revertedFile)
					}
				}
			}
		}
	}

	return reverted, nil
}

func processProvisionalPath(ctx context.Context, workDir, fullPath, relPath string, isGit bool) (string, bool) {
	if isGit {
		checkCmd := exec.CommandContext(ctx, "git", "-C", workDir, "ls-files", "--error-unmatch", relPath)
		windowgate.ConfigureBackgroundCommand(checkCmd)
		if checkCmd.Run() == nil {
			// Committed/tracked in git: preserve progress!
			// Check if modified locally; if so, restore to committed state
			diffCmd := exec.CommandContext(ctx, "git", "-C", workDir, "diff", "--name-only", relPath)
			windowgate.ConfigureBackgroundCommand(diffCmd)
			out, _ := diffCmd.Output()
			if len(bytes.TrimSpace(out)) > 0 {
				restoreCmd := exec.CommandContext(ctx, "git", "-C", workDir, "checkout", "--", relPath)
				windowgate.ConfigureBackgroundCommand(restoreCmd)
				if restoreCmd.Run() == nil {
					return relPath, true
				}
			}
			return "", false
		}
	}

	// Untracked/uncommitted scratch file: safely remove
	if err := os.Remove(fullPath); err == nil {
		return relPath, true
	}
	return "", false
}
