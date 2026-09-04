package ops

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"syscall"

	"github.com/anthony-chaudhary/fak/internal/procguard"
)

// ProcessReapResult captures the details of an executed process reap pass.
type ProcessReapResult struct {
	PIDsReaped []int    `json:"pids_reaped"`
	Names      []string `json:"names"`
	Reasons    []string `json:"reasons"`
	Errors     []string `json:"errors"`
}

// ProcessManager identifies and reaps orphan helpers, dead-parent shells, and resource runaways.
type ProcessManager struct {
	Config Config
	Killer func(int) (bool, string)
}

// NewProcessManager creates a ProcessManager.
func NewProcessManager(cfg Config) *ProcessManager {
	return &ProcessManager{
		Config: cfg,
		Killer: DefaultProcessKiller,
	}
}

// DefaultProcessKiller terminates a process by PID.
func DefaultProcessKiller(pid int) (bool, string) {
	proc, err := os.FindProcess(pid)
	if err != nil {
		return false, err.Error()
	}
	if runtime.GOOS == "windows" {
		err = proc.Kill()
	} else {
		err = proc.Signal(syscall.SIGKILL)
	}
	if err != nil {
		return false, err.Error()
	}
	return true, "killed"
}

// SweepProcessRunaways identifies orphan helpers, runaway thread processes, and executes reaping if enabled.
func (pm *ProcessManager) SweepProcessRunaways(ctx context.Context, dryRun bool) (ProcessReapResult, error) {
	var res ProcessReapResult

	procs, collectErr := procguard.CollectProcesses()
	if collectErr != "" && len(procs) == 0 {
		return res, fmt.Errorf("collect processes: %s", collectErr)
	}

	th := procguard.Thresholds{
		MaxThreads: pm.Config.MaxThreads,
	}

	opts := procguard.Options{
		Thresholds: th,
		Enact:      !dryRun && pm.Config.OrphanReapEnabled,
		Killer:     pm.Killer,
		Platform:   runtime.GOOS,
	}

	payload := procguard.Build(procs, opts)

	for _, en := range payload.Enacted {
		if en.OK {
			res.PIDsReaped = append(res.PIDsReaped, en.PID)
			res.Names = append(res.Names, en.Name)
			res.Reasons = append(res.Reasons, en.Detail)
		} else {
			res.Errors = append(res.Errors, fmt.Sprintf("pid %d (%s): %s", en.PID, en.Name, en.Detail))
		}
	}

	return res, nil
}
