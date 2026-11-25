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
	// Skipped names each host-reaching step declined for lack of HostReach.
	Skipped []ScopeSkip `json:"skipped,omitempty"`
}

// ProcessManager identifies and reaps orphan helpers, dead-parent shells, and resource runaways.
type ProcessManager struct {
	Config Config
	Killer func(int) (bool, string)
	// Collect is the process census; nil (the NewProcessManager default)
	// skips the sweep, because the only census there is spans the whole host.
	Collect func() ([]procguard.Proc, string)
	// EnactAuthorized grants kill authority on top of Config.OrphanReapEnabled.
	EnactAuthorized bool
}

// NewProcessManager creates a ProcessManager that neither scans nor kills
// until Collect and EnactAuthorized are granted (see HostReach).
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

	if pm.Collect == nil {
		res.Skipped = append(res.Skipped, SkipProcessCensusUnauthorized)
		return res, nil
	}
	procs, collectErr := pm.Collect()
	if collectErr != "" && len(procs) == 0 {
		return res, fmt.Errorf("collect processes: %s", collectErr)
	}

	th := procguard.Thresholds{
		MaxThreads: pm.Config.MaxThreads,
	}

	wantEnact := !dryRun && pm.Config.OrphanReapEnabled
	if wantEnact && !pm.EnactAuthorized {
		res.Skipped = append(res.Skipped, SkipProcessReapUnauthorized)
	}
	opts := procguard.Options{
		Thresholds: th,
		Enact:      wantEnact && pm.EnactAuthorized,
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
