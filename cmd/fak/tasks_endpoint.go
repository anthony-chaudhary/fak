package main

import (
	"os"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/taskmgr"
)

// init wires the read-only GET /v1/fak/tasks endpoint into the gateway. The
// provider is gated on FAK_TASKMGR at request time, so installing it here is inert
// for every subcommand and for `fak serve` until an operator opts in: the served
// request path is unchanged and no task-manager overhead is paid by default.
func init() {
	gateway.SetTasksSnapshotProvider(serveTasksSnapshot)
}

var (
	procTaskMgrOnce sync.Once
	procTaskMgr     *taskmgr.Manager
)

// taskManagerEnabled reports whether the operator has turned the process task
// manager on via FAK_TASKMGR (1/true/yes/on). Default off keeps the endpoint inert.
func taskManagerEnabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("FAK_TASKMGR"))) {
	case "1", "true", "yes", "on":
		return true
	}
	return false
}

// processTaskManager lazily builds the process-global manager the first time the
// endpoint is hit while enabled, so a disabled serve never constructs it. It seeds
// one representative process task; deliberate per-request-phase instrumentation is
// a separate integration step.
func processTaskManager() *taskmgr.Manager {
	procTaskMgrOnce.Do(func() {
		procTaskMgr = taskmgr.NewManager()
		if task, err := procTaskMgr.StartTask(taskmgr.TaskSpec{TaskID: "fak_serve", Title: "fak serve process"}); err == nil {
			_, _ = task.StartObserveStep("serve_observe", "serve process observe")
		}
	})
	return procTaskMgr
}

// serveTasksSnapshot is the gateway provider: it returns the live snapshot and
// true only when FAK_TASKMGR is set, and (nil, false) otherwise so the endpoint
// stays inert.
func serveTasksSnapshot() (any, bool) {
	if !taskManagerEnabled() {
		return nil, false
	}
	return processTaskManager().Snapshot(), true
}
