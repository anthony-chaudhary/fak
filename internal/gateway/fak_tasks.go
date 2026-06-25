package gateway

import (
	"encoding/json"
	"net/http"
)

// tasksSnapshotProvider, when non-nil, supplies the read-only task-manager
// snapshot served at GET /v1/fak/tasks. It is nil by default, which keeps the
// endpoint inert: a disabled task manager adds no operator surface and serves 404.
// A host (cmd/fak serve) installs a provider via SetTasksSnapshotProvider only when
// the operator opts in, so the served request path is otherwise unchanged.
//
// The snapshot type is decoupled to any so this package does not import the
// foundation-tier taskmgr package; the handler only marshals what the host hands it.
var tasksSnapshotProvider func() (any, bool)

// SetTasksSnapshotProvider installs (or, with nil, clears) the task-manager
// snapshot source for GET /v1/fak/tasks. The provider returns the snapshot to
// serve and whether the task manager is currently enabled; a false second value
// keeps the endpoint inert.
func SetTasksSnapshotProvider(provider func() (any, bool)) { tasksSnapshotProvider = provider }

// TasksSnapshotProviderInstalled reports whether a provider has been installed. It
// exists so a host can prove its wiring ran without exporting the provider itself.
func TasksSnapshotProviderInstalled() bool { return tasksSnapshotProvider != nil }

// handleFakTasks serves the read-only process task-manager snapshot. It answers
// 404 unless a provider is installed AND reports the task manager enabled, so the
// endpoint is inert by default. The snapshot carries task/step and resource
// accounting only — never prompt or tool-result payload bytes.
func (s *Server) handleFakTasks(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.Header().Set("Allow", http.MethodGet)
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	provider := tasksSnapshotProvider
	if provider == nil {
		writeTasksDisabled(w)
		return
	}
	snap, enabled := provider()
	if !enabled {
		writeTasksDisabled(w)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(snap)
}

func writeTasksDisabled(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusNotFound)
	_, _ = w.Write([]byte(`{"error":"task manager disabled; enable with FAK_TASKMGR=1"}` + "\n"))
}
