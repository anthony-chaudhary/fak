package gateway

import (
	"encoding/json"
	"net/http"
	"strings"
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
	filter, ok := parseTasksWitnessFilter(r.URL.Query().Get("origin_witness"))
	if !ok {
		http.Error(w, "bad origin_witness filter; use failed, refused, unavailable, unwitnessed, or all", http.StatusBadRequest)
		return
	}
	if filter != tasksWitnessAll {
		snap = filterTasksSnapshotByWitness(snap, filter)
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(snap)
}

func writeTasksDisabled(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusNotFound)
	_, _ = w.Write([]byte(`{"error":"task manager disabled; enable with FAK_TASKMGR=1"}` + "\n"))
}

type tasksWitnessFilter string

const (
	tasksWitnessAll         tasksWitnessFilter = ""
	tasksWitnessFailed      tasksWitnessFilter = "failed"
	tasksWitnessRefused     tasksWitnessFilter = "refused"
	tasksWitnessUnavailable tasksWitnessFilter = "unavailable"
	tasksWitnessUnwitnessed tasksWitnessFilter = "unwitnessed"
)

func parseTasksWitnessFilter(raw string) (tasksWitnessFilter, bool) {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "", "all":
		return tasksWitnessAll, true
	case "failed", "failure", "failures", "problem", "problems":
		return tasksWitnessFailed, true
	case "refused":
		return tasksWitnessRefused, true
	case "unavailable":
		return tasksWitnessUnavailable, true
	case "unwitnessed":
		return tasksWitnessUnwitnessed, true
	default:
		return tasksWitnessAll, false
	}
}

func filterTasksSnapshotByWitness(snap any, filter tasksWitnessFilter) any {
	b, err := json.Marshal(snap)
	if err != nil {
		return snap
	}
	var doc map[string]any
	if err := json.Unmarshal(b, &doc); err != nil {
		return snap
	}
	tasks, ok := doc["tasks"].([]any)
	if !ok {
		return doc
	}
	filtered := make([]any, 0, len(tasks))
	for _, task := range tasks {
		if taskMatchesWitnessFilter(task, filter) {
			filtered = append(filtered, task)
		}
	}
	doc["tasks"] = filtered
	doc["origin_witness_filter"] = string(filter)
	doc["origin_witness_filter_count"] = len(filtered)
	return doc
}

func taskMatchesWitnessFilter(task any, filter tasksWitnessFilter) bool {
	obj, ok := task.(map[string]any)
	if !ok {
		return false
	}
	if witnessStateMatches(witnessState(obj["witness"]), filter) {
		return true
	}
	steps, _ := obj["steps"].([]any)
	for _, step := range steps {
		stepObj, ok := step.(map[string]any)
		if ok && witnessStateMatches(witnessState(stepObj["witness"]), filter) {
			return true
		}
	}
	return false
}

func witnessState(witness any) string {
	obj, ok := witness.(map[string]any)
	if !ok || obj == nil {
		return ""
	}
	state, _ := obj["verified_state"].(string)
	return strings.TrimSpace(state)
}

func witnessStateMatches(state string, filter tasksWitnessFilter) bool {
	switch filter {
	case tasksWitnessFailed:
		return state == "verified_refused" || state == "verified_unavailable"
	case tasksWitnessRefused:
		return state == "verified_refused"
	case tasksWitnessUnavailable:
		return state == "verified_unavailable"
	case tasksWitnessUnwitnessed:
		return state == "" || state == "verified_unknown"
	default:
		return true
	}
}
