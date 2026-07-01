package gateway

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestHandleFakTasksInertWhenNoProvider(t *testing.T) {
	defer restoreTasksProvider(tasksSnapshotProvider)
	tasksSnapshotProvider = nil
	rr := httptest.NewRecorder()
	(&Server{}).handleFakTasks(rr, httptest.NewRequest(http.MethodGet, "/v1/fak/tasks", nil))
	if rr.Code != http.StatusNotFound {
		t.Fatalf("nil provider: status = %d, want 404", rr.Code)
	}
}

func TestHandleFakTasksInertWhenDisabled(t *testing.T) {
	defer restoreTasksProvider(tasksSnapshotProvider)
	tasksSnapshotProvider = func() (any, bool) { return nil, false }
	rr := httptest.NewRecorder()
	(&Server{}).handleFakTasks(rr, httptest.NewRequest(http.MethodGet, "/v1/fak/tasks", nil))
	if rr.Code != http.StatusNotFound {
		t.Fatalf("disabled provider: status = %d, want 404", rr.Code)
	}
}

func TestHandleFakTasksServesSnapshotWhenEnabled(t *testing.T) {
	defer restoreTasksProvider(tasksSnapshotProvider)
	tasksSnapshotProvider = func() (any, bool) {
		return map[string]any{"schema": "fak.task-manager-snapshot.v1", "tasks": []any{}}, true
	}
	rr := httptest.NewRecorder()
	(&Server{}).handleFakTasks(rr, httptest.NewRequest(http.MethodGet, "/v1/fak/tasks", nil))
	if rr.Code != http.StatusOK {
		t.Fatalf("enabled: status = %d, want 200 (body=%s)", rr.Code, rr.Body.String())
	}
	if ct := rr.Header().Get("Content-Type"); ct != "application/json" {
		t.Fatalf("content-type = %q, want application/json", ct)
	}
	var body map[string]any
	if err := json.Unmarshal(rr.Body.Bytes(), &body); err != nil {
		t.Fatalf("body not json: %v", err)
	}
	if body["schema"] != "fak.task-manager-snapshot.v1" {
		t.Fatalf("schema = %v, want snapshot schema", body["schema"])
	}
}

func TestHandleFakTasksWitnessFilterReturnsOnlyFailures(t *testing.T) {
	defer restoreTasksProvider(tasksSnapshotProvider)
	tasksSnapshotProvider = func() (any, bool) {
		return map[string]any{
			"schema": "fak.task-manager-snapshot.v1",
			"tasks": []any{
				map[string]any{"task_id": "ok", "witness": map[string]any{"verified_state": "verified_done"}},
				map[string]any{"task_id": "refused", "witness": map[string]any{"verified_state": "verified_refused"}},
				map[string]any{"task_id": "unavailable", "witness": map[string]any{"verified_state": "verified_unavailable"}},
				map[string]any{"task_id": "step_refused", "steps": []any{
					map[string]any{"step_id": "shape", "witness": map[string]any{"verified_state": "verified_refused"}},
				}},
				map[string]any{"task_id": "unwitnessed"},
			},
		}, true
	}

	rr := httptest.NewRecorder()
	(&Server{}).handleFakTasks(rr, httptest.NewRequest(http.MethodGet, "/v1/fak/tasks?origin_witness=failed", nil))
	if rr.Code != http.StatusOK {
		t.Fatalf("filtered: status = %d, want 200 (body=%s)", rr.Code, rr.Body.String())
	}
	var body map[string]any
	if err := json.Unmarshal(rr.Body.Bytes(), &body); err != nil {
		t.Fatalf("body not json: %v", err)
	}
	if body["origin_witness_filter"] != "failed" || body["origin_witness_filter_count"].(float64) != 3 {
		t.Fatalf("filter metadata = %v/%v, want failed/3", body["origin_witness_filter"], body["origin_witness_filter_count"])
	}
	got := taskIDs(body["tasks"].([]any))
	for _, want := range []string{"refused", "step_refused", "unavailable"} {
		if !hasString(got, want) {
			t.Fatalf("filtered tasks = %+v, missing %q", got, want)
		}
	}
	for _, reject := range []string{"ok", "unwitnessed"} {
		if hasString(got, reject) {
			t.Fatalf("filtered tasks = %+v, unexpectedly included %q", got, reject)
		}
	}
}

func TestHandleFakTasksWitnessFilterRejectsBadValue(t *testing.T) {
	defer restoreTasksProvider(tasksSnapshotProvider)
	tasksSnapshotProvider = func() (any, bool) {
		return map[string]any{"schema": "fak.task-manager-snapshot.v1", "tasks": []any{}}, true
	}
	rr := httptest.NewRecorder()
	(&Server{}).handleFakTasks(rr, httptest.NewRequest(http.MethodGet, "/v1/fak/tasks?origin_witness=maybe", nil))
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("bad filter: status = %d, want 400", rr.Code)
	}
}

func TestHandleFakTasksRejectsNonGet(t *testing.T) {
	defer restoreTasksProvider(tasksSnapshotProvider)
	tasksSnapshotProvider = func() (any, bool) { return map[string]any{}, true }
	rr := httptest.NewRecorder()
	(&Server{}).handleFakTasks(rr, httptest.NewRequest(http.MethodPost, "/v1/fak/tasks", nil))
	if rr.Code != http.StatusMethodNotAllowed {
		t.Fatalf("POST: status = %d, want 405", rr.Code)
	}
}

func restoreTasksProvider(f func() (any, bool)) { tasksSnapshotProvider = f }

func taskIDs(tasks []any) []string {
	out := make([]string, 0, len(tasks))
	for _, task := range tasks {
		obj, _ := task.(map[string]any)
		id, _ := obj["task_id"].(string)
		if id != "" {
			out = append(out, id)
		}
	}
	return out
}

func hasString(items []string, want string) bool {
	for _, item := range items {
		if item == want {
			return true
		}
	}
	return false
}
