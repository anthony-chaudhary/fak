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
