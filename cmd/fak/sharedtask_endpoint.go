package main

import (
	"encoding/json"
	"net/http"
	"os"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/sharedtask"
)

// init wires the shared-task record co-editing subtree (/v1/fak/sharedtask/)
// into the gateway — the live consumer of the internal/sharedtask fold
// (#3885): a running `fak serve` exposes co-editing of a task record through
// the fold's write-gate. Concurrent clients POST patch envelopes against the
// same record and the fold adjudicates accept / conflict / deny / quarantine
// per docs/shared-task-record-contract.md, while GET views (record and
// historical event catch-up) are redacted by the caller's reader scope. The
// provider is gated on FAK_SHAREDTASK at request time, so installing it here is
// inert for every subcommand and for `fak serve` until an operator opts in.
func init() {
	gateway.SetSharedTaskProvider(serveSharedTask)
}

var (
	procSharedTaskOnce  = &sync.Once{}
	procSharedTaskStore *sharedtask.Store
)

// sharedTaskCoEditEnabled reports whether the operator has turned the shared-task
// co-editing surface on via FAK_SHAREDTASK (1/true/yes/on). Default off keeps the
// endpoint inert.
func sharedTaskCoEditEnabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("FAK_SHAREDTASK"))) {
	case "1", "true", "yes", "on":
		return true
	}
	return false
}

// processSharedTaskStore lazily builds the process-global record store the first
// time the surface is hit while enabled, so a disabled serve never constructs
// it. Policy MaxScope tenant admits agent/fleet/tenant-scoped writes; a
// public-scoped write stays refused (SCOPE_WIDEN_FORBIDDEN).
func processSharedTaskStore() *sharedtask.Store {
	procSharedTaskOnce.Do(func() {
		procSharedTaskStore = sharedtask.NewStore(sharedtask.Policy{MaxScope: sharedtask.ScopeTenant})
	})
	return procSharedTaskStore
}

// serveSharedTask is the gateway provider: it adjudicates against the
// process-global store only when FAK_SHAREDTASK is set, and reports (0, nil)
// otherwise so the endpoint stays inert.
func serveSharedTask(req gateway.SharedTaskRequest) (int, any) {
	if !sharedTaskCoEditEnabled() {
		return 0, nil
	}
	return serveSharedTaskOn(processSharedTaskStore(), req)
}

// serveSharedTaskOn adjudicates one surface call against the given store — the
// full served request path minus the env gate, split out so tests drive the
// exact co-editing path against a fresh store.
func serveSharedTaskOn(store *sharedtask.Store, req gateway.SharedTaskRequest) (int, any) {
	taskID, rest, _ := strings.Cut(strings.Trim(req.Path, "/"), "/")
	if taskID == "" {
		return http.StatusBadRequest, map[string]any{"error": "missing task id: /v1/fak/sharedtask/{task_id}[/patch|/events]"}
	}
	viewPolicy := sharedtask.ViewPolicy{MaxScope: sharedtask.Scope(strings.TrimSpace(req.Scope))}
	switch {
	case req.Method == http.MethodGet && rest == "":
		view, ok := store.View(taskID, viewPolicy)
		if !ok {
			return http.StatusNotFound, map[string]any{"error": "no such shared task"}
		}
		return http.StatusOK, view
	case req.Method == http.MethodGet && rest == "events":
		events, ok := store.EventsView(taskID, viewPolicy)
		if !ok {
			return http.StatusNotFound, map[string]any{"error": "no such shared task"}
		}
		return http.StatusOK, events
	case req.Method == http.MethodPost && rest == "":
		var record sharedtask.TaskRecord
		if err := json.Unmarshal(req.Body, &record); err != nil {
			return http.StatusBadRequest, map[string]any{"error": "bad task record envelope: " + err.Error()}
		}
		if record.TaskID == "" {
			record.TaskID = taskID
		}
		if record.TaskID != taskID {
			return http.StatusBadRequest, map[string]any{"error": "task_id in body does not match URL"}
		}
		created, err := store.Create(record)
		if err != nil {
			return http.StatusUnprocessableEntity, map[string]any{"error": err.Error()}
		}
		return http.StatusCreated, created
	case req.Method == http.MethodPost && rest == "patch":
		var patch sharedtask.Patch
		if err := json.Unmarshal(req.Body, &patch); err != nil {
			return http.StatusBadRequest, map[string]any{"error": "bad patch envelope: " + err.Error()}
		}
		if patch.TaskID == "" {
			patch.TaskID = taskID
		}
		if patch.TaskID != taskID {
			return http.StatusBadRequest, map[string]any{"error": "task_id in patch does not match URL"}
		}
		result := store.Apply(patch)
		return sharedTaskResultStatus(result), result
	}
	return http.StatusNotFound, map[string]any{"error": "unsupported shared-task route/method (GET {task_id}, GET {task_id}/events, POST {task_id}, POST {task_id}/patch)"}
}

// sharedTaskResultStatus maps the fold's closed verdict vocabulary onto the
// HTTP statuses the surface documents: accepted 200, typed conflict 409, denied
// 403, quarantined/needs-approval held as 202.
func sharedTaskResultStatus(result sharedtask.PatchResult) int {
	switch result.Verdict {
	case sharedtask.VerdictAccepted:
		return http.StatusOK
	case sharedtask.VerdictConflict:
		return http.StatusConflict
	case sharedtask.VerdictDenied:
		return http.StatusForbidden
	case sharedtask.VerdictQuarantined, sharedtask.VerdictNeedsApproval:
		return http.StatusAccepted
	}
	return http.StatusUnprocessableEntity
}
