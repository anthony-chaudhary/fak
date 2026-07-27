package main

import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/sharedtask"
)

// sharedTaskFixture reads one contract fixture from examples/shared-task-record/
// so the served surface is exercised with the exact envelopes the contract doc
// pins (docs/shared-task-record-contract.md).
func sharedTaskFixture(t *testing.T, name string) []byte {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("..", "..", "examples", "shared-task-record", name))
	if err != nil {
		t.Fatalf("read contract fixture %s: %v", name, err)
	}
	return b
}

// TestSharedTaskEndpointProviderInstalled proves the init() wiring ran: the
// gateway holds the cmd/fak shared-task provider, so a running `fak serve`
// actually serves /v1/fak/sharedtask/ through the internal/sharedtask fold.
func TestSharedTaskEndpointProviderInstalled(t *testing.T) {
	if !gateway.SharedTaskProviderInstalled() {
		t.Fatal("init() did not install the shared-task provider on the gateway")
	}
}

// TestSharedTaskEndpointDisabledIsInert pins the opt-in gate: without
// FAK_SHAREDTASK the provider reports disabled and the endpoint stays 404.
func TestSharedTaskEndpointDisabledIsInert(t *testing.T) {
	t.Setenv("FAK_SHAREDTASK", "")
	status, body := serveSharedTask(gateway.SharedTaskRequest{Method: http.MethodGet, Path: "task_shared_demo"})
	if status != 0 || body != nil {
		t.Fatalf("disabled surface must be inert (0, nil); got (%d, %#v)", status, body)
	}
}

// TestSharedTaskEndpointCoEditAdjudication drives the co-editing scenario the
// issue names — two clients patch the same task record through the served
// surface and the fold adjudicates accept / conflict / redact:
//
//   - the record is created from the contract's task fixture;
//   - client A's replace /title at the current base is accepted (revision
//     advances, an event row is emitted);
//   - client B's stale replace /title returns the typed conflict carrying
//     base / current / proposed;
//   - client B's stale append-only tenant note still merges (id-newness rule);
//   - a fleet-scoped reader gets the tenant note redacted from both the record
//     view and the historical event catch-up; a tenant-scoped reader sees it.
func TestSharedTaskEndpointCoEditAdjudication(t *testing.T) {
	store := sharedtask.NewStore(sharedtask.Policy{MaxScope: sharedtask.ScopeTenant})
	const taskID = "task_shared_demo"

	// Create the record from the contract fixture through the served path.
	status, body := serveSharedTaskOn(store, gateway.SharedTaskRequest{
		Method: http.MethodPost, Path: taskID, Body: sharedTaskFixture(t, "01-task.json"),
	})
	if status != http.StatusCreated {
		t.Fatalf("create: want 201, got %d (%#v)", status, body)
	}
	created, ok := body.(sharedtask.TaskRecord)
	if !ok || created.Rev == "" {
		t.Fatalf("create: want a TaskRecord with a rev, got %#v", body)
	}

	// Client A: replace /title at the current base — accepted through the write-gate.
	status, body = serveSharedTaskOn(store, gateway.SharedTaskRequest{
		Method: http.MethodPost, Path: taskID + "/patch", Body: sharedTaskFixture(t, "02-title-patch.json"),
	})
	if status != http.StatusOK {
		t.Fatalf("client A title patch: want 200 accepted, got %d (%#v)", status, body)
	}
	resA := body.(sharedtask.PatchResult)
	if resA.Verdict != sharedtask.VerdictAccepted || resA.CurrentRev == created.Rev || resA.EventID == "" {
		t.Fatalf("client A: want accepted verdict advancing the rev with an event row, got %#v", resA)
	}

	// Client B: a stale non-commuting write against the base client A already
	// advanced past — the fold returns the typed conflict, not a lost update.
	patchB, err := json.Marshal(sharedtask.Patch{
		Schema:  sharedtask.SchemaPatch,
		TaskID:  taskID,
		BaseRev: created.Rev,
		Actor:   sharedtask.Actor{Kind: "agent", ID: "client-b"},
		Scope:   sharedtask.ScopeFleet,
		Ops:     []sharedtask.Op{{Op: "replace", Path: "/title", Value: "Client B rename"}},
		Message: "stale rename",
	})
	if err != nil {
		t.Fatalf("marshal client B patch: %v", err)
	}
	status, body = serveSharedTaskOn(store, gateway.SharedTaskRequest{
		Method: http.MethodPost, Path: taskID + "/patch", Body: patchB,
	})
	if status != http.StatusConflict {
		t.Fatalf("client B stale title patch: want 409 conflict, got %d (%#v)", status, body)
	}
	resB := body.(sharedtask.PatchResult)
	if resB.Verdict != sharedtask.VerdictConflict || resB.Conflict == nil {
		t.Fatalf("client B: want a typed conflict, got %#v", resB)
	}
	if resB.Conflict.BaseValue != "Coordinate the shared release checklist" ||
		resB.Conflict.CurrentValue != "Coordinate the scoped release checklist" ||
		resB.Conflict.ProposedValue != "Client B rename" {
		t.Fatalf("conflict must carry base/current/proposed, got %#v", resB.Conflict)
	}

	// Client B recovers with a stale append-only tenant note — id-newness merges it.
	status, body = serveSharedTaskOn(store, gateway.SharedTaskRequest{
		Method: http.MethodPost, Path: taskID + "/patch", Body: sharedTaskFixture(t, "04-tenant-note-patch.json"),
	})
	if status != http.StatusOK {
		t.Fatalf("client B stale note append: want 200 accepted (append-only merge), got %d (%#v)", status, body)
	}
	if res := body.(sharedtask.PatchResult); res.Verdict != sharedtask.VerdictAccepted {
		t.Fatalf("client B note append: want accepted, got %#v", res)
	}

	// A fleet-scoped reader (default) gets the tenant note redacted...
	status, body = serveSharedTaskOn(store, gateway.SharedTaskRequest{Method: http.MethodGet, Path: taskID})
	if status != http.StatusOK {
		t.Fatalf("fleet view: want 200, got %d (%#v)", status, body)
	}
	fleetView := body.(sharedtask.TaskView)
	if fleetView.RedactedNotes != 1 || len(fleetView.Record.Notes) != 0 {
		t.Fatalf("fleet reader must have the tenant note redacted, got %#v", fleetView)
	}
	if fleetView.Record.Title != "Coordinate the scoped release checklist" {
		t.Fatalf("fleet view must carry client A's accepted title, got %q", fleetView.Record.Title)
	}

	// ...while a tenant-scoped reader sees it.
	status, body = serveSharedTaskOn(store, gateway.SharedTaskRequest{Method: http.MethodGet, Path: taskID, Scope: "tenant"})
	if status != http.StatusOK {
		t.Fatalf("tenant view: want 200, got %d (%#v)", status, body)
	}
	tenantView := body.(sharedtask.TaskView)
	if tenantView.RedactedNotes != 0 || len(tenantView.Record.Notes) != 1 {
		t.Fatalf("tenant reader must see the tenant note, got %#v", tenantView)
	}

	// Historical catch-up applies the same reader policy: the tenant-scoped event
	// row is redacted for the fleet reader and visible to the tenant reader.
	status, body = serveSharedTaskOn(store, gateway.SharedTaskRequest{Method: http.MethodGet, Path: taskID + "/events"})
	if status != http.StatusOK {
		t.Fatalf("fleet events view: want 200, got %d (%#v)", status, body)
	}
	fleetEvents := body.(sharedtask.EventLogView)
	if len(fleetEvents.Events) != 1 || fleetEvents.RedactedEvents != 1 {
		t.Fatalf("fleet reader must see 1 event with 1 redacted, got %#v", fleetEvents)
	}
	status, body = serveSharedTaskOn(store, gateway.SharedTaskRequest{Method: http.MethodGet, Path: taskID + "/events", Scope: "tenant"})
	if status != http.StatusOK {
		t.Fatalf("tenant events view: want 200, got %d (%#v)", status, body)
	}
	tenantEvents := body.(sharedtask.EventLogView)
	if len(tenantEvents.Events) != 2 || tenantEvents.RedactedEvents != 0 {
		t.Fatalf("tenant reader must see both event rows, got %#v", tenantEvents)
	}
}
