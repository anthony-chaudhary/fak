package sharedtask

import "testing"

func TestAcceptedPatchAdvancesRecordAndEmitsEvent(t *testing.T) {
	store := NewStore(Policy{MaxScope: ScopeFleet})
	task := mustCreate(t, store)

	result := store.Apply(Patch{
		TaskID:     task.TaskID,
		BaseRev:    task.Rev,
		Actor:      Actor{Kind: "human", ID: "operator"},
		Scope:      ScopeFleet,
		Durability: DurabilitySession,
		Ops: []Op{{
			Op:   "append",
			Path: "/open_decisions",
			Value: Decision{
				DecisionID: "dec_release_owner",
				State:      "input_required",
				Reason:     "release owner must approve the public note",
			},
		}},
		Message: "Pause publication until approved.",
	})

	if result.Verdict != VerdictAccepted {
		t.Fatalf("verdict = %s, want accepted", result.Verdict)
	}
	if result.CurrentRev == task.Rev || result.EventID == "" || result.RecordRef != result.CurrentRev {
		t.Fatalf("accepted result = %+v, base rev %s", result, task.Rev)
	}
	got, ok := store.Get(task.TaskID)
	if !ok || got.Rev != result.CurrentRev || len(got.OpenDecisions) != 1 {
		t.Fatalf("stored record = %+v ok=%v", got, ok)
	}
	events := store.Events(task.TaskID)
	if len(events) != 1 || events[0].BaseRev != task.Rev || events[0].NextRev != result.CurrentRev {
		t.Fatalf("events = %+v", events)
	}
}

func TestReplaceTitleAdvancesRecordAndEmitsEvent(t *testing.T) {
	store := NewStore(Policy{MaxScope: ScopeFleet})
	task := mustCreate(t, store)

	result := store.Apply(Patch{
		TaskID:     task.TaskID,
		BaseRev:    task.Rev,
		Actor:      Actor{Kind: "human", ID: "editor"},
		Scope:      ScopeFleet,
		Durability: DurabilitySession,
		Ops:        []Op{{Op: "replace", Path: "/title", Value: "Coordinate the scoped release checklist"}},
		Message:    "Rename the collaborative task.",
	})

	got, _ := store.Get(task.TaskID)
	if result.Verdict != VerdictAccepted || got.Title != "Coordinate the scoped release checklist" {
		t.Fatalf("result=%+v record=%+v", result, got)
	}
	if result.CurrentRev == task.Rev || len(store.Events(task.TaskID)) != 1 {
		t.Fatalf("result=%+v events=%d, want advanced rev and one event", result, len(store.Events(task.TaskID)))
	}
}

func TestStaleReplaceTitleReturnsConflictValue(t *testing.T) {
	store := NewStore(Policy{MaxScope: ScopeFleet})
	task := mustCreate(t, store)
	first := store.Apply(Patch{
		TaskID:     task.TaskID,
		BaseRev:    task.Rev,
		Actor:      Actor{Kind: "human", ID: "editor"},
		Scope:      ScopeFleet,
		Durability: DurabilitySession,
		Ops:        []Op{{Op: "replace", Path: "/title", Value: "Coordinate the scoped release checklist"}},
		Message:    "Rename the collaborative task.",
	})
	if first.Verdict != VerdictAccepted {
		t.Fatalf("first verdict = %s, want accepted", first.Verdict)
	}

	result := store.Apply(Patch{
		TaskID:     task.TaskID,
		BaseRev:    task.Rev,
		Actor:      Actor{Kind: "human", ID: "editor-2"},
		Scope:      ScopeFleet,
		Durability: DurabilitySession,
		Ops:        []Op{{Op: "replace", Path: "/title", Value: "Publish the release checklist"}},
		Message:    "Rename from a stale editor.",
	})

	if result.Verdict != VerdictConflict || result.CurrentRev != first.CurrentRev || result.Conflict == nil {
		t.Fatalf("result = %+v, want title conflict at current rev %s", result, first.CurrentRev)
	}
	if result.Conflict.Path != "/title" ||
		result.Conflict.BaseValue != "Coordinate the shared release checklist" ||
		result.Conflict.CurrentValue != "Coordinate the scoped release checklist" ||
		result.Conflict.ProposedValue != "Publish the release checklist" {
		t.Fatalf("conflict body = %+v", result.Conflict)
	}
}

func TestStaleAppendOpenDecisionAutoMerges(t *testing.T) {
	store := NewStore(Policy{MaxScope: ScopeFleet})
	task := mustCreate(t, store)
	first := store.Apply(Patch{
		TaskID:     task.TaskID,
		BaseRev:    task.Rev,
		Actor:      Actor{Kind: "agent", ID: "runner"},
		Scope:      ScopeFleet,
		Durability: DurabilitySession,
		Ops:        []Op{{Op: "replace", Path: "/state", Value: "input_required"}},
		Message:    "Mark blocked on input.",
	})
	if first.Verdict != VerdictAccepted {
		t.Fatalf("first verdict = %s, want accepted", first.Verdict)
	}

	result := store.Apply(Patch{
		TaskID:     task.TaskID,
		BaseRev:    task.Rev,
		Actor:      Actor{Kind: "human", ID: "operator"},
		Scope:      ScopeFleet,
		Durability: DurabilitySession,
		Ops: []Op{{
			Op:   "append",
			Path: "/open_decisions",
			Value: Decision{
				DecisionID: "dec_release_owner",
				State:      "input_required",
				Reason:     "release owner must approve the public note",
			},
		}},
		Message: "Append a decision from a stale editor.",
	})

	got, _ := store.Get(task.TaskID)
	if result.Verdict != VerdictAccepted || got.State != "input_required" || len(got.OpenDecisions) != 1 {
		t.Fatalf("result=%+v record=%+v", result, got)
	}
}

func TestResolveOpenDecisionAdvancesRecord(t *testing.T) {
	store := NewStore(Policy{MaxScope: ScopeFleet})
	task := mustCreate(t, store)
	first := store.Apply(Patch{
		TaskID:     task.TaskID,
		BaseRev:    task.Rev,
		Actor:      Actor{Kind: "human", ID: "operator"},
		Scope:      ScopeFleet,
		Durability: DurabilitySession,
		Ops: []Op{{
			Op:   "append",
			Path: "/open_decisions",
			Value: Decision{
				DecisionID: "dec_release_owner",
				State:      "input_required",
				Reason:     "release owner must approve the public note",
			},
		}},
		Message: "Append a decision.",
	})
	if first.Verdict != VerdictAccepted {
		t.Fatalf("first verdict = %s, want accepted", first.Verdict)
	}

	result := store.Apply(Patch{
		TaskID:     task.TaskID,
		BaseRev:    first.CurrentRev,
		Actor:      Actor{Kind: "human", ID: "operator"},
		Scope:      ScopeFleet,
		Durability: DurabilitySession,
		Ops:        []Op{{Op: "replace", Path: "/open_decisions/dec_release_owner/state", Value: "completed"}},
		Message:    "Resolve the decision.",
	})

	got, _ := store.Get(task.TaskID)
	if result.Verdict != VerdictAccepted || got.OpenDecisions[0].State != "completed" {
		t.Fatalf("result=%+v record=%+v", result, got)
	}
}

func TestDisaggregatedArtifactMissingDeletionWitnessIsHeld(t *testing.T) {
	store := NewStore(Policy{MaxScope: ScopeTenant})
	task := mustCreate(t, store)

	result := store.Apply(Patch{
		TaskID:     task.TaskID,
		BaseRev:    task.Rev,
		Actor:      Actor{Kind: "agent", ID: "runner"},
		Scope:      ScopeTenant,
		Durability: DurabilitySession,
		Ops: []Op{{
			Op:   "append",
			Path: "/artifacts",
			Value: ArtifactRef{
				ArtifactID: "art_remote_trace",
				Ref:        "sha256:remoteartifact001",
				MediaType:  "application/json",
				Taint:      TaintTainted,
				Scope:      ScopeTenant,
				Store:      "l3-kv",
			},
		}},
		Message: "Attach a remote trace.",
	})

	if result.Verdict != VerdictQuarantined || result.Reason != ReasonArtifactWitness || result.Quarantine == nil {
		t.Fatalf("result = %+v, want quarantined artifact witness", result)
	}
	assertRecordUnchanged(t, store, task)
}

func TestReplaceBodyRefWithDeletionWitnessIsAccepted(t *testing.T) {
	store := NewStore(Policy{MaxScope: ScopeTenant})
	task := mustCreate(t, store)

	result := store.Apply(Patch{
		TaskID:     task.TaskID,
		BaseRev:    task.Rev,
		Actor:      Actor{Kind: "human", ID: "operator"},
		Scope:      ScopeTenant,
		Durability: DurabilitySession,
		Ops:        []Op{{Op: "replace", Path: "/body_ref", Value: externalBodyRef("sha256:bodyremote001", ScopeTenant, "sha256:deletebody001")}},
		Message:    "Move the task body to remote storage.",
	})

	got, _ := store.Get(task.TaskID)
	if result.Verdict != VerdictAccepted || got.BodyRef.Store != "l3-kv" {
		t.Fatalf("result=%+v record=%+v", result, got)
	}
}

func TestViewFiltersNotesOutsideReaderScope(t *testing.T) {
	store := NewStore(Policy{MaxScope: ScopeTenant})
	task := mustCreate(t, store)
	fleetNote := store.Apply(Patch{
		TaskID:     task.TaskID,
		BaseRev:    task.Rev,
		Actor:      Actor{Kind: "human", ID: "operator"},
		Scope:      ScopeFleet,
		Durability: DurabilitySession,
		Ops:        []Op{{Op: "append", Path: "/notes", Value: noteValue("note_fleet", ScopeFleet)}},
		Message:    "Add a fleet-scoped note.",
	})
	tenantNote := store.Apply(Patch{
		TaskID:     task.TaskID,
		BaseRev:    fleetNote.CurrentRev,
		Actor:      Actor{Kind: "human", ID: "operator"},
		Scope:      ScopeTenant,
		Durability: DurabilitySession,
		Ops:        []Op{{Op: "append", Path: "/notes", Value: noteValue("note_tenant", ScopeTenant)}},
		Message:    "Add a tenant-scoped note.",
	})
	if fleetNote.Verdict != VerdictAccepted || tenantNote.Verdict != VerdictAccepted {
		t.Fatalf("note verdicts fleet=%+v tenant=%+v", fleetNote, tenantNote)
	}

	view, ok := store.View(task.TaskID, ViewPolicy{MaxScope: ScopeFleet})
	if !ok || len(view.Record.Notes) != 1 || view.Record.Notes[0].NoteID != "note_fleet" || view.RedactedNotes != 1 {
		t.Fatalf("fleet view = %+v ok=%v", view, ok)
	}
	tenantView, _ := store.View(task.TaskID, ViewPolicy{MaxScope: ScopeTenant})
	if len(tenantView.Record.Notes) != 2 || tenantView.RedactedNotes != 0 {
		t.Fatalf("tenant view = %+v", tenantView)
	}
}

func TestEventsViewFiltersEventsOutsideReaderScope(t *testing.T) {
	store := NewStore(Policy{MaxScope: ScopeTenant})
	task := mustCreate(t, store)
	fleetEdit := store.Apply(Patch{
		TaskID:     task.TaskID,
		BaseRev:    task.Rev,
		Actor:      Actor{Kind: "human", ID: "editor"},
		Scope:      ScopeFleet,
		Durability: DurabilitySession,
		Ops:        []Op{{Op: "replace", Path: "/title", Value: "Coordinate the scoped release checklist"}},
		Message:    "Rename the task for fleet readers.",
	})
	tenantEdit := store.Apply(Patch{
		TaskID:     task.TaskID,
		BaseRev:    fleetEdit.CurrentRev,
		Actor:      Actor{Kind: "human", ID: "tenant-editor"},
		Scope:      ScopeTenant,
		Durability: DurabilitySession,
		Ops:        []Op{{Op: "append", Path: "/notes", Value: noteValue("note_tenant_history", ScopeTenant)}},
		Message:    "Add a tenant-scoped note.",
	})
	if fleetEdit.Verdict != VerdictAccepted || tenantEdit.Verdict != VerdictAccepted {
		t.Fatalf("edits fleet=%+v tenant=%+v", fleetEdit, tenantEdit)
	}

	fleetView, ok := store.EventsView(task.TaskID, ViewPolicy{MaxScope: ScopeFleet})
	if !ok || len(fleetView.Events) != 1 || fleetView.Events[0].EventID != fleetEdit.EventID || fleetView.RedactedEvents != 1 {
		t.Fatalf("fleet event view = %+v ok=%v", fleetView, ok)
	}
	tenantView, _ := store.EventsView(task.TaskID, ViewPolicy{MaxScope: ScopeTenant})
	if len(tenantView.Events) != 2 || tenantView.RedactedEvents != 0 {
		t.Fatalf("tenant event view = %+v", tenantView)
	}
}

func TestJournalRoundTripRestoresCurrentRecordAndEvents(t *testing.T) {
	store := NewStore(Policy{MaxScope: ScopeFleet})
	task := mustCreate(t, store)
	first := store.Apply(Patch{
		TaskID:     task.TaskID,
		BaseRev:    task.Rev,
		Actor:      Actor{Kind: "agent", ID: "runner"},
		Scope:      ScopeFleet,
		Durability: DurabilitySession,
		Ops:        []Op{{Op: "replace", Path: "/state", Value: "input_required"}},
		Message:    "Mark the task blocked on input.",
	})
	second := store.Apply(Patch{
		TaskID:     task.TaskID,
		BaseRev:    first.CurrentRev,
		Actor:      Actor{Kind: "human", ID: "editor"},
		Scope:      ScopeFleet,
		Durability: DurabilitySession,
		Ops:        []Op{{Op: "replace", Path: "/title", Value: "Coordinate the scoped release checklist"}},
		Message:    "Rename the task.",
	})
	if first.Verdict != VerdictAccepted || second.Verdict != VerdictAccepted {
		t.Fatalf("results first=%+v second=%+v", first, second)
	}

	journal, ok := store.Journal(task.TaskID)
	if !ok {
		t.Fatalf("journal missing")
	}
	if err := journal.Verify(); err != nil {
		t.Fatalf("verify: %v", err)
	}
	restored, err := LoadJournal(journal, Policy{MaxScope: ScopeFleet})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	got, _ := restored.Get(task.TaskID)
	if got.Rev != second.CurrentRev || len(restored.Events(task.TaskID)) != 2 {
		t.Fatalf("restored record=%+v events=%+v", got, restored.Events(task.TaskID))
	}
}

func mustCreate(t *testing.T, store *Store) TaskRecord {
	t.Helper()
	task, err := store.Create(TaskRecord{
		TaskID: "task_shared_demo",
		State:  "working",
		Title:  "Coordinate the shared release checklist",
		BodyRef: BodyRef{
			Kind:       "cas",
			Digest:     "sha256:body001",
			Bytes:      512,
			Taint:      TaintTainted,
			Scope:      ScopeFleet,
			Durability: DurabilitySession,
		},
		Artifacts: []ArtifactRef{{
			ArtifactID: "art_release_notes",
			Ref:        "sha256:artifact001",
			MediaType:  "text/markdown",
			Taint:      TaintTrusted,
			Scope:      ScopeFleet,
			Store:      "local-cas",
		}},
		UpdatedBy: Actor{Kind: "agent", ID: "planner"},
		UpdatedAt: "2026-06-25T00:00:00Z",
	})
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	return task
}

func noteValue(noteID string, scope Scope) Note {
	return Note{
		NoteID: noteID,
		Kind:   "comment",
		BodyRef: BodyRef{
			Kind:       "cas",
			Digest:     "sha256:notebody" + noteID,
			Bytes:      128,
			Taint:      TaintTainted,
			Scope:      scope,
			Durability: DurabilitySession,
		},
		Author:    Actor{Kind: "human", ID: "operator"},
		CreatedAt: "2026-06-25T00:04:00Z",
	}
}

func externalBodyRef(digest string, scope Scope, deletionCertificate string) BodyRef {
	return BodyRef{
		Kind:                "external",
		Digest:              digest,
		Bytes:               512,
		Taint:               TaintTainted,
		Scope:               scope,
		Durability:          DurabilitySession,
		Store:               "l3-kv",
		DeletionCertificate: deletionCertificate,
	}
}

func assertRecordUnchanged(t *testing.T, store *Store, want TaskRecord) {
	t.Helper()
	got, ok := store.Get(want.TaskID)
	if !ok {
		t.Fatalf("missing task %s", want.TaskID)
	}
	if got.Rev != want.Rev || got.State != want.State || got.Title != want.Title ||
		len(got.OpenDecisions) != len(want.OpenDecisions) ||
		len(got.Artifacts) != len(want.Artifacts) || len(got.Notes) != len(want.Notes) {
		t.Fatalf("record changed\ngot:  %+v\nwant: %+v", got, want)
	}
	if events := store.Events(want.TaskID); len(events) != 0 {
		t.Fatalf("held/refused write emitted events: %+v", events)
	}
}
