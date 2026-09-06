package sharedtask

import (
	"context"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/a2achan"
	"github.com/anthony-chaudhary/fak/internal/abi"
)

var (
	benchRecordSink  TaskRecord
	benchResultSink  PatchResult
	benchViewSink    TaskView
	benchJournalSink Journal
	benchEventsSink  []Event
	benchStoreSink   *Store
	benchStringSink  string
	benchRefSink     abi.Ref
)

// BenchmarkComputeRev measures SHA256 task-record revision calculation.
func BenchmarkComputeRev(b *testing.B) {
	record := TaskRecord{
		TaskID: "task_bench_rev",
		State:  "working",
		Title:  "Collaborative task revision benchmark",
		BodyRef: BodyRef{
			Kind:       "cas",
			Digest:     "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
			Bytes:      1024,
			Taint:      TaintTrusted,
			Scope:      ScopeFleet,
			Durability: DurabilitySession,
		},
		Artifacts: []ArtifactRef{
			{
				ArtifactID: "art_1",
				Ref:        "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
				MediaType:  "application/json",
				Taint:      TaintTrusted,
				Scope:      ScopeFleet,
				Store:      "local-cas",
			},
		},
		Notes: []Note{
			{
				NoteID: "note_1",
				Kind:   "comment",
				BodyRef: BodyRef{
					Kind:       "cas",
					Digest:     "sha256:ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb",
					Bytes:      256,
					Taint:      TaintTrusted,
					Scope:      ScopeFleet,
					Durability: DurabilitySession,
				},
				Author:    Actor{Kind: "agent", ID: "worker-1"},
				CreatedAt: "2026-09-05T12:00:00Z",
			},
		},
		OpenDecisions: []Decision{
			{
				DecisionID: "dec_1",
				State:      "input_required",
				Reason:     "approval needed",
			},
		},
		UpdatedBy: Actor{Kind: "agent", ID: "worker-1"},
		UpdatedAt: "2026-09-05T12:00:00Z",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchStringSink = ComputeRev(record)
	}
}

// BenchmarkStore_Create measures task creation, body-ref admission, and store indexing.
func BenchmarkStore_Create(b *testing.B) {
	template := TaskRecord{
		State: "working",
		Title: "Task creation benchmark",
		BodyRef: BodyRef{
			Kind:       "cas",
			Digest:     "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
			Bytes:      512,
			Taint:      TaintTrusted,
			Scope:      ScopeFleet,
			Durability: DurabilitySession,
		},
	}
	store := NewStore(Policy{MaxScope: ScopeFleet})

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		template.TaskID = "task_bench"
		rec, err := store.Create(template)
		if err != nil {
			b.Fatal(err)
		}
		benchRecordSink = rec
		delete(store.tasks, template.TaskID)
		delete(store.initial, template.TaskID)
		delete(store.history, rec.Rev)
	}
}

// BenchmarkStore_Get measures task record lookup and defensive cloning.
func BenchmarkStore_Get(b *testing.B) {
	store := NewStore(Policy{MaxScope: ScopeFleet})
	task, err := store.Create(TaskRecord{
		TaskID: "task_get_bench",
		State:  "working",
		Title:  "Store get benchmark",
		BodyRef: BodyRef{
			Kind:       "cas",
			Digest:     "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
			Bytes:      512,
			Taint:      TaintTrusted,
			Scope:      ScopeFleet,
			Durability: DurabilitySession,
		},
		Artifacts: []ArtifactRef{{
			ArtifactID: "art_1",
			Ref:        "sha256:artifact001",
			MediaType:  "text/markdown",
			Taint:      TaintTrusted,
			Scope:      ScopeFleet,
			Store:      "local-cas",
		}},
	})
	if err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rec, ok := store.Get(task.TaskID)
		if !ok {
			b.Fatal("task not found")
		}
		benchRecordSink = rec
	}
}

// BenchmarkStore_Apply_ReplaceTitle measures linear non-commuting patch application.
func BenchmarkStore_Apply_ReplaceTitle(b *testing.B) {
	store := NewStore(Policy{MaxScope: ScopeFleet})
	initialTask, err := store.Create(TaskRecord{
		TaskID: "task_apply_title",
		State:  "working",
		Title:  "Initial title",
		BodyRef: BodyRef{
			Kind:       "cas",
			Digest:     "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
			Bytes:      512,
			Taint:      TaintTrusted,
			Scope:      ScopeFleet,
			Durability: DurabilitySession,
		},
	})
	if err != nil {
		b.Fatal(err)
	}
	template := TaskRecord{
		TaskID:  initialTask.TaskID,
		Rev:     initialTask.Rev,
		State:   "working",
		Title:   "Initial title",
		BodyRef: initialTask.BodyRef,
	}
	actor := Actor{Kind: "agent", ID: "editor"}
	patch := Patch{
		TaskID:     initialTask.TaskID,
		BaseRev:    initialTask.Rev,
		Actor:      actor,
		Scope:      ScopeFleet,
		Durability: DurabilitySession,
		Ops:        []Op{{Op: "replace", Path: "/title", Value: "Updated title"}},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		store.tasks[initialTask.TaskID] = template
		store.events[initialTask.TaskID] = nil
		delete(store.history, patch.BaseRev)
		res := store.Apply(patch)
		if res.Verdict != VerdictAccepted {
			b.Fatalf("verdict: %v", res.Verdict)
		}
		benchResultSink = res
	}
}

// BenchmarkStore_Apply_AppendNote measures commuting patch append operations on a representative task.
func BenchmarkStore_Apply_AppendNote(b *testing.B) {
	store := NewStore(Policy{MaxScope: ScopeFleet})
	initialTask, err := store.Create(TaskRecord{
		TaskID: "task_apply_note",
		State:  "working",
		Title:  "Initial title",
		BodyRef: BodyRef{
			Kind:       "cas",
			Digest:     "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
			Bytes:      512,
			Taint:      TaintTrusted,
			Scope:      ScopeFleet,
			Durability: DurabilitySession,
		},
	})
	if err != nil {
		b.Fatal(err)
	}
	template := TaskRecord{
		TaskID:  initialTask.TaskID,
		Rev:     initialTask.Rev,
		State:   "working",
		Title:   "Initial title",
		BodyRef: initialTask.BodyRef,
	}
	actor := Actor{Kind: "agent", ID: "worker"}
	patch := Patch{
		TaskID:     initialTask.TaskID,
		BaseRev:    initialTask.Rev,
		Actor:      actor,
		Scope:      ScopeFleet,
		Durability: DurabilitySession,
		Ops: []Op{{
			Op:   "append",
			Path: "/notes",
			Value: Note{
				NoteID: "note_sample",
				Kind:   "comment",
				BodyRef: BodyRef{
					Kind:       "cas",
					Digest:     "sha256:notebodysample",
					Bytes:      128,
					Taint:      TaintTrusted,
					Scope:      ScopeFleet,
					Durability: DurabilitySession,
				},
				Author:    actor,
				CreatedAt: "2026-09-05T12:00:00Z",
			},
		}},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		store.tasks[initialTask.TaskID] = template
		store.events[initialTask.TaskID] = nil
		delete(store.history, patch.BaseRev)
		res := store.Apply(patch)
		if res.Verdict != VerdictAccepted {
			b.Fatalf("verdict: %v", res.Verdict)
		}
		benchResultSink = res
	}
}

// BenchmarkStore_Apply_AppendDecision measures commuting decision append operations on a representative task.
func BenchmarkStore_Apply_AppendDecision(b *testing.B) {
	store := NewStore(Policy{MaxScope: ScopeFleet})
	initialTask, err := store.Create(TaskRecord{
		TaskID: "task_apply_decision",
		State:  "working",
		Title:  "Initial title",
		BodyRef: BodyRef{
			Kind:       "cas",
			Digest:     "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
			Bytes:      512,
			Taint:      TaintTrusted,
			Scope:      ScopeFleet,
			Durability: DurabilitySession,
		},
	})
	if err != nil {
		b.Fatal(err)
	}
	template := TaskRecord{
		TaskID:  initialTask.TaskID,
		Rev:     initialTask.Rev,
		State:   "working",
		Title:   "Initial title",
		BodyRef: initialTask.BodyRef,
	}
	actor := Actor{Kind: "human", ID: "operator"}
	patch := Patch{
		TaskID:     initialTask.TaskID,
		BaseRev:    initialTask.Rev,
		Actor:      actor,
		Scope:      ScopeFleet,
		Durability: DurabilitySession,
		Ops: []Op{{
			Op:   "append",
			Path: "/open_decisions",
			Value: Decision{
				DecisionID: "dec_review",
				State:      "input_required",
				Reason:     "operator review needed",
			},
		}},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		store.tasks[initialTask.TaskID] = template
		store.events[initialTask.TaskID] = nil
		delete(store.history, patch.BaseRev)
		res := store.Apply(patch)
		if res.Verdict != VerdictAccepted {
			b.Fatalf("verdict: %v", res.Verdict)
		}
		benchResultSink = res
	}
}

// BenchmarkStore_Apply_StaleConflict measures conflict adjudication when a patch arrives on a stale base.
func BenchmarkStore_Apply_StaleConflict(b *testing.B) {
	store := NewStore(Policy{MaxScope: ScopeFleet})
	task, err := store.Create(TaskRecord{
		TaskID: "task_apply_conflict",
		State:  "working",
		Title:  "Initial title",
		BodyRef: BodyRef{
			Kind:       "cas",
			Digest:     "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
			Bytes:      512,
			Taint:      TaintTrusted,
			Scope:      ScopeFleet,
			Durability: DurabilitySession,
		},
	})
	if err != nil {
		b.Fatal(err)
	}
	first := store.Apply(Patch{
		TaskID:     task.TaskID,
		BaseRev:    task.Rev,
		Actor:      Actor{Kind: "agent", ID: "editor1"},
		Scope:      ScopeFleet,
		Durability: DurabilitySession,
		Ops:        []Op{{Op: "replace", Path: "/title", Value: "New title"}},
	})
	if first.Verdict != VerdictAccepted {
		b.Fatal("setup failed")
	}

	stalePatch := Patch{
		TaskID:     task.TaskID,
		BaseRev:    task.Rev,
		Actor:      Actor{Kind: "agent", ID: "editor2"},
		Scope:      ScopeFleet,
		Durability: DurabilitySession,
		Ops:        []Op{{Op: "replace", Path: "/title", Value: "Conflicting title"}},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res := store.Apply(stalePatch)
		if res.Verdict != VerdictConflict {
			b.Fatalf("expected conflict, got %v", res.Verdict)
		}
		benchResultSink = res
	}
}

// BenchmarkStore_View measures scope and taint projection with redactions.
func BenchmarkStore_View(b *testing.B) {
	store := NewStore(Policy{MaxScope: ScopeTenant})
	task, err := store.Create(TaskRecord{
		TaskID: "task_view_bench",
		State:  "working",
		Title:  "Task view benchmark",
		BodyRef: BodyRef{
			Kind:       "cas",
			Digest:     "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
			Bytes:      512,
			Taint:      TaintTrusted,
			Scope:      ScopeFleet,
			Durability: DurabilitySession,
		},
	})
	if err != nil {
		b.Fatal(err)
	}
	rev := task.Rev
	for i := 0; i < 10; i++ {
		scope := ScopeFleet
		if i%2 == 1 {
			scope = ScopeTenant
		}
		res := store.Apply(Patch{
			TaskID:     task.TaskID,
			BaseRev:    rev,
			Actor:      Actor{Kind: "agent", ID: "worker"},
			Scope:      scope,
			Durability: DurabilitySession,
			Ops: []Op{{
				Op:   "append",
				Path: "/notes",
				Value: Note{
					NoteID: "note_" + strconv.Itoa(i),
					Kind:   "comment",
					BodyRef: BodyRef{
						Kind:       "cas",
						Digest:     "sha256:notebody" + strconv.Itoa(i),
						Bytes:      128,
						Taint:      TaintTrusted,
						Scope:      scope,
						Durability: DurabilitySession,
					},
					Author:    Actor{Kind: "human", ID: "operator"},
					CreatedAt: "2026-09-05T12:00:00Z",
				},
			}},
		})
		if res.Verdict != VerdictAccepted {
			b.Fatal("setup note failed")
		}
		rev = res.CurrentRev
	}

	policy := ViewPolicy{MaxScope: ScopeFleet}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		view, ok := store.View(task.TaskID, policy)
		if !ok {
			b.Fatal("view failed")
		}
		benchViewSink = view
	}
}

// BenchmarkStore_EventsView measures event log filtering by scope and taint.
func BenchmarkStore_EventsView(b *testing.B) {
	store := NewStore(Policy{MaxScope: ScopeTenant})
	task, err := store.Create(TaskRecord{
		TaskID: "task_events_bench",
		State:  "working",
		Title:  "Events view benchmark",
		BodyRef: BodyRef{
			Kind:       "cas",
			Digest:     "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
			Bytes:      512,
			Taint:      TaintTrusted,
			Scope:      ScopeFleet,
			Durability: DurabilitySession,
		},
	})
	if err != nil {
		b.Fatal(err)
	}
	rev := task.Rev
	for i := 0; i < 20; i++ {
		scope := ScopeFleet
		if i%2 == 1 {
			scope = ScopeTenant
		}
		res := store.Apply(Patch{
			TaskID:     task.TaskID,
			BaseRev:    rev,
			Actor:      Actor{Kind: "agent", ID: "worker"},
			Scope:      scope,
			Durability: DurabilitySession,
			Ops: []Op{{
				Op:   "append",
				Path: "/notes",
				Value: Note{
					NoteID: "note_" + strconv.Itoa(i),
					Kind:   "comment",
					BodyRef: BodyRef{
						Kind:       "cas",
						Digest:     "sha256:notebody" + strconv.Itoa(i),
						Bytes:      128,
						Taint:      TaintTrusted,
						Scope:      scope,
						Durability: DurabilitySession,
					},
					Author:    Actor{Kind: "human", ID: "operator"},
					CreatedAt: "2026-09-05T12:00:00Z",
				},
			}},
		})
		if res.Verdict != VerdictAccepted {
			b.Fatal("setup event failed")
		}
		rev = res.CurrentRev
	}

	policy := ViewPolicy{MaxScope: ScopeFleet}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		evView, ok := store.EventsView(task.TaskID, policy)
		if !ok {
			b.Fatal("events view failed")
		}
		benchEventsSink = evView.Events
	}
}

// BenchmarkJournal_Generate measures journal snapshot serialization and hash chaining.
func BenchmarkJournal_Generate(b *testing.B) {
	store := NewStore(Policy{MaxScope: ScopeFleet})
	task, err := store.Create(TaskRecord{
		TaskID: "task_journal_bench",
		State:  "working",
		Title:  "Journal benchmark",
		BodyRef: BodyRef{
			Kind:       "cas",
			Digest:     "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
			Bytes:      512,
			Taint:      TaintTrusted,
			Scope:      ScopeFleet,
			Durability: DurabilitySession,
		},
	})
	if err != nil {
		b.Fatal(err)
	}
	rev := task.Rev
	for i := 0; i < 5; i++ {
		res := store.Apply(Patch{
			TaskID:     task.TaskID,
			BaseRev:    rev,
			Actor:      Actor{Kind: "agent", ID: "worker"},
			Scope:      ScopeFleet,
			Durability: DurabilitySession,
			Ops: []Op{{
				Op:   "append",
				Path: "/notes",
				Value: Note{
					NoteID: "note_" + strconv.Itoa(i),
					Kind:   "comment",
					BodyRef: BodyRef{
						Kind:       "cas",
						Digest:     "sha256:notebody" + strconv.Itoa(i),
						Bytes:      128,
						Taint:      TaintTrusted,
						Scope:      ScopeFleet,
						Durability: DurabilitySession,
					},
					Author:    Actor{Kind: "human", ID: "operator"},
					CreatedAt: "2026-09-05T12:00:00Z",
				},
			}},
		})
		if res.Verdict != VerdictAccepted {
			b.Fatal("setup failed")
		}
		rev = res.CurrentRev
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		j, ok := store.Journal(task.TaskID)
		if !ok {
			b.Fatal("journal failed")
		}
		benchJournalSink = j
	}
}

// BenchmarkJournal_Verify measures full cryptographic integrity verification over a journal.
func BenchmarkJournal_Verify(b *testing.B) {
	store := NewStore(Policy{MaxScope: ScopeFleet})
	task, err := store.Create(TaskRecord{
		TaskID: "task_journal_verify",
		State:  "working",
		Title:  "Journal verify benchmark",
		BodyRef: BodyRef{
			Kind:       "cas",
			Digest:     "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
			Bytes:      512,
			Taint:      TaintTrusted,
			Scope:      ScopeFleet,
			Durability: DurabilitySession,
		},
	})
	if err != nil {
		b.Fatal(err)
	}
	rev := task.Rev
	for i := 0; i < 5; i++ {
		res := store.Apply(Patch{
			TaskID:     task.TaskID,
			BaseRev:    rev,
			Actor:      Actor{Kind: "agent", ID: "worker"},
			Scope:      ScopeFleet,
			Durability: DurabilitySession,
			Ops: []Op{{
				Op:   "append",
				Path: "/notes",
				Value: Note{
					NoteID: "note_" + strconv.Itoa(i),
					Kind:   "comment",
					BodyRef: BodyRef{
						Kind:       "cas",
						Digest:     "sha256:notebody" + strconv.Itoa(i),
						Bytes:      128,
						Taint:      TaintTrusted,
						Scope:      ScopeFleet,
						Durability: DurabilitySession,
					},
					Author:    Actor{Kind: "human", ID: "operator"},
					CreatedAt: "2026-09-05T12:00:00Z",
				},
			}},
		})
		if res.Verdict != VerdictAccepted {
			b.Fatal("setup failed")
		}
		rev = res.CurrentRev
	}
	journal, ok := store.Journal(task.TaskID)
	if !ok {
		b.Fatal("journal creation failed")
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := journal.Verify(); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkJournal_Load measures reconstructing a stateful Store from a verified journal.
func BenchmarkJournal_Load(b *testing.B) {
	store := NewStore(Policy{MaxScope: ScopeFleet})
	task, err := store.Create(TaskRecord{
		TaskID: "task_journal_load",
		State:  "working",
		Title:  "Journal load benchmark",
		BodyRef: BodyRef{
			Kind:       "cas",
			Digest:     "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
			Bytes:      512,
			Taint:      TaintTrusted,
			Scope:      ScopeFleet,
			Durability: DurabilitySession,
		},
	})
	if err != nil {
		b.Fatal(err)
	}
	rev := task.Rev
	for i := 0; i < 5; i++ {
		res := store.Apply(Patch{
			TaskID:     task.TaskID,
			BaseRev:    rev,
			Actor:      Actor{Kind: "agent", ID: "worker"},
			Scope:      ScopeFleet,
			Durability: DurabilitySession,
			Ops: []Op{{
				Op:   "append",
				Path: "/notes",
				Value: Note{
					NoteID: "note_" + strconv.Itoa(i),
					Kind:   "comment",
					BodyRef: BodyRef{
						Kind:       "cas",
						Digest:     "sha256:notebody" + strconv.Itoa(i),
						Bytes:      128,
						Taint:      TaintTrusted,
						Scope:      ScopeFleet,
						Durability: DurabilitySession,
					},
					Author:    Actor{Kind: "human", ID: "operator"},
					CreatedAt: "2026-09-05T12:00:00Z",
				},
			}},
		})
		if res.Verdict != VerdictAccepted {
			b.Fatal("setup failed")
		}
		rev = res.CurrentRev
	}
	journal, ok := store.Journal(task.TaskID)
	if !ok {
		b.Fatal("journal creation failed")
	}
	policy := Policy{MaxScope: ScopeFleet}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s, err := LoadJournal(journal, policy)
		if err != nil {
			b.Fatal(err)
		}
		benchStoreSink = s
	}
}

// BenchmarkLive_EventRef measures marshalling an Event to inline ABI capability format.
func BenchmarkLive_EventRef(b *testing.B) {
	event := Event{
		Schema:      SchemaEvent,
		EventID:     "evt_bench",
		TaskID:      "task_bench_live",
		EventKind:   "patch_accepted",
		Actor:       Actor{Kind: "agent", ID: "coordinator"},
		BaseRev:     "sha256:base",
		NextRev:     "sha256:next",
		Scope:       ScopeFleet,
		Durability:  DurabilitySession,
		Taint:       TaintTrusted,
		PatchDigest: "sha256:patch",
		Verdict:     VerdictAccepted,
		TS:          "logical:1",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ref, err := EventRef(event)
		if err != nil {
			b.Fatal(err)
		}
		benchRefSink = ref
	}
}

// BenchmarkLive_PublishEvent measures live bus publication with capability checking.
func BenchmarkLive_PublishEvent(b *testing.B) {
	ctx := context.Background()
	bus := a2achan.NewBus()
	bus.SetRateLimit(-1, 0)
	topic := EventTopic("task_live_bench")
	inbox, cancel := bus.Subscribe(topic)
	defer cancel()

	event := Event{
		Schema:      SchemaEvent,
		EventID:     "evt_bench",
		TaskID:      "task_live_bench",
		EventKind:   "patch_accepted",
		Actor:       Actor{Kind: "agent", ID: "coordinator"},
		BaseRev:     "sha256:base",
		NextRev:     "sha256:next",
		Scope:       ScopeFleet,
		Durability:  DurabilitySession,
		Taint:       TaintTrusted,
		PatchDigest: "sha256:patch",
		Verdict:     VerdictAccepted,
		TS:          "logical:1",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		verdict, n, err := PublishEvent(ctx, bus, "coordinator", event, a2achan.CapA2ASend)
		if err != nil || verdict.Kind != abi.VerdictAllow || n != 1 {
			b.Fatalf("publish failed: verdict=%v, n=%d, err=%v", verdict, n, err)
		}
		bus.TryRecv(ctx, inbox, a2achan.CapA2ARecv)
	}
}

// BenchmarkLive_ApplyAndPublish measures combined patch store application and live bus dispatch.
func BenchmarkLive_ApplyAndPublish(b *testing.B) {
	ctx := context.Background()
	bus := a2achan.NewBus()
	bus.SetRateLimit(-1, 0)
	store := NewStore(Policy{MaxScope: ScopeFleet})
	task, err := store.Create(TaskRecord{
		TaskID: "task_live_apply_pub",
		State:  "working",
		Title:  "Initial title",
		BodyRef: BodyRef{
			Kind:       "cas",
			Digest:     "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
			Bytes:      512,
			Taint:      TaintTrusted,
			Scope:      ScopeFleet,
			Durability: DurabilitySession,
		},
	})
	if err != nil {
		b.Fatal(err)
	}
	template := TaskRecord{
		TaskID:  task.TaskID,
		Rev:     task.Rev,
		State:   "working",
		Title:   "Initial title",
		BodyRef: task.BodyRef,
	}
	topic := EventTopic(task.TaskID)
	inbox, cancel := bus.Subscribe(topic)
	defer cancel()

	actor := Actor{Kind: "agent", ID: "operator"}
	patch := Patch{
		TaskID:     task.TaskID,
		BaseRev:    task.Rev,
		Actor:      actor,
		Scope:      ScopeFleet,
		Durability: DurabilitySession,
		Ops:        []Op{{Op: "replace", Path: "/title", Value: "Updated title"}},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		store.tasks[task.TaskID] = template
		store.events[task.TaskID] = nil
		delete(store.history, patch.BaseRev)
		res, verdict, n, err := store.ApplyAndPublish(ctx, bus, "coordinator", patch, a2achan.CapA2ASend)
		if err != nil || res.Verdict != VerdictAccepted || verdict.Kind != abi.VerdictAllow || n != 1 {
			b.Fatalf("apply and publish failed: res=%v, verdict=%v, n=%d, err=%v", res, verdict, n, err)
		}
		bus.TryRecv(ctx, inbox, a2achan.CapA2ARecv)
	}
}

// BenchmarkContract_ValidateValue measures recursive JSON schema contract validation over TaskRecord.
func BenchmarkContract_ValidateValue(b *testing.B) {
	schemaDir := findSchemaDir(b)
	schema, err := LoadContractSchema(schemaDir, SchemaTask)
	if err != nil {
		b.Fatalf("load schema: %v", err)
	}
	taskJSON := `{
		"schema": "fak.shared-task.v1",
		"task_id": "task_bench_validate",
		"rev": "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
		"state": "working",
		"title": "Validate task benchmark",
		"body_ref": {
			"kind": "cas",
			"digest": "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
			"bytes": 512,
			"taint": "trusted",
			"scope": "fleet",
			"durability": "session"
		},
		"artifacts": [
			{
				"artifact_id": "art_1",
				"ref": "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
				"media_type": "text/markdown",
				"taint": "trusted",
				"scope": "fleet",
				"store": "local-cas"
			}
		],
		"notes": [],
		"open_decisions": [],
		"updated_by": {
			"kind": "agent",
			"id": "planner"
		},
		"updated_at": "2026-09-05T12:00:00Z"
	}`
	instance, err := decodeContractJSON([]byte(taskJSON))
	if err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := validateValue(instance, schema, schema, "$"); err != nil {
			b.Fatal(err)
		}
	}
}

func findSchemaDir(b *testing.B) string {
	b.Helper()
	dir, err := os.Getwd()
	if err != nil {
		b.Fatalf("getwd: %v", err)
	}
	for {
		candidate := filepath.Join(dir, "tools", "schemas")
		if info, err := os.Stat(candidate); err == nil && info.IsDir() {
			return candidate
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			b.Fatal("could not find tools/schemas above benchmark dir")
		}
		dir = parent
	}
}
