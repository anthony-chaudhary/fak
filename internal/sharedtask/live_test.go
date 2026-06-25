package sharedtask

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/a2achan"
	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestPublishEventFansOutAcceptedEvent(t *testing.T) {
	ctx := context.Background()
	bus := a2achan.NewBus()
	task := mustCreate(t, NewStore(Policy{MaxScope: ScopeFleet}))
	topic := EventTopic(task.TaskID)
	inbox, cancel := bus.Subscribe(topic)
	defer cancel()

	event := Event{
		Schema:      SchemaEvent,
		EventID:     "evt_test",
		TaskID:      task.TaskID,
		EventKind:   "patch_accepted",
		Actor:       Actor{Kind: "human", ID: "operator"},
		BaseRev:     task.Rev,
		NextRev:     "sha256:next",
		Scope:       ScopeFleet,
		Durability:  DurabilitySession,
		Taint:       TaintTainted,
		PatchDigest: "sha256:patch",
		Verdict:     VerdictAccepted,
		TS:          "logical:1",
	}

	verdict, n, err := PublishEvent(ctx, bus, "coordinator", event, a2achan.CapA2ASend)
	if err != nil {
		t.Fatalf("publish event: %v", err)
	}
	if verdict.Kind != abi.VerdictAllow || n != 1 {
		t.Fatalf("publish verdict=%+v subscribers=%d, want allow/1", verdict, n)
	}

	msg, recv, ok := bus.TryRecv(ctx, inbox, a2achan.CapA2ARecv)
	if !ok || recv.Kind != abi.VerdictAllow {
		t.Fatalf("recv ok=%v verdict=%+v", ok, recv)
	}
	if msg.Body.Scope != abi.ScopeFleet || msg.Body.Taint != abi.TaintTainted {
		t.Fatalf("body metadata scope=%v taint=%v", msg.Body.Scope, msg.Body.Taint)
	}
	var got Event
	if err := json.Unmarshal(msg.Body.Inline, &got); err != nil {
		t.Fatalf("unmarshal event body: %v", err)
	}
	if got.EventID != event.EventID || got.TaskID != event.TaskID || got.NextRev != event.NextRev {
		t.Fatalf("event body = %+v, want %+v", got, event)
	}
}

func TestPublishEventScopedFiltersReaderTopics(t *testing.T) {
	ctx := context.Background()
	bus := a2achan.NewBus()
	task := mustCreate(t, NewStore(Policy{MaxScope: ScopeTenant}))
	fleetInbox, fleetCancel := bus.Subscribe(ScopedEventTopic(task.TaskID, ScopeFleet))
	defer fleetCancel()
	tenantInbox, tenantCancel := bus.Subscribe(ScopedEventTopic(task.TaskID, ScopeTenant))
	defer tenantCancel()
	publicInbox, publicCancel := bus.Subscribe(ScopedEventTopic(task.TaskID, ScopePublic))
	defer publicCancel()

	event := Event{
		Schema:      SchemaEvent,
		EventID:     "evt_tenant",
		TaskID:      task.TaskID,
		EventKind:   "patch_accepted",
		Actor:       Actor{Kind: "human", ID: "operator"},
		BaseRev:     task.Rev,
		NextRev:     "sha256:next",
		Scope:       ScopeTenant,
		Durability:  DurabilitySession,
		Taint:       TaintTainted,
		PatchDigest: "sha256:patch",
		Verdict:     VerdictAccepted,
		TS:          "logical:1",
	}

	verdict, n, err := PublishEventScoped(ctx, bus, "coordinator", event, a2achan.CapA2ASend)
	if err != nil {
		t.Fatalf("publish scoped event: %v", err)
	}
	if verdict.Kind != abi.VerdictAllow || n != 2 {
		t.Fatalf("publish verdict=%+v subscribers=%d, want allow/2", verdict, n)
	}
	if bus.Len(fleetInbox) != 0 {
		t.Fatalf("fleet-scoped inbox received tenant event")
	}
	for name, inbox := range map[string]a2achan.ChannelKey{"tenant": tenantInbox, "public": publicInbox} {
		msg, recv, ok := bus.TryRecv(ctx, inbox, a2achan.CapA2ARecv)
		if !ok || recv.Kind != abi.VerdictAllow {
			t.Fatalf("%s recv ok=%v verdict=%+v", name, ok, recv)
		}
		var got Event
		if err := json.Unmarshal(msg.Body.Inline, &got); err != nil {
			t.Fatalf("%s unmarshal event body: %v", name, err)
		}
		if got.EventID != event.EventID || got.Scope != ScopeTenant {
			t.Fatalf("%s event body = %+v, want %+v", name, got, event)
		}
	}
}

func TestPublishEventRequiresSendCapability(t *testing.T) {
	ctx := context.Background()
	bus := a2achan.NewBus()
	event := Event{
		EventID: "evt_test",
		TaskID:  "task_shared_demo",
		Scope:   ScopeFleet,
		Taint:   TaintTainted,
		Verdict: VerdictAccepted,
	}

	verdict, n, err := PublishEvent(ctx, bus, "coordinator", event)
	if err != nil {
		t.Fatalf("publish event: %v", err)
	}
	if verdict.Kind != abi.VerdictDeny || n != 0 {
		t.Fatalf("publish without cap verdict=%+v subscribers=%d, want deny/0", verdict, n)
	}
}

func TestPublishEventDoesNotPublishHeldVerdict(t *testing.T) {
	ctx := context.Background()
	bus := a2achan.NewBus()
	topic := EventTopic("task_shared_demo")
	inbox, cancel := bus.Subscribe(topic)
	defer cancel()

	event := Event{
		EventID: "evt_held",
		TaskID:  "task_shared_demo",
		Scope:   ScopeFleet,
		Taint:   TaintTainted,
		Verdict: VerdictDenied,
	}
	verdict, n, err := PublishEvent(ctx, bus, "coordinator", event, a2achan.CapA2ASend)
	if err != nil {
		t.Fatalf("publish event: %v", err)
	}
	if verdict.Kind != abi.VerdictDefer || n != 0 {
		t.Fatalf("held verdict publish=%+v subscribers=%d, want defer/0", verdict, n)
	}
	if bus.Len(inbox) != 0 {
		t.Fatalf("held verdict queued %d messages", bus.Len(inbox))
	}
}

func TestPublishEventRefusesPrivateOrQuarantinedBody(t *testing.T) {
	ctx := context.Background()
	bus := a2achan.NewBus()
	tests := []Event{
		{EventID: "evt_private", TaskID: "task_shared_demo", Scope: ScopeAgent, Taint: TaintTainted, Verdict: VerdictAccepted},
		{EventID: "evt_quarantined", TaskID: "task_shared_demo", Scope: ScopeFleet, Taint: TaintQuarantined, Verdict: VerdictAccepted},
	}
	for _, event := range tests {
		verdict, n, err := PublishEvent(ctx, bus, "coordinator", event, a2achan.CapA2ASend)
		if err != nil {
			t.Fatalf("%s: publish event: %v", event.EventID, err)
		}
		if verdict.Kind != abi.VerdictDeny || n != 0 {
			t.Fatalf("%s: publish verdict=%+v subscribers=%d, want deny/0", event.EventID, verdict, n)
		}
	}
}

func TestPublishEventScopedRefusesPrivateOrQuarantinedBody(t *testing.T) {
	ctx := context.Background()
	bus := a2achan.NewBus()
	topic := ScopedEventTopic("task_shared_demo", ScopeFleet)
	inbox, cancel := bus.Subscribe(topic)
	defer cancel()
	tests := []Event{
		{EventID: "evt_private", TaskID: "task_shared_demo", Scope: ScopeAgent, Taint: TaintTainted, Verdict: VerdictAccepted},
		{EventID: "evt_quarantined", TaskID: "task_shared_demo", Scope: ScopeFleet, Taint: TaintQuarantined, Verdict: VerdictAccepted},
	}
	for _, event := range tests {
		verdict, n, err := PublishEventScoped(ctx, bus, "coordinator", event, a2achan.CapA2ASend)
		if err != nil {
			t.Fatalf("%s: publish scoped event: %v", event.EventID, err)
		}
		if verdict.Kind != abi.VerdictDeny || n != 0 {
			t.Fatalf("%s: publish verdict=%+v subscribers=%d, want deny/0", event.EventID, verdict, n)
		}
		if bus.Len(inbox) != 0 {
			t.Fatalf("%s: denied scoped event queued %d messages", event.EventID, bus.Len(inbox))
		}
	}
}

func TestAcceptedStoreEventPublishesToLiveSubscribers(t *testing.T) {
	ctx := context.Background()
	bus := a2achan.NewBus()
	store := NewStore(Policy{MaxScope: ScopeFleet})
	task := mustCreate(t, store)
	inbox, cancel := bus.Subscribe(EventTopic(task.TaskID))
	defer cancel()

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
		t.Fatalf("apply verdict = %s, want accepted", result.Verdict)
	}
	events := store.Events(task.TaskID)
	if len(events) != 1 {
		t.Fatalf("events = %d, want 1", len(events))
	}

	verdict, n, err := PublishEvent(ctx, bus, "coordinator", events[0], a2achan.CapA2ASend)
	if err != nil {
		t.Fatalf("publish event: %v", err)
	}
	if verdict.Kind != abi.VerdictAllow || n != 1 {
		t.Fatalf("publish verdict=%+v subscribers=%d, want allow/1", verdict, n)
	}
	msg, recv, ok := bus.TryRecv(ctx, inbox, a2achan.CapA2ARecv)
	if !ok || recv.Kind != abi.VerdictAllow {
		t.Fatalf("recv ok=%v verdict=%+v", ok, recv)
	}
	var got Event
	if err := json.Unmarshal(msg.Body.Inline, &got); err != nil {
		t.Fatalf("unmarshal event body: %v", err)
	}
	if got.EventID != result.EventID || got.NextRev != result.CurrentRev {
		t.Fatalf("live event = %+v, result = %+v", got, result)
	}
}

func TestApplyAndPublishAcceptedPatchFansOut(t *testing.T) {
	ctx := context.Background()
	bus := a2achan.NewBus()
	store := NewStore(Policy{MaxScope: ScopeFleet})
	task := mustCreate(t, store)
	inbox, cancel := bus.Subscribe(EventTopic(task.TaskID))
	defer cancel()

	result, verdict, n, err := store.ApplyAndPublish(ctx, bus, "coordinator", Patch{
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
	}, a2achan.CapA2ASend)
	if err != nil {
		t.Fatalf("apply and publish: %v", err)
	}
	if result.Verdict != VerdictAccepted || verdict.Kind != abi.VerdictAllow || n != 1 {
		t.Fatalf("result=%+v publish=%+v subscribers=%d, want accepted/allow/1", result, verdict, n)
	}

	msg, recv, ok := bus.TryRecv(ctx, inbox, a2achan.CapA2ARecv)
	if !ok || recv.Kind != abi.VerdictAllow {
		t.Fatalf("recv ok=%v verdict=%+v", ok, recv)
	}
	var got Event
	if err := json.Unmarshal(msg.Body.Inline, &got); err != nil {
		t.Fatalf("unmarshal event body: %v", err)
	}
	if got.EventID != result.EventID || got.NextRev != result.CurrentRev {
		t.Fatalf("published event = %+v, result = %+v", got, result)
	}
}

func TestApplyAndPublishScopedUsesReaderScopeTopics(t *testing.T) {
	ctx := context.Background()
	bus := a2achan.NewBus()
	store := NewStore(Policy{MaxScope: ScopeTenant})
	task := mustCreate(t, store)

	fleetSub, ok := store.SubscribeScopedView(bus, task.TaskID, ViewPolicy{MaxScope: ScopeFleet})
	if !ok {
		t.Fatalf("fleet subscribe missing")
	}
	defer fleetSub.Cancel()
	tenantSub, ok := store.SubscribeScopedView(bus, task.TaskID, ViewPolicy{MaxScope: ScopeTenant})
	if !ok {
		t.Fatalf("tenant subscribe missing")
	}
	defer tenantSub.Cancel()
	if fleetSub.Topic != ScopedEventTopic(task.TaskID, ScopeFleet) ||
		tenantSub.Topic != ScopedEventTopic(task.TaskID, ScopeTenant) {
		t.Fatalf("topics fleet=%+v tenant=%+v", fleetSub.Topic, tenantSub.Topic)
	}

	result, verdict, n, err := store.ApplyAndPublishScoped(ctx, bus, "coordinator", Patch{
		TaskID:     task.TaskID,
		BaseRev:    task.Rev,
		Actor:      Actor{Kind: "human", ID: "operator"},
		Scope:      ScopeTenant,
		Durability: DurabilitySession,
		Ops: []Op{{
			Op:   "append",
			Path: "/open_decisions",
			Value: Decision{
				DecisionID: "dec_tenant_review",
				State:      "input_required",
				Reason:     "tenant reviewer must approve the update",
			},
		}},
		Message: "Ask tenant reviewer for approval.",
	}, a2achan.CapA2ASend)
	if err != nil {
		t.Fatalf("apply and publish scoped: %v", err)
	}
	if result.Verdict != VerdictAccepted || verdict.Kind != abi.VerdictAllow || n != 1 {
		t.Fatalf("result=%+v publish=%+v subscribers=%d, want accepted/allow/1", result, verdict, n)
	}
	if bus.Len(fleetSub.Inbox) != 0 {
		t.Fatalf("fleet scoped inbox received tenant event")
	}
	msg, recv, ok := bus.TryRecv(ctx, tenantSub.Inbox, a2achan.CapA2ARecv)
	if !ok || recv.Kind != abi.VerdictAllow {
		t.Fatalf("tenant recv ok=%v verdict=%+v", ok, recv)
	}
	var got Event
	if err := json.Unmarshal(msg.Body.Inline, &got); err != nil {
		t.Fatalf("unmarshal event body: %v", err)
	}
	if got.EventID != result.EventID || got.Scope != ScopeTenant || got.NextRev != result.CurrentRev {
		t.Fatalf("event body = %+v, result = %+v", got, result)
	}
}

func TestApplyAndPublishHeldPatchDoesNotPublish(t *testing.T) {
	ctx := context.Background()
	bus := a2achan.NewBus()
	store := NewStore(Policy{MaxScope: ScopeFleet, RequireApproval: true})
	task := mustCreate(t, store)
	inbox, cancel := bus.Subscribe(EventTopic(task.TaskID))
	defer cancel()

	result, verdict, n, err := store.ApplyAndPublish(ctx, bus, "coordinator", Patch{
		TaskID:     task.TaskID,
		BaseRev:    task.Rev,
		Actor:      Actor{Kind: "human", ID: "operator"},
		Scope:      ScopeFleet,
		Durability: DurabilitySession,
		Ops:        []Op{{Op: "replace", Path: "/state", Value: "completed"}},
		Message:    "Publish after approval.",
	}, a2achan.CapA2ASend)
	if err != nil {
		t.Fatalf("apply and publish: %v", err)
	}
	if result.Verdict != VerdictNeedsApproval || verdict.Kind != abi.VerdictDefer || n != 0 {
		t.Fatalf("result=%+v publish=%+v subscribers=%d, want needs_approval/defer/0", result, verdict, n)
	}
	if bus.Len(inbox) != 0 {
		t.Fatalf("held patch published %d message(s)", bus.Len(inbox))
	}
}

func TestSubscribeViewReturnsScopedSnapshotAndFutureEvents(t *testing.T) {
	ctx := context.Background()
	bus := a2achan.NewBus()
	store := NewStore(Policy{MaxScope: ScopeTenant})
	task := mustCreate(t, store)
	artifactResult := store.Apply(Patch{
		TaskID:     task.TaskID,
		BaseRev:    task.Rev,
		Actor:      Actor{Kind: "agent", ID: "runner"},
		Scope:      ScopeTenant,
		Durability: DurabilitySession,
		Ops: []Op{{
			Op:   "append",
			Path: "/artifacts",
			Value: ArtifactRef{
				ArtifactID:          "art_remote_trace",
				Ref:                 "sha256:remoteartifact001",
				MediaType:           "application/json",
				Taint:               TaintTainted,
				Scope:               ScopeTenant,
				Store:               "l3-kv",
				DeletionCertificate: "sha256:deletecert001",
			},
		}},
		Message: "Attach a tenant-scoped remote trace.",
	})
	if artifactResult.Verdict != VerdictAccepted {
		t.Fatalf("artifact verdict = %s, want accepted", artifactResult.Verdict)
	}

	sub, ok := store.SubscribeView(bus, task.TaskID, ViewPolicy{MaxScope: ScopeFleet})
	if !ok {
		t.Fatalf("subscribe view missing")
	}
	defer sub.Cancel()
	if sub.Topic != EventTopic(task.TaskID) || bus.Subscribers(sub.Topic) != 1 {
		t.Fatalf("subscription topic=%+v subscribers=%d", sub.Topic, bus.Subscribers(sub.Topic))
	}
	if len(sub.View.Record.Artifacts) != 1 || sub.View.RedactedArtifacts != 1 {
		t.Fatalf("scoped snapshot = %+v", sub.View)
	}

	result, verdict, n, err := store.ApplyAndPublish(ctx, bus, "coordinator", Patch{
		TaskID:     task.TaskID,
		BaseRev:    artifactResult.CurrentRev,
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
	}, a2achan.CapA2ASend)
	if err != nil {
		t.Fatalf("apply and publish: %v", err)
	}
	if result.Verdict != VerdictAccepted || verdict.Kind != abi.VerdictAllow || n != 1 {
		t.Fatalf("result=%+v publish=%+v subscribers=%d, want accepted/allow/1", result, verdict, n)
	}
	msg, recv, ok := bus.TryRecv(ctx, sub.Inbox, a2achan.CapA2ARecv)
	if !ok || recv.Kind != abi.VerdictAllow {
		t.Fatalf("recv ok=%v verdict=%+v", ok, recv)
	}
	var got Event
	if err := json.Unmarshal(msg.Body.Inline, &got); err != nil {
		t.Fatalf("unmarshal event body: %v", err)
	}
	if got.EventID != result.EventID || got.NextRev != result.CurrentRev {
		t.Fatalf("event body = %+v, result = %+v", got, result)
	}
}

func TestSubscribeScopedViewMissingTaskCancelsSubscription(t *testing.T) {
	bus := a2achan.NewBus()
	store := NewStore(Policy{MaxScope: ScopeFleet})
	topic := ScopedEventTopic("task_missing", ScopeFleet)

	if _, ok := store.SubscribeScopedView(bus, "task_missing", ViewPolicy{MaxScope: ScopeFleet}); ok {
		t.Fatalf("missing task subscribed")
	}
	if bus.Subscribers(topic) != 0 {
		t.Fatalf("missing task left %d subscriber(s)", bus.Subscribers(topic))
	}
}

func TestSubscribeViewMissingTaskCancelsSubscription(t *testing.T) {
	bus := a2achan.NewBus()
	store := NewStore(Policy{MaxScope: ScopeFleet})
	topic := EventTopic("task_missing")

	if _, ok := store.SubscribeView(bus, "task_missing", ViewPolicy{MaxScope: ScopeFleet}); ok {
		t.Fatalf("missing task subscribed")
	}
	if bus.Subscribers(topic) != 0 {
		t.Fatalf("missing task left %d subscriber(s)", bus.Subscribers(topic))
	}
}
