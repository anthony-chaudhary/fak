package sharedtask

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/a2achan"
	"github.com/anthony-chaudhary/fak/internal/abi"
)

const liveTopicPrefix = "sharedtask:"

type TaskSubscription struct {
	Topic  a2achan.ChannelKey
	Inbox  a2achan.ChannelKey
	View   TaskView
	Cancel func()
}

func EventTopic(taskID string) a2achan.ChannelKey {
	return a2achan.ChannelKey{Locale: a2achan.InKernel, ID: liveTopicPrefix + taskID}
}

func ScopedEventTopic(taskID string, maxScope Scope) a2achan.ChannelKey {
	if maxScope == "" {
		maxScope = ScopeFleet
	}
	return a2achan.ChannelKey{Locale: a2achan.InKernel, ID: liveTopicPrefix + taskID + ":scope:" + string(maxScope)}
}

func EventRef(event Event) (abi.Ref, error) {
	body, err := json.Marshal(event)
	if err != nil {
		return abi.Ref{}, err
	}
	return a2achan.Inline(body, abiScope(event.Scope), abiTaint(event.Taint)), nil
}

// preparePublish resolves the default bus and builds the event ref shared by
// the publish entrypoints. ok is false when the caller should return early with
// the supplied verdict/err (event not accepted, or marshal failure).
func preparePublish(bus *a2achan.Bus, event Event) (resolved *a2achan.Bus, ref abi.Ref, verdict abi.Verdict, err error, ok bool) {
	if bus == nil {
		bus = a2achan.Default
	}
	if event.Verdict != VerdictAccepted {
		return bus, abi.Ref{}, abi.Verdict{Kind: abi.VerdictDefer, By: "sharedtask/live"}, nil, false
	}
	ref, err = EventRef(event)
	if err != nil {
		return bus, abi.Ref{}, abi.Verdict{}, err, false
	}
	return bus, ref, abi.Verdict{}, nil, true
}

func PublishEvent(ctx context.Context, bus *a2achan.Bus, from string, event Event, caps ...abi.Capability) (abi.Verdict, int, error) {
	bus, ref, verdict, err, ok := preparePublish(bus, event)
	if !ok {
		return verdict, 0, err
	}
	verdict, n := bus.Publish(ctx, from, EventTopic(event.TaskID), ref, caps...)
	return verdict, n, nil
}

func PublishEventScoped(ctx context.Context, bus *a2achan.Bus, from string, event Event, caps ...abi.Capability) (abi.Verdict, int, error) {
	bus, ref, earlyVerdict, err, ok := preparePublish(bus, event)
	if !ok {
		return earlyVerdict, 0, err
	}
	topics := scopedEventTopics(event.TaskID, event.Scope)
	if len(topics) == 0 {
		return abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonTrustViolation, By: "sharedtask/live"}, 0, nil
	}

	total := 0
	var last abi.Verdict
	for _, topic := range topics {
		verdict, n := bus.Publish(ctx, from, topic, ref, caps...)
		if verdict.Kind != abi.VerdictAllow {
			return verdict, total, nil
		}
		last = verdict
		total += n
	}
	return last, total, nil
}

type eventPublisher func(ctx context.Context, bus *a2achan.Bus, from string, event Event, caps ...abi.Capability) (abi.Verdict, int, error)

func (s *Store) applyAndPublish(ctx context.Context, bus *a2achan.Bus, from string, patch Patch, publish eventPublisher, caps ...abi.Capability) (PatchResult, abi.Verdict, int, error) {
	result := s.Apply(patch)
	if result.Verdict != VerdictAccepted {
		return result, abi.Verdict{Kind: abi.VerdictDefer, By: "sharedtask/live"}, 0, nil
	}
	event, ok := s.Event(result.TaskID, result.EventID)
	if !ok {
		return result, abi.Verdict{}, 0, fmt.Errorf("sharedtask: accepted event %q missing", result.EventID)
	}
	verdict, n, err := publish(ctx, bus, from, event, caps...)
	return result, verdict, n, err
}

func (s *Store) ApplyAndPublish(ctx context.Context, bus *a2achan.Bus, from string, patch Patch, caps ...abi.Capability) (PatchResult, abi.Verdict, int, error) {
	return s.applyAndPublish(ctx, bus, from, patch, PublishEvent, caps...)
}

func (s *Store) ApplyAndPublishScoped(ctx context.Context, bus *a2achan.Bus, from string, patch Patch, caps ...abi.Capability) (PatchResult, abi.Verdict, int, error) {
	return s.applyAndPublish(ctx, bus, from, patch, PublishEventScoped, caps...)
}

func (s *Store) subscribeTopic(bus *a2achan.Bus, taskID string, topic a2achan.ChannelKey, policy ViewPolicy) (TaskSubscription, bool) {
	if bus == nil {
		bus = a2achan.Default
	}
	inbox, cancel := bus.Subscribe(topic)
	view, ok := s.View(taskID, policy)
	if !ok {
		cancel()
		return TaskSubscription{}, false
	}
	return TaskSubscription{Topic: topic, Inbox: inbox, View: view, Cancel: cancel}, true
}

func (s *Store) SubscribeView(bus *a2achan.Bus, taskID string, policy ViewPolicy) (TaskSubscription, bool) {
	return s.subscribeTopic(bus, taskID, EventTopic(taskID), policy)
}

func (s *Store) SubscribeScopedView(bus *a2achan.Bus, taskID string, policy ViewPolicy) (TaskSubscription, bool) {
	policy = normalizeViewPolicy(policy)
	return s.subscribeTopic(bus, taskID, ScopedEventTopic(taskID, policy.MaxScope), policy)
}

func scopedEventTopics(taskID string, eventScope Scope) []a2achan.ChannelKey {
	var topics []a2achan.ChannelKey
	for _, maxScope := range []Scope{ScopeAgent, ScopeFleet, ScopeTenant, ScopePublic} {
		if scopeAllowed(eventScope, maxScope) {
			topics = append(topics, ScopedEventTopic(taskID, maxScope))
		}
	}
	return topics
}

func abiScope(scope Scope) abi.ShareScope {
	switch scope {
	case ScopeAgent:
		return abi.ScopeAgent
	case ScopeTenant, ScopePublic:
		return abi.ScopeTenant
	case ScopeFleet:
		return abi.ScopeFleet
	default:
		return abi.ScopeAgent
	}
}

func abiTaint(taint Taint) abi.TaintLabel {
	switch taint {
	case TaintTrusted:
		return abi.TaintTrusted
	case TaintQuarantined:
		return abi.TaintQuarantined
	case TaintTainted:
		return abi.TaintTainted
	default:
		return abi.TaintTainted
	}
}
