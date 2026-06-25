package sharedtask

type ViewPolicy struct {
	MaxScope           Scope
	IncludeQuarantined bool
}

type TaskView struct {
	Record            TaskRecord
	BodyVisible       bool
	RedactedArtifacts int
	RedactedNotes     int
}

type EventLogView struct {
	Events         []Event
	RedactedEvents int
}

func (s *Store) View(taskID string, policy ViewPolicy) (TaskView, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	record, ok := s.tasks[taskID]
	if !ok {
		return TaskView{}, false
	}
	policy = normalizeViewPolicy(policy)

	view := TaskView{Record: cloneRecord(record), BodyVisible: true}
	if !scopeAllowed(view.Record.BodyRef.Scope, policy.MaxScope) ||
		(view.Record.BodyRef.Taint == TaintQuarantined && !policy.IncludeQuarantined) {
		view.BodyVisible = false
		view.Record.BodyRef = redactedBodyRef(record.BodyRef)
	}

	artifacts := make([]ArtifactRef, 0, len(view.Record.Artifacts))
	for _, artifact := range view.Record.Artifacts {
		if !scopeAllowed(artifact.Scope, policy.MaxScope) ||
			(artifact.Taint == TaintQuarantined && !policy.IncludeQuarantined) {
			view.RedactedArtifacts++
			continue
		}
		artifacts = append(artifacts, artifact)
	}
	view.Record.Artifacts = artifacts

	notes := make([]Note, 0, len(view.Record.Notes))
	for _, note := range view.Record.Notes {
		if !scopeAllowed(note.BodyRef.Scope, policy.MaxScope) ||
			(note.BodyRef.Taint == TaintQuarantined && !policy.IncludeQuarantined) {
			view.RedactedNotes++
			continue
		}
		notes = append(notes, note)
	}
	view.Record.Notes = notes
	return view, true
}

func (s *Store) EventsView(taskID string, policy ViewPolicy) (EventLogView, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.tasks[taskID]; !ok {
		return EventLogView{}, false
	}
	policy = normalizeViewPolicy(policy)

	events := s.events[taskID]
	view := EventLogView{Events: make([]Event, 0, len(events))}
	for _, event := range events {
		if !eventVisible(event, policy) {
			view.RedactedEvents++
			continue
		}
		view.Events = append(view.Events, event)
	}
	return view, true
}

func normalizeViewPolicy(policy ViewPolicy) ViewPolicy {
	if policy.MaxScope == "" {
		policy.MaxScope = ScopeFleet
	}
	return policy
}

func eventVisible(event Event, policy ViewPolicy) bool {
	return scopeAllowed(event.Scope, policy.MaxScope) &&
		(event.Taint != TaintQuarantined || policy.IncludeQuarantined)
}

func redactedBodyRef(ref BodyRef) BodyRef {
	return BodyRef{
		Kind:       "external",
		Digest:     "sha256:redacted",
		Bytes:      0,
		Taint:      TaintQuarantined,
		Scope:      ref.Scope,
		Durability: ref.Durability,
	}
}
