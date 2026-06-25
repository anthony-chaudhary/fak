package sharedtask

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
)

const (
	SchemaTask        = "fak.shared-task.v1"
	SchemaEvent       = "fak.shared-event.v1"
	SchemaPatch       = "fak.shared-patch.v1"
	SchemaPatchResult = "fak.shared-patch-result.v1"
)

type Scope string

const (
	ScopeAgent  Scope = "agent"
	ScopeFleet  Scope = "fleet"
	ScopeTenant Scope = "tenant"
	ScopePublic Scope = "public"
)

type Durability string

const (
	DurabilityTurn    Durability = "turn"
	DurabilitySession Durability = "session"
	DurabilityBounded Durability = "bounded"
	DurabilityDurable Durability = "durable"
)

type Taint string

const (
	TaintTrusted     Taint = "trusted"
	TaintTainted     Taint = "tainted"
	TaintQuarantined Taint = "quarantined"
)

type Verdict string

const (
	VerdictAccepted      Verdict = "accepted"
	VerdictConflict      Verdict = "conflict"
	VerdictQuarantined   Verdict = "quarantined"
	VerdictDenied        Verdict = "denied"
	VerdictNeedsApproval Verdict = "needs_approval"
)

const (
	ReasonApprovalRequired   = "APPROVAL_REQUIRED"
	ReasonArtifactQuarantine = "ARTIFACT_QUARANTINED"
	ReasonArtifactWitness    = "ARTIFACT_WITNESS_MISSING"
	ReasonBodyQuarantine     = "BODY_QUARANTINED"
	ReasonBodyWitness        = "BODY_WITNESS_MISSING"
	ReasonMissingDecision    = "MISSING_DECISION"
	ReasonScopeDenied        = "SCOPE_WIDEN_FORBIDDEN"
	ReasonStaleBase          = "STALE_BASE"
	ReasonUnsupportedPatch   = "UNSUPPORTED_PATCH"
)

type Actor struct {
	Kind string `json:"kind"`
	ID   string `json:"id"`
}

type BodyRef struct {
	Kind                string     `json:"kind"`
	Digest              string     `json:"digest"`
	Bytes               int64      `json:"bytes"`
	Taint               Taint      `json:"taint"`
	Scope               Scope      `json:"scope"`
	Durability          Durability `json:"durability"`
	Store               string     `json:"store,omitempty"`
	DeletionCertificate string     `json:"deletion_certificate,omitempty"`
}

type ArtifactRef struct {
	ArtifactID          string `json:"artifact_id"`
	Ref                 string `json:"ref"`
	MediaType           string `json:"media_type"`
	Taint               Taint  `json:"taint"`
	Scope               Scope  `json:"scope"`
	Store               string `json:"store"`
	DeletionCertificate string `json:"deletion_certificate,omitempty"`
}

type Decision struct {
	DecisionID string `json:"decision_id"`
	State      string `json:"state"`
	Reason     string `json:"reason"`
}

type Note struct {
	NoteID    string  `json:"note_id"`
	Kind      string  `json:"kind"`
	BodyRef   BodyRef `json:"body_ref"`
	Author    Actor   `json:"author"`
	CreatedAt string  `json:"created_at"`
}

type TaskRecord struct {
	Schema        string        `json:"schema"`
	TaskID        string        `json:"task_id"`
	Rev           string        `json:"rev"`
	State         string        `json:"state"`
	Title         string        `json:"title"`
	BodyRef       BodyRef       `json:"body_ref"`
	Artifacts     []ArtifactRef `json:"artifacts"`
	Notes         []Note        `json:"notes"`
	OpenDecisions []Decision    `json:"open_decisions"`
	UpdatedBy     Actor         `json:"updated_by"`
	UpdatedAt     string        `json:"updated_at"`
}

type Event struct {
	Schema      string     `json:"schema"`
	EventID     string     `json:"event_id"`
	TaskID      string     `json:"task_id"`
	PrevEvent   string     `json:"prev_event"`
	EventKind   string     `json:"event_kind"`
	Actor       Actor      `json:"actor"`
	BaseRev     string     `json:"base_rev"`
	NextRev     string     `json:"next_rev"`
	Scope       Scope      `json:"scope"`
	Durability  Durability `json:"durability"`
	Taint       Taint      `json:"taint"`
	PatchDigest string     `json:"patch_digest"`
	Verdict     Verdict    `json:"verdict"`
	Reason      string     `json:"reason"`
	TS          string     `json:"ts"`
}

type Patch struct {
	Schema     string     `json:"schema"`
	TaskID     string     `json:"task_id"`
	BaseRev    string     `json:"base_rev"`
	Actor      Actor      `json:"actor"`
	Scope      Scope      `json:"scope"`
	Durability Durability `json:"durability"`
	Ops        []Op       `json:"ops"`
	Message    string     `json:"message"`
}

type Op struct {
	Op    string `json:"op"`
	Path  string `json:"path"`
	Value any    `json:"value,omitempty"`
}

type PatchResult struct {
	Schema     string      `json:"schema"`
	TaskID     string      `json:"task_id"`
	BaseRev    string      `json:"base_rev"`
	CurrentRev string      `json:"current_rev"`
	Verdict    Verdict     `json:"verdict"`
	Reason     string      `json:"reason"`
	EventID    string      `json:"event_id,omitempty"`
	RecordRef  string      `json:"record_ref,omitempty"`
	Approval   *Approval   `json:"approval,omitempty"`
	Denial     *Denial     `json:"denial,omitempty"`
	Quarantine *Quarantine `json:"quarantine,omitempty"`
	Conflict   *Conflict   `json:"conflict,omitempty"`
}

type Approval struct {
	DecisionID string `json:"decision_id"`
	State      string `json:"state"`
	Reason     string `json:"reason"`
}

type Denial struct {
	Policy        string `json:"policy"`
	Actor         Actor  `json:"actor"`
	RequiredScope Scope  `json:"required_scope"`
}

type Quarantine struct {
	QuarantineID string `json:"quarantine_id"`
	ArtifactID   string `json:"artifact_id"`
	SubjectKind  string `json:"subject_kind,omitempty"`
	SubjectID    string `json:"subject_id,omitempty"`
	SafeSummary  string `json:"safe_summary"`
}

type Conflict struct {
	Path          string `json:"path"`
	BaseValue     any    `json:"base_value"`
	CurrentValue  any    `json:"current_value"`
	ProposedValue any    `json:"proposed_value"`
	Resolution    string `json:"resolution"`
}

type Policy struct {
	MaxScope           Scope
	RequireApproval    bool
	ApprovalDecisionID string
	ApprovalReason     string
	DenialPolicy       string
	QuarantineSummary  string
}

type Store struct {
	mu      sync.Mutex
	policy  Policy
	tasks   map[string]TaskRecord
	initial map[string]TaskRecord
	history map[string]TaskRecord
	events  map[string][]Event
	seq     uint64
}

func NewStore(policy Policy) *Store {
	if policy.MaxScope == "" {
		policy.MaxScope = ScopeFleet
	}
	if policy.DenialPolicy == "" {
		policy.DenialPolicy = "shared-task.write.scope"
	}
	if policy.ApprovalDecisionID == "" {
		policy.ApprovalDecisionID = "dec_shared_task_approval"
	}
	if policy.ApprovalReason == "" {
		policy.ApprovalReason = "write requires approval"
	}
	if policy.QuarantineSummary == "" {
		policy.QuarantineSummary = "ref is held out of context until a deletion/witness certificate exists"
	}
	return &Store{
		policy:  policy,
		tasks:   map[string]TaskRecord{},
		initial: map[string]TaskRecord{},
		history: map[string]TaskRecord{},
		events:  map[string][]Event{},
	}
}

func (s *Store) Create(record TaskRecord) (TaskRecord, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if record.TaskID == "" {
		return TaskRecord{}, fmt.Errorf("sharedtask: missing task id")
	}
	if reason := bodyRefAdmissionReason(record.BodyRef); reason != "" {
		return TaskRecord{}, fmt.Errorf("sharedtask: body ref refused: %s", reason)
	}
	if _, exists := s.tasks[record.TaskID]; exists {
		return TaskRecord{}, fmt.Errorf("sharedtask: task %q already exists", record.TaskID)
	}
	record.Schema = defaultString(record.Schema, SchemaTask)
	if record.Rev == "" {
		record.Rev = ComputeRev(record)
	}
	cp := cloneRecord(record)
	s.tasks[record.TaskID] = cp
	s.initial[record.TaskID] = cp
	s.history[record.Rev] = cp
	return cloneRecord(cp), nil
}

func (s *Store) Get(taskID string) (TaskRecord, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	record, ok := s.tasks[taskID]
	if !ok {
		return TaskRecord{}, false
	}
	return cloneRecord(record), true
}

func (s *Store) Events(taskID string) []Event {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]Event, len(s.events[taskID]))
	copy(out, s.events[taskID])
	return out
}

func (s *Store) Event(taskID, eventID string) (Event, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, event := range s.events[taskID] {
		if event.EventID == eventID {
			return event, true
		}
	}
	return Event{}, false
}

func (s *Store) Apply(patch Patch) PatchResult {
	s.mu.Lock()
	defer s.mu.Unlock()
	patch.Schema = defaultString(patch.Schema, SchemaPatch)
	task, ok := s.tasks[patch.TaskID]
	if !ok {
		return denied(patch, patch.BaseRev, ReasonUnsupportedPatch, s.policy)
	}
	if patch.BaseRev == "" {
		patch.BaseRev = task.Rev
	}
	if s.policy.RequireApproval {
		return needsApproval(patch, s.policy)
	}
	if !scopeAllowed(patch.Scope, s.policy.MaxScope) {
		return denied(patch, patch.BaseRev, ReasonScopeDenied, s.policy)
	}
	if result, held := s.admitRefs(patch); held {
		return result
	}

	next := cloneRecord(task)
	for _, op := range patch.Ops {
		if patch.BaseRev != task.Rev && !commutes(op) {
			return s.conflict(patch, task, op)
		}
		if conflict, reason, ok := applyOp(&next, op); !ok {
			return resultForUnsupported(patch, task.Rev)
		} else if conflict != nil {
			return PatchResult{
				Schema:     SchemaPatchResult,
				TaskID:     patch.TaskID,
				BaseRev:    patch.BaseRev,
				CurrentRev: task.Rev,
				Verdict:    VerdictConflict,
				Reason:     reason,
				Conflict:   conflict,
			}
		}
	}

	next.UpdatedBy = patch.Actor
	next.Rev = ComputeRev(next)
	s.tasks[patch.TaskID] = cloneRecord(next)
	s.history[next.Rev] = cloneRecord(next)
	event := s.eventFor(patch, task, next)
	s.events[patch.TaskID] = append(s.events[patch.TaskID], event)
	return PatchResult{
		Schema:     SchemaPatchResult,
		TaskID:     patch.TaskID,
		BaseRev:    patch.BaseRev,
		CurrentRev: next.Rev,
		Verdict:    VerdictAccepted,
		EventID:    event.EventID,
		RecordRef:  next.Rev,
	}
}

func ComputeRev(record TaskRecord) string {
	record.Rev = ""
	b, _ := json.Marshal(record)
	sum := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(sum[:])
}

func (s *Store) eventFor(patch Patch, before, after TaskRecord) Event {
	s.seq++
	patchBytes, _ := json.Marshal(patch)
	patchSum := sha256.Sum256(patchBytes)
	idSum := sha256.Sum256([]byte(fmt.Sprintf("%s:%s:%s:%d", patch.TaskID, patch.BaseRev, after.Rev, s.seq)))
	prev := ""
	if events := s.events[patch.TaskID]; len(events) > 0 {
		prev = digest(events[len(events)-1])
	}
	return Event{
		Schema:      SchemaEvent,
		EventID:     "evt_" + hex.EncodeToString(idSum[:8]),
		TaskID:      patch.TaskID,
		PrevEvent:   prev,
		EventKind:   "patch_accepted",
		Actor:       patch.Actor,
		BaseRev:     patch.BaseRev,
		NextRev:     after.Rev,
		Scope:       patch.Scope,
		Durability:  patch.Durability,
		Taint:       before.BodyRef.Taint,
		PatchDigest: "sha256:" + hex.EncodeToString(patchSum[:]),
		Verdict:     VerdictAccepted,
		TS:          fmt.Sprintf("logical:%d", s.seq),
	}
}

func (s *Store) conflict(patch Patch, current TaskRecord, op Op) PatchResult {
	base := s.history[patch.BaseRev]
	baseValue, _ := valueAt(base, op.Path)
	currentValue, _ := valueAt(current, op.Path)
	return PatchResult{
		Schema:     SchemaPatchResult,
		TaskID:     patch.TaskID,
		BaseRev:    patch.BaseRev,
		CurrentRev: current.Rev,
		Verdict:    VerdictConflict,
		Reason:     ReasonStaleBase,
		Conflict: &Conflict{
			Path:          op.Path,
			BaseValue:     baseValue,
			CurrentValue:  currentValue,
			ProposedValue: op.Value,
			Resolution:    "rebase_or_human_decide",
		},
	}
}

func applyOp(record *TaskRecord, op Op) (*Conflict, string, bool) {
	if op.Op == "replace" {
		if conflict, reason, ok := replaceDecisionState(record, op); ok {
			return conflict, reason, true
		}
	}
	switch op.Op + " " + op.Path {
	case "replace /state":
		state, ok := op.Value.(string)
		if !ok || state == "" {
			return nil, "", false
		}
		record.State = state
		return nil, "", true
	case "replace /title":
		title, ok := op.Value.(string)
		if !ok || title == "" {
			return nil, "", false
		}
		record.Title = title
		return nil, "", true
	case "replace /body_ref":
		ref, ok := decodeValue[BodyRef](op.Value)
		if !ok || bodyRefAdmissionReason(ref) != "" {
			return nil, "", false
		}
		record.BodyRef = ref
		return nil, "", true
	case "append /open_decisions":
		decision, ok := decodeValue[Decision](op.Value)
		if !ok || decision.DecisionID == "" {
			return nil, "", false
		}
		for _, existing := range record.OpenDecisions {
			if existing.DecisionID == decision.DecisionID {
				return &Conflict{
					Path:          "/open_decisions",
					BaseValue:     existing,
					CurrentValue:  existing,
					ProposedValue: decision,
					Resolution:    "manual",
				}, "DUPLICATE_DECISION", true
			}
		}
		record.OpenDecisions = append(record.OpenDecisions, decision)
		return nil, "", true
	case "append /notes":
		note, ok := decodeValue[Note](op.Value)
		if !ok || !noteShapeOK(note) {
			return nil, "", false
		}
		for _, existing := range record.Notes {
			if existing.NoteID == note.NoteID {
				return &Conflict{
					Path:          "/notes",
					BaseValue:     existing,
					CurrentValue:  existing,
					ProposedValue: note,
					Resolution:    "manual",
				}, "DUPLICATE_NOTE", true
			}
		}
		record.Notes = append(record.Notes, note)
		return nil, "", true
	case "append /artifacts":
		artifact, ok := decodeValue[ArtifactRef](op.Value)
		if !ok || artifact.ArtifactID == "" {
			return nil, "", false
		}
		record.Artifacts = append(record.Artifacts, artifact)
		return nil, "", true
	}
	return nil, "", false
}

func replaceDecisionState(record *TaskRecord, op Op) (*Conflict, string, bool) {
	decisionID, ok := decisionStatePath(op.Path)
	if !ok {
		return nil, "", false
	}
	state, ok := op.Value.(string)
	if !ok || state == "" {
		return nil, "", false
	}
	for i, decision := range record.OpenDecisions {
		if decision.DecisionID == decisionID {
			record.OpenDecisions[i].State = state
			return nil, "", true
		}
	}
	return &Conflict{
		Path:          op.Path,
		BaseValue:     nil,
		CurrentValue:  nil,
		ProposedValue: state,
		Resolution:    "manual",
	}, ReasonMissingDecision, true
}

func commutes(op Op) bool {
	return op.Op == "append" && (op.Path == "/open_decisions" || op.Path == "/notes" || op.Path == "/artifacts")
}

func valueAt(record TaskRecord, path string) (any, bool) {
	if decisionID, ok := decisionStatePath(path); ok {
		for _, decision := range record.OpenDecisions {
			if decision.DecisionID == decisionID {
				return decision.State, true
			}
		}
		return nil, true
	}
	switch path {
	case "/body_ref":
		return record.BodyRef, true
	case "/state":
		return record.State, true
	case "/title":
		return record.Title, true
	case "/open_decisions":
		return record.OpenDecisions, true
	case "/notes":
		return record.Notes, true
	case "/artifacts":
		return record.Artifacts, true
	}
	return nil, false
}

func decisionStatePath(path string) (string, bool) {
	const prefix = "/open_decisions/"
	const suffix = "/state"
	if !strings.HasPrefix(path, prefix) || !strings.HasSuffix(path, suffix) {
		return "", false
	}
	decisionID := strings.TrimSuffix(strings.TrimPrefix(path, prefix), suffix)
	if decisionID == "" || strings.Contains(decisionID, "/") {
		return "", false
	}
	return decisionID, true
}

func (s *Store) admitRefs(patch Patch) (PatchResult, bool) {
	for _, op := range patch.Ops {
		switch op.Path {
		case "/body_ref":
			if op.Op != "replace" {
				continue
			}
			ref, ok := decodeValue[BodyRef](op.Value)
			if !ok {
				return resultForUnsupported(patch, patch.BaseRev), true
			}
			if reason := bodyRefAdmissionReason(ref); reason != "" {
				return quarantinedBody(patch, reason, s.policy), true
			}
		case "/artifacts":
			if op.Op != "append" {
				continue
			}
			artifact, ok := decodeValue[ArtifactRef](op.Value)
			if !ok {
				return resultForUnsupported(patch, patch.BaseRev), true
			}
			if artifact.Taint == TaintQuarantined {
				return quarantinedArtifact(patch, artifact, ReasonArtifactQuarantine, s.policy), true
			}
			if !artifactShapeOK(artifact) {
				return quarantinedArtifact(patch, artifact, ReasonArtifactWitness, s.policy), true
			}
			if disaggregatedStore(artifact.Store) && !digestShape(artifact.DeletionCertificate) {
				return quarantinedArtifact(patch, artifact, ReasonArtifactWitness, s.policy), true
			}
		case "/notes":
			if op.Op != "append" {
				continue
			}
			note, ok := decodeValue[Note](op.Value)
			if !ok {
				return resultForUnsupported(patch, patch.BaseRev), true
			}
			if note.BodyRef.Taint == TaintQuarantined {
				return quarantinedNote(patch, note, ReasonBodyQuarantine, s.policy), true
			}
			if !noteShapeOK(note) {
				return resultForUnsupported(patch, patch.BaseRev), true
			}
			if disaggregatedBodyRef(note.BodyRef) && !digestShape(note.BodyRef.DeletionCertificate) {
				return quarantinedNote(patch, note, ReasonBodyWitness, s.policy), true
			}
		}
	}
	return PatchResult{}, false
}

func artifactShapeOK(artifact ArtifactRef) bool {
	return artifact.ArtifactID != "" &&
		digestShape(artifact.Ref) &&
		artifact.MediaType != "" &&
		artifact.Store != "" &&
		artifact.Taint != "" &&
		artifact.Scope != ""
}

func noteShapeOK(note Note) bool {
	return note.NoteID != "" &&
		(note.Kind == "comment" || note.Kind == "progress") &&
		note.Author.Kind != "" &&
		note.Author.ID != "" &&
		note.CreatedAt != "" &&
		bodyRefShapeOK(note.BodyRef) &&
		note.BodyRef.Taint != TaintQuarantined
}

func bodyRefShapeOK(ref BodyRef) bool {
	return ref.Kind != "" &&
		digestShape(ref.Digest) &&
		ref.Taint != "" &&
		ref.Scope != "" &&
		ref.Durability != "" &&
		(ref.Kind != "external" || ref.Store != "")
}

func bodyRefAdmissionReason(ref BodyRef) string {
	if ref.Taint == TaintQuarantined {
		return ReasonBodyQuarantine
	}
	if !bodyRefShapeOK(ref) {
		return ReasonBodyWitness
	}
	if disaggregatedBodyRef(ref) && !digestShape(ref.DeletionCertificate) {
		return ReasonBodyWitness
	}
	return ""
}

func disaggregatedBodyRef(ref BodyRef) bool {
	return ref.Kind == "external" || disaggregatedStore(ref.Store)
}

func disaggregatedStore(store string) bool {
	return store != "" && store != "local-cas"
}

func digestShape(value string) bool {
	return strings.HasPrefix(value, "sha256:") && len(value) > len("sha256:")
}

func needsApproval(patch Patch, policy Policy) PatchResult {
	return PatchResult{
		Schema:     SchemaPatchResult,
		TaskID:     patch.TaskID,
		BaseRev:    patch.BaseRev,
		CurrentRev: patch.BaseRev,
		Verdict:    VerdictNeedsApproval,
		Reason:     ReasonApprovalRequired,
		Approval: &Approval{
			DecisionID: policy.ApprovalDecisionID,
			State:      "input_required",
			Reason:     policy.ApprovalReason,
		},
	}
}

func denied(patch Patch, rev, reason string, policy Policy) PatchResult {
	return PatchResult{
		Schema:     SchemaPatchResult,
		TaskID:     patch.TaskID,
		BaseRev:    rev,
		CurrentRev: rev,
		Verdict:    VerdictDenied,
		Reason:     reason,
		Denial: &Denial{
			Policy:        policy.DenialPolicy,
			Actor:         patch.Actor,
			RequiredScope: patch.Scope,
		},
	}
}

func quarantinedArtifact(patch Patch, artifact ArtifactRef, reason string, policy Policy) PatchResult {
	return PatchResult{
		Schema:     SchemaPatchResult,
		TaskID:     patch.TaskID,
		BaseRev:    patch.BaseRev,
		CurrentRev: patch.BaseRev,
		Verdict:    VerdictQuarantined,
		Reason:     reason,
		Quarantine: &Quarantine{
			QuarantineID: "q_" + artifact.ArtifactID,
			ArtifactID:   artifact.ArtifactID,
			SubjectKind:  "artifact",
			SubjectID:    artifact.ArtifactID,
			SafeSummary:  policy.QuarantineSummary,
		},
	}
}

func quarantinedNote(patch Patch, note Note, reason string, policy Policy) PatchResult {
	return PatchResult{
		Schema:     SchemaPatchResult,
		TaskID:     patch.TaskID,
		BaseRev:    patch.BaseRev,
		CurrentRev: patch.BaseRev,
		Verdict:    VerdictQuarantined,
		Reason:     reason,
		Quarantine: &Quarantine{
			QuarantineID: "q_" + note.NoteID,
			SubjectKind:  "note",
			SubjectID:    note.NoteID,
			SafeSummary:  policy.QuarantineSummary,
		},
	}
}

func quarantinedBody(patch Patch, reason string, policy Policy) PatchResult {
	return PatchResult{
		Schema:     SchemaPatchResult,
		TaskID:     patch.TaskID,
		BaseRev:    patch.BaseRev,
		CurrentRev: patch.BaseRev,
		Verdict:    VerdictQuarantined,
		Reason:     reason,
		Quarantine: &Quarantine{
			QuarantineID: "q_body_ref",
			SubjectKind:  "body",
			SubjectID:    "body_ref",
			SafeSummary:  policy.QuarantineSummary,
		},
	}
}

func resultForUnsupported(patch Patch, currentRev string) PatchResult {
	return PatchResult{
		Schema:     SchemaPatchResult,
		TaskID:     patch.TaskID,
		BaseRev:    patch.BaseRev,
		CurrentRev: currentRev,
		Verdict:    VerdictDenied,
		Reason:     ReasonUnsupportedPatch,
		Denial: &Denial{
			Policy:        "shared-task.patch.shape",
			Actor:         patch.Actor,
			RequiredScope: patch.Scope,
		},
	}
}

func scopeAllowed(got, max Scope) bool {
	gr, gok := scopeRank(got)
	mr, mok := scopeRank(max)
	return gok && mok && gr <= mr
}

func scopeRank(scope Scope) (int, bool) {
	switch scope {
	case ScopeAgent:
		return 0, true
	case ScopeFleet:
		return 1, true
	case ScopeTenant:
		return 2, true
	case ScopePublic:
		return 3, true
	}
	return 0, false
}

func decodeValue[T any](value any) (T, bool) {
	var out T
	b, err := json.Marshal(value)
	if err != nil {
		return out, false
	}
	if err := json.Unmarshal(b, &out); err != nil {
		return out, false
	}
	return out, true
}

func cloneRecord(record TaskRecord) TaskRecord {
	b, _ := json.Marshal(record)
	var out TaskRecord
	_ = json.Unmarshal(b, &out)
	return out
}

func digest(v any) string {
	b, _ := json.Marshal(v)
	sum := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(sum[:])
}

func defaultString(got, fallback string) string {
	if got != "" {
		return got
	}
	return fallback
}
