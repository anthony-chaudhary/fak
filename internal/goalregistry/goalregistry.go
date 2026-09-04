// Package goalregistry stores durable user intent independently from any execution tree.
// trajctl objectives remain execution controllers and bind through namespace fak:trajctl;
// this package is the sole authority for cross-harness goal identity and lifecycle.
package goalregistry

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/flock"
)

// Schema identifies the canonical JSON persistence format for goal registry storage.
const Schema = "fak-goal-registry/1"

// DefaultEvidencePolicy enforces third-party verification for terminal state transitions.
const DefaultEvidencePolicy = "independent_witness_required"

// Lifecycle represents the current operational disposition of a tracked goal.
type Lifecycle string

const (
	// Active indicates a goal is actively pursued by agents or operators.
	Active Lifecycle = "active"
	// Achieved indicates a goal reached successful completion with witness evidence.
	Achieved Lifecycle = "achieved"
	// Abandoned indicates a goal was intentionally discontinued prior to completion.
	Abandoned Lifecycle = "abandoned"
	// Superseded indicates a goal was replaced by a subsequent or refined objective.
	Superseded Lifecycle = "superseded"
	// Blocked indicates progress is halted by an external dependency or obstacle.
	Blocked Lifecycle = "blocked"
	// Paused indicates active execution is temporarily suspended without termination.
	Paused Lifecycle = "paused"
)

// Provenance records origin metadata including initiating actor, authority source, and audit witness.
type Provenance struct {
	Actor     string `json:"actor"`
	Authority string `json:"authority"`
	Witness   string `json:"witness,omitempty"`
}

// Relation expresses a structural lineage or replacement link between registered goals.
type Relation struct {
	Kind   string `json:"kind"`
	GoalID string `json:"goal_id"`
}

// EvidenceClass categorizes the trust level and verification origin of an outcome assertion.
type EvidenceClass string

const (
	// HarnessAssertion designates an outcome claimed directly by an execution harness.
	HarnessAssertion EvidenceClass = "harness_assertion"
	// AgentAssertion designates an outcome reported by an autonomous model worker.
	AgentAssertion EvidenceClass = "agent_assertion"
	// OperatorDeclaration designates an outcome explicitly asserted by a human operator.
	OperatorDeclaration EvidenceClass = "operator_declaration"
	// IndependentWitness designates an outcome corroborated by independent verification artifacts.
	IndependentWitness EvidenceClass = "independent_witness"
)

// OutcomeEvidence captures auditable witness records associated with lifecycle changes.
type OutcomeEvidence struct {
	GoalID     string        `json:"goal_id"`
	Lifecycle  Lifecycle     `json:"lifecycle"`
	Class      EvidenceClass `json:"class"`
	Author     string        `json:"author"`
	Reference  string        `json:"reference"`
	RecordedAt time.Time     `json:"recorded_at"`
}

// Goal holds durable intent, lifecycle state, provenance, and lineage links.
type Goal struct {
	GoalID         string     `json:"goal_id"`
	Title          string     `json:"title"`
	Summary        string     `json:"summary,omitempty"`
	Lifecycle      Lifecycle  `json:"lifecycle"`
	CreatedAt      time.Time  `json:"created_at"`
	UpdatedAt      time.Time  `json:"updated_at"`
	Provenance     Provenance `json:"provenance"`
	EvidencePolicy string     `json:"evidence_policy,omitempty"`
	Relations      []Relation `json:"relations,omitempty"`
}

// Binding maps an external harness or tracker identifier into a canonical goal.
type Binding struct {
	GoalID         string     `json:"goal_id"`
	Namespace      string     `json:"namespace"`
	ExternalID     string     `json:"external_id"`
	Revision       string     `json:"revision,omitempty"`
	BoundAt        time.Time  `json:"bound_at"`
	Provenance     Provenance `json:"provenance"`
	EvidencePolicy string     `json:"evidence_policy,omitempty"`
}

// Registry encapsulates persisted goals, external bindings, and historical outcome evidence.
type Registry struct {
	Schema          string            `json:"schema"`
	Goals           []Goal            `json:"goals"`
	Bindings        []Binding         `json:"bindings"`
	OutcomeEvidence []OutcomeEvidence `json:"outcome_evidence,omitempty"`
}

// Store manages thread-safe read and write operations against the goal registry backing file.
type Store struct {
	Path string
	Now  func() time.Time
}

func (s Store) withWriteLock(fn func() error) error {
	if err := os.MkdirAll(filepath.Dir(s.Path), 0o700); err != nil {
		return err
	}
	lock, err := os.OpenFile(s.Path+".lock", os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return err
	}
	defer lock.Close()
	deadline := time.Now().Add(10 * time.Second)
	for {
		err = flock.TryLock(lock)
		if err == nil {
			defer flock.Unlock(lock)
			return fn()
		}
		if !errors.Is(err, flock.ErrLockBusy) || time.Now().After(deadline) {
			return fmt.Errorf("goal registry lock: %w", err)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// DefaultPath returns the resolved file path for goal persistence, honoring environment overrides.
func DefaultPath() string {
	if p := strings.TrimSpace(os.Getenv("FAK_GOAL_REGISTRY")); p != "" {
		return p
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return filepath.Join(".fak", "goals.json")
	}
	return filepath.Join(home, ".fak", "goals.json")
}

// Load reads and decodes the registry contents from disk, initializing an empty state if nonexistent.
func (s Store) Load() (Registry, error) {
	r := Registry{Schema: Schema, Goals: []Goal{}, Bindings: []Binding{}, OutcomeEvidence: []OutcomeEvidence{}}
	b, err := os.ReadFile(s.Path)
	if errors.Is(err, os.ErrNotExist) {
		return r, nil
	}
	if err != nil {
		return r, err
	}
	if err := json.Unmarshal(b, &r); err != nil {
		return r, fmt.Errorf("decode goal registry: %w", err)
	}
	if r.Schema != Schema {
		return r, fmt.Errorf("unsupported schema %q", r.Schema)
	}
	return r, nil
}

// Create allocates an opaque goal identifier and persists a new active goal entry.
func (s Store) Create(title, summary string, provenance Provenance, relations []Relation) (Goal, error) {
	var out Goal
	err := s.withWriteLock(func() error {
		var err error
		out, err = s.create(title, summary, provenance, relations)
		return err
	})
	return out, err
}

func (s Store) create(title, summary string, provenance Provenance, relations []Relation) (Goal, error) {
	title = strings.TrimSpace(title)
	if title == "" {
		return Goal{}, errors.New("title is required")
	}
	if err := validateProvenance(provenance); err != nil {
		return Goal{}, err
	}
	for _, rel := range relations {
		if !validRelation(rel) {
			return Goal{}, fmt.Errorf("invalid relation %+v", rel)
		}
	}
	id, err := newID()
	if err != nil {
		return Goal{}, err
	}
	now := s.now()
	g := Goal{GoalID: id, Title: title, Summary: strings.TrimSpace(summary), Lifecycle: Active, CreatedAt: now, UpdatedAt: now, Provenance: provenance, EvidencePolicy: DefaultEvidencePolicy, Relations: relations}
	r, err := s.Load()
	if err != nil {
		return Goal{}, err
	}
	r.Goals = append(r.Goals, g)
	return g, s.save(r)
}

// Resolve returns the single explicitly recorded external binding. An omitted
// revision is a wildcard only when it identifies exactly one binding; callers
// must name the revision when external history would otherwise be ambiguous.
func (s Store) Resolve(namespace, externalID, revision string) (Goal, Binding, error) {
	namespace = strings.TrimSpace(namespace)
	externalID = strings.TrimSpace(externalID)
	revision = strings.TrimSpace(revision)
	if namespace == "" || externalID == "" {
		return Goal{}, Binding{}, errors.New("namespace and external ID are required")
	}
	r, err := s.Load()
	if err != nil {
		return Goal{}, Binding{}, err
	}
	var matches []Binding
	for _, b := range r.Bindings {
		if b.Namespace == namespace && b.ExternalID == externalID && (revision == "" || b.Revision == revision) {
			matches = append(matches, b)
		}
	}
	if len(matches) == 0 {
		return Goal{}, Binding{}, fmt.Errorf("binding not found: %s:%s revision %q", namespace, externalID, revision)
	}
	if len(matches) != 1 {
		return Goal{}, Binding{}, fmt.Errorf("binding is ambiguous: %s:%s matches %d revisions; specify --revision", namespace, externalID, len(matches))
	}
	for _, g := range r.Goals {
		if g.GoalID == matches[0].GoalID {
			return g, matches[0], nil
		}
	}
	return Goal{}, Binding{}, fmt.Errorf("binding references missing goal %q", matches[0].GoalID)
}

// Show retrieves a goal along with all registered external bindings linked to its identifier.
func (s Store) Show(id string) (Goal, []Binding, error) {
	r, err := s.Load()
	if err != nil {
		return Goal{}, nil, err
	}
	for _, g := range r.Goals {
		if g.GoalID == id {
			var bindings []Binding
			for _, b := range r.Bindings {
				if b.GoalID == id {
					bindings = append(bindings, b)
				}
			}
			return g, bindings, nil
		}
	}
	return Goal{}, nil, fmt.Errorf("goal %q not found", id)
}

// List returns all registered goals sorted chronologically by creation timestamp.
func (s Store) List() ([]Goal, error) {
	r, err := s.Load()
	if err != nil {
		return nil, err
	}
	sort.Slice(r.Goals, func(i, j int) bool { return r.Goals[i].CreatedAt.Before(r.Goals[j].CreatedAt) })
	return r.Goals, nil
}

// Update modifies title, summary, and non-terminal operational state without requiring witness proofs.
func (s Store) Update(id, title, summary string, lifecycle Lifecycle) (Goal, error) {
	var out Goal
	err := s.withWriteLock(func() error {
		var err error
		out, err = s.update(id, title, summary, lifecycle)
		return err
	})
	return out, err
}

func (s Store) update(id, title, summary string, lifecycle Lifecycle) (Goal, error) {
	r, err := s.Load()
	if err != nil {
		return Goal{}, err
	}
	if title = strings.TrimSpace(title); title == "" {
		return Goal{}, errors.New("title is required")
	}
	if !validLifecycle(lifecycle) {
		return Goal{}, fmt.Errorf("invalid lifecycle %q", lifecycle)
	}
	if lifecycle != Active && lifecycle != Paused {
		return Goal{}, errors.New("terminal lifecycle requires typed outcome evidence; use Transition")
	}
	for i := range r.Goals {
		if r.Goals[i].GoalID != id {
			continue
		}
		r.Goals[i].Title, r.Goals[i].Summary, r.Goals[i].Lifecycle, r.Goals[i].UpdatedAt = title, strings.TrimSpace(summary), lifecycle, s.now()
		return r.Goals[i], s.save(r)
	}
	return Goal{}, fmt.Errorf("goal %q not found", id)
}

// Transition applies one evidence-bearing lifecycle decision. Terminal conflicts
// require an explicit reopen first, preserving every prior report.
func (s Store) Transition(goalID string, lifecycle Lifecycle, evidence OutcomeEvidence) (Goal, error) {
	if !validLifecycle(lifecycle) {
		return Goal{}, fmt.Errorf("invalid lifecycle %q", lifecycle)
	}
	if lifecycle == Active || lifecycle == Paused {
		return Goal{}, errors.New("use Reopen for a non-terminal transition")
	}
	if evidence.Class != IndependentWitness {
		return Goal{}, fmt.Errorf("policy %s requires independent_witness evidence", DefaultEvidencePolicy)
	}
	if strings.TrimSpace(evidence.Author) == "" || strings.TrimSpace(evidence.Reference) == "" {
		return Goal{}, errors.New("evidence author and reference are required")
	}
	var out Goal
	err := s.withWriteLock(func() error {
		r, err := s.Load()
		if err != nil {
			return err
		}
		for i := range r.Goals {
			if r.Goals[i].GoalID != goalID {
				continue
			}
			if r.Goals[i].Lifecycle != Active && r.Goals[i].Lifecycle != Paused && r.Goals[i].Lifecycle != lifecycle {
				return fmt.Errorf("terminal lifecycle conflict: goal is %s; reopen before %s", r.Goals[i].Lifecycle, lifecycle)
			}
			evidence.GoalID, evidence.Lifecycle, evidence.RecordedAt = goalID, lifecycle, s.now()
			evidence.Author, evidence.Reference = strings.TrimSpace(evidence.Author), strings.TrimSpace(evidence.Reference)
			r.OutcomeEvidence = append(r.OutcomeEvidence, evidence)
			r.Goals[i].Lifecycle, r.Goals[i].UpdatedAt = lifecycle, s.now()
			out = r.Goals[i]
			return s.save(r)
		}
		return fmt.Errorf("goal %q not found", goalID)
	})
	return out, err
}

// Reopen transitions a terminated goal back to active status backed by operator declaration.
func (s Store) Reopen(goalID, author, reference string) (Goal, error) {
	if strings.TrimSpace(author) == "" || strings.TrimSpace(reference) == "" {
		return Goal{}, errors.New("reopen author and reference are required")
	}
	var out Goal
	err := s.withWriteLock(func() error {
		r, err := s.Load()
		if err != nil {
			return err
		}
		for i := range r.Goals {
			if r.Goals[i].GoalID == goalID {
				r.OutcomeEvidence = append(r.OutcomeEvidence, OutcomeEvidence{GoalID: goalID, Lifecycle: Active, Class: OperatorDeclaration, Author: strings.TrimSpace(author), Reference: strings.TrimSpace(reference), RecordedAt: s.now()})
				r.Goals[i].Lifecycle, r.Goals[i].UpdatedAt = Active, s.now()
				out = r.Goals[i]
				return s.save(r)
			}
		}
		return fmt.Errorf("goal %q not found", goalID)
	})
	return out, err
}

// OutcomeEvidence returns historical outcome verification records associated with the specified goal.
func (s Store) OutcomeEvidence(goalID string) ([]OutcomeEvidence, error) {
	r, err := s.Load()
	if err != nil {
		return nil, err
	}
	var out []OutcomeEvidence
	for _, e := range r.OutcomeEvidence {
		if e.GoalID == goalID {
			out = append(out, e)
		}
	}
	return out, nil
}

// RequireGoal verifies that an opaque canonical goal already exists.
func (s Store) RequireGoal(goalID string) (Goal, error) {
	goalID = strings.TrimSpace(goalID)
	if goalID == "" {
		return Goal{}, errors.New("goal ID is required")
	}
	r, err := s.Load()
	if err != nil {
		return Goal{}, err
	}
	for _, g := range r.Goals {
		if g.GoalID == goalID {
			return g, nil
		}
	}
	return Goal{}, fmt.Errorf("goal %q not found", goalID)
}

// Bind associates an external namespace identifier and optional revision with a canonical goal.
func (s Store) Bind(goalID, namespace, externalID, revision string, provenance Provenance) (Binding, error) {
	var out Binding
	err := s.withWriteLock(func() error {
		var err error
		out, err = s.bind(goalID, namespace, externalID, revision, provenance)
		return err
	})
	return out, err
}

func (s Store) bind(goalID, namespace, externalID, revision string, provenance Provenance) (Binding, error) {
	namespace, externalID = strings.TrimSpace(namespace), strings.TrimSpace(externalID)
	if namespace == "" || externalID == "" {
		return Binding{}, errors.New("namespace and external ID are required")
	}
	if err := validateProvenance(provenance); err != nil {
		return Binding{}, err
	}
	r, err := s.Load()
	if err != nil {
		return Binding{}, err
	}
	found := false
	for _, g := range r.Goals {
		if g.GoalID == goalID {
			found = true
			break
		}
	}
	if !found {
		return Binding{}, fmt.Errorf("goal %q not found", goalID)
	}
	for _, b := range r.Bindings {
		if b.Namespace == namespace && b.ExternalID == externalID && b.Revision == revision {
			if b.GoalID == goalID {
				return b, nil
			}
			return Binding{}, fmt.Errorf("binding collision: %s:%s revision %q already belongs to %s", namespace, externalID, revision, b.GoalID)
		}
	}
	b := Binding{GoalID: goalID, Namespace: namespace, ExternalID: externalID, Revision: strings.TrimSpace(revision), BoundAt: s.now(), Provenance: provenance}
	r.Bindings = append(r.Bindings, b)
	return b, s.save(r)
}

// Unbind removes a specific namespace and external identifier association from the registry.
func (s Store) Unbind(goalID, namespace, externalID, revision string) error {
	return s.withWriteLock(func() error { return s.unbind(goalID, namespace, externalID, revision) })
}

func (s Store) unbind(goalID, namespace, externalID, revision string) error {
	r, err := s.Load()
	if err != nil {
		return err
	}
	out := r.Bindings[:0]
	removed := false
	for _, b := range r.Bindings {
		if b.GoalID == goalID && b.Namespace == namespace && b.ExternalID == externalID && b.Revision == revision {
			removed = true
			continue
		}
		out = append(out, b)
	}
	if !removed {
		return errors.New("binding not found")
	}
	r.Bindings = out
	return s.save(r)
}

func (s Store) now() time.Time {
	if s.Now != nil {
		return s.Now().UTC()
	}
	return time.Now().UTC()
}

func (s Store) save(r Registry) error {
	r.Schema = Schema
	if err := os.MkdirAll(filepath.Dir(s.Path), 0o700); err != nil {
		return err
	}
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(s.Path), ".goals-*.tmp")
	if err != nil {
		return err
	}
	name := tmp.Name()
	defer os.Remove(name)
	if err := tmp.Chmod(0o600); err != nil {
		tmp.Close()
		return err
	}
	if _, err := tmp.Write(append(b, '\n')); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(name, s.Path)
}

func newID() (string, error) {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", err
	}
	return "goal_" + hex.EncodeToString(b[:]), nil
}
func validateProvenance(p Provenance) error {
	if strings.TrimSpace(p.Actor) == "" || strings.TrimSpace(p.Authority) == "" {
		return errors.New("provenance actor and authority are required")
	}
	return nil
}
func validLifecycle(v Lifecycle) bool {
	return v == Active || v == Achieved || v == Abandoned || v == Superseded || v == Blocked || v == Paused
}
func validRelation(r Relation) bool {
	return (r.Kind == "parent_goal" || r.Kind == "derived_from" || r.Kind == "supersedes") && strings.TrimSpace(r.GoalID) != ""
}
