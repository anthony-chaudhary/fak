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

const Schema = "fak-goal-registry/1"

type Lifecycle string

const (
	Active     Lifecycle = "active"
	Achieved   Lifecycle = "achieved"
	Abandoned  Lifecycle = "abandoned"
	Superseded Lifecycle = "superseded"
	Blocked    Lifecycle = "blocked"
	Paused     Lifecycle = "paused"
)

type Provenance struct {
	Actor     string `json:"actor"`
	Authority string `json:"authority"`
	Witness   string `json:"witness,omitempty"`
}

type Relation struct {
	Kind   string `json:"kind"`
	GoalID string `json:"goal_id"`
}

type Goal struct {
	GoalID     string     `json:"goal_id"`
	Title      string     `json:"title"`
	Summary    string     `json:"summary,omitempty"`
	Lifecycle  Lifecycle  `json:"lifecycle"`
	CreatedAt  time.Time  `json:"created_at"`
	UpdatedAt  time.Time  `json:"updated_at"`
	Provenance Provenance `json:"provenance"`
	Relations  []Relation `json:"relations,omitempty"`
}

type Binding struct {
	GoalID     string     `json:"goal_id"`
	Namespace  string     `json:"namespace"`
	ExternalID string     `json:"external_id"`
	Revision   string     `json:"revision,omitempty"`
	BoundAt    time.Time  `json:"bound_at"`
	Provenance Provenance `json:"provenance"`
}

type Registry struct {
	Schema   string    `json:"schema"`
	Goals    []Goal    `json:"goals"`
	Bindings []Binding `json:"bindings"`
}

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

func (s Store) Load() (Registry, error) {
	r := Registry{Schema: Schema, Goals: []Goal{}, Bindings: []Binding{}}
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
	g := Goal{GoalID: id, Title: title, Summary: strings.TrimSpace(summary), Lifecycle: Active, CreatedAt: now, UpdatedAt: now, Provenance: provenance, Relations: relations}
	r, err := s.Load()
	if err != nil {
		return Goal{}, err
	}
	r.Goals = append(r.Goals, g)
	return g, s.save(r)
}

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

func (s Store) List() ([]Goal, error) {
	r, err := s.Load()
	if err != nil {
		return nil, err
	}
	sort.Slice(r.Goals, func(i, j int) bool { return r.Goals[i].CreatedAt.Before(r.Goals[j].CreatedAt) })
	return r.Goals, nil
}

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
	for i := range r.Goals {
		if r.Goals[i].GoalID != id {
			continue
		}
		r.Goals[i].Title, r.Goals[i].Summary, r.Goals[i].Lifecycle, r.Goals[i].UpdatedAt = title, strings.TrimSpace(summary), lifecycle, s.now()
		return r.Goals[i], s.save(r)
	}
	return Goal{}, fmt.Errorf("goal %q not found", id)
}

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
