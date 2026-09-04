// Package learningobservation stores content-addressed learning records and typed lineage edges.
package learningobservation

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Schema identifies the JSON schema for learning observation stores.
const Schema = "fak.learning-observation.v1"

// Kind distinguishes the lifecycle stage of a learning record.
type Kind string

// Supported observation record kinds.
const (
	KindObservation Kind = "observation"
	KindCandidate   Kind = "candidate"
	KindWitness     Kind = "witness"
	KindVerdict     Kind = "verdict"
)

// Valid reports whether k is a known record kind.
func (k Kind) Valid() bool {
	switch k {
	case KindObservation, KindCandidate, KindWitness, KindVerdict:
		return true
	default:
		return false
	}
}

// Outcome defines the evaluation decision for a verdict record.
type Outcome string

// Supported verdict outcomes.
const (
	OutcomeKept     Outcome = "kept"
	OutcomeRejected Outcome = "rejected"
)

// Valid reports whether o is a supported verdict outcome.
func (o Outcome) Valid() bool { return o == OutcomeKept || o == OutcomeRejected }

// Relation defines a typed directed lineage connection between records.
type Relation string

// Supported lineage relations.
const (
	ObservedFrom Relation = "observed-from"
	Supports     Relation = "supports"
	Contradicts  Relation = "contradicts"
	Proposes     Relation = "proposes"
	TestedBy     Relation = "tested-by"
	KeptAs       Relation = "kept-as"
	RejectedAs   Relation = "rejected-as"
)

// Relations lists all supported lineage relation types.
var Relations = []Relation{ObservedFrom, Supports, Contradicts, Proposes, TestedBy, KeptAs, RejectedAs}

// Valid reports whether r is in Relations.
func (r Relation) Valid() bool {
	for _, known := range Relations {
		if r == known {
			return true
		}
	}
	return false
}

// Record is a content-addressed learning entity with normalized provenance.
type Record struct {
	ID      string  `json:"id"`
	Kind    Kind    `json:"kind"`
	Source  string  `json:"source"`
	Content string  `json:"content"`
	Outcome Outcome `json:"outcome,omitempty"`
}

// Edge is a directed lineage relationship from one record to another.
type Edge struct {
	From     string   `json:"from"`
	Relation Relation `json:"relation"`
	To       string   `json:"to"`
}

// Store contains content-addressed learning records and their lineage edges.
type Store struct {
	Schema  string   `json:"schema"`
	Records []Record `json:"records"`
	Edges   []Edge   `json:"edges"`
}

// Error sentinels returned by Store operations.
var (
	ErrConflict        = errors.New("conflicting content")
	ErrDanglingID      = errors.New("dangling id")
	ErrUnknownRelation = errors.New("unknown relation")
	ErrCycle           = errors.New("lineage cycle")
)

// Load reads a store from disk. If path does not exist, an empty store is returned.
func Load(path string) (*Store, error) {
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return &Store{Schema: Schema, Records: []Record{}, Edges: []Edge{}}, nil
	}
	if err != nil {
		return nil, err
	}
	var s Store
	if err := json.Unmarshal(data, &s); err != nil {
		return nil, fmt.Errorf("decode store: %w", err)
	}
	if s.Schema != Schema {
		return nil, fmt.Errorf("store schema %q, want %q", s.Schema, Schema)
	}
	return &s, nil
}

// Add inserts a record or returns an existing identical record. Conflicting content returns ErrConflict.
func (s *Store) Add(kind Kind, source, content string, outcome Outcome) (Record, bool, error) {
	source = normalize(source)
	content = normalize(content)
	if !kind.Valid() {
		return Record{}, false, fmt.Errorf("unknown kind %q", kind)
	}
	if source == "" || content == "" {
		return Record{}, false, errors.New("source and content are required")
	}
	if kind == KindVerdict {
		if !outcome.Valid() {
			return Record{}, false, fmt.Errorf("verdict outcome must be %q or %q", OutcomeKept, OutcomeRejected)
		}
	} else if outcome != "" {
		return Record{}, false, errors.New("outcome is only valid for verdict records")
	}
	for _, record := range s.Records {
		if record.Kind == kind && normalize(record.Source) == source {
			if normalize(record.Content) == content && record.Outcome == outcome {
				return record, false, nil
			}
			return Record{}, false, fmt.Errorf("%w for kind %q source %q", ErrConflict, kind, source)
		}
	}
	record := Record{ID: stableID(kind, source, content, outcome), Kind: kind, Source: source, Content: content, Outcome: outcome}
	s.Records = append(s.Records, record)
	sort.Slice(s.Records, func(i, j int) bool { return s.Records[i].ID < s.Records[j].ID })
	return record, true, nil
}

// Link connects two existing records with a directed lineage edge, rejecting unknown relations, dangling IDs, and cycles.
func (s *Store) Link(from string, relation Relation, to string) (bool, error) {
	if !relation.Valid() {
		return false, fmt.Errorf("%w %q", ErrUnknownRelation, relation)
	}
	if s.find(from) == nil || s.find(to) == nil {
		return false, fmt.Errorf("%w: from=%q to=%q", ErrDanglingID, from, to)
	}
	for _, edge := range s.Edges {
		if edge.From == from && edge.Relation == relation && edge.To == to {
			return false, nil
		}
	}
	if from == to || s.reachable(to, from) {
		return false, fmt.Errorf("%w: %s -> %s", ErrCycle, from, to)
	}
	s.Edges = append(s.Edges, Edge{From: from, Relation: relation, To: to})
	sort.Slice(s.Edges, func(i, j int) bool {
		a, b := s.Edges[i], s.Edges[j]
		return a.From+a.Relation.String()+a.To < b.From+b.Relation.String()+b.To
	})
	return true, nil
}

// String returns the string representation of the relation.
func (r Relation) String() string { return string(r) }

// Trace traverses the lineage reachable from candidateID, returning all reached records and edges.
func (s *Store) Trace(candidateID string) ([]Record, []Edge, error) {
	root := s.find(candidateID)
	if root == nil {
		return nil, nil, fmt.Errorf("%w: %q", ErrDanglingID, candidateID)
	}
	if root.Kind != KindCandidate {
		return nil, nil, fmt.Errorf("trace root %q is %q, want candidate", candidateID, root.Kind)
	}
	seen := map[string]bool{candidateID: true}
	queue := []string{candidateID}
	var edges []Edge
	for len(queue) > 0 {
		id := queue[0]
		queue = queue[1:]
		for _, edge := range s.Edges {
			if edge.From != id {
				continue
			}
			edges = append(edges, edge)
			if !seen[edge.To] {
				seen[edge.To] = true
				queue = append(queue, edge.To)
			}
		}
	}
	records := make([]Record, 0, len(seen))
	for _, record := range s.Records {
		if seen[record.ID] {
			records = append(records, record)
		}
	}
	return records, edges, nil
}

// Save writes the store atomically to path with 0600 permissions.
func (s *Store) Save(path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	tmp, err := os.CreateTemp(filepath.Dir(path), ".learning-observation-*")
	if err != nil {
		return err
	}
	name := tmp.Name()
	defer os.Remove(name)
	if _, err = tmp.Write(data); err == nil {
		err = tmp.Sync()
	}
	if closeErr := tmp.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	if err := os.Chmod(name, 0o600); err != nil {
		return err
	}
	return os.Rename(name, path)
}

func (s *Store) find(id string) *Record {
	for i := range s.Records {
		if s.Records[i].ID == id {
			return &s.Records[i]
		}
	}
	return nil
}

func (s *Store) reachable(from, target string) bool {
	seen := map[string]bool{from: true}
	queue := []string{from}
	for len(queue) > 0 {
		id := queue[0]
		queue = queue[1:]
		if id == target {
			return true
		}
		for _, edge := range s.Edges {
			if edge.From == id && !seen[edge.To] {
				seen[edge.To] = true
				queue = append(queue, edge.To)
			}
		}
	}
	return false
}

func stableID(kind Kind, source, content string, outcome Outcome) string {
	sum := sha256.Sum256([]byte(string(kind) + "\x00" + source + "\x00" + content + "\x00" + string(outcome)))
	return "lo_" + hex.EncodeToString(sum[:16])
}

func normalize(value string) string { return strings.Join(strings.Fields(value), " ") }
