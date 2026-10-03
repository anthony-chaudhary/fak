// inkernel_incremental.go — a Go port of OpenCode's SystemContext Source / Registry
// / Snapshot incremental-context-update pattern (opencode@4eb29a6, MIT): models
// privileged system context as independently refreshable typed sources so a resident
// context is re-baselined only when a source genuinely changes, and a source that is
// merely unavailable keeps its admitted snapshot instead of being mistaken for removed.
//
// Semantics mirror the upstream exactly. A Source is one typed value with a
// namespaced Key, a Load observer, a Baseline render for a fresh generation, an
// Update render for the incremental delta when the value changes, and an optional
// Removed render for when the source leaves the registry. The registry composes
// sources in order and rejects duplicate keys. The Snapshot is the durable
// comparison state; reconciliation renders ONLY the changed / new / removed sources,
// never a full re-render.

package agent

import (
	"encoding/json"
	"fmt"
	"reflect"
	"regexp"
	"sort"
	"strings"
)

// ContextKey is the stable, namespaced identity of one independently refreshable
// system-context source. Valid keys match the upstream pattern (a namespace segment,
// a slash, then a path-like segment).
type ContextKey string

var contextKeyPattern = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._/-]*$`)

// ValidContextKey reports whether k is a well-formed namespaced system-context key.
func ValidContextKey(k ContextKey) bool {
	return contextKeyPattern.MatchString(string(k))
}

// comparedKind is the closed outcome of comparing one observed value against a
// stored snapshot value.
type comparedKind int

const (
	// comparedUnchanged: the decoded stored value is DeepEqual to the observed value.
	comparedUnchanged comparedKind = iota
	// comparedIncompatible: the stored value failed to decode; replacement is required.
	comparedIncompatible
	// comparedUpdated: the value changed; text carries the incremental render.
	comparedUpdated
)

// compared is the type-erased result of one source comparison.
type compared struct {
	kind     comparedKind
	text     string
	snapshot SourceSnapshot
}

// packedLoad is the type-erased, already-observed form of one source: the closures
// below close over the value observed at `load()` time. `unavailable` records that
// observation failed temporarily (which is NOT removal).
type packedLoad struct {
	unavailable bool
	baseline    func() (string, SourceSnapshot, error)
	compare     func(prev json.RawMessage) (compared, error)
}

// PackedContextSource hides one source's value type behind an interface. The
// unexported load method means only this package can implement or observe it; callers
// construct sources exclusively through MakeContextSource.
type PackedContextSource interface {
	Key() ContextKey
	load() packedLoad
}

// SourceSnapshot is the durable comparison state for one admitted source: the
// JSON-encoded value plus the optional render text captured for its eventual removal.
type SourceSnapshot struct {
	Value   json.RawMessage `json:"value"`
	Removed string          `json:"removed,omitempty"`
}

// ContextSnapshot maps each admitted source key to its durable comparison state.
type ContextSnapshot map[ContextKey]SourceSnapshot

// ContextGeneration is the immutable baseline and durable snapshot of a fresh
// context generation.
type ContextGeneration struct {
	Baseline string          `json:"baseline"`
	Snapshot ContextSnapshot `json:"snapshot"`
}

// ReconcileKind is the closed outcome vocabulary of a reconciliation.
type ReconcileKind string

const (
	// ReconcileUnchanged: nothing changed, was added, or was removed.
	ReconcileUnchanged ReconcileKind = "unchanged"
	// ReconcileUpdated: Text carries only the changed / new / removed renderers.
	ReconcileUpdated ReconcileKind = "updated"
	// ReconcileReplacementReady: a full new generation is required and is set.
	ReconcileReplacementReady ReconcileKind = "replacement_ready"
	// ReconcileReplacementBlocked: a replacement is required but admitted context
	// is temporarily unavailable, so the caller must wait.
	ReconcileReplacementBlocked ReconcileKind = "replacement_blocked"
)

// ReconcileResult is the outcome of reconciling a registry against a snapshot.
type ReconcileResult struct {
	Kind       ReconcileKind
	Text       string            // set only when Kind == ReconcileUpdated
	Snapshot   ContextSnapshot   // new snapshot for unchanged/updated/replacement_ready
	Generation ContextGeneration // set only when Kind == ReconcileReplacementReady
}

// DuplicateContextKeyError is returned when two composed registries share a key.
type DuplicateContextKeyError struct{ Key ContextKey }

func (e *DuplicateContextKeyError) Error() string {
	return "Duplicate system context key: " + string(e.Key)
}

// InitializationBlockedError is returned when any source is unavailable during
// initialization; Keys is in registry order.
type InitializationBlockedError struct{ Keys []ContextKey }

func (e *InitializationBlockedError) Error() string {
	keys := make([]string, 0, len(e.Keys))
	for _, k := range e.Keys {
		keys = append(keys, string(k))
	}
	return "System context initialization blocked by unavailable sources: " + strings.Join(keys, ", ")
}

// contextSource is the concrete typed source a MakeContextSource call closes over.
type contextSource[A any] struct {
	key      ContextKey
	loadFn   func() (A, bool)
	baseline func(A) string
	update   func(prev, cur A) string
	removed  func(prev A) (string, bool)
}

// Key reports the source's namespaced identity.
func (s *contextSource[A]) Key() ContextKey { return s.key }

// snapshot encodes v and captures its optional removal render. An empty removal
// render is refused (matching the upstream requireText guard) so a stored removal
// can never be confused with "no removal renderer".
func (s *contextSource[A]) snapshot(v A) (SourceSnapshot, error) {
	enc, err := json.Marshal(v)
	if err != nil {
		return SourceSnapshot{}, fmt.Errorf("system context source %s encode: %w", s.key, err)
	}
	snap := SourceSnapshot{Value: enc}
	if s.removed != nil {
		if text, ok := s.removed(v); ok {
			if text == "" {
				return SourceSnapshot{}, fmt.Errorf("system context source %s rendered an empty removal", s.key)
			}
			snap.Removed = text
		}
	}
	return snap, nil
}

// load observes the source once and returns its type-erased closures. The user load
// function's ok==false is the temporary-unavailable signal, not removal.
func (s *contextSource[A]) load() packedLoad {
	value, ok := s.loadFn()
	if !ok {
		return packedLoad{unavailable: true}
	}
	return packedLoad{
		baseline: func() (string, SourceSnapshot, error) {
			text := s.baseline(value)
			if text == "" {
				return "", SourceSnapshot{}, fmt.Errorf("system context source %s rendered an empty baseline", s.key)
			}
			snap, err := s.snapshot(value)
			if err != nil {
				return "", SourceSnapshot{}, err
			}
			return text, snap, nil
		},
		compare: func(prev json.RawMessage) (compared, error) {
			var decoded A
			if err := json.Unmarshal(prev, &decoded); err != nil {
				return compared{kind: comparedIncompatible}, nil
			}
			if reflect.DeepEqual(decoded, value) {
				return compared{kind: comparedUnchanged}, nil
			}
			text := s.update(decoded, value)
			if text == "" {
				return compared{}, fmt.Errorf("system context source %s rendered an empty update", s.key)
			}
			snap, err := s.snapshot(value)
			if err != nil {
				return compared{}, err
			}
			return compared{kind: comparedUpdated, text: text, snapshot: snap}, nil
		},
	}
}

// MakeContextSource closes one typed source into a registry-composable
// PackedContextSource. load returns (value, ok); ok==false means the source could not
// be observed and is treated as temporarily unavailable (never as removed). removed is
// optional (nil disables removal rendering); its bool reports whether a removal render
// is provided for the observed value.
func MakeContextSource[A any](
	key ContextKey,
	load func() (A, bool),
	baseline func(A) string,
	update func(prev, cur A) string,
	removed func(prev A) (string, bool),
) (PackedContextSource, error) {
	if !ValidContextKey(key) {
		return nil, fmt.Errorf("invalid system context key %q", string(key))
	}
	if load == nil || baseline == nil || update == nil {
		return nil, fmt.Errorf("system context source %s requires load, baseline, and update renderers", string(key))
	}
	return &contextSource[A]{key: key, loadFn: load, baseline: baseline, update: update, removed: removed}, nil
}

// IncrementalContext is an ordered, duplicate-free composition of packed sources.
type IncrementalContext struct {
	sources []PackedContextSource
}

// EmptyIncrementalContext returns the identity registry (no sources).
func EmptyIncrementalContext() *IncrementalContext {
	return &IncrementalContext{}
}

// MakeIncrementalContext returns a registry holding exactly one source.
func MakeIncrementalContext(source PackedContextSource) *IncrementalContext {
	if source == nil {
		return &IncrementalContext{}
	}
	return &IncrementalContext{sources: []PackedContextSource{source}}
}

// CombineIncrementalContext concatenates registries in order, rejecting any duplicate
// key immediately with a *DuplicateContextKeyError.
func CombineIncrementalContext(values ...*IncrementalContext) (*IncrementalContext, error) {
	var sources []PackedContextSource
	seen := map[ContextKey]bool{}
	for _, v := range values {
		if v == nil {
			continue
		}
		for _, s := range v.sources {
			k := s.Key()
			if seen[k] {
				return nil, &DuplicateContextKeyError{Key: k}
			}
			seen[k] = true
			sources = append(sources, s)
		}
	}
	return &IncrementalContext{sources: sources}, nil
}

// observedEntry is one registry entry observed once.
type observedEntry struct {
	key  ContextKey
	load packedLoad
}

// observe reads every source once, in registry order.
func observe(ctx *IncrementalContext) []observedEntry {
	if ctx == nil {
		return nil
	}
	out := make([]observedEntry, 0, len(ctx.sources))
	for _, s := range ctx.sources {
		out = append(out, observedEntry{key: s.Key(), load: s.load()})
	}
	return out
}

// render joins rendered parts with the canonical blank-line separator.
func render(parts []string) string { return strings.Join(parts, "\n\n") }

// sortedSnapshotKeys returns the snapshot's keys in deterministic sorted order.
func sortedSnapshotKeys(snapshot ContextSnapshot) []ContextKey {
	keys := make([]ContextKey, 0, len(snapshot))
	for k := range snapshot {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
	return keys
}

// initializeObservation builds a fresh generation from already-observed entries,
// skipping unavailable entries (the caller decides whether that is legal).
func initializeObservation(entries []observedEntry) (ContextGeneration, error) {
	snapshot := ContextSnapshot{}
	var parts []string
	for _, e := range entries {
		if e.load.unavailable {
			continue
		}
		text, snap, err := e.load.baseline()
		if err != nil {
			return ContextGeneration{}, err
		}
		parts = append(parts, text)
		snapshot[e.key] = snap
	}
	return ContextGeneration{Baseline: render(parts), Snapshot: snapshot}, nil
}

// InitializeIncrementalContext creates the immutable baseline and durable snapshot for
// a new generation. Any unavailable source refuses the whole generation.
func InitializeIncrementalContext(ctx *IncrementalContext) (ContextGeneration, error) {
	entries := observe(ctx)
	var blocked []ContextKey
	for _, e := range entries {
		if e.load.unavailable {
			blocked = append(blocked, e.key)
		}
	}
	if len(blocked) > 0 {
		return ContextGeneration{}, &InitializationBlockedError{Keys: blocked}
	}
	return initializeObservation(entries)
}

// ReconcileIncrementalContext reconciles current source values against one active
// snapshot. A required-but-blocked replacement falls through to ReplaceIncrementalContext.
func ReconcileIncrementalContext(ctx *IncrementalContext, previous ContextSnapshot) (ReconcileResult, error) {
	entries := observe(ctx)
	res, replace, err := reconcileObservation(entries, previous)
	if err != nil {
		return ReconcileResult{}, err
	}
	if replace {
		return replaceObservation(entries, previous)
	}
	return res, nil
}

// reconcileObservation computes the incremental result. replace==true means the caller
// must fall through to a full replacement.
func reconcileObservation(entries []observedEntry, previous ContextSnapshot) (ReconcileResult, bool, error) {
	keys := make(map[ContextKey]bool, len(entries))
	for _, e := range entries {
		keys[e.key] = true
	}
	comparisons := make(map[ContextKey]compared, len(entries))
	for _, e := range entries {
		if e.load.unavailable {
			continue
		}
		stored, ok := previous[e.key]
		if !ok {
			continue
		}
		c, err := e.load.compare(stored.Value)
		if err != nil {
			return ReconcileResult{}, false, err
		}
		if c.kind == comparedIncompatible {
			return ReconcileResult{}, true, nil
		}
		comparisons[e.key] = c
	}
	// A source removed from the registry with no removal renderer forces a full
	// replacement rather than silently dropping admitted context.
	for _, key := range sortedSnapshotKeys(previous) {
		if keys[key] {
			continue
		}
		if previous[key].Removed == "" {
			return ReconcileResult{}, true, nil
		}
	}

	snapshot := ContextSnapshot{}
	var updates []string
	for _, e := range entries {
		stored, has := previous[e.key]
		if e.load.unavailable {
			if has {
				snapshot[e.key] = stored
			}
			continue
		}
		if !has {
			text, snap, err := e.load.baseline()
			if err != nil {
				return ReconcileResult{}, false, err
			}
			updates = append(updates, text)
			snapshot[e.key] = snap
			continue
		}
		c := comparisons[e.key]
		if c.kind == comparedUnchanged {
			snapshot[e.key] = stored
			continue
		}
		updates = append(updates, c.text)
		snapshot[e.key] = c.snapshot
	}
	for _, key := range sortedSnapshotKeys(previous) {
		if keys[key] {
			continue
		}
		updates = append(updates, previous[key].Removed)
	}
	if len(updates) == 0 {
		return ReconcileResult{Kind: ReconcileUnchanged, Snapshot: previous}, false, nil
	}
	return ReconcileResult{Kind: ReconcileUpdated, Text: render(updates), Snapshot: snapshot}, false, nil
}

// ReplaceIncrementalContext builds a complete replacement generation, or blocks while
// an admitted source is temporarily unavailable.
func ReplaceIncrementalContext(ctx *IncrementalContext, previous ContextSnapshot) (ReconcileResult, error) {
	return replaceObservation(observe(ctx), previous)
}

// replaceObservation returns ReplacementBlocked when an unavailable source has an
// admitted snapshot, else a fresh ReplacementReady generation.
func replaceObservation(entries []observedEntry, previous ContextSnapshot) (ReconcileResult, error) {
	for _, e := range entries {
		if !e.load.unavailable {
			continue
		}
		if _, ok := previous[e.key]; ok {
			return ReconcileResult{Kind: ReconcileReplacementBlocked}, nil
		}
	}
	generation, err := initializeObservation(entries)
	if err != nil {
		return ReconcileResult{}, err
	}
	return ReconcileResult{
		Kind:       ReconcileReplacementReady,
		Snapshot:   generation.Snapshot,
		Generation: generation,
	}, nil
}
