// MessageType is the OUTPUT-to-peer half of the grammar rung: the symmetric twin
// of the INPUT-to-tool Grammar. Where Grammar names and validates the argument
// shape a model HANDS a tool, MessageType names and validates the structured
// payload an agent HANDS a peer — a fold result, a gather item, a witness claim —
// the typed thing that today is an ad-hoc map[string]string Meta blob.
//
// MPI analogue: MPI_Type_create_struct / MPI_Pack — a named, registered
// description of a structured payload. HONESTY CAVEAT (this is a Go SCHEMA
// registry, not MPI): MessageType reuses Grammar's content-addressing and the
// existing structural repair-rung shape only. It is NOT MPI's binary type map,
// NOT zero-copy packing into a contiguous buffer, and makes NO wire-format claim.
// Validation is structural (every required field present), deterministic, and
// replay-checkable — the same discipline as the grammar repair rung — not a
// performance/packing semantics claim.
//
// Determinism: a type is content-addressed by Grammar.digest() over its fields,
// so registering the same field set twice dedupes to one canonical entry (the
// byDigest discipline of unit 57). A payload either structurally matches a
// registered type (every required field present) or it doesn't; the verdict is a
// pure function of (type, payload), reusing the unexported countMissing.
package grammar

import (
	"encoding/json"
	"sort"
	"sync"
)

// MessageType is a named, content-addressed description of a typed inter-agent
// payload. Fields reuses Grammar's Param so the same digest()/countMissing logic
// validates a peer message the way it validates a tool call. Name is the
// human/registry handle; Digest is the content address (derived, not trusted).
type MessageType struct {
	Name   string  `json:"name"`
	Fields []Param `json:"fields"`
	Digest string  `json:"digest"`
}

// grammar projects a MessageType's fields onto a Grammar so the registry can
// reuse the unexported digest() content-addressing and countMissing validation
// — the same rungs the tool-input side uses, pointed at the peer-output side.
func (t MessageType) grammar() Grammar {
	return Grammar{Params: t.Fields}
}

// TypeRegistry is the content-addressed registry of MessageTypes — the
// MPI_Type_create_struct analogue. It mirrors Rung's byTool/byDigest split: a
// name index (byName: type name -> digest) over a deduped canonical store
// (byDigest: digest -> MessageType), so registering identical field sets under
// different names collapses to one canonical entry.
type TypeRegistry struct {
	mu       sync.RWMutex
	byName   map[string]string      // type name -> field-set digest
	byDigest map[string]MessageType // digest -> canonical type (deduped)
}

// NewTypeRegistry builds an empty registry with its name->digest and
// digest->type indexes.
func NewTypeRegistry() *TypeRegistry {
	return &TypeRegistry{byName: map[string]string{}, byDigest: map[string]MessageType{}}
}

// Register installs a typed payload description under name, content-addressing it
// by Grammar.digest() over its fields and deduping identical field sets to one
// canonical entry. The returned MessageType carries the derived Digest. Fields
// are sorted by name first so a field set is order-invariant: the same fields in
// a different declaration order content-address to the SAME type (a structural,
// not positional, identity).
func (tr *TypeRegistry) Register(name string, fields []Param) MessageType {
	sorted := make([]Param, len(fields))
	copy(sorted, fields)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].Name < sorted[j].Name })

	d := Grammar{Params: sorted}.digest()
	mt := MessageType{Name: name, Fields: sorted, Digest: d}

	tr.mu.Lock()
	defer tr.mu.Unlock()
	if existing, ok := tr.byDigest[d]; ok {
		// Identical field set already canonical: keep the canonical fields,
		// just bind this name to the shared digest (the byDigest dedup of unit 57).
		mt.Fields = existing.Fields
	} else {
		tr.byDigest[d] = mt
	}
	tr.byName[name] = d
	return mt
}

// LookupByName returns the MessageType registered under name, or false.
func (tr *TypeRegistry) LookupByName(name string) (MessageType, bool) {
	tr.mu.RLock()
	defer tr.mu.RUnlock()
	d, ok := tr.byName[name]
	if !ok {
		return MessageType{}, false
	}
	mt, ok := tr.byDigest[d]
	return mt, ok
}

// LookupByDigest returns the canonical MessageType for a content address, or
// false. This is the content-addressing rung: two names that registered the same
// field set resolve to one type here.
func (tr *TypeRegistry) LookupByDigest(digest string) (MessageType, bool) {
	tr.mu.RLock()
	defer tr.mu.RUnlock()
	mt, ok := tr.byDigest[digest]
	return mt, ok
}

// UniqueTypeCount is the number of distinct (deduped) registered types — the
// byDigest cardinality, the type-side twin of Rung.UniqueGrammarCount.
func (tr *TypeRegistry) UniqueTypeCount() int {
	tr.mu.RLock()
	defer tr.mu.RUnlock()
	return len(tr.byDigest)
}

// Validate structurally checks a peer payload against a registered type, by
// NAME. It reuses the unexported countMissing: a payload is valid iff every
// required field is present (a present "named:X" or bare "X" key satisfies field
// X — the same satisfaction rule the tool-input rung uses). Returns the missing
// required-field names (nil/empty => valid) and whether the type was registered.
// An unregistered type FAILS CLOSED on the peer-output side: a typed message
// whose declared type is unknown cannot be vouched for (unlike the tool-input
// rung, which fails OPEN so it never over-refuses a call it has no grammar for).
func (tr *TypeRegistry) Validate(typeName string, payload map[string]any) (missing []string, known bool) {
	mt, ok := tr.LookupByName(typeName)
	if !ok {
		return nil, false
	}
	for _, p := range mt.grammar().Params {
		if !p.Required {
			continue
		}
		if _, ok := payload["named:"+p.Name]; ok {
			continue
		}
		if _, ok := payload[p.Name]; ok {
			continue
		}
		missing = append(missing, p.Name)
	}
	return missing, true
}

// Pack serializes a typed payload to a deterministic, self-describing envelope:
// the type name, its content address, and the field values — the MPI_Pack
// analogue (a NAMED description travelling WITH the data, NOT a contiguous binary
// buffer, NOT a wire-format guarantee). The payload is validated structurally
// first; an invalid or unknown-type payload is refused so a peer never receives a
// message that doesn't match its declared type. The JSON is canonical (map keys
// sorted by encoding/json) so the same (type, payload) packs byte-identically.
func (tr *TypeRegistry) Pack(typeName string, payload map[string]any) ([]byte, bool) {
	missing, known := tr.Validate(typeName, payload)
	if !known || len(missing) > 0 {
		return nil, false
	}
	mt, _ := tr.LookupByName(typeName)
	env := struct {
		Type    string         `json:"type"`
		Digest  string         `json:"digest"`
		Payload map[string]any `json:"payload"`
	}{Type: mt.Name, Digest: mt.Digest, Payload: payload}
	b, err := json.Marshal(env)
	if err != nil {
		return nil, false
	}
	return b, true
}

// Unpack parses a Pack envelope and structurally re-validates the payload against
// the registry, closing the roundtrip: Pack -> bytes -> Unpack yields the same
// payload only when the declared type is still registered and the payload still
// matches it. Returns the type name, the payload, and whether the message is a
// valid, registry-known typed payload.
func (tr *TypeRegistry) Unpack(b []byte) (typeName string, payload map[string]any, ok bool) {
	var env struct {
		Type    string         `json:"type"`
		Digest  string         `json:"digest"`
		Payload map[string]any `json:"payload"`
	}
	if err := json.Unmarshal(b, &env); err != nil {
		return "", nil, false
	}
	// The envelope's declared digest must match the registered type's content
	// address: a forged/stale digest does not get to claim a type it isn't.
	mt, known := tr.LookupByName(env.Type)
	if !known || env.Digest != mt.Digest {
		return "", nil, false
	}
	missing, _ := tr.Validate(env.Type, env.Payload)
	if len(missing) > 0 {
		return "", nil, false
	}
	return env.Type, env.Payload, true
}

// DefaultTypes is the registered inter-agent message-type registry — the
// peer-output twin of Default (the tool-input grammar rung).
var DefaultTypes = NewTypeRegistry()
