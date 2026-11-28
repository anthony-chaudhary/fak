// Package scorecard is the in-process concept-disambiguation scorecard: the Go
// under the data directory against the tracked tree into the same payload,
// terminal views and generated doc folder the Python tool emits, byte for byte.
package scorecard

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strconv"
)

const (
	Schema          = "fak-concept-disambiguation-scorecard/1"
	DataDirRel      = "tools/concept_disambiguation_scorecard.data"
	GeneratedDocDir = "docs/concept-disambiguation-scorecard"
	CLIRefRel       = "docs/cli-reference.md"
)

var kindNames = []string{"cli-verb", "concept", "config", "doc-term", "metric", "subsystem", "symbol"}

var kinds = map[string]bool{
	"concept": true, "subsystem": true, "symbol": true, "config": true,
	"metric": true, "cli-verb": true, "doc-term": true,
}

var groundingKindNames = []string{"claims", "doc", "metric", "path", "symbol", "verb"}

var groundingKinds = map[string]bool{
	"symbol": true, "path": true, "claims": true, "doc": true, "metric": true, "verb": true,
}

// Verdicts is the clarity ladder, best to worst.
var Verdicts = []string{"crystal", "defined", "drifting", "entangled", "colliding", "undocumented"}

var verdictRank = map[string]int{
	"crystal": 0, "defined": 1, "drifting": 2, "entangled": 3, "colliding": 4, "undocumented": 5,
}

// Groups orders the KPI groups the debt is reported by.
var Groups = []string{"well-formed", "distinctness", "separation", "indexed", "grounded", "honesty"}

var kpiGroup = map[string]string{
	"well_formed":        "well-formed",
	"canonical_unique":   "distinctness",
	"defined":            "distinctness",
	"disambiguated":      "distinctness",
	"reference_resolves": "separation",
	"pair_separated":     "separation",
	"pair_mutual":        "separation",
	"grounded":           "grounded",
	"alias_resolved":     "grounded",
	"anchored":           "grounded",
	"index_resolves":     "indexed",
	"clarity_consistent": "honesty",
}

type kpiWeight struct {
	name   string
	weight float64
}

var kpiWeights = []kpiWeight{
	{"well_formed", 0.08},
	{"canonical_unique", 0.14},
	{"defined", 0.10},
	{"disambiguated", 0.15},
	{"reference_resolves", 0.07},
	{"pair_separated", 0.12},
	{"pair_mutual", 0.05},
	{"grounded", 0.05},
	{"alias_resolved", 0.05},
	{"anchored", 0.05},
	{"index_resolves", 0.07},
	{"clarity_consistent", 0.07},
}

var kpiPenalty = map[string]int{
	"well_formed":        12,
	"canonical_unique":   25,
	"defined":            16,
	"disambiguated":      20,
	"reference_resolves": 14,
	"pair_separated":     20,
	"pair_mutual":        10,
	"grounded":           16,
	"alias_resolved":     12,
	"anchored":           12,
	"index_resolves":     18,
	"clarity_consistent": 18,
}

const (
	clarityWeight  = 0.35
	coverageWeight = 0.65
)

var requiredFields = []string{
	"id", "canonical", "family", "kind", "definition", "distinction",
	"distinct_from", "aliases", "grounding", "grounding_kind",
	"glossary_anchor", "verdict", "gaps",
}

const (
	maxPairEdits   = 2
	minPairLen     = 6
	minSharedAffix = 5
)

var pairKinds = []string{"homonym", "permuted", "near"}

var pairRank = map[string]int{"homonym": 0, "permuted": 1, "near": 2}

var verdictMark = map[string]string{
	"crystal": "*", "defined": "o", "drifting": "~", "entangled": "=",
	"colliding": "x", "undocumented": ".",
}

func markOf(v Value) string {
	if s, ok := v.(string); ok {
		if m, ok := verdictMark[s]; ok {
			return m
		}
	}
	return " "
}

// rankOf is VERDICT_RANK.get(v, missing).
func rankOf(v Value, missing int) int {
	if s, ok := v.(string); ok {
		if r, ok := verdictRank[s]; ok {
			return r
		}
	}
	return missing
}

func get(r *Object, key string) Value {
	v, _ := r.Get(key)
	return v
}

// getOr is dict.get(key, def).
func getOr(r *Object, key string, def Value) Value {
	if v, ok := r.Get(key); ok {
		return v
	}
	return def
}

func getString(r *Object, key string) string {
	s, _ := get(r, key).(string)
	return s
}

// ridIndexed is `r.get("id") if _nonempty(r.get("id")) else f"row[{i}]"`.
func ridIndexed(r *Object, i int) string {
	if id := get(r, "id"); nonempty(id) {
		return id.(string)
	}
	return fmt.Sprintf("row[%d]", i)
}

// ridDefault is `r.get("id", i)` interpolated into a message.
func ridDefault(r *Object, i int) string {
	if v, ok := r.Get("id"); ok {
		return pyStr(v)
	}
	return strconv.Itoa(i)
}

// ridTruthy is `r.get("id") or f"row[{i}]"`.
func ridTruthy(r *Object, i int) string {
	if id := get(r, "id"); truthy(id) {
		return pyStr(id)
	}
	return fmt.Sprintf("row[%d]", i)
}

// idString is the row id when it is a JSON string.
func idString(r *Object) (string, bool) {
	s, ok := get(r, "id").(string)
	return s, ok
}

// orEmptyList is `v or []`.
func orEmptyList(v Value) Value {
	if truthy(v) {
		return v
	}
	return []Value{}
}

// KV is one entry of an insertion-ordered string-to-int mapping.
type KV struct {
	Key string
	N   int
}

// OrderedCounts serializes as a JSON object in slice order.
type OrderedCounts []KV

func (o OrderedCounts) MarshalJSON() ([]byte, error) {
	var b bytes.Buffer
	b.WriteByte('{')
	for i, kv := range o {
		if i > 0 {
			b.WriteByte(',')
		}
		writeJSONString(&b, kv.Key)
		b.WriteByte(':')
		b.WriteString(strconv.Itoa(kv.N))
	}
	b.WriteByte('}')
	return b.Bytes(), nil
}

// Get returns the count under key, 0 when absent.
func (o OrderedCounts) Get(key string) int {
	for _, kv := range o {
		if kv.Key == key {
			return kv.N
		}
	}
	return 0
}

// SourcesKV is one target of an index entry with the surfaces it answers by.
type SourcesKV struct {
	ID      string
	Sources []string
}

// OrderedSources serializes as {id: [source, ...]} in slice order.
type OrderedSources []SourcesKV

func (o OrderedSources) MarshalJSON() ([]byte, error) {
	var b bytes.Buffer
	b.WriteByte('{')
	for i, kv := range o {
		if i > 0 {
			b.WriteByte(',')
		}
		writeJSONString(&b, kv.ID)
		b.WriteByte(':')
		enc, err := json.Marshal(nonNil(kv.Sources))
		if err != nil {
			return nil, err
		}
		b.Write(enc)
	}
	b.WriteByte('}')
	return b.Bytes(), nil
}

// Get returns the sources for id, nil when absent.
func (o OrderedSources) Get(id string) []string {
	for _, kv := range o {
		if kv.ID == id {
			return kv.Sources
		}
	}
	return nil
}

// KeyTargets is one by_key entry.
type KeyTargets struct {
	Key     string
	Targets []string
}

// OrderedKeyTargets serializes as {key: [target, ...]} in slice order.
type OrderedKeyTargets []KeyTargets

func (o OrderedKeyTargets) MarshalJSON() ([]byte, error) {
	var b bytes.Buffer
	b.WriteByte('{')
	for i, kv := range o {
		if i > 0 {
			b.WriteByte(',')
		}
		writeJSONString(&b, kv.Key)
		b.WriteByte(':')
		b.WriteByte('[')
		for j, t := range kv.Targets {
			if j > 0 {
				b.WriteByte(',')
			}
			writeJSONString(&b, t)
		}
		b.WriteByte(']')
	}
	b.WriteByte('}')
	return b.Bytes(), nil
}

func nonNil(s []string) []string {
	if s == nil {
		return []string{}
	}
	return s
}
