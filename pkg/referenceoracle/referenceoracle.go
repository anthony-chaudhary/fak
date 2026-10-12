// Package referenceoracle defines fak.reference_oracle.v1: what a runnable
// reference engine produced on a pinned checkpoint, so a model bring-up can diff
// fak against it token by token and layer by layer.
//
// It is a PUBLIC package (github.com/anthony-chaudhary/fak/pkg/referenceoracle)
// so any bring-up tooling, in this repository or outside it, reads one schema
// instead of a per-model copy.
//
// An oracle records:
//
//   - reference: the engine build (repo, commit, license, binary digest);
//   - artifact: the checkpoint it ran (repo, revision, shard digests);
//   - prompts: per prompt, the exact input token ids, the greedy continuation
//     and the top-5 log-probabilities at each generated position;
//   - self_parity: whether a second identical run reproduced the first;
//   - trace_refs: the per-layer activation trace captured for each prompt
//     (a computetrace artifact converted with a named llama.cpp stage map).
//
// Decode also reads the earlier per-model receipt shape through an adapter:
// any schema "fak.<model>_reference_oracle.v1" (for example
// fak.v41_reference_oracle.v1) whose prompts are an object keyed by prompt text.
// See adaptLegacy for the fields it reads.
package referenceoracle

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"regexp"
)

// Schema is the schema id of a native oracle document.
const Schema = "fak.reference_oracle.v1"

// LegacySchemaV41 is the DeepSeek V4.1 receipt schema the adapter reads.
const LegacySchemaV41 = "fak.v41_reference_oracle.v1"

var legacySchema = regexp.MustCompile(`^fak\.[a-z0-9][a-z0-9_]*_reference_oracle\.v1$`)

// Errors returned by Decode and Validate.
var (
	ErrUnknownSchema = errors.New("referenceoracle: unknown schema")
	ErrInvalid       = errors.New("referenceoracle: invalid oracle")
)

// Oracle is one fak.reference_oracle.v1 document.
type Oracle struct {
	Schema string `json:"schema"`
	// SourceSchema names the schema an adapted document was read from.
	SourceSchema  string     `json:"source_schema,omitempty"`
	EvidenceLabel string     `json:"evidence_label,omitempty"`
	Host          string     `json:"host,omitempty"`
	CapturedUTC   string     `json:"captured_utc,omitempty"`
	Reference     Reference  `json:"reference"`
	Artifact      Artifact   `json:"artifact"`
	Prompts       []Prompt   `json:"prompts"`
	SelfParity    SelfParity `json:"self_parity"`
	TraceRefs     []TraceRef `json:"trace_refs,omitempty"`
}

// Reference is the engine build that produced the oracle.
type Reference struct {
	Repo         string `json:"repo"`
	SHA          string `json:"sha"`
	License      string `json:"license,omitempty"`
	BinarySHA256 string `json:"binary_sha256,omitempty"`
}

// Artifact is the checkpoint the reference ran.
type Artifact struct {
	Repo  string `json:"repo"`
	Rev   string `json:"rev,omitempty"`
	Quant string `json:"quant,omitempty"`
	// ShardCount is how many files the checkpoint has; Shards may list only
	// the ones whose digest was taken.
	ShardCount int     `json:"shard_count,omitempty"`
	Shards     []Shard `json:"shards,omitempty"`
}

// Shard is one checkpoint file and its digest.
type Shard struct {
	Name      string `json:"name"`
	SHA256    string `json:"sha256"`
	SizeBytes int64  `json:"size_bytes,omitempty"`
}

// Prompt is one prompt's exact input and the reference's greedy continuation.
// Top5[i] lists the five most likely ids at generated position i, most likely
// first, as log-softmax of the raw logits.
type Prompt struct {
	Text          string        `json:"text"`
	InputTokenIDs []int         `json:"input_token_ids"`
	GreedyIDs     []int         `json:"greedy_ids"`
	Top5          [][]TokenProb `json:"top5,omitempty"`
}

// TokenProb is one candidate token and its log-probability.
type TokenProb struct {
	ID      int     `json:"id"`
	Piece   string  `json:"piece,omitempty"`
	Logprob float64 `json:"logprob"`
}

// SelfParity records whether a second identical reference run reproduced the
// first. Checked means the second run covered every prompt.
type SelfParity struct {
	Method    string `json:"method,omitempty"`
	Checked   bool   `json:"checked"`
	Identical bool   `json:"identical"`
}

// TraceRef locates the per-layer activation trace captured for one prompt.
// Path is relative to the oracle document's directory.
type TraceRef struct {
	Prompt   int    `json:"prompt"`
	Path     string `json:"path"`
	SHA256   string `json:"sha256,omitempty"`
	StageMap string `json:"stage_map,omitempty"`
	Layers   int    `json:"layers,omitempty"`
	Tokens   int    `json:"tokens,omitempty"`
}

// IsOracleSchema reports whether Decode reads a document with this schema id.
func IsOracleSchema(schema string) bool {
	return schema == Schema || legacySchema.MatchString(schema)
}

// ReadFile decodes the oracle at path.
func ReadFile(path string) (Oracle, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Oracle{}, err
	}
	return Parse(data)
}

// Decode reads and validates one oracle document.
func Decode(r io.Reader) (Oracle, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return Oracle{}, err
	}
	return Parse(data)
}

// Parse decodes and validates one oracle document, adapting a legacy
// per-model receipt to fak.reference_oracle.v1.
func Parse(data []byte) (Oracle, error) {
	var head struct {
		Schema string `json:"schema"`
	}
	if err := json.Unmarshal(data, &head); err != nil {
		return Oracle{}, fmt.Errorf("%w: %v", ErrInvalid, err)
	}
	var (
		o   Oracle
		err error
	)
	switch {
	case head.Schema == Schema:
		dec := json.NewDecoder(bytes.NewReader(data))
		err = dec.Decode(&o)
	case legacySchema.MatchString(head.Schema):
		o, err = adaptLegacy(data)
	default:
		return Oracle{}, fmt.Errorf("%w: %q", ErrUnknownSchema, head.Schema)
	}
	if err != nil {
		return Oracle{}, fmt.Errorf("%w: %v", ErrInvalid, err)
	}
	if err := o.Validate(); err != nil {
		return Oracle{}, err
	}
	return o, nil
}

func invalid(format string, args ...any) error {
	return fmt.Errorf("%w: "+format, append([]any{ErrInvalid}, args...)...)
}

// Validate checks the structural contract every consumer relies on: a named
// reference and artifact, at least one prompt with input and greedy ids, top-5
// rows that are finite, descending and led by the greedy id, and trace refs that
// name a prompt and a path local to the document.
func (o Oracle) Validate() error {
	if o.Schema != Schema {
		return invalid("schema %q, want %q", o.Schema, Schema)
	}
	if o.Reference.Repo == "" || o.Reference.SHA == "" {
		return invalid("reference repo and sha are required")
	}
	if o.Artifact.Repo == "" {
		return invalid("artifact repo is required")
	}
	if len(o.Prompts) == 0 {
		return invalid("no prompts")
	}
	for i, p := range o.Prompts {
		if len(p.InputTokenIDs) == 0 || len(p.GreedyIDs) == 0 {
			return invalid("prompt %d: input_token_ids and greedy_ids are required", i)
		}
		if len(p.Top5) != 0 && len(p.Top5) != len(p.GreedyIDs) {
			return invalid("prompt %d: %d top5 rows for %d greedy ids", i, len(p.Top5), len(p.GreedyIDs))
		}
		for pos, row := range p.Top5 {
			if len(row) == 0 {
				return invalid("prompt %d position %d: empty top5 row", i, pos)
			}
			if row[0].ID != p.GreedyIDs[pos] {
				return invalid("prompt %d position %d: top5 leads with id %d, greedy id is %d", i, pos, row[0].ID, p.GreedyIDs[pos])
			}
			for k, c := range row {
				if math.IsNaN(c.Logprob) || math.IsInf(c.Logprob, 0) || c.Logprob > 1e-6 {
					return invalid("prompt %d position %d: logprob %v is not a finite log-probability", i, pos, c.Logprob)
				}
				if k > 0 && c.Logprob > row[k-1].Logprob+1e-9 {
					return invalid("prompt %d position %d: top5 is not in descending order", i, pos)
				}
			}
		}
	}
	seen := map[int]bool{}
	for i, t := range o.TraceRefs {
		if seen[t.Prompt] {
			return invalid("trace_refs[%d]: duplicate trace ref for prompt %d", i, t.Prompt)
		}
		seen[t.Prompt] = true
		if t.Prompt < 0 || t.Prompt >= len(o.Prompts) {
			return invalid("trace_refs[%d]: prompt %d out of range", i, t.Prompt)
		}
		if t.Path == "" || !filepath.IsLocal(t.Path) {
			return invalid("trace_refs[%d]: path %q is not local to the oracle", i, t.Path)
		}
	}
	return nil
}

// TraceRef returns the activation trace reference of prompt i.
func (o Oracle) TraceRef(prompt int) (TraceRef, bool) {
	for _, t := range o.TraceRefs {
		if t.Prompt == prompt {
			return t, true
		}
	}
	return TraceRef{}, false
}
