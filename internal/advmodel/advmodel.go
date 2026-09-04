package advmodel

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// Invariant: advanced model router maintains deterministic model selection without latency regressions.
// Invariant: The advisory adjudicator may only emit VerdictDeny or VerdictDefer, never VerdictAllow.
// Contract: Fail-closed evaluation guarantees that misconfigured or uninitialized models default to VerdictDefer.
// Precondition: Serialized model artifact must have schema identifier matching ArtifactSchema.

// ArtifactSchema is the JSON schema marker the loader requires. Bumping it is an
// additive version gate (a future trainer writes v2; this loader still reads v1).
const ArtifactSchema = "fak-advmodel/v1"

// Artifact is the trained advisory model — a logistic-regression classifier over
// a bag of call tokens (tool + args), serialized as JSON so the Go loader and the
// Python trainer share one on-disk shape. It is produced by train.py over the
// frozen harvest corpus (testdata/corpus.jsonl).
type Artifact struct {
	// Schema specifies the JSON schema marker required for artifact compatibility.
	Schema string `json:"schema"`
	// Bias specifies the baseline intercept term added to token feature weight sums.
	Bias float64 `json:"bias"`
	// Threshold specifies the logit decision boundary for emitting VerdictDeny; 0 == sigmoid 0.5.
	Threshold float64 `json:"threshold"`
	// Features maps individual lowercased call tokens to their learned classification weights.
	Features map[string]float64 `json:"features"`
	// Meta encapsulates training provenance, dataset partition counts, and evaluation metrics.
	Meta ArtifactMeta `json:"meta"`
}

// ArtifactMeta carries the reproducibility witness: the held-out eval vs the
// stock reference (the untrained artifact, which always defers). Every number is
// produced by train.py and re-checkable by re-running it; nothing here is
// self-asserted at load time.
type ArtifactMeta struct {
	// TrainRows indicates the count of examples utilized during model training.
	TrainRows int `json:"train_rows"`
	// HeldRows indicates the count of examples evaluated in the held-out validation set.
	HeldRows int `json:"held_rows"`
	// Precision records the positive predictive value on held-out validation data.
	Precision float64 `json:"precision"`
	// Recall records the true positive sensitivity on held-out validation data.
	Recall float64 `json:"recall"`
	// F1 records the harmonic mean of precision and recall on the validation set.
	F1 float64 `json:"f1"`
	// StockF1 records the baseline reference F1 score for the untrained model on the held split.
	StockF1 float64 `json:"stock_ref_f1"`
	// MajorityF1 records the baseline F1 score achievable by constant deny prediction.
	MajorityF1 float64 `json:"majority_f1"`
	// TrainF1 records the fitted F1 score measured over the training partition.
	TrainF1 float64 `json:"train_f1"`
	// Trained specifies the UTC timestamp when the model artifact was generated.
	Trained string `json:"trained"`
}

// tokenRe is the shared featurizer: a call is lower-cased (tool + args JSON) and
// split into alphanumeric+underscore runs. This regex is the CONTRACT between the
// Go scorer and train.py — both must extract the identical token set per call, or
// the loaded weights score the wrong features. train.py mirrors it verbatim.
var tokenRe = regexp.MustCompile(`[a-z0-9_]+`)

// Tokens returns the bag-of-words feature set for one call: the unique
// alphanumeric/underscore runs of lower-case(tool + "\x00" + args). Duplicates
// collapse (binary features, not counts) so a repeated token adds no weight —
// keeping the Go/Python parity exact and robust to argument ordering.
func Tokens(tool string, args []byte) []string {
	s := strings.ToLower(tool) + "\x00" + strings.ToLower(string(args))
	seen := make(map[string]struct{})
	var out []string
	for _, m := range tokenRe.FindAllString(s, -1) {
		if _, ok := seen[m]; ok {
			continue
		}
		seen[m] = struct{}{}
		out = append(out, m)
	}
	return out
}

// Score computes the raw logit (pre-sigmoid): bias + sum of the learned weights of the
// call's tokens. Unseen tokens (absent from Features) contribute nothing. A nil
// or featureless artifact scores 0 — the inert baseline.
func (a *Artifact) Score(tool string, args []byte) float64 {
	if a == nil || len(a.Features) == 0 {
		return 0
	}
	z := a.Bias
	for _, tok := range Tokens(tool, args) {
		if w, ok := a.Features[tok]; ok {
			z += w
		}
	}
	return z
}

// Denies reports whether the model corroborates a deny for this call: the logit
// meets the threshold. A nil/featureless (inert) artifact NEVER denies — it
// defers on everything, so a mis-loaded or untrained model is a no-op, never an
// authority-widening or authority-narrowing hole beyond the floor.
func (a *Artifact) Denies(tool string, args []byte) bool {
	if a == nil || len(a.Features) == 0 {
		return false
	}
	return a.Score(tool, args) >= a.Threshold
}

// LoadBytes parses a trained artifact. It rejects an unknown schema rather than
// coercing, so a future v2 artifact can never be silently mis-scored by this v1
// loader.
func LoadBytes(b []byte) (*Artifact, error) {
	var a Artifact
	if err := json.Unmarshal(b, &a); err != nil {
		return nil, fmt.Errorf("advmodel: parse artifact: %w", err)
	}
	if a.Schema != ArtifactSchema {
		return nil, fmt.Errorf("advmodel: unknown artifact schema %q (want %q)", a.Schema, ArtifactSchema)
	}
	return &a, nil
}

// Load reads a trained artifact from a file path.
func Load(path string) (*Artifact, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("advmodel: read artifact %s: %w", path, err)
	}
	return LoadBytes(b)
}

// Adjudicator is the opt-in abi.Adjudicator that folds a trained Artifact into
// the kernel's decision chain. It is FAIL-CLOSED: it returns only VerdictDeny
// (corroborate) or VerdictDefer (no opinion), NEVER VerdictAllow — so under the
// kernel's restrictiveness fold it can only TIGHTEN a decision (add a deny),
// never weaken the deterministic floor. Construct with NewAdjudicator; wire with
// kernel.WithAdjudicators or abi.RegisterAdjudicator (the package never
// self-registers — default-off).
type Adjudicator struct {
	art *Artifact
}

// NewAdjudicator wraps a trained artifact as an abi.Adjudicator. A nil artifact
// yields an inert adjudicator that defers on every call (the default-off no-op).
func NewAdjudicator(a *Artifact) *Adjudicator { return &Adjudicator{art: a} }

// Caps advertises no special capabilities (the baseline advisory link).
func (d *Adjudicator) Caps() []abi.Capability { return nil }

// Adjudicate is the fail-closed decision: Deny (corroborate) when the learned
// logit meets the threshold, otherwise Defer. It never returns Allow, so it can
// never weaken the floor. By:"advmodel" + a Meta score give forensics without
// leaking the weight vector (only the scalar logit).
func (d *Adjudicator) Adjudicate(ctx context.Context, c *abi.ToolCall) abi.Verdict {
	_ = ctx
	if d == nil || d.art == nil || !d.art.Denies(c.Tool, c.Args.Inline) {
		return abi.Verdict{Kind: abi.VerdictDefer, By: "advmodel"}
	}
	z := d.art.Score(c.Tool, c.Args.Inline)
	return abi.Verdict{
		Kind:   abi.VerdictDeny,
		Reason: abi.ReasonPolicyBlock,
		By:     "advmodel",
		Meta:   map[string]string{"logit": fmt.Sprintf("%.4f", z)},
	}
}

// Descriptor summarizes the operational parameters, feature volume, and
// evaluation metrics of an advisory adjudication model.
type Descriptor struct {
	// Schema specifies the model format specification version identifier.
	Schema string `json:"schema"`
	// FeatureCount indicates the total number of learned feature weights.
	FeatureCount int `json:"feature_count"`
	// Bias specifies the classification intercept value.
	Bias float64 `json:"bias"`
	// Threshold specifies the logit threshold for classification.
	Threshold float64 `json:"threshold"`
	// Precision records validation positive predictive value.
	Precision float64 `json:"precision"`
	// Recall records validation sensitivity.
	Recall float64 `json:"recall"`
	// F1 records validation harmonic mean score.
	F1 float64 `json:"f1"`
	// Trained records the generation timestamp.
	Trained string `json:"trained,omitempty"`
}

// Valid reports whether the descriptor represents a valid, non-empty model specification.
func (d Descriptor) Valid() bool {
	return d.Schema == ArtifactSchema && d.FeatureCount > 0
}

// Descriptor extracts the operational descriptor summarizing the model artifact.
// If the receiver is nil or uninitialized, an empty descriptor is returned.
func (a *Artifact) Descriptor() Descriptor {
	if a == nil {
		return Descriptor{}
	}
	return Descriptor{
		Schema:       a.Schema,
		FeatureCount: len(a.Features),
		Bias:         a.Bias,
		Threshold:    a.Threshold,
		Precision:    a.Meta.Precision,
		Recall:       a.Meta.Recall,
		F1:           a.Meta.F1,
		Trained:      a.Meta.Trained,
	}
}

// Descriptor returns the operational descriptor of the underlying artifact,
// or an empty descriptor if the adjudicator has no configured artifact.
func (d *Adjudicator) Descriptor() Descriptor {
	if d == nil || d.art == nil {
		return Descriptor{}
	}
	return d.art.Descriptor()
}

// Model returns the underlying Artifact associated with the adjudicator, or nil
// if none was configured.
func (d *Adjudicator) Model() *Artifact {
	if d == nil {
		return nil
	}
	return d.art
}

// Resolve parses and validates serialized artifact bytes into an Artifact and its
// operational Descriptor.
func Resolve(b []byte) (*Artifact, Descriptor, error) {
	art, err := LoadBytes(b)
	if err != nil {
		return nil, Descriptor{}, err
	}
	return art, art.Descriptor(), nil
}

// ResolveModel parses and validates serialized artifact bytes into an Artifact and its
// operational Descriptor. It is an alias for Resolve.
func ResolveModel(b []byte) (*Artifact, Descriptor, error) {
	return Resolve(b)
}

// ResolvePath reads a model artifact from disk and resolves its operational descriptor.
func ResolvePath(path string) (*Artifact, Descriptor, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, Descriptor{}, fmt.Errorf("advmodel: read artifact %s: %w", path, err)
	}
	return Resolve(b)
}
