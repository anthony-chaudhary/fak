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

// ArtifactSchema is the JSON schema marker the loader requires. Bumping it is an
// additive version gate (a future trainer writes v2; this loader still reads v1).
const ArtifactSchema = "fak-advmodel/v1"

// Artifact is the trained advisory model — a logistic-regression classifier over
// a bag of call tokens (tool + args), serialized as JSON so the Go loader and the
// Python trainer share one on-disk shape. It is produced by train.py over the
// frozen harvest corpus (testdata/corpus.jsonl).
type Artifact struct {
	Schema    string             `json:"schema"`
	Bias      float64            `json:"bias"`
	Threshold float64            `json:"threshold"` // decision boundary on the logit; 0 == sigmoid 0.5
	Features  map[string]float64 `json:"features"`  // token -> learned weight
	Meta      ArtifactMeta       `json:"meta"`
}

// ArtifactMeta carries the reproducibility witness: the held-out eval vs the
// stock reference (the untrained artifact, which always defers). Every number is
// produced by train.py and re-checkable by re-running it; nothing here is
// self-asserted at load time.
type ArtifactMeta struct {
	TrainRows  int     `json:"train_rows"`
	HeldRows   int     `json:"held_rows"`
	Precision  float64 `json:"precision"`
	Recall     float64 `json:"recall"`
	F1         float64 `json:"f1"`
	StockF1    float64 `json:"stock_ref_f1"` // stock reference (untrained) F1 on the same held split
	MajorityF1 float64 `json:"majority_f1"`  // majority-class (deny-all) baseline F1, for context
	TrainF1    float64 `json:"train_f1"`     // train-split fit (sanity, not a generalization claim)
	Trained    string  `json:"trained"`      // UTC stamp train.py wrote the artifact
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

// Score is the raw logit (pre-sigmoid): bias + sum of the learned weights of the
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
