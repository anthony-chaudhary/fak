package quality

import (
	"encoding/json"
	"fmt"
	"strings"
)

// batch_invariance.go is the batch-invariance child of the quality spine (#4532):
// a request's output must be invariant to batch composition, order, and position.
// The reference decodes the target request ALONE; the engine decodes the same
// target embedded in a sweep of batches (first, middle, last, with neighbors).
// Every batched decode of the target must match the alone decode token by token —
// a token that depends on a neighbor request or on the target's batch position is
// a cross-request state leak, and the oracle localizes it to the offending batch
// position and token index.
//
// The sweep needs richer per-case data than Tokens/Logits carry, so it rides the
// existing Trace seam additively: the case's Prompt holds the batch plan as JSON,
// and the engine trace's Text holds the per-placement results as JSON, parsed
// inside Judge. Judge's signature is untouched.

// batchVocab is the small fixed vocabulary the deterministic decode draws from.
// Eight entries keep the token space tiny while making an accidental collision
// between two different requests' streams vanishingly unlikely over a few steps.
var batchVocab = []string{"anchor", "breeze", "copper", "dune", "ellipse", "fjord", "glacier", "hollow"}

// batchDraw maps (request id, step) to one deterministic draw: an FNV-1a fold of
// the request id seeds a splitmix64-style mix with the step counter. Because the
// draw is a pure function of (request, step) — no carried state, no ambient
// entropy — a FAITHFUL engine's token i for a request depends only on that
// request, which is exactly the batch-invariance contract the oracle enforces.
func batchDraw(req string, step int) uint64 {
	h := uint64(0xcbf29ce484222325)
	for i := 0; i < len(req); i++ {
		h ^= uint64(req[i])
		h *= 0x100000001b3
	}
	z := h + (uint64(step)+1)*0x9e3779b97f4a7c15
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

// batchDecode is the shared per-request decode path: steps draws for req, each
// mapped onto the fixed vocab. It is what a faithful engine computes for the
// target regardless of which batch the target rides in.
func batchDecode(req string, steps int) []string {
	toks := make([]string, 0, steps)
	for i := 0; i < steps; i++ {
		toks = append(toks, batchVocab[batchDraw(req, i)%uint64(len(batchVocab))])
	}
	return toks
}

// batchRotate returns the vocab entry k places after tok, wrapping. For any k not
// a multiple of the vocab size, rotation can never map a token to itself, so an
// injected leak is GUARANTEED to diverge at exactly its step — the mutant cannot
// accidentally reproduce the alone decode.
func batchRotate(tok string, k int) string {
	for i, v := range batchVocab {
		if v == tok {
			return batchVocab[(i+k)%len(batchVocab)]
		}
	}
	return batchVocab[k%len(batchVocab)]
}

// batchLeakStep is the step the injected leaks corrupt: mid-sequence, so the
// passing prefix proves the localization is doing work (the failure pins to a
// token index within a named batch position, not to "the whole sweep looked off").
const batchLeakStep = 2

// batchLeakNeighbor is the neighbor whose presence BEFORE the target triggers the
// injected cross-request state leak, and batchNeighborShift is the rotation that
// leak applies. The shift differs from every position shift the sweep can produce
// at the same batch position, so the two defect classes stay distinguishable.
const (
	batchLeakNeighbor  = "req-south"
	batchNeighborShift = 3
)

// batchPlan is the batch sweep a case declares, carried as JSON in the case's
// Prompt: the target request id and the batches to embed it in. Each batch is a
// slice of request ids; the target's index within a batch is its position.
type batchPlan struct {
	Target  string     `json:"target"`
	Batches [][]string `json:"batches"`
}

// batchPlacement is one entry of the engine's sweep result, carried as JSON in
// the engine trace's Text: where the target sat, alongside whom, and the token
// stream the engine produced for the target THERE.
type batchPlacement struct {
	Position int      `json:"position"`
	Batch    []string `json:"batch"`
	Tokens   []string `json:"tokens"`
}

// batchParsePlan decodes a case Prompt into its batch plan.
func batchParsePlan(prompt string) (batchPlan, error) {
	var p batchPlan
	if err := json.Unmarshal([]byte(prompt), &p); err != nil {
		return batchPlan{}, fmt.Errorf("batch plan: %w", err)
	}
	if p.Target == "" || len(p.Batches) == 0 {
		return batchPlan{}, fmt.Errorf("batch plan: empty target or no batches")
	}
	return p, nil
}

// batchParsePlacements decodes an engine trace Text into its sweep results.
func batchParsePlacements(text string) ([]batchPlacement, error) {
	var ps []batchPlacement
	if err := json.Unmarshal([]byte(text), &ps); err != nil {
		return nil, fmt.Errorf("batch sweep: %w", err)
	}
	return ps, nil
}

// BatchInvarianceRunner decodes the case's target request once per batch in the
// case's plan and reports the whole sweep. The zero value is a faithful engine
// (per-request decode is a pure function of the request, so every placement
// reproduces the alone decode); the defect field (set via BatchInvarianceEngine)
// injects a cross-request leak. Trace.Tokens carries the first placement's
// stream; Trace.Text carries the full sweep as JSON for the oracle.
type BatchInvarianceRunner struct {
	Label  string
	defect string
}

func (r BatchInvarianceRunner) Name() string {
	if r.Label != "" {
		return r.Label
	}
	return "batched-engine"
}

func (r BatchInvarianceRunner) Run(c QualityCase) (Trace, error) {
	plan, err := batchParsePlan(c.Prompt)
	if err != nil {
		return Trace{}, err
	}
	placements := make([]batchPlacement, 0, len(plan.Batches))
	for _, batch := range plan.Batches {
		pos := -1
		for i, id := range batch {
			if id == plan.Target {
				pos = i
				break
			}
		}
		if pos < 0 {
			return Trace{}, fmt.Errorf("batch %v does not contain target %q", batch, plan.Target)
		}
		toks := batchDecode(plan.Target, c.Params.MaxTokens)
		switch r.defect {
		case "position-leak":
			// The target's decode depends on WHERE it sits: any non-leading
			// position perturbs one step by the position itself.
			if pos > 0 && batchLeakStep < len(toks) {
				toks[batchLeakStep] = batchRotate(toks[batchLeakStep], pos)
			}
		case "neighbor-leak":
			// Cross-request state leak: a specific neighbor decoded BEFORE the
			// target contaminates the target's step (a KV/cache bleed stand-in).
			for i := 0; i < pos; i++ {
				if batch[i] == batchLeakNeighbor && batchLeakStep < len(toks) {
					toks[batchLeakStep] = batchRotate(toks[batchLeakStep], batchNeighborShift)
					break
				}
			}
		}
		placements = append(placements, batchPlacement{Position: pos, Batch: batch, Tokens: toks})
	}
	sweep, err := json.Marshal(placements)
	if err != nil {
		return Trace{}, fmt.Errorf("marshal batch sweep: %w", err)
	}
	return Trace{Runner: r.Name(), Tokens: placements[0].Tokens, Text: string(sweep)}, nil
}

// BatchInvarianceEngine returns a batched engine runner with an optional injected
// defect: "" decodes every placement faithfully (batch composition cannot reach
// the target's stream); "position-leak" makes the target's token at step 2 depend
// on its batch position; "neighbor-leak" makes it depend on a specific neighbor
// being scheduled before the target. These are the deterministic mutant sources
// the tests use to prove the invariance gate trips.
func BatchInvarianceEngine(defect string) BatchInvarianceRunner {
	switch defect {
	case "position-leak":
		return BatchInvarianceRunner{Label: "engine-position-leak", defect: defect}
	case "neighbor-leak":
		return BatchInvarianceRunner{Label: "engine-neighbor-leak", defect: defect}
	default:
		return BatchInvarianceRunner{Label: "engine-batched-clean"}
	}
}

// BatchInvarianceCase builds the demo sweep: the target decoded alone, then
// leading, middle, and trailing a batch of three. The reference is the target's
// ALONE decode; the case asserts every batched placement reproduces it exactly.
func BatchInvarianceCase() QualityCase {
	plan := batchPlan{
		Target: "req-target",
		Batches: [][]string{
			{"req-target"},
			{"req-target", "req-north", "req-south"},
			{"req-north", "req-target", "req-south"},
			{"req-north", "req-south", "req-target"},
		},
	}
	prompt, err := json.Marshal(plan)
	if err != nil {
		panic("quality: marshal batch-invariance plan: " + err.Error())
	}
	const steps = 6
	alone := batchDecode(plan.Target, steps)
	return QualityCase{
		Schema:  CaseSchema,
		ID:      "batch-invariance-demo",
		Version: 1,
		Prompt:  string(prompt),
		Params:  SamplingParams{Temperature: 0, MaxTokens: steps},
		Reference: Trace{
			Tokens: alone,
			Text:   strings.Join(alone, " "),
		},
		Oracles: []string{"batch-invariance"},
	}
}

// BatchInvariance is the differential oracle for batch invariance (#4532): every
// placement in the engine's sweep must reproduce the reference (alone) token
// stream exactly. Any mismatch is a cross-request state leak, reported as the
// FIRST divergence with the offending BATCH POSITION named in Detail — so "the
// batched output looked off" localizes to "at batch position 1 of 3, token 2 was
// 'fjord' where the alone decode emitted 'dune'". The oracle fails closed: a
// trace carrying no sweep, or a sweep that never actually batches the target,
// cannot prove invariance and does not pass.
type BatchInvariance struct{}

func (BatchInvariance) Name() string { return "batch-invariance" }
func (BatchInvariance) Kind() string { return "differential" }

func init() { Register(BatchInvariance{}) }

func (BatchInvariance) Judge(ref, eng Trace, c QualityCase) Verdict {
	v := Verdict{Oracle: "batch-invariance", Kind: "differential", Pass: true}
	placements, err := batchParsePlacements(eng.Text)
	if err != nil || len(placements) == 0 {
		v.Pass = false
		v.Detail = "engine trace carries no batch sweep (Text must be a JSON array of placements); invariance cannot be verified"
		return v
	}
	target := ""
	if plan, perr := batchParsePlan(c.Prompt); perr == nil {
		target = plan.Target
	}
	maxBatch := 0
	for _, p := range placements {
		if p.Position < 0 || p.Position >= len(p.Batch) {
			v.Pass = false
			v.Detail = fmt.Sprintf("malformed placement: position %d outside batch of %d", p.Position, len(p.Batch))
			return v
		}
		if target != "" && p.Batch[p.Position] != target {
			v.Pass = false
			v.Detail = fmt.Sprintf("malformed placement: batch position %d holds %q, not the target %q",
				p.Position, p.Batch[p.Position], target)
			return v
		}
		if len(p.Batch) > maxBatch {
			maxBatch = len(p.Batch)
		}
		n := len(ref.Tokens)
		if len(p.Tokens) < n {
			n = len(p.Tokens)
		}
		for i := 0; i < n; i++ {
			if ref.Tokens[i] != p.Tokens[i] {
				v.Pass = false
				v.FirstDivergence = &Divergence{Index: i, Reference: ref.Tokens[i], Engine: p.Tokens[i]}
				v.Detail = fmt.Sprintf("target output depends on batch composition: at batch position %d (batch of %d: %s) token %d diverged: alone %q, batched %q",
					p.Position, len(p.Batch), strings.Join(p.Batch, ","), i, ref.Tokens[i], p.Tokens[i])
				return v
			}
		}
		if len(ref.Tokens) != len(p.Tokens) {
			v.Pass = false
			v.FirstDivergence = &Divergence{Index: n, Reference: tokenAt(ref.Tokens, n), Engine: tokenAt(p.Tokens, n)}
			v.Detail = fmt.Sprintf("target output depends on batch composition: at batch position %d (batch of %d: %s) length diverged at %d: alone has %d tokens, batched has %d",
				p.Position, len(p.Batch), strings.Join(p.Batch, ","), n, len(ref.Tokens), len(p.Tokens))
			return v
		}
	}
	if len(placements) < 2 || maxBatch < 2 {
		v.Pass = false
		v.Detail = fmt.Sprintf("batch sweep too thin to witness invariance: %d placement(s), max batch size %d (need >=2 placements and a real batch)",
			len(placements), maxBatch)
		return v
	}
	v.Detail = fmt.Sprintf("target decode invariant across %d batch placements (max batch size %d): every batched stream matched the alone reference",
		len(placements), maxBatch)
	return v
}
