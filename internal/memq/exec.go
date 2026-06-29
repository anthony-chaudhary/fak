package memq

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
)

// ErrSealed is returned by a Backend.Materialize that refuses a page-in because the
// cell is quarantined by the trust gate. memq core never imports a mechanism package;
// a backend translates its own seal error (e.g. recall.ErrSealed) into this so the
// executor can label the refusal without the coupling.
var ErrSealed = errors.New("memq: cell sealed by the trust gate")

// ErrStale is returned by a Backend.Materialize that refuses a page-in because
// recall re-verified a concrete artifact claim at read time and found it stale.
var ErrStale = errors.New("memq: recall artifact stale")

// Backend supplies cells and trust-gated byte access. The two required methods are
// reads; the optional Tombstoner/Pruner add the durable mutations a backend chooses
// to support. A backend that implements neither is read/propose-only — the safe floor.
type Backend interface {
	// Cells returns the page table as cells, carrying only SAFE metadata (never the
	// bytes of a sealed span). It is a snapshot; the executor does not mutate it.
	Cells(ctx context.Context) ([]Cell, error)
	// Materialize pages one cell's bytes in THROUGH THE TRUST GATE. A sealed/refused
	// cell returns an error wrapping ErrSealed; the bytes never cross the gate.
	Materialize(ctx context.Context, id string) ([]byte, error)
}

// Tombstoner is the optional negative-only suppression capability (recall's
// RequestContextChange + persist). Tombstone reports whether the suppression was
// actually applied (false if the backend declined, e.g. the cell was already gone).
type Tombstoner interface {
	Tombstone(ctx context.Context, id, reason, by string) (bool, error)
}

// Pruner is the optional unreferenced-storage GC capability. Prune with apply=false
// is a dry-run count; apply=true reclaims. It never touches model-visible pages.
type Pruner interface {
	Prune(ctx context.Context, apply bool) (blobs int, bytes int64, err error)
}

// Caps is the capability grant that lets Run APPLY a durable mutation. Without a grant
// for an effect kind, that effect is recorded as a PROPOSAL and the backend is not
// touched — the fail-closed default. Apply is the master switch; Allow names the
// specific mutation kinds permitted.
type Caps struct {
	Apply bool            `json:"apply"`
	Allow map[string]bool `json:"allow,omitempty"`
}

func (c Caps) may(kind string) bool { return c.Apply && c.Allow[kind] }

// AllowAll is a convenience Caps granting every mutation — for an operator-driven
// `--apply` at the CLI, never the MCP default.
func AllowAll() Caps {
	return Caps{Apply: true, Allow: map[string]bool{OpTombstone: true, OpPrune: true}}
}

// StepTrace records one executed op: the working-set size in and out, plus a note.
type StepTrace struct {
	Index int    `json:"index"`
	Kind  string `json:"kind"`
	In    int    `json:"in"`
	Out   int    `json:"out"`
	Note  string `json:"note,omitempty"`
}

// RenderItem is one cell materialized into context by an OpRender.
type RenderItem struct {
	ID         string `json:"id"`
	Step       int    `json:"step"`
	Role       string `json:"role,omitempty"`
	Descriptor string `json:"descriptor"`
	Bytes      int64  `json:"bytes"`
	Tokens     int    `json:"tokens"`
}

// Refusal is a cell an effect declined to touch (sealed/tombstoned/page-in refused).
type Refusal struct {
	ID     string `json:"id"`
	Step   int    `json:"step"`
	Role   string `json:"role,omitempty"`
	Reason string `json:"reason"`
}

// Effect is one mutation/derivation the pipeline performed or proposed. Applied is
// true only when a Caps grant let Run mutate durable backend state. Derived carries
// a consolidation's artifact (its bytes are in DerivedBytes, json-excluded).
type Effect struct {
	Kind         string   `json:"kind"`
	Applied      bool     `json:"applied"`
	Reason       string   `json:"reason,omitempty"`
	Cells        []string `json:"cells,omitempty"`
	Derived      *Cell    `json:"derived,omitempty"`
	DerivedBytes []byte   `json:"-"`
	Note         string   `json:"note,omitempty"`
}

// Stats is the run accounting.
type Stats struct {
	CellsScanned    int   `json:"cells_scanned"`
	CellsSelected   int   `json:"cells_selected"` // final working-set size
	Rendered        int   `json:"rendered"`
	RenderedBytes   int64 `json:"rendered_bytes"`
	EstimatedTokens int   `json:"estimated_tokens"`
	Refused         int   `json:"refused"`
	EffectsProposed int   `json:"effects_proposed"`
	EffectsApplied  int   `json:"effects_applied"`
}

// Result is the outcome of Run.
type Result struct {
	Intent   string       `json:"intent,omitempty"`
	Steps    []StepTrace  `json:"steps"`
	Rendered []RenderItem `json:"rendered,omitempty"`
	Effects  []Effect     `json:"effects,omitempty"`
	Refused  []Refusal    `json:"refused,omitempty"`
	Working  []Cell       `json:"working,omitempty"` // final working set (safe metadata only)
	Stats    Stats        `json:"stats"`
}

// Run executes an authored query against a backend. It validates first (an invalid
// query never runs), loads the cells once, computes the global refcount + relevance
// scores, then threads a working set through the pipeline. Mutations are applied only
// where caps permit; otherwise they are proposed. The whole pass is deterministic:
// the same (backend cells, query, caps) yields a byte-identical Result.
func Run(ctx context.Context, b Backend, q Query, caps Caps) (Result, error) {
	if err := Validate(q); err != nil {
		return Result{}, err
	}
	all, err := b.Cells(ctx)
	if err != nil {
		return Result{}, err
	}
	for i := range all {
		all[i].Durability = NormDurability(all[i].Durability)
	}
	refcount := computeRefcount(all)
	score := make(map[string]int, len(all))
	qterms := tokenize(q.Intent)
	for _, c := range all {
		score[c.ID] = overlap(qterms, tokenize(c.Role+" "+c.Descriptor))
	}

	res := Result{Intent: q.Intent}
	res.Stats.CellsScanned = len(all)
	work := append([]Cell(nil), all...)

	for i, op := range q.Ops {
		in := len(work)
		note := ""
		switch op.Kind {
		case OpScan:
			work = append([]Cell(nil), all...)
		case OpFilter:
			kept := work[:0:0]
			for _, c := range work {
				if op.Pred.eval(c, refcount[c.ID]) {
					kept = append(kept, c)
				}
			}
			work = kept
		case OpRank:
			sortByRank(work, op.By, op.Desc, score)
		case OpLimit:
			if op.K < len(work) {
				work = work[:op.K]
			}
		case OpBudget:
			work, note = applyBudget(work, op.Bytes)
		case OpRender:
			note = renderInto(ctx, b, &res, work)
		case OpTombstone:
			note = applyTombstone(ctx, b, &res, work, op, caps)
		case OpConsolidate:
			note = applyConsolidate(ctx, b, &res, work)
		case OpReclassify:
			note = applyReclassify(&res, work, op.By)
		case OpPrune:
			note = applyPrune(ctx, b, &res, caps)
		}
		res.Steps = append(res.Steps, StepTrace{Index: i, Kind: op.Kind, In: in, Out: len(work), Note: note})
	}

	res.Working = work
	finalizeStats(&res)
	return res, nil
}

// computeRefcount returns, per cell ID, how many OTHER cells reference its content —
// either by naming its digest in their Refs, or by being a duplicate (alias) of the
// same digest. refcount==0 means unique, unreferenced content: the genuine GC /
// compaction signal on both an in-memory store and a recall core image (where
// duplicate-digest aliases are exactly what recall.Dream surfaces).
func computeRefcount(cells []Cell) map[string]int {
	digestCount := map[string]int{}
	for _, c := range cells {
		if c.Digest != "" {
			digestCount[c.Digest]++
		}
	}
	refsTo := map[string]int{}
	for _, c := range cells {
		seen := map[string]bool{}
		for _, r := range c.Refs {
			if !seen[r] {
				seen[r] = true
				refsTo[r]++
			}
		}
	}
	out := make(map[string]int, len(cells))
	for _, c := range cells {
		alias := 0
		if c.Digest != "" {
			alias = digestCount[c.Digest] - 1 // other cells sharing this content
		}
		out[c.ID] = alias + refsTo[c.Digest] + refsTo[c.ID]
	}
	return out
}

// applyBudget keeps the prefix whose cumulative size stays within cap (0 = unbounded).
func applyBudget(work []Cell, cap int64) ([]Cell, string) {
	if cap <= 0 {
		return work, ""
	}
	used := int64(0)
	kept := work[:0:0]
	dropped := 0
	for _, c := range work {
		if used+c.Bytes > cap {
			dropped++
			continue
		}
		used += c.Bytes
		kept = append(kept, c)
	}
	if dropped == 0 {
		return kept, ""
	}
	return kept, fmt.Sprintf("dropped %d cell(s) over the %d-byte budget", dropped, cap)
}

// pageIn routes one cell through the backend's trust gate for rendering/folding. It
// returns the materialized body when the cell is admissible; otherwise it records the
// refusal on res (sealed, tombstoned, or a page-in error) and returns ok=false. This is
// the single poison-never-enters-context page-in gate shared by renderInto and
// applyConsolidate, so the refusal vocabulary stays defined in exactly one place.
func pageIn(ctx context.Context, b Backend, res *Result, c Cell) (body []byte, ok bool) {
	if c.Sealed {
		res.Refused = append(res.Refused, Refusal{ID: c.ID, Step: c.Step, Role: c.Role, Reason: "sealed_by_trust_gate"})
		return nil, false
	}
	if c.Tombstoned {
		res.Refused = append(res.Refused, Refusal{ID: c.ID, Step: c.Step, Role: c.Role, Reason: "tombstoned_by_context_control"})
		return nil, false
	}
	body, err := b.Materialize(ctx, c.ID)
	if err != nil {
		reason := "page_in_refused"
		if errors.Is(err, ErrSealed) {
			reason = "sealed_by_trust_gate"
		} else if errors.Is(err, ErrStale) {
			reason = "stale_recall_artifact"
		}
		res.Refused = append(res.Refused, Refusal{ID: c.ID, Step: c.Step, Role: c.Role, Reason: reason})
		return nil, false
	}
	return body, true
}

// renderInto materializes the working set into context, routing every page-in through
// the backend's trust gate. A sealed or tombstoned cell is REFUSED, not rendered — the
// poison-never-enters-context invariant. Render does not change the working set.
func renderInto(ctx context.Context, b Backend, res *Result, work []Cell) string {
	for _, c := range work {
		body, ok := pageIn(ctx, b, res, c)
		if !ok {
			continue
		}
		res.Rendered = append(res.Rendered, RenderItem{
			ID: c.ID, Step: c.Step, Role: c.Role, Descriptor: c.Descriptor,
			Bytes: int64(len(body)), Tokens: tokenEstimate(len(body)),
		})
	}
	return ""
}

// applyTombstone proposes (or, with caps, applies) a negative-only suppression for
// every non-sealed, non-already-tombstoned cell in the set. A sealed cell is left to
// the trust gate; tombstoning it would add nothing (it is already refused on page-in).
func applyTombstone(ctx context.Context, b Backend, res *Result, work []Cell, op Op, caps Caps) string {
	reason := op.Reason
	if reason == "" {
		reason = "memq tombstone"
	}
	ts, canApply := b.(Tombstoner)
	apply := caps.may(OpTombstone) && canApply
	var ids []string
	appliedAny := false
	for _, c := range work {
		if c.Tombstoned || c.Sealed {
			continue
		}
		ids = append(ids, c.ID)
		if apply {
			if ok, err := ts.Tombstone(ctx, c.ID, reason, "memq"); err == nil && ok {
				appliedAny = true
			}
		}
	}
	note := ""
	if !canApply {
		note = "backend does not support tombstone; proposal only"
	} else if !apply {
		note = "no caps granted; proposal only (run with --apply to suppress)"
	}
	res.Effects = append(res.Effects, Effect{
		Kind: OpTombstone, Applied: appliedAny, Reason: reason, Cells: ids, Note: note,
	})
	return fmt.Sprintf("%d cell(s) tombstoned=%v", len(ids), appliedAny)
}

// applyConsolidate folds the set into ONE derived disposition: a deterministic
// extractive summary built by paging each non-sealed cell in through the gate and
// concatenating a bounded head of each. The artifact is REAL (an agent can render it
// into context); its durable write-back to a recall image is the rung-2 follow-on, so
// Applied stays false. Sealed/tombstoned cells are refused, never folded in.
func applyConsolidate(ctx context.Context, b Backend, res *Result, work []Cell) string {
	var parts [][]byte
	var srcIDs []string
	folded := 0
	for _, c := range work {
		body, ok := pageIn(ctx, b, res, c)
		if !ok {
			continue
		}
		head := headLine(body, 160)
		parts = append(parts, []byte(fmt.Sprintf("- [%s] %s", c.Role, head)))
		srcIDs = append(srcIDs, c.ID)
		folded++
	}
	summary := []byte(strings.Join(byteLines(parts), "\n"))
	digest := Digest(summary)
	derived := &Cell{
		ID:         "consolidate:" + digest[:12],
		Step:       -1,
		Role:       "memq/consolidate",
		Kind:       "disposition",
		Descriptor: headLine(summary, 120),
		Digest:     digest,
		Bytes:      int64(len(summary)),
		// A derived disposition is at most session-durable without an earned, witnessed
		// promotion — expire-by-default applies to derivations too.
		Durability: DurabilitySession,
		Refs:       srcIDs,
	}
	res.Effects = append(res.Effects, Effect{
		Kind: OpConsolidate, Applied: false, Cells: srcIDs, Derived: derived, DerivedBytes: summary,
		Note: fmt.Sprintf("folded %d cell(s) into a derived %s disposition; artifact produced, durable write-back is rung 2", folded, derived.Durability),
	})
	return fmt.Sprintf("folded %d cell(s) -> %s", folded, derived.ID)
}

// applyReclassify proposes a durability change for the set. It can only HOLD or LOWER
// a class (the effective target is the less-durable of the request and the cell's
// current class) — a promotion toward a longer-lived class is refused, since promotion
// must be earned. Write-back is the rung-2 follow-on, so Applied stays false.
func applyReclassify(res *Result, work []Cell, target string) string {
	target = NormDurability(target)
	var ids []string
	capped := 0
	for _, c := range work {
		cur := NormDurability(c.Durability)
		eff := target
		if durabilityRank[eff] > durabilityRank[cur] {
			eff = cur // refuse the promotion
			capped++
		}
		ids = append(ids, c.ID)
		_ = eff
	}
	res.Effects = append(res.Effects, Effect{
		Kind: OpReclassify, Applied: false, Cells: ids,
		Note: fmt.Sprintf("target=%s; %d promotion(s) refused (capped at current class); durable write-back is rung 2", target, capped),
	})
	return fmt.Sprintf("%d cell(s) proposed -> %s (%d promotions refused)", len(ids), target, capped)
}

// applyPrune reclaims unreferenced storage if the backend supports it. Dry-run counts;
// a caps grant applies. A backend with no Pruner is proposal-only (the recall image's
// CAS GC lives in fak dream, which memq complements rather than duplicates).
func applyPrune(ctx context.Context, b Backend, res *Result, caps Caps) string {
	pr, ok := b.(Pruner)
	if !ok {
		res.Effects = append(res.Effects, Effect{
			Kind: OpPrune, Applied: false,
			Note: "backend does not support prune (use `fak dream` for recall-image CAS GC)",
		})
		return "prune unsupported by backend"
	}
	apply := caps.may(OpPrune)
	blobs, bytes, err := pr.Prune(ctx, apply)
	note := fmt.Sprintf("%d unreferenced blob(s), %d byte(s)", blobs, bytes)
	if err != nil {
		note = "prune error: " + err.Error()
	} else if !apply {
		note += " (proposal only; run with --apply to reclaim)"
	}
	res.Effects = append(res.Effects, Effect{Kind: OpPrune, Applied: apply && err == nil, Note: note})
	return note
}

func finalizeStats(res *Result) {
	res.Stats.CellsSelected = len(res.Working)
	res.Stats.Rendered = len(res.Rendered)
	for _, r := range res.Rendered {
		res.Stats.RenderedBytes += r.Bytes
		res.Stats.EstimatedTokens += r.Tokens
	}
	res.Stats.Refused = len(res.Refused)
	for _, e := range res.Effects {
		res.Stats.EffectsProposed++
		if e.Applied {
			res.Stats.EffectsApplied++
		}
	}
}

// Digest is the canonical content address (sha256 hex), the same scheme recall/blob
// use, so a memq-derived digest and a recall digest are interchangeable.
func Digest(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

func tokenEstimate(n int) int {
	if n <= 0 {
		return 0
	}
	return (n + 3) / 4
}

// headLine returns a single-line, bounded, trimmed head of a byte body — the same
// extractive descriptor shape recall uses, so a consolidation carries no more than a
// faithful prefix of its sources (no model, no hallucination surface).
func headLine(b []byte, max int) string {
	s := strings.TrimSpace(string(b))
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		s = s[:i]
	}
	s = strings.TrimSpace(s)
	if len(s) > max {
		s = s[:max]
	}
	return s
}

func byteLines(parts [][]byte) []string {
	out := make([]string, len(parts))
	for i, p := range parts {
		out[i] = string(p)
	}
	return out
}
