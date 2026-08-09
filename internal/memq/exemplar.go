package memq

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/simhash"
	"github.com/anthony-chaudhary/fak/internal/trajquery"
)

// ExemplarVerdict types why a candidate trajectory was or was not injected.
type ExemplarVerdict string

const (
	ExemplarInjected ExemplarVerdict = "INJECTED"
	ExemplarStale    ExemplarVerdict = "STALE_RECALL"
	ExemplarSealed   ExemplarVerdict = "SEALED_RECALL"
	ExemplarMissing  ExemplarVerdict = "MISSING_RECALL"
	ExemplarBudget   ExemplarVerdict = "BUDGET_WITHHELD"
)

// ExemplarDecision is safe provenance for one trajectory candidate. It never
// contains the trajectory body or backend error (which may carry paths).
type ExemplarDecision struct {
	TraceID string          `json:"trace_id"`
	CellID  string          `json:"cell_id"`
	Verdict ExemplarVerdict `json:"verdict"`
	Score   float64         `json:"score,omitempty"`
}

// ExemplarRequest selects concrete prior trajectory turns. Corpus rows are a
// confined metadata index, not trusted page-in bytes. Required columns are
// trace_id, cell_id and task; content is always read through Backend.Materialize.
type ExemplarRequest struct {
	Task       string
	TopK       int
	ByteBudget int
	Notes      []byte // already-admitted notes; exemplars share this total budget
	View       trajquery.View
	Corpus     []trajquery.Row
	Backend    Backend
}

// ExemplarBlock is a budget-bounded orientation block plus safe decisions.
type ExemplarBlock struct {
	Bytes     []byte             `json:"-"`
	Decisions []ExemplarDecision `json:"decisions"`
	UsedBytes int                `json:"used_bytes"`
}

type exemplarCandidate struct {
	traceID string
	cellID  string
	score   float64
}

// RetrieveExemplars retrieves by deterministic embedding, validates and executes
// the read through trajquery's confined View, then pages every selected body in
// through the existing memq trust gate. Stale, sealed and missing pages are
// withheld. Notes and exemplars consume one shared byte budget.
func RetrieveExemplars(ctx context.Context, req ExemplarRequest) (ExemplarBlock, error) {
	if req.Backend == nil {
		return ExemplarBlock{}, errors.New("memq: exemplar backend is required")
	}
	if req.TopK <= 0 || req.ByteBudget <= len(req.Notes) {
		return ExemplarBlock{UsedBytes: len(req.Notes)}, nil
	}
	q := trajquery.Query{From: req.View.Name, Columns: []string{"trace_id", "cell_id", "task"}}
	report := req.View.Validate(q, req.Corpus)
	if !report.Valid {
		return ExemplarBlock{}, fmt.Errorf("memq: exemplar view refused metadata query: %s", strings.Join(report.Violations, "; "))
	}
	rewritten, err := req.View.Rewrite(q)
	if err != nil {
		return ExemplarBlock{}, fmt.Errorf("memq: exemplar metadata rewrite: %w", err)
	}
	rows, err := rewritten.Execute(req.Corpus)
	if err != nil {
		return ExemplarBlock{}, fmt.Errorf("memq: exemplar metadata query: %w", err)
	}
	queryVec := simhash.Embed(req.Task)
	candidates := make([]exemplarCandidate, 0, len(rows))
	for _, row := range rows {
		traceID, _ := row["trace_id"].(string)
		cellID, _ := row["cell_id"].(string)
		task, _ := row["task"].(string)
		if traceID == "" || cellID == "" || task == "" {
			continue
		}
		candidates = append(candidates, exemplarCandidate{traceID: traceID, cellID: cellID, score: simhash.Cosine(queryVec, simhash.Embed(task))})
	}
	sort.SliceStable(candidates, func(i, j int) bool {
		if candidates[i].score == candidates[j].score {
			return candidates[i].traceID < candidates[j].traceID
		}
		return candidates[i].score > candidates[j].score
	})
	if len(candidates) > req.TopK {
		candidates = candidates[:req.TopK]
	}

	out := ExemplarBlock{UsedBytes: len(req.Notes)}
	var b strings.Builder
	for _, c := range candidates {
		body, materializeErr := req.Backend.Materialize(ctx, c.cellID)
		d := ExemplarDecision{TraceID: c.traceID, CellID: c.cellID, Score: c.score}
		switch {
		case errors.Is(materializeErr, ErrStale):
			d.Verdict = ExemplarStale
		case errors.Is(materializeErr, ErrSealed):
			d.Verdict = ExemplarSealed
		case materializeErr != nil:
			d.Verdict = ExemplarMissing
		default:
			entry := fmt.Sprintf("[trajectory exemplar trace=%s]\n%s\n", strconv.Quote(c.traceID), body)
			if out.UsedBytes+b.Len()+len(entry) > req.ByteBudget {
				d.Verdict = ExemplarBudget
			} else {
				d.Verdict = ExemplarInjected
				b.WriteString(entry)
			}
		}
		out.Decisions = append(out.Decisions, d)
	}
	out.Bytes = []byte(b.String())
	out.UsedBytes += len(out.Bytes)
	return out, nil
}
