package gateway

import (
	"errors"
	"regexp"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

var ctxRestoreTombstoneIDRE = regexp.MustCompile(`\bid=([0-9a-f]{64})\b`)

// CtxRestoreTombstoneIDs extracts unique content handles in first-seen order.
func CtxRestoreTombstoneIDs(text string) []string {
	seen := make(map[string]struct{})
	var ids []string
	for _, match := range ctxRestoreTombstoneIDRE.FindAllStringSubmatch(text, -1) {
		if _, ok := seen[match[1]]; ok {
			continue
		}
		seen[match[1]] = struct{}{}
		ids = append(ids, match[1])
	}
	return ids
}

type CtxRestoreBoundedRequest struct {
	TraceID    string   `json:"trace_id"`
	IDs        []string `json:"ids"`
	ByteBudget int      `json:"byte_budget"`
}

type CtxRestoreSpanOutcome struct {
	ID         string `json:"id"`
	Bytes      string `json:"bytes,omitempty"`
	Excerpt    string `json:"excerpt,omitempty"`
	Provenance string `json:"provenance,omitempty"`
	Status     string `json:"status"`
	Reason     string `json:"reason,omitempty"`
}

type CtxRestoreBoundedResult struct {
	Spans     []CtxRestoreSpanOutcome `json:"spans"`
	UsedBytes int                     `json:"used_bytes"`
	Elided    int                     `json:"elided"`
	Misses    int                     `json:"misses"`
	Refused   int                     `json:"refused"`
}

// restoreContextBounded is a read-only worker reseed over the existing exact-ID
// trust gate. Misses and refusals are rows, allowing a worker to continue with
// the spans that remain admissible.
func (s *Server) restoreContextBounded(caller string, req CtxRestoreBoundedRequest) CtxRestoreBoundedResult {
	budget := req.ByteBudget
	if budget < 0 {
		budget = 0
	}
	result := CtxRestoreBoundedResult{Spans: make([]CtxRestoreSpanOutcome, 0, len(req.IDs))}
	seen := make(map[string]struct{})
	for _, rawID := range req.IDs {
		id := strings.TrimSpace(rawID)
		if id == "" {
			continue
		}
		if _, ok := seen[id]; ok {
			continue
		}
		seen[id] = struct{}{}
		span, err := s.restoreContext(caller, ContextRestoreRequest{ID: id, TraceID: req.TraceID})
		if err != nil {
			row := CtxRestoreSpanOutcome{ID: id, Status: "MISS", Reason: err.Error()}
			switch {
			case errors.Is(err, ErrRestoreRefused), errors.Is(err, ctxplan.ErrSealed), errors.Is(err, ctxplan.ErrTombstoned):
				row.Status = "REFUSED"
				result.Refused++
			default:
				result.Misses++
			}
			result.Spans = append(result.Spans, row)
			continue
		}
		need := len(span.Bytes)
		if need > budget-result.UsedBytes {
			result.Spans = append(result.Spans, CtxRestoreSpanOutcome{ID: id, Excerpt: span.Excerpt, Status: "ELIDED", Reason: "byte budget exhausted"})
			result.Elided++
			continue
		}
		result.UsedBytes += need
		result.Spans = append(result.Spans, CtxRestoreSpanOutcome{
			ID: id, Bytes: span.Bytes, Excerpt: span.Excerpt, Provenance: span.Provenance, Status: "RESTORED",
		})
	}
	return result
}
