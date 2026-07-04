package contextq

import (
	"cmp"
	"fmt"
	"sort"
	"strings"
)

// AssumedContext is an explicit assumption row supplied by the caller. The
// renderer does not infer assumptions from materialized pages; inferred facts
// must be passed in as assumptions so known evidence and assumed context stay
// visibly separate.
type AssumedContext struct {
	Key        string  `json:"key"`
	Statement  string  `json:"statement"`
	Source     string  `json:"source,omitempty"`
	Confidence float64 `json:"confidence,omitempty"`
	Action     string  `json:"action,omitempty"`
	Reason     string  `json:"reason,omitempty"`
}

// KnownContextRow is one materialized evidence handle that may enter context.
type KnownContextRow struct {
	Step           int                 `json:"step"`
	Role           string              `json:"role"`
	Descriptor     string              `json:"descriptor"`
	ViewID         string              `json:"view_id,omitempty"`
	ViewDigest     string              `json:"view_digest,omitempty"`
	SourceDigest   string              `json:"source_digest,omitempty"`
	SourceMedia    string              `json:"source_media,omitempty"`
	Bytes          int64               `json:"bytes"`
	MaterializedBy MaterializationKind `json:"materialized_by"`
}

// UnknownContextRow is one referenced handle that is not known/materialized.
type UnknownContextRow struct {
	Kind         string `json:"kind"` // refusal or omission
	Step         int    `json:"step"`
	Role         string `json:"role"`
	Descriptor   string `json:"descriptor"`
	Reason       string `json:"reason"`
	SourceDigest string `json:"source_digest,omitempty"`
	SourceMedia  string `json:"source_media,omitempty"`
}

// ContextRender is the operator-facing known/unknown/assumed split for a query
// result. It is a projection of Result, not a second materializer.
type ContextRender struct {
	Query   string              `json:"query"`
	Known   []KnownContextRow   `json:"known"`
	Unknown []UnknownContextRow `json:"unknown"`
	Assumed []AssumedContext    `json:"assumed"`
}

// RenderKnownUnknownAssumedContext projects a query Result into three explicit
// buckets: known materialized evidence, unknown/refused/omitted handles, and
// caller-supplied assumptions. Unknown rows are never promoted into known rows.
func RenderKnownUnknownAssumedContext(res Result, assumed []AssumedContext) ContextRender {
	views := map[string]MemoryViewRecord{}
	for _, v := range res.Views {
		views[v.ViewID] = v
	}
	out := ContextRender{
		Query:   res.Query,
		Assumed: cleanAssumptions(assumed),
	}
	for _, sl := range res.Slices {
		row := KnownContextRow{
			Step:           sl.Step,
			Role:           sl.Role,
			Descriptor:     sl.Descriptor,
			ViewID:         sl.ViewID,
			SourceDigest:   sl.Source.Digest,
			SourceMedia:    string(sl.Source.MediaType),
			Bytes:          sl.Bytes,
			MaterializedBy: sl.MaterializedBy,
		}
		if v, ok := views[sl.ViewID]; ok {
			row.ViewDigest = v.CacheEntry.ID.Digest
		}
		out.Known = append(out.Known, row)
	}
	for _, r := range res.Refused {
		out.Unknown = append(out.Unknown, UnknownContextRow{
			Kind:         "refusal",
			Step:         r.Step,
			Role:         r.Role,
			Descriptor:   r.Descriptor,
			Reason:       r.Reason,
			SourceDigest: r.Entry.Digest,
			SourceMedia:  string(r.Entry.MediaType),
		})
	}
	for _, o := range res.Omissions {
		out.Unknown = append(out.Unknown, UnknownContextRow{
			Kind:         "omission",
			Step:         o.Step,
			Role:         o.Role,
			Descriptor:   o.Descriptor,
			Reason:       o.Reason,
			SourceDigest: o.Entry.Digest,
			SourceMedia:  string(o.Entry.MediaType),
		})
	}
	sortKnownContext(out.Known)
	sortUnknownContext(out.Unknown)
	sortAssumedContext(out.Assumed)
	return out
}

// Markdown renders the three buckets as a compact operator transcript.
func (r ContextRender) Markdown() string {
	var b strings.Builder
	fmt.Fprintf(&b, "# context evidence split\n\n")
	fmt.Fprintf(&b, "- query: %q\n- known: %d\n- unknown: %d\n- assumed: %d\n\n",
		r.Query, len(r.Known), len(r.Unknown), len(r.Assumed))

	fmt.Fprintf(&b, "## known\n\n")
	fmt.Fprintf(&b, "| step | role | descriptor | view | source | materialized | bytes |\n|---|---|---|---|---|---|---|\n")
	for _, row := range r.Known {
		fmt.Fprintf(&b, "| %d | %s | %s | %s | %s | %s | %d |\n",
			row.Step, mdCell(row.Role), mdCell(row.Descriptor), mdCell(firstNonEmptyContext(row.ViewID, row.ViewDigest)),
			mdCell(shortDigest(row.SourceDigest)), row.MaterializedBy, row.Bytes)
	}

	fmt.Fprintf(&b, "\n## unknown\n\n")
	fmt.Fprintf(&b, "| kind | step | role | descriptor | reason | source |\n|---|---|---|---|---|---|\n")
	for _, row := range r.Unknown {
		fmt.Fprintf(&b, "| %s | %d | %s | %s | %s | %s |\n",
			mdCell(row.Kind), row.Step, mdCell(row.Role), mdCell(row.Descriptor), mdCell(row.Reason), mdCell(shortDigest(row.SourceDigest)))
	}

	fmt.Fprintf(&b, "\n## assumed\n\n")
	fmt.Fprintf(&b, "| key | statement | source | confidence | action | reason |\n|---|---|---|---|---|---|\n")
	for _, row := range r.Assumed {
		fmt.Fprintf(&b, "| %s | %s | %s | %.2f | %s | %s |\n",
			mdCell(row.Key), mdCell(row.Statement), mdCell(row.Source), row.Confidence, mdCell(row.Action), mdCell(row.Reason))
	}
	return b.String()
}

func cleanAssumptions(in []AssumedContext) []AssumedContext {
	out := make([]AssumedContext, 0, len(in))
	for _, a := range in {
		a.Key = strings.TrimSpace(a.Key)
		a.Statement = strings.TrimSpace(a.Statement)
		a.Source = strings.TrimSpace(a.Source)
		a.Action = strings.TrimSpace(a.Action)
		a.Reason = strings.TrimSpace(a.Reason)
		if a.Key == "" && a.Statement == "" {
			continue
		}
		if a.Confidence < 0 {
			a.Confidence = 0
		}
		if a.Confidence > 1 {
			a.Confidence = 1
		}
		out = append(out, a)
	}
	return out
}

// cmpOrdered returns -1, 0, or 1 for a vs b — the numeric core every chained
// tie-break comparison below reduces to.
func cmpOrdered[T cmp.Ordered](a, b T) int {
	switch {
	case a < b:
		return -1
	case a > b:
		return 1
	default:
		return 0
	}
}

// chainLess composes per-key comparators (each returning cmpOrdered's -1/0/1)
// into one sort.SliceStable Less func: the first comparator to report non-equal
// decides — exactly the "if rows[i].F != rows[j].F { return rows[i].F <
// rows[j].F }" cascade every row sorter in this file otherwise repeats by hand.
func chainLess(cmps ...func(i, j int) int) func(i, j int) bool {
	return func(i, j int) bool {
		for _, c := range cmps {
			if v := c(i, j); v != 0 {
				return v < 0
			}
		}
		return false
	}
}

func sortKnownContext(rows []KnownContextRow) {
	sort.SliceStable(rows, chainLess(
		func(i, j int) int { return cmpOrdered(rows[i].Step, rows[j].Step) },
		func(i, j int) int { return cmpOrdered(rows[i].Role, rows[j].Role) },
		func(i, j int) int { return cmpOrdered(rows[i].Descriptor, rows[j].Descriptor) },
	))
}

func sortUnknownContext(rows []UnknownContextRow) {
	sort.SliceStable(rows, chainLess(
		func(i, j int) int { return cmpOrdered(rows[i].Step, rows[j].Step) },
		func(i, j int) int { return cmpOrdered(rows[i].Kind, rows[j].Kind) },
		func(i, j int) int { return cmpOrdered(rows[i].Role, rows[j].Role) },
		func(i, j int) int { return cmpOrdered(rows[i].Reason, rows[j].Reason) },
	))
}

func sortAssumedContext(rows []AssumedContext) {
	sort.SliceStable(rows, chainLess(
		func(i, j int) int { return cmpOrdered(rows[i].Key, rows[j].Key) },
		func(i, j int) int { return cmpOrdered(rows[i].Statement, rows[j].Statement) },
	))
}

func mdCell(s string) string {
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.ReplaceAll(s, "|", "\\|")
	return strings.TrimSpace(s)
}

func shortDigest(s string) string {
	if len(s) > 12 {
		return s[:12]
	}
	return s
}

func firstNonEmptyContext(xs ...string) string {
	for _, x := range xs {
		if strings.TrimSpace(x) != "" {
			return x
		}
	}
	return ""
}
