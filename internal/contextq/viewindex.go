package contextq

import (
	"context"
	"errors"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/cdb"
	"github.com/anthony-chaudhary/fak/internal/recall"
)

// The recall-image view indexer (acceptance #2 of issue #437). Where Query
// answers ONE request with a working-set of views, IndexViews walks an attached
// image and emits, for every benign source page, the full multi-view set the
// memo names: a descriptor card, an extracted facts list, an ordered timeline,
// a templated QA pair, and a bounded summary. Each view is a provenance-bound
// MemoryViewRecord carrying its source page ids and a digest-addressed cache
// entry, so the same fused model — immutable pages, many recomputable views —
// is materialized over a whole image rather than a single query.
//
// Every producer is deterministic and extractive: a view's substantive tokens
// are verbatim from the source page bytes or the frame's own descriptor
// metadata, with only fixed structural scaffolding added (ordinal markers, a
// QA template). There is no model in the loop and thus no hallucination surface,
// so FaithfulnessProbe is 1.0; the lossy axis is Coverage (extracted source
// bytes / page bytes). Sealed and tombstoned pages fail closed: they are refused
// before any byte is paged in, so no derived view can leak a quarantined page.

// Additional derived view types the indexer emits alongside ViewSummary. They
// join ViewSnippet/ViewSummary as the multi-view set acceptance #2 names.
const (
	// ViewDescriptor is a metadata-only card (role, descriptor, step, length).
	// It pages in no bytes; its content is the frame's own descriptor metadata.
	ViewDescriptor ViewType = "descriptor"
	// ViewFacts is the page's "fact-like" lines (key:value, key=value, or a
	// line carrying a digit) extracted verbatim and in order.
	ViewFacts ViewType = "facts"
	// ViewTimeline is the page's non-empty lines in chronological (page) order,
	// verbatim, bounded — the event sequence the page records.
	ViewTimeline ViewType = "timeline"
	// ViewQA is a deterministic question/answer pair: the question is templated
	// over the frame descriptor, the answer is the extractive summary.
	ViewQA ViewType = "qa"
)

// IndexViewTypes is the ordered multi-view set IndexViews emits per benign page
// when IndexRequest.Views is empty. It is exactly the set acceptance #2 names.
var IndexViewTypes = []ViewType{ViewDescriptor, ViewFacts, ViewTimeline, ViewQA, ViewSummary}

// maxIndexViewBytes bounds the extracted (facts/timeline) views. A page shorter
// than this contributes its whole eligible content.
const maxIndexViewBytes = 512

// IndexRequest configures a view-indexing pass over an attached image.
type IndexRequest struct {
	// Producer stamps every emitted view (default "contextq-index").
	Producer string `json:"producer,omitempty"`
	// PolicyVersion is the freshness axis carried onto every view, identical to
	// the Query path's contract.
	PolicyVersion string `json:"policy_version,omitempty"`
	// Scope is stamped onto every view's cache entry.
	Scope abi.ShareScope `json:"scope"`
	// Views selects which view types to emit per page; empty -> IndexViewTypes.
	Views []ViewType `json:"views,omitempty"`
}

// IndexResult is the typed multi-view index over an image's benign pages.
type IndexResult struct {
	Views    []MemoryViewRecord       `json:"views"`
	Verdicts []MaterializationVerdict `json:"verdicts"`
	Refused  []Refusal                `json:"refused"`
	Stats    IndexStats               `json:"stats"`
}

// IndexStats reports the index's coverage and fail-closed accounting.
type IndexStats struct {
	PagesTotal    int              `json:"pages_total"`
	PagesIndexed  int              `json:"pages_indexed"`
	PagesRefused  int              `json:"pages_refused"`
	BytesPagedIn  int64            `json:"bytes_paged_in"`
	ViewsEmitted  int              `json:"views_emitted"`
	ViewsByType   map[ViewType]int `json:"views_by_type"`
	SourcePageIDs []int            `json:"source_page_ids"`
}

// IndexViews emits the multi-view set over every benign page of an attached
// image. Sealed and tombstoned pages are refused (fail closed) before any byte
// is paged in. Each emitted MemoryViewRecord is a FAULT (built from a raw
// page-in or, for the descriptor card, from frame metadata) and carries its
// source page id plus a digest-addressed cache entry.
func IndexViews(ctx context.Context, im *cdb.Image, req IndexRequest) IndexResult {
	if req.Producer == "" {
		req.Producer = "contextq-index"
	}
	want := req.Views
	if len(want) == 0 {
		want = IndexViewTypes
	}

	frames := im.Backtrace()
	entries := map[int]cachemeta.Entry{}
	for i, e := range im.PageCacheEntries() {
		entries[i] = e
	}

	res := IndexResult{Stats: IndexStats{
		PagesTotal:  len(frames),
		ViewsByType: map[ViewType]int{},
	}}
	covered := map[int]bool{}

	for _, f := range frames {
		e := entries[f.Step]

		// Fail closed: a sealed or tombstoned source never yields a view, and we
		// refuse it WITHOUT paging in its bytes so nothing downstream can leak it.
		if f.Sealed {
			res.refuse(f, e.ID, "sealed_by_trust_gate")
			res.Stats.PagesRefused++
			continue
		}
		if f.Tombstoned {
			res.refuse(f, e.ID, "tombstoned_by_context_control")
			res.Stats.PagesRefused++
			continue
		}

		// Page in once per page; derive every requested content view from the
		// same bytes. The descriptor card needs no bytes but we still gate on a
		// successful page-in so a page that refuses on examine yields no views.
		b, err := im.Examine(ctx, f.Step)
		if err != nil {
			reason := "page_in_refused"
			if errors.Is(err, recall.ErrSealed) {
				reason = "sealed_by_trust_gate"
			}
			res.refuse(f, e.ID, reason)
			res.Stats.PagesRefused++
			continue
		}
		res.Stats.BytesPagedIn += int64(len(b))
		res.Stats.PagesIndexed++
		covered[f.Step] = true

		for _, vt := range want {
			payload, extracted := produceView(vt, f, b)
			view := makeIndexedView(req, f, e, int64(len(b)), vt, payload, extracted)
			res.Views = append(res.Views, view)
			res.Stats.ViewsEmitted++
			res.Stats.ViewsByType[vt]++
			res.Verdicts = append(res.Verdicts, MaterializationVerdict{
				Kind:   MaterializationFault,
				Reason: "view_indexed_from_page",
				Step:   f.Step,
				ViewID: view.ViewID,
				Entry:  view.CacheEntry.ID,
			})
		}
	}

	for step := range covered {
		res.Stats.SourcePageIDs = append(res.Stats.SourcePageIDs, step)
	}
	sort.Ints(res.Stats.SourcePageIDs)
	return res
}

func (r *IndexResult) refuse(f cdb.Frame, id cachemeta.EntryID, reason string) {
	r.Refused = append(r.Refused, Refusal{
		Step: f.Step, Role: f.Role, Descriptor: f.Descriptor, Reason: reason, Entry: id,
	})
	r.Verdicts = append(r.Verdicts, MaterializationVerdict{
		Kind: MaterializationRefuse, Reason: reason, Step: f.Step, Entry: id,
	})
}

// produceView returns the rendered payload for a view type plus the count of
// verbatim source bytes it carries (the numerator of Coverage). The descriptor
// card reports its full descriptor metadata as "covered" since it is not a
// content slice of the page bytes.
func produceView(vt ViewType, f cdb.Frame, page []byte) (payload []byte, extracted int64) {
	switch vt {
	case ViewDescriptor:
		p := buildDescriptor(f)
		return p, int64(len(p))
	case ViewFacts:
		return buildFacts(page)
	case ViewTimeline:
		return buildTimeline(page)
	case ViewQA:
		return buildQA(f, page)
	default: // ViewSummary and any unknown type fall back to the bounded summary.
		p := buildSummary(page)
		return p, int64(len(p))
	}
}

// makeIndexedView lowers one derived view into a provenance-bound record. The
// descriptor card's Coverage is 1.0 (it fully captures the frame descriptor);
// content views report extracted/page-len coverage.
func makeIndexedView(req IndexRequest, f cdb.Frame, source cachemeta.Entry, pageLen int64, vt ViewType, payload []byte, extracted int64) MemoryViewRecord {
	viewID := "view-" + string(vt) + "-step-" + strconv.Itoa(f.Step) + "-" + short(source.ID.Digest)
	coverage := 1.0
	if vt != ViewDescriptor && pageLen > 0 {
		coverage = float64(extracted) / float64(pageLen)
		if coverage > 1.0 {
			coverage = 1.0
		}
		if coverage < 0 {
			coverage = 0
		}
	}
	v := cachemeta.MemoryView{
		ViewID:            viewID,
		ViewType:          string(vt),
		Length:            int64(len(payload)),
		SourceRefs:        []cachemeta.EntryID{source.ID},
		Producer:          req.Producer,
		PolicyVersion:     req.PolicyVersion,
		Scope:             req.Scope,
		Taint:             source.Security.Taint,
		Coverage:          coverage,
		FaithfulnessProbe: 1.0,
		Witness:           source.Validity.Witness,
	}
	entry := cachemeta.FromMemoryView(v)
	return MemoryViewRecord{
		ViewID:            viewID,
		ViewType:          vt,
		SourcePageIDs:     []int{f.Step},
		SourceDigests:     []string{source.ID.Digest},
		SourceLen:         pageLen,
		Producer:          req.Producer,
		PolicyVersion:     req.PolicyVersion,
		Scope:             req.Scope,
		Taint:             source.Security.Taint,
		Coverage:          coverage,
		FaithfulnessProbe: 1.0,
		CacheEntry:        entry,
		Labels: map[string]string{
			"role":       f.Role,
			"descriptor": f.Descriptor,
			"view_type":  string(vt),
		},
	}
}

// buildDescriptor renders the metadata card. Every value is verbatim frame
// metadata; no page bytes are read.
func buildDescriptor(f cdb.Frame) []byte {
	var sb strings.Builder
	sb.WriteString("role: ")
	sb.WriteString(f.Role)
	sb.WriteString("\ndescriptor: ")
	sb.WriteString(f.Descriptor)
	sb.WriteString("\nstep: ")
	sb.WriteString(strconv.Itoa(f.Step))
	sb.WriteString("\nlen: ")
	sb.WriteString(strconv.FormatInt(f.Len, 10))
	return []byte(sb.String())
}

// buildFacts extracts the page's fact-like lines verbatim: a line carrying a
// "key: value" / "key=value" shape or a digit. Bounded by maxIndexViewBytes.
func buildFacts(page []byte) ([]byte, int64) {
	s := strings.ToValidUTF8(string(page), "")
	var picked []string
	used := 0
	for _, line := range strings.Split(s, "\n") {
		t := strings.TrimSpace(line)
		if t == "" || !factLike(t) {
			continue
		}
		if used+len(t) > maxIndexViewBytes {
			break
		}
		picked = append(picked, t)
		used += len(t)
	}
	out := strings.Join(picked, "\n")
	return []byte(out), int64(len(out))
}

// factLike reports whether a trimmed line looks like a fact: a key:value or
// key=value pair, or any line containing a digit.
func factLike(t string) bool {
	if i := strings.IndexAny(t, ":="); i > 0 && i < len(t)-1 {
		return true
	}
	return strings.ContainsAny(t, "0123456789")
}

// buildTimeline renders the page's non-empty lines verbatim in page order,
// bounded by maxIndexViewBytes — the event sequence the page records.
func buildTimeline(page []byte) ([]byte, int64) {
	s := strings.ToValidUTF8(string(page), "")
	var picked []string
	used := 0
	for _, line := range strings.Split(s, "\n") {
		t := strings.TrimSpace(line)
		if t == "" {
			continue
		}
		if used+len(t) > maxIndexViewBytes {
			break
		}
		picked = append(picked, t)
		used += len(t)
	}
	out := strings.Join(picked, "\n")
	return []byte(out), int64(len(out))
}

// buildQA renders a deterministic question/answer pair. The question is a fixed
// template over the verbatim frame descriptor; the answer is the extractive
// summary. Both contain only source-derived text plus fixed scaffolding, so the
// view carries no invented facts.
func buildQA(f cdb.Frame, page []byte) ([]byte, int64) {
	desc := strings.TrimSpace(f.Descriptor)
	if desc == "" {
		desc = f.Role
	}
	answer := buildSummary(page)
	var sb strings.Builder
	sb.WriteString("Q: What does ")
	sb.WriteString(desc)
	sb.WriteString(" record?\nA: ")
	sb.Write(answer)
	// Coverage numerator is the verbatim answer bytes (the question is metadata).
	return []byte(sb.String()), int64(len(answer))
}
