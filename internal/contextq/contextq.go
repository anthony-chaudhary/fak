// Package contextq is the on-demand context materializer over CDB images.
// It does not own session bytes. CDB remains the page-fault provider and recall
// remains the trust gate; this package only turns a request into typed handles,
// views, materialization verdicts, omissions, and a render plan.
//
// Two materialization paths ship:
//
//   - the default raw-page path (no PreferView set): every selected benign page
//     is demand-paged through the trust gate and wrapped as a snippet view. Every
//     materialization is a FAULT. This is the v1 behavior and is unchanged.
//   - the derived-view path (PreferView set, e.g. "summary"): selected pages are
//     resolved through a ViewCache. A fresh view is a FAULT (the raw page must be
//     paged in to build it); a cached-but-stale view is a RECOMPUTE (policy axis
//     drifted, rebuild); a cached-and-fresh view is a HIT (serve without paging).
//     HIT is the economic point: a derived view is reused as an adjudicated cache
//     artifact instead of re-faulting the raw page.
//
// All five MaterializationVerdict kinds are reachable across the two paths:
// HIT, FAULT, RECOMPUTE, REFUSE, ABSTAIN.
package contextq

import (
	"context"
	"errors"
	"sort"
	"strconv"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/cdb"
	"github.com/anthony-chaudhary/fak/internal/recall"
)

// Request is one context-materialization query: the search query and K, the byte
// budget, scope, pin/exclude patterns, and the optional derived-view selector
// (PreferView + ViewCache) that switches Query from the raw-page path to the
// view-cache path.
type Request struct {
	Query string `json:"query"`
	K     int    `json:"k,omitempty"`
	// BudgetBytes caps the RENDERED context (what would enter a prompt), not just
	// paged-in bytes. A HIT still costs its rendered length against the budget; a
	// FAULT/RECOMPUTE costs the same rendered length but additionally faults the
	// raw page (counted in Stats.BytesPagedIn). 0 = unbounded.
	BudgetBytes int64          `json:"budget_bytes,omitempty"`
	Scope       abi.ShareScope `json:"scope"`
	Pins        []string       `json:"pins,omitempty"`
	Excludes    []string       `json:"excludes,omitempty"`
	// PolicyVersion stamps every built view and is the freshness axis for the
	// derived-view path. A cached view whose PolicyVersion differs from the
	// request's is stale -> RECOMPUTE.
	PolicyVersion string `json:"policy_version,omitempty"`
	Producer      string `json:"producer,omitempty"`
	// PreferView selects the derived-view path. When set together with ViewCache,
	// Query resolves each selected page through the cache (HIT/RECOMPUTE/FAULT)
	// instead of always faulting the raw page. Empty keeps the v1 raw-page path.
	PreferView ViewType `json:"prefer_view,omitempty"`
	// ViewCache is the optional shared view store consulted on the derived-view
	// path. It is safe for concurrent use; a nil cache forces the raw-page path
	// even when PreferView is set.
	ViewCache *ViewCache `json:"-"`
}

// Result is the outcome of a Query: the selected frames, slice handles, built/served
// views, per-page materialization verdicts (HIT/FAULT/RECOMPUTE/REFUSE/ABSTAIN),
// refusals and omissions, the render plan, and the materializer's byte/fault stats.
type Result struct {
	Query       string                   `json:"query"`
	BudgetBytes int64                    `json:"budget_bytes,omitempty"`
	Frames      []cdb.Frame              `json:"frames"`
	Slices      []SliceRef               `json:"slices"`
	Views       []MemoryViewRecord       `json:"views"`
	Verdicts    []MaterializationVerdict `json:"verdicts"`
	Refused     []Refusal                `json:"refused"`
	Omissions   []Omission               `json:"omissions"`
	RenderPlan  RenderPlan               `json:"render_plan"`
	Stats       Stats                    `json:"stats"`
}

// Stats accounts one Query's materialization: pages touched/total/benign, sealed and
// tombstoned skips, bytes paged in vs rendered, residency, faults avoided, and the
// view-cache hit/recompute counts.
type Stats struct {
	PagesTouched      int     `json:"pages_touched"`
	PagesTotal        int     `json:"pages_total"`
	PagesBenign       int     `json:"pages_benign"`
	SealedSkipped     int     `json:"sealed_skipped"`
	TombstonedSkipped int     `json:"tombstoned_skipped"`
	BytesPagedIn      int64   `json:"bytes_paged_in"`
	RenderedBytes     int64   `json:"rendered_bytes"`
	ResidentBytes     int64   `json:"resident_bytes"`
	ResidencyPct      float64 `json:"residency_pct"`
	FaultsAvoided     int     `json:"faults_avoided"`
	ViewHits          int     `json:"view_hits"`
	ViewRecomputes    int     `json:"view_recomputes"`
	PoisonInSet       bool    `json:"poison_in_set"`
}

// SliceRef is a handle to one materialized context slice: the source step/role, its
// rendered byte and token size, the source cache entry and view id, and the
// MaterializationKind (HIT/FAULT/RECOMPUTE) that produced it.
type SliceRef struct {
	Step           int                 `json:"step"`
	Role           string              `json:"role"`
	Descriptor     string              `json:"descriptor"`
	Bytes          int64               `json:"bytes"`
	TokenEstimate  int                 `json:"token_estimate"`
	Source         cachemeta.EntryID   `json:"source"`
	ViewID         string              `json:"view_id"`
	MaterializedBy MaterializationKind `json:"materialized_by"`
}

// ViewType names a derived-view rendering of a source page: snippet (the verbatim
// raw page), summary (a bounded extractive head), or playbook (the agent-edited
// strategy store). It is both the cache key axis and the PreferView selector.
type ViewType string

const (
	ViewSnippet ViewType = "snippet"
	// ViewSummary is a deterministic extractive summary: a bounded head-of-page
	// slice taken at a line boundary. It is faithful by construction (every byte
	// is verbatim from the source page, so FaithfulnessProbe = 1.0) and lossy
	// only in coverage (Coverage = rendered / source bytes).
	ViewSummary ViewType = "summary"
	// ViewPlaybook is the ContextPlaybook view (#538): NOT a read-only projection
	// of an immutable recorded page like the other views, but the rendered surface
	// of an agent-edited, accumulating, counter-ranked strategy store (ACE, arXiv
	// 2510.04618). It is materialized from ContextPlaybook.Snapshot; every bullet
	// that enters it was admit-gated by ctxmmu and every counter on it was earned
	// from an independent witness, never self-asserted. See playbook.go.
	ViewPlaybook ViewType = "playbook"
)

// maxSummaryBytes bounds an extractive summary. A page smaller than this is
// summarized whole (Coverage = 1.0).
const maxSummaryBytes = 256

// MemoryViewRecord is the metadata for one derived view: its id, type, source
// page(s)/digests, producer, policy version, scope, taint, coverage and faithfulness
// probe, and the cachemeta entry that addresses its rendered bytes.
type MemoryViewRecord struct {
	ViewID            string            `json:"view_id"`
	ViewType          ViewType          `json:"view_type"`
	SourcePageIDs     []int             `json:"source_page_ids"`
	SourceDigests     []string          `json:"source_digests"`
	SourceLen         int64             `json:"source_len,omitempty"`
	Producer          string            `json:"producer"`
	PolicyVersion     string            `json:"policy_version,omitempty"`
	Scope             abi.ShareScope    `json:"scope"`
	Taint             abi.TaintLabel    `json:"taint"`
	Coverage          float64           `json:"coverage"`
	FaithfulnessProbe float64           `json:"faithfulness_probe"`
	CacheEntry        cachemeta.Entry   `json:"cache_entry"`
	Labels            map[string]string `json:"labels,omitempty"`
}

// Entry returns the cachemeta entry that addresses this view's rendered bytes.
func (v MemoryViewRecord) Entry() cachemeta.Entry { return v.CacheEntry }

type MaterializationKind string

const (
	MaterializationHit       MaterializationKind = "HIT"
	MaterializationFault     MaterializationKind = "FAULT"
	MaterializationRecompute MaterializationKind = "RECOMPUTE"
	MaterializationRefuse    MaterializationKind = "REFUSE"
	MaterializationAbstain   MaterializationKind = "ABSTAIN"
)

type MaterializationVerdict struct {
	Kind   MaterializationKind `json:"kind"`
	Reason string              `json:"reason"`
	Step   int                 `json:"step,omitempty"`
	ViewID string              `json:"view_id,omitempty"`
	Entry  cachemeta.EntryID   `json:"entry,omitempty"`
}

// Refusal records a referenced page the materializer would not serve (sealed,
// tombstoned, excluded by request, or page-in refused) with the reason and source
// entry; it pairs with a REFUSE verdict.
type Refusal struct {
	Step       int               `json:"step"`
	Role       string            `json:"role"`
	Descriptor string            `json:"descriptor"`
	Reason     string            `json:"reason"`
	Entry      cachemeta.EntryID `json:"entry,omitempty"`
}

// Omission records a query-relevant page that was NOT materialized for a benign
// reason (not selected by the ranker, or the byte budget was exhausted) with the
// reason and source entry.
type Omission struct {
	Step       int               `json:"step"`
	Role       string            `json:"role"`
	Descriptor string            `json:"descriptor"`
	Reason     string            `json:"reason"`
	Entry      cachemeta.EntryID `json:"entry,omitempty"`
}

// RenderPlan is the prompt-assembly layout the materializer emits: the ordered render
// items split into a stable prefix (reused/derived views) and a volatile tail (raw-page
// faults), with the provider-cache and local-KV shape hints.
type RenderPlan struct {
	Target             string       `json:"target"`
	Items              []RenderItem `json:"items"`
	EstimatedTokens    int          `json:"estimated_tokens"`
	StablePrefixItems  int          `json:"stable_prefix_items"`
	VolatileTailItems  int          `json:"volatile_tail_items"`
	ProviderCacheShape string       `json:"provider_cache_shape"`
	LocalKVShape       string       `json:"local_kv_shape"`
}

// RenderItem kinds. A stable_prefix item is a reused/derived view (HIT or a
// freshly built view); a working_set/volatile item is a raw-page fault.
const (
	RenderRawPage    = "raw_page"
	RenderMemoryView = "memory_view"
)

// RenderItem is one entry in the render plan: its kind (raw_page vs memory_view), the
// source step, view id and cache entry, its token estimate, and its prompt position
// (stable_prefix vs working_set).
type RenderItem struct {
	Kind          string            `json:"kind"`
	Step          int               `json:"step,omitempty"`
	ViewID        string            `json:"view_id,omitempty"`
	Source        cachemeta.EntryID `json:"source"`
	TokenEstimate int               `json:"token_estimate"`
	Position      string            `json:"position"`
}

// ViewCache is a concurrency-safe store of derived view artifacts (metadata +
// rendered bytes). It is the "view = adjudicated cache artifact" substrate: a
// view built once can be served later as a HIT without re-faulting its source
// page. It is keyed by (source step, view type, producer) and deliberately
// EXCLUDES policy version, so a policy drift is observed as a stale hit
// (RECOMPUTE) rather than a silent miss.
type ViewCache struct {
	mu    sync.Mutex
	store map[string]viewCacheEntry
}

type viewCacheEntry struct {
	record  MemoryViewRecord
	payload []byte
}

// NewViewCache returns an empty, concurrency-safe ViewCache ready to back the
// derived-view materialization path.
func NewViewCache() *ViewCache {
	return &ViewCache{store: make(map[string]viewCacheEntry)}
}

func viewKey(step int, viewType ViewType, producer string) string {
	return string(viewType) + ":" + producer + ":step-" + strconv.Itoa(step)
}

// Get returns the cached view record and its rendered payload for the given
// source step. The bool is false when no view is cached.
func (c *ViewCache) Get(step int, viewType ViewType, producer string) (MemoryViewRecord, []byte, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	e, ok := c.store[viewKey(step, viewType, producer)]
	if !ok {
		return MemoryViewRecord{}, nil, false
	}
	return e.record, append([]byte(nil), e.payload...), true
}

// Put stores or replaces a view artifact. Replacing on the same key is how a
// RECOMPUTE lands the refreshed (new policy) view.
func (c *ViewCache) Put(v MemoryViewRecord, payload []byte) {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.store[viewKey(firstSourcePage(v), v.ViewType, v.Producer)] = viewCacheEntry{
		record:  v,
		payload: append([]byte(nil), payload...),
	}
}

// Invalidate drops a single source step's view, if any.
func (c *ViewCache) Invalidate(step int, viewType ViewType, producer string) {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.store, viewKey(step, viewType, producer))
}

func firstSourcePage(v MemoryViewRecord) int {
	if len(v.SourcePageIDs) > 0 {
		return v.SourcePageIDs[0]
	}
	return -1
}

// Query resolves a context request against an attached image. See the package
// doc for the two materialization paths.
func Query(ctx context.Context, im *cdb.Image, req Request) Result {
	if req.Producer == "" {
		req.Producer = "contextq"
	}
	frames := im.Backtrace()
	entries := pageEntries(im)

	ws := im.WorkingSet(ctx, req.Query, req.K)
	res := Result{
		Query:       req.Query,
		BudgetBytes: req.BudgetBytes,
		Frames:      frames,
		Stats: Stats{
			PagesTouched:      ws.PagesTouched,
			PagesTotal:        ws.PagesTotal,
			PagesBenign:       ws.PagesBenign,
			SealedSkipped:     ws.SealedSkipped,
			TombstonedSkipped: ws.TombstonedSkipped,
			BytesPagedIn:      ws.BytesPagedIn,
			ResidentBytes:     ws.ResidentBytes,
			ResidencyPct:      ws.ResidencyPct,
			FaultsAvoided:     ws.FaultsAvoided,
			PoisonInSet:       ws.PoisonInSet,
		},
		RenderPlan: RenderPlan{
			Target:             "prompt",
			ProviderCacheShape: "stable handles/views first, volatile raw-page faults last",
			LocalKVShape:       "exact prefix/radix only; non-prefix KV requires separate audit",
		},
	}

	// WorkingSet already paged its candidates to rank them, so its byte accounting
	// is for the ranking pass. For the materializer's own accounting we re-zero
	// BytesPagedIn and accrue only what THIS resolution actually faults.
	res.Stats.BytesPagedIn = 0
	res.Stats.ResidencyPct = 0

	excluded := matchingSteps(frames, req.Excludes)
	pinned := orderedMatchingSteps(frames, req.Pins)
	selected := make([]int, 0, len(pinned)+len(ws.Slices))
	seen := map[int]bool{}
	add := func(step int) {
		if step < 0 || step >= len(frames) || seen[step] {
			return
		}
		seen[step] = true
		selected = append(selected, step)
	}
	for _, step := range pinned {
		add(step)
	}
	for _, sl := range ws.Slices {
		add(sl.Step)
	}

	// Refuse sealed / tombstoned / explicitly-excluded pages that the request
	// touches. This runs on both paths; REFUSE is orthogonal to view resolution.
	for _, f := range frames {
		e := entries[f.Step]
		if excluded[f.Step] {
			res.refuse(f, e.ID, "excluded_by_request")
			continue
		}
		if f.Sealed && frameReferenced(f, req) {
			res.refuse(f, e.ID, "sealed_by_trust_gate")
			continue
		}
		if f.Tombstoned && frameReferenced(f, req) {
			res.refuse(f, e.ID, "tombstoned_by_context_control")
			continue
		}
	}

	useViews := req.PreferView != "" && req.ViewCache != nil
	if useViews {
		materializeWithViews(ctx, im, req, &res, frames, entries, selected, excluded)
	} else {
		materializeRaw(ctx, im, req, &res, frames, entries, selected, excluded)
	}
	finalizeStats(&res)
	return res
}

// materializeRaw is the v1 path: every selected page is demand-paged and wrapped
// as a snippet view; every materialization is a FAULT.
func materializeRaw(ctx context.Context, im *cdb.Image, req Request, res *Result, frames []cdb.Frame, entries map[int]cachemeta.Entry, selected []int, excluded map[int]bool) {
	usedBytes := int64(0)
	for _, step := range selected {
		f := frames[step]
		e := entries[step]
		if excluded[step] || f.Sealed || f.Tombstoned {
			continue
		}
		if req.BudgetBytes > 0 && usedBytes+f.Len > req.BudgetBytes {
			res.omitBudgetExhausted(f, e.ID)
			continue
		}
		b, reason, ok := examineStep(ctx, im, step)
		if !ok {
			res.refuse(f, e.ID, reason)
			continue
		}
		usedBytes += int64(len(b))
		res.Stats.BytesPagedIn += int64(len(b))
		view := makeSnippetView(req, f, e, int64(len(b)))
		res.Views = append(res.Views, view)
		res.Slices = append(res.Slices, SliceRef{
			Step:           f.Step,
			Role:           f.Role,
			Descriptor:     f.Descriptor,
			Bytes:          int64(len(b)),
			TokenEstimate:  tokenEstimate(len(b)),
			Source:         e.ID,
			ViewID:         view.ViewID,
			MaterializedBy: MaterializationFault,
		})
		res.Verdicts = append(res.Verdicts, MaterializationVerdict{
			Kind: MaterializationFault, Reason: "raw_page_fault", Step: f.Step, ViewID: view.ViewID, Entry: e.ID,
		})
		res.RenderPlan.Items = append(res.RenderPlan.Items, RenderItem{
			Kind:          RenderRawPage,
			Step:          f.Step,
			ViewID:        view.ViewID,
			Source:        e.ID,
			TokenEstimate: tokenEstimate(len(b)),
			Position:      renderPosition(len(res.RenderPlan.Items)),
		})
		res.RenderPlan.EstimatedTokens += tokenEstimate(len(b))
	}

	selectedSet := map[int]bool{}
	for _, step := range selected {
		selectedSet[step] = true
	}
	omitUnselected(req, res, frames, entries, selectedSet, excluded)
	res.Stats.RenderedBytes = usedBytes
}

// materializeWithViews is the derived-view path. For each selected page it
// consults the ViewCache and emits HIT / RECOMPUTE / FAULT, budgeting against
// rendered (prompt-bound) bytes. HIT avoids the raw-page fault entirely.
func materializeWithViews(ctx context.Context, im *cdb.Image, req Request, res *Result, frames []cdb.Frame, entries map[int]cachemeta.Entry, selected []int, excluded map[int]bool) {
	cache := req.ViewCache
	renderedBytes := int64(0)

	selectedSet := map[int]bool{}
	for _, step := range selected {
		selectedSet[step] = true
	}

	for _, step := range selected {
		f := frames[step]
		e := entries[step]
		if excluded[step] || f.Sealed || f.Tombstoned {
			continue
		}

		// Estimate rendered cost BEFORE resolving, so an over-budget item is
		// omitted without paying a raw fault. A cached view's rendered size is
		// known; an uncached one is estimated at the source page length.
		estimate := f.Len
		if cached, _, ok := cache.Get(step, req.PreferView, req.Producer); ok {
			if cached.CacheEntry.ID.Length > 0 {
				estimate = cached.CacheEntry.ID.Length
			}
		}
		if req.BudgetBytes > 0 && renderedBytes+estimate > req.BudgetBytes {
			res.omitBudgetExhausted(f, e.ID)
			continue
		}

		cached, cachedPayload, ok := cache.Get(step, req.PreferView, req.Producer)
		stale := ok && isStale(cached, req)

		switch {
		case ok && !stale:
			// HIT: serve the cached artifact, page nothing.
			renderedBytes += int64(len(cachedPayload))
			res.Views = append(res.Views, cached)
			res.Slices = append(res.Slices, SliceRef{
				Step:           f.Step,
				Role:           f.Role,
				Descriptor:     f.Descriptor,
				Bytes:          int64(len(cachedPayload)),
				TokenEstimate:  tokenEstimate(len(cachedPayload)),
				Source:         cached.CacheEntry.ID,
				ViewID:         cached.ViewID,
				MaterializedBy: MaterializationHit,
			})
			res.Verdicts = append(res.Verdicts, MaterializationVerdict{
				Kind: MaterializationHit, Reason: "view_cache_hit", Step: f.Step, ViewID: cached.ViewID, Entry: cached.CacheEntry.ID,
			})
			res.Stats.ViewHits++
			res.appendRenderItem(RenderMemoryView, f.Step, cached.ViewID, cached.CacheEntry.ID, len(cachedPayload))
		default:
			// Must page in the source to build (FAULT) or rebuild (RECOMPUTE).
			b, reason, ok := examineStep(ctx, im, step)
			if !ok {
				res.refuse(f, e.ID, reason)
				continue
			}
			res.Stats.BytesPagedIn += int64(len(b))
			payload := buildSummary(b)
			view := makeSummaryView(req, f, e, int64(len(b)), payload)
			cache.Put(view, payload)
			renderedBytes += int64(len(payload))

			kind, reason := MaterializationFault, "view_build_raw_page_fault"
			if stale {
				kind, reason = MaterializationRecompute, "view_stale_policy_mismatch"
				res.Stats.ViewRecomputes++
			}
			res.Views = append(res.Views, view)
			res.Slices = append(res.Slices, SliceRef{
				Step:           f.Step,
				Role:           f.Role,
				Descriptor:     f.Descriptor,
				Bytes:          int64(len(payload)),
				TokenEstimate:  tokenEstimate(len(payload)),
				Source:         view.CacheEntry.ID,
				ViewID:         view.ViewID,
				MaterializedBy: kind,
			})
			res.Verdicts = append(res.Verdicts, MaterializationVerdict{
				Kind: kind, Reason: reason, Step: f.Step, ViewID: view.ViewID, Entry: view.CacheEntry.ID,
			})
			res.appendRenderItem(RenderMemoryView, f.Step, view.ViewID, view.CacheEntry.ID, len(payload))
		}
	}

	omitUnselected(req, res, frames, entries, selectedSet, excluded)
	res.Stats.RenderedBytes = renderedBytes
}

func (r *Result) appendRenderItem(kind string, step int, viewID string, source cachemeta.EntryID, bytes int) {
	r.RenderPlan.Items = append(r.RenderPlan.Items, RenderItem{
		Kind:          kind,
		Step:          step,
		ViewID:        viewID,
		Source:        source,
		TokenEstimate: tokenEstimate(bytes),
		Position:      renderPosition(len(r.RenderPlan.Items)),
	})
	r.RenderPlan.EstimatedTokens += tokenEstimate(bytes)
}

// isStale is the freshness test for a cached view against the current request.
// Today the only staleness axis is policy version (the memo's contract: a view is
// stale when its policy axis drifts). Additional axes (witness refutation, TTL,
// source-page digest change) slot in here without touching the resolver.
func isStale(v MemoryViewRecord, req Request) bool {
	if req.PolicyVersion == "" || v.PolicyVersion == "" {
		return false
	}
	return v.PolicyVersion != req.PolicyVersion
}

func finalizeStats(res *Result) {
	// Stable prefix = reused or derived views that did not require a raw fault in
	// this pass; volatile tail = raw-page faults. HIT-only passes fill the stable
	// prefix, which is the provider-cache-friendly layout.
	stable, volatile := 0, 0
	for _, it := range res.RenderPlan.Items {
		if it.Kind == RenderMemoryView {
			stable++
		} else {
			volatile++
		}
	}
	res.RenderPlan.StablePrefixItems = stable
	res.RenderPlan.VolatileTailItems = volatile
	res.Stats.PagesTouched = len(res.Slices)
	if res.Stats.ResidentBytes > 0 {
		res.Stats.ResidencyPct = 100 * float64(res.Stats.BytesPagedIn) / float64(res.Stats.ResidentBytes)
	}
	// FaultsAvoided counts benign pages that were served without a raw fault this
	// pass (HITs) plus benign pages the working set never selected. It can go
	// negative only if selection exceeds benign, which the gate prevents; clamp
	// anyway for robustness.
	servedNoFault := res.Stats.ViewHits
	res.Stats.FaultsAvoided = res.Stats.PagesBenign - res.Stats.TombstonedSkipped - pagesFaulted(res)
	if res.Stats.FaultsAvoided < 0 {
		res.Stats.FaultsAvoided = 0
	}
	res.Stats.FaultsAvoided += servedNoFault
}

func pagesFaulted(res *Result) int {
	n := 0
	for _, sl := range res.Slices {
		if sl.MaterializedBy == MaterializationFault || sl.MaterializedBy == MaterializationRecompute {
			n++
		}
	}
	return n
}

func makeSnippetView(req Request, f cdb.Frame, source cachemeta.Entry, byteLen int64) MemoryViewRecord {
	viewID := "view-step-" + strconv.Itoa(f.Step) + "-" + short(source.ID.Digest)
	v := cachemeta.MemoryView{
		ViewID:            viewID,
		ViewType:          string(ViewSnippet),
		Length:            byteLen,
		SourceRefs:        []cachemeta.EntryID{source.ID},
		Producer:          req.Producer,
		PolicyVersion:     req.PolicyVersion,
		Scope:             req.Scope,
		Taint:             source.Security.Taint,
		Coverage:          1.0,
		FaithfulnessProbe: 1.0,
		Witness:           source.Validity.Witness,
	}
	entry := cachemeta.FromMemoryView(v)
	return MemoryViewRecord{
		ViewID:            viewID,
		ViewType:          ViewSnippet,
		SourcePageIDs:     []int{f.Step},
		SourceDigests:     []string{source.ID.Digest},
		SourceLen:         byteLen,
		Producer:          req.Producer,
		PolicyVersion:     req.PolicyVersion,
		Scope:             req.Scope,
		Taint:             source.Security.Taint,
		Coverage:          1.0,
		FaithfulnessProbe: 1.0,
		CacheEntry:        entry,
		Labels: map[string]string{
			"role":       f.Role,
			"descriptor": f.Descriptor,
		},
	}
}

// makeSummaryView builds an extractive summary view over one source page. The
// view's cachemeta entry carries the rendered length, source refs, coverage, and
// a faithfulness probe of 1.0 (extractive -> no hallucination surface).
func makeSummaryView(req Request, f cdb.Frame, source cachemeta.Entry, sourceLen int64, payload []byte) MemoryViewRecord {
	viewID := "view-summary-step-" + strconv.Itoa(f.Step) + "-" + short(source.ID.Digest)
	coverage := 1.0
	if sourceLen > 0 {
		coverage = float64(len(payload)) / float64(sourceLen)
		if coverage > 1.0 {
			coverage = 1.0
		}
	}
	v := cachemeta.MemoryView{
		ViewID:            viewID,
		ViewType:          string(ViewSummary),
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
		ViewType:          ViewSummary,
		SourcePageIDs:     []int{f.Step},
		SourceDigests:     []string{source.ID.Digest},
		SourceLen:         sourceLen,
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
		},
	}
}

// buildSummary is the deterministic extractive summary producer: a bounded,
// line-boundary-trimmed head of the source page. No model dependency, no
// hallucination surface — the bytes are a verbatim prefix of the source.
func buildSummary(page []byte) []byte {
	s := strings.ToValidUTF8(string(page), "")
	s = strings.TrimSpace(s)
	if len(s) <= maxSummaryBytes {
		return []byte(s)
	}
	cut := maxSummaryBytes
	if i := strings.LastIndexAny(s[:cut], "\n"); i > 0 {
		cut = i
	}
	return []byte(strings.TrimRight(s[:cut], " \t\r\n"))
}

func (r *Result) refuse(f cdb.Frame, id cachemeta.EntryID, reason string) {
	r.Refused = append(r.Refused, Refusal{
		Step: f.Step, Role: f.Role, Descriptor: f.Descriptor, Reason: reason, Entry: id,
	})
	r.Verdicts = append(r.Verdicts, MaterializationVerdict{
		Kind: MaterializationRefuse, Reason: reason, Step: f.Step, Entry: id,
	})
}

func (r *Result) omit(f cdb.Frame, id cachemeta.EntryID, reason string) {
	r.Omissions = append(r.Omissions, Omission{
		Step: f.Step, Role: f.Role, Descriptor: f.Descriptor, Reason: reason, Entry: id,
	})
}

// pageEntries snapshots an image's page-cache entries keyed by step. Both Query
// and IndexViews need this map; building it once here keeps the two resolvers in
// lockstep.
func pageEntries(im *cdb.Image) map[int]cachemeta.Entry {
	entries := map[int]cachemeta.Entry{}
	for i, e := range im.PageCacheEntries() {
		entries[i] = e
	}
	return entries
}

// examineStep pages in one step through the trust gate. On failure it returns the
// canonical refusal reason ("sealed_by_trust_gate" for a sealed page, else
// "page_in_refused") so every caller refuses with identical wording; ok is false
// when the page-in was refused.
func examineStep(ctx context.Context, im *cdb.Image, step int) (b []byte, reason string, ok bool) {
	b, err := im.Examine(ctx, step)
	if err != nil {
		reason = "page_in_refused"
		if errors.Is(err, recall.ErrSealed) {
			reason = "sealed_by_trust_gate"
		}
		return nil, reason, false
	}
	return b, "", true
}

// omitBudgetExhausted records the omission + abstain verdict for a frame the
// budget can no longer admit. Shared by the raw and view materialization paths.
func (r *Result) omitBudgetExhausted(f cdb.Frame, id cachemeta.EntryID) {
	r.omit(f, id, "budget_exhausted")
	r.Verdicts = append(r.Verdicts, MaterializationVerdict{
		Kind: MaterializationAbstain, Reason: "budget_exhausted", Step: f.Step, Entry: id,
	})
}

// omitUnselected omits every referenced benign frame the ranker did not select,
// after a materialization pass. Shared by the raw and view paths.
func omitUnselected(req Request, res *Result, frames []cdb.Frame, entries map[int]cachemeta.Entry, selectedSet, excluded map[int]bool) {
	for _, f := range frames {
		if selectedSet[f.Step] || excluded[f.Step] || f.Sealed || f.Tombstoned {
			continue
		}
		if frameReferenced(f, req) {
			e := entries[f.Step]
			res.omit(f, e.ID, "not_selected_by_ranker")
		}
	}
}

func matchingSteps(frames []cdb.Frame, patterns []string) map[int]bool {
	out := map[int]bool{}
	for _, p := range patterns {
		p = strings.TrimSpace(strings.ToLower(p))
		if p == "" {
			continue
		}
		for _, f := range frames {
			if frameContains(f, p) {
				out[f.Step] = true
			}
		}
	}
	return out
}

func orderedMatchingSteps(frames []cdb.Frame, patterns []string) []int {
	m := matchingSteps(frames, patterns)
	out := make([]int, 0, len(m))
	for _, f := range frames {
		if m[f.Step] {
			out = append(out, f.Step)
		}
	}
	return out
}

func frameReferenced(f cdb.Frame, req Request) bool {
	if len(req.Pins) > 0 && matchingSteps([]cdb.Frame{f}, req.Pins)[f.Step] {
		return true
	}
	if len(req.Excludes) > 0 && matchingSteps([]cdb.Frame{f}, req.Excludes)[f.Step] {
		return true
	}
	return overlap(tokens(req.Query), tokens(f.Role+" "+f.Descriptor)) > 0
}

func frameContains(f cdb.Frame, needle string) bool {
	hay := strings.ToLower(f.Role + " " + f.Descriptor + " " + f.Digest + " step:" + strconv.Itoa(f.Step))
	return strings.Contains(hay, needle)
}

func tokens(s string) []string {
	raw := strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return !((r >= 'a' && r <= 'z') || (r >= '0' && r <= '9'))
	})
	out := raw[:0]
	for _, t := range raw {
		if len(t) > 2 && !queryStopwords[t] {
			out = append(out, t)
		}
	}
	return out
}

var queryStopwords = map[string]bool{
	"the": true, "and": true, "for": true, "did": true, "show": true,
	"what": true, "with": true, "from": true, "this": true, "that": true,
	"user": true, "users": true, "page": true, "pages": true, "using": true,
}

func overlap(query, doc []string) int {
	q := make(map[string]bool, len(query))
	for _, t := range query {
		q[t] = true
	}
	seen := map[string]bool{}
	n := 0
	for _, t := range doc {
		if q[t] && !seen[t] {
			seen[t] = true
			n++
		}
	}
	return n
}

func tokenEstimate(bytes int) int {
	if bytes <= 0 {
		return 0
	}
	return (bytes + 3) / 4
}

func renderPosition(i int) string {
	if i == 0 {
		return "stable_prefix"
	}
	return "working_set"
}

func short(d string) string {
	if len(d) > 12 {
		return d[:12]
	}
	return d
}

// C3: Per-capability residency tracking and eviction.
// See SKILL-LOADER-QUERY-EPIC.md issue #1106.

// CapState mirrors ctxresidency.State for capability residency.
type CapState string

const (
	CapStateResident  CapState = "resident"
	CapStateEvictable CapState = "evictable"
	CapStateHeld      CapState = "held"
)

// CapSpan records a single capability's residency state and metrics.
type CapSpan struct {
	CapRef    CapRef   `json:"cap_ref"`
	State     CapState `json:"state"`
	Faults    int      `json:"faults"`
	LastFault int64    `json:"last_fault"`
	Bytes     int      `json:"bytes"`
}

// CapSnapshot is a consistent snapshot of all tracked capabilities.
type CapSnapshot struct {
	Spans []CapSpan `json:"spans"`
}

// CapabilityLedger tracks per-capability residency for C3 eviction.
// It uses fault count as a recency signal (fewer faults = colder).
type CapabilityLedger struct {
	mu sync.RWMutex

	caps map[CapRef]*capResidency
}

type capResidency struct {
	state     CapState
	faults    int
	lastFault int64
	bytes     int
}

// NewCapabilityLedger creates a new capability residency ledger.
func NewCapabilityLedger() *CapabilityLedger {
	return &CapabilityLedger{
		caps: make(map[CapRef]*capResidency),
	}
}

// RecordFault records a capability fault, transitioning it to resident.
func (l *CapabilityLedger) RecordFault(ref CapRef, now int64) {
	if l == nil {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()

	cr := l.caps[ref]
	if cr == nil {
		cr = &capResidency{
			bytes: len(ref.Name) * 4, // rough token estimate
		}
		l.caps[ref] = cr
	}

	cr.state = CapStateResident
	cr.faults++
	cr.lastFault = now
}

// setState transitions a tracked capability to the given state under the ledger
// lock. A nil ledger or an unknown ref is a no-op. Shared by the single-field
// state transitions (RecordEvict, MarkEvictable).
func (l *CapabilityLedger) setState(ref CapRef, state CapState) {
	if l == nil {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()

	if cr := l.caps[ref]; cr != nil {
		cr.state = state
	}
}

// RecordEvict records an eviction, transitioning the capability to held.
func (l *CapabilityLedger) RecordEvict(ref CapRef) { l.setState(ref, CapStateHeld) }

// MarkEvictable marks a capability as evictable.
func (l *CapabilityLedger) MarkEvictable(ref CapRef) { l.setState(ref, CapStateEvictable) }

// Query returns a consistent snapshot of all tracked capabilities.
func (l *CapabilityLedger) Query() CapSnapshot {
	if l == nil {
		return CapSnapshot{Spans: []CapSpan{}}
	}
	l.mu.RLock()
	defer l.mu.RUnlock()

	snap := CapSnapshot{
		Spans: make([]CapSpan, 0, len(l.caps)),
	}

	for ref, cr := range l.caps {
		snap.Spans = append(snap.Spans, CapSpan{
			CapRef:    ref,
			State:     cr.state,
			Faults:    cr.faults,
			LastFault: cr.lastFault,
			Bytes:     cr.bytes,
		})
	}

	return snap
}

// EvictColdest evicts up to n capabilities in coldest-first order
// (fewest recent faults).
func (l *CapabilityLedger) EvictColdest(n int) []CapRef {
	if l == nil || n <= 0 {
		return nil
	}
	l.mu.Lock()
	defer l.mu.Unlock()

	type evictCandidate struct {
		ref CapRef
		cr  *capResidency
	}
	var evictable []evictCandidate
	for ref, cr := range l.caps {
		if cr.state == CapStateEvictable {
			evictable = append(evictable, evictCandidate{ref: ref, cr: cr})
		}
	}

	// Sort by fault count (coldest = fewest faults)
	sort.Slice(evictable, func(i, j int) bool {
		if evictable[i].cr.faults != evictable[j].cr.faults {
			return evictable[i].cr.faults < evictable[j].cr.faults
		}
		return evictable[i].ref.Name < evictable[j].ref.Name
	})

	evicted := make([]CapRef, 0, min(n, len(evictable)))
	for i := 0; i < min(n, len(evictable)); i++ {
		evicted = append(evicted, evictable[i].ref)
		evictable[i].cr.state = CapStateHeld
	}

	return evicted
}

// EvictUnderBudget evicts evictable capabilities until total bytes < budget.
func (l *CapabilityLedger) EvictUnderBudget(budgetBytes int) []CapRef {
	if l == nil {
		return nil
	}
	l.mu.Lock()
	defer l.mu.Unlock()

	// Calculate current total bytes
	var totalBytes int
	type evictCandidate struct {
		ref CapRef
		cr  *capResidency
	}
	var evictable []evictCandidate
	for ref, cr := range l.caps {
		totalBytes += cr.bytes
		if cr.state == CapStateEvictable {
			evictable = append(evictable, evictCandidate{ref: ref, cr: cr})
		}
	}

	if totalBytes <= budgetBytes {
		return nil
	}

	// Sort by fault count (coldest = fewest faults)
	sort.Slice(evictable, func(i, j int) bool {
		if evictable[i].cr.faults != evictable[j].cr.faults {
			return evictable[i].cr.faults < evictable[j].cr.faults
		}
		return evictable[i].ref.Name < evictable[j].ref.Name
	})

	evicted := make([]CapRef, 0, len(evictable))
	for i := 0; i < len(evictable) && totalBytes > budgetBytes; i++ {
		totalBytes -= evictable[i].cr.bytes
		evicted = append(evicted, evictable[i].ref)
		evictable[i].cr.state = CapStateHeld
	}

	return evicted
}

// CapRef identifies a capability (from C2 QueryCapabilities).
type CapRef struct {
	Name    string `json:"name"`
	Source  string `json:"source"`   // skill name or system
	Card    int    `json:"card"`     // 0 = scalar, >0 = collection
	IsQuery bool   `json:"is_query"` // true if from a query parameter
}

// CapSkill is the skill metadata from SKILL-LOADER-QUERY-EPIC.
type CapSkill struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	File        string `json:"file"`
}

// CapResult is a resolved capability reference (from C2).
type CapResult struct {
	Ref   CapRef    `json:"ref"`
	Skill *CapSkill `json:"skill,omitempty"`
}

// C2: in-kernel query front-end — model-emitted intent+budget → rank cards → fault winners.
// See SKILL-LOADER-QUERY-EPIC.md issue #1105.

// CapKind names a capability kind: skill, mcp-tool, a2a-agent, or future protocols.
type CapKind string

const (
	CapKindSkill    CapKind = "skill"
	CapKindMCPTool  CapKind = "mcp-tool"
	CapKindA2AAgent CapKind = "a2a-agent"
)

// CapCard is the tiny, at-rest descriptor for a capability: its name, version,
// trigger clause (what the model searches for), and tags. It is the O(1) query
// surface; only winners' full bodies are faulted in.
type CapCard struct {
	Name          string   `json:"name"`
	Kind          CapKind  `json:"kind"`
	Version       string   `json:"version"`
	Trigger       string   `json:"trigger"` // search query text
	Tags          []string `json:"tags"`
	EstimateBytes int      `json:"estimate_bytes"` // size hint for budgeting
}

// Capability is a full capability descriptor: its reference, digest (content hash),
// the CapCard (indexable metadata), and a Resolve function that pages in the body
// on FAULT. The body is lazily loaded; only queried winners are materialized.
type Capability struct {
	Ref     CapRef         `json:"ref"`
	Digest  string         `json:"digest"`
	Card    CapCard        `json:"card"`
	Resolve func() []byte  `json:"-"`
	Scope   abi.ShareScope `json:"scope"`
}

// Resolver is the protocol-generic capability lookup seam. Every protocol
// (skill, MCP, A2A, future) implements Index() and Fault().
type Resolver interface {
	// Index returns the at-rest CapCard set — cheap metadata only.
	Index() []CapCard
	// Fault resolves a capability reference into a full Capability, loading its
	// body. The same CapRef must resolve to the same Capability (cacheability).
	Fault(ref CapRef) Capability
}

// CapQueryRequest is a model-emitted intent+budget query over the capability space.
type CapQueryRequest struct {
	Intent      string `json:"intent"`       // natural-language query from the model
	BudgetBytes int64  `json:"budget_bytes"` // max rendered bytes for faulted winners
	K           int    `json:"k,omitempty"`  // max winners to return (0 = unbounded)
}

// CapQueryResult is the outcome of QueryCapabilities: ranked winners with their
// faulted bodies, plus omissions for budget-exhausted items.
type CapQueryResult struct {
	Winners   []Capability `json:"winners"`
	Omitted   []CapCard    `json:"ommitted"`
	BudgetHit bool         `json:"budget_hit"` // true if budget exhausted mid-rank
}

// QueryCapabilities accepts a model-emitted intent+budget query, ranks CapCards
// by relevance, faults in winners up to the budget, and returns the resolved
// capabilities. This is the MCP-Zero active-discovery move: the model emits a query,
// and only the N winners are materialized into context. The query is in-kernel and
// witnessed (every fault lands in the CapabilityLedger via RecordFault).
func QueryCapabilities(resolvers []Resolver, req CapQueryRequest, ledger *CapabilityLedger) CapQueryResult {
	if len(resolvers) == 0 {
		return CapQueryResult{}
	}

	now := int64(0)
	if ledger != nil {
		now = 0
	}

	cards := indexAll(resolvers)
	ranked := rankByIntent(cards, req.Intent)

	var winners []Capability
	var omitted []CapCard
	usedBytes := int64(0)

	for _, card := range ranked {
		if req.K > 0 && len(winners) >= req.K {
			omitted = append(omitted, card)
			continue
		}
		if req.BudgetBytes > 0 && usedBytes+int64(card.EstimateBytes) > req.BudgetBytes {
			omitted = append(omitted, card)
			continue
		}

		var cap Capability
		for _, r := range resolvers {
			cap = r.Fault(CapRef{Name: card.Name, Source: string(card.Kind)})
			if cap.Resolve != nil {
				break
			}
		}
		if cap.Resolve != nil {
			winners = append(winners, cap)
			usedBytes += int64(card.EstimateBytes)
			if ledger != nil {
				ledger.RecordFault(cap.Ref, now)
			}
		}
	}

	return CapQueryResult{
		Winners:   winners,
		Omitted:   omitted,
		BudgetHit: len(omitted) > 0,
	}
}

// indexAll concatenates all resolver indices into one CapCard slice.
func indexAll(resolvers []Resolver) []CapCard {
	var all []CapCard
	for _, r := range resolvers {
		all = append(all, r.Index()...)
	}
	return all
}

// rankByIntent orders cards by token overlap with the intent query.
// Higher overlap = higher rank.
func rankByIntent(cards []CapCard, intent string) []CapCard {
	queryTokens := tokens(intent)
	type scoredCard struct {
		card  CapCard
		score int
	}
	var scored []scoredCard
	for _, card := range cards {
		score := overlap(queryTokens, tokens(card.Trigger+" "+card.Name))
		for _, tag := range card.Tags {
			score += overlap(queryTokens, tokens(tag))
		}
		scored = append(scored, scoredCard{card: card, score: score})
	}
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score
	})
	result := make([]CapCard, len(scored))
	for i, sc := range scored {
		result[i] = sc.card
	}
	return result
}
