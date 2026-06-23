package cdb

import (
	"context"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/recall"
)

// Image is an ATTACHED core image: a finished session reloaded as a debuggable
// target. It wraps a recall.Session (the frozen page table over a private CAS, with
// the trust gate enforced on every page-in) and adds the debugger's inspection +
// demand-paging surface. Construct it with Attach.
//
// Everything below is read-mostly: the only mutation is Clear (a witness clearance),
// which is exactly as load-bearing here as in recall — necessary but not sufficient,
// because Examine still re-screens the bytes through a fresh gate on the way in.
type Image struct {
	sess *recall.Session
	dir  string

	// on-disk decomposition, captured at attach (the core-dump view):
	manifestBytes int64 // the page table you always carry — small
	casFileBytes  int64 // the swap device on disk (base64+JSON inflated)
}

// Attach loads the persisted core image under dir (manifest.json + cas.json) and
// returns a debugger bound to it. Like recall.Load it verifies every CAS blob against
// its digest address, so a tampered swap device fails closed at attach. The returned
// Image resolves only against the on-disk bytes — never the process that produced
// them — so attaching to a session whose run is long dead is the whole point.
func Attach(dir string) (*Image, error) {
	s, err := recall.Load(dir)
	if err != nil {
		return nil, err
	}
	im := &Image{sess: s, dir: dir}
	if fi, e := os.Stat(filepath.Join(dir, "manifest.json")); e == nil {
		im.manifestBytes = fi.Size()
	}
	if fi, e := os.Stat(filepath.Join(dir, "cas.json")); e == nil {
		im.casFileBytes = fi.Size()
	}
	return im, nil
}

// Info is the core-image decomposition: a flat "N-token session" re-presented as a
// small page table over a cold, dedup'd swap device. This is the headline structural
// claim — that the heavy bytes were already paged out at write time, so what you
// carry to answer a follow-up is the manifest, not the transcript.
type Info struct {
	SessionID string `json:"session_id"`
	Version   string `json:"version"`
	WorldVer  uint64 `json:"world_ver"` // frozen: a finished session never writes again

	Pages      int `json:"pages"`
	Benign     int `json:"benign"`
	Sealed     int `json:"sealed"`     // quarantined out of any context
	Tombstoned int `json:"tombstoned"` // suppressed from model-visible recall
	Cleared    int `json:"cleared"`    // sealed pages a witness has cleared (still re-screened on page-in)
	Heavy      int `json:"heavy_pages"`

	RawBytes      int64 `json:"raw_bytes"`   // sum of page lengths (the flat-transcript size)
	CASBytes      int64 `json:"cas_bytes"`   // distinct cold bytes on the swap device (post-dedup)
	DedupSaved    int64 `json:"dedup_saved"` // RawBytes - CASBytes (content-addressed dedup win)
	DistinctBlobs int   `json:"distinct_blobs"`
	ResidentBytes int64 `json:"resident_bytes"` // benign distinct bytes — the demand-pageable universe

	ManifestFileBytes int64 `json:"manifest_file_bytes"` // the page table on disk (small)
	CASFileBytes      int64 `json:"cas_file_bytes"`      // the swap device on disk (base64-inflated)
}

// Info computes the decomposition from the loaded page table (no bytes are paged in).
func (im *Image) Info() Info {
	st := im.sess.Stats()
	info := Info{
		SessionID: st.SessionID, Version: st.Version, WorldVer: im.sess.Manifest.WorldVer,
		Pages: st.Pages, Benign: st.Benign, Sealed: st.Quarantined, Tombstoned: st.Tombstoned, Cleared: st.Cleared,
		CASBytes: st.CASBytes, ManifestFileBytes: im.manifestBytes, CASFileBytes: im.casFileBytes,
	}
	distinct := map[string]int64{}    // digest -> len
	benignDigest := map[string]bool{} // digest seen on a benign page
	for _, p := range im.sess.Pages() {
		info.RawBytes += p.Len
		if p.Len > ctxmmu.OversizeBytes {
			info.Heavy++
		}
		distinct[p.Digest] = p.Len
		if !p.Quarantined && !im.sess.Tombstoned(p.Step) {
			benignDigest[p.Digest] = true
		}
	}
	info.DistinctBlobs = len(distinct)
	info.DedupSaved = info.RawBytes - info.CASBytes
	for d := range benignDigest {
		info.ResidentBytes += distinct[d]
	}
	return info
}

// Frame is one page-table row — the debugger's stack frame / memory-map line. It
// carries NO bytes: a sealed page's frame is the safe sealed-metadata descriptor only,
// so inspecting the map can never surface poison.
type Frame struct {
	Step       int    `json:"step"`
	Role       string `json:"role"`
	Descriptor string `json:"descriptor"`
	Len        int64  `json:"len"`
	Digest     string `json:"digest"` // short content address
	Taint      uint8  `json:"taint"`
	Sealed     bool   `json:"sealed"`
	Tombstoned bool   `json:"tombstoned"`
	Heavy      bool   `json:"heavy"` // paged out to the CAS at write time (> OversizeBytes)
	QID        string `json:"qid,omitempty"`
	Reason     string `json:"reason,omitempty"`
}

// Backtrace returns the full page table — the `bt` / memory-map of the core image.
func (im *Image) Backtrace() []Frame {
	pages := im.sess.Pages()
	out := make([]Frame, 0, len(pages))
	for _, p := range pages {
		out = append(out, frameOf(p, im.sess.Tombstoned(p.Step)))
	}
	return out
}

func frameOf(p recall.Page, tombstoned bool) Frame {
	d := p.Digest
	if len(d) > 12 {
		d = d[:12]
	}
	return Frame{
		Step: p.Step, Role: p.Role, Descriptor: p.Descriptor, Len: p.Len, Digest: d,
		Taint: p.Taint, Sealed: p.Quarantined, Tombstoned: tombstoned, Heavy: p.Len > ctxmmu.OversizeBytes,
		QID: p.QID, Reason: p.Reason,
	}
}

// PageCacheEntry lowers one page-table row into cache metadata without paging in
// its bytes. The entry is suitable as a handle in higher-level context
// materializers.
func (im *Image) PageCacheEntry(step int) (cachemeta.Entry, bool) {
	pages := im.sess.Pages()
	if step < 0 || step >= len(pages) {
		return cachemeta.Entry{}, false
	}
	return pages[step].CacheEntry(im.sess.Manifest.SessionID), true
}

// PageCacheEntries returns cache metadata for every page-table row without
// materializing page bytes.
func (im *Image) PageCacheEntries() []cachemeta.Entry {
	pages := im.sess.Pages()
	out := make([]cachemeta.Entry, 0, len(pages))
	for _, p := range pages {
		out = append(out, p.CacheEntry(im.sess.Manifest.SessionID))
	}
	return out
}

// Examine demand-pages ONE page's bytes through the gate — the debugger's `x`
// (examine memory at an address). A benign page round-trips byte-identical (and is
// re-screened, defense in depth); a sealed page is REFUSED unless a witness Clear()
// ran AND the bytes pass a fresh content re-screen. The error wraps recall.ErrSealed,
// so a caller can branch on "the gate held" vs a plain lookup miss.
func (im *Image) Examine(ctx context.Context, step int) ([]byte, error) {
	return im.sess.Resolve(ctx, step)
}

// Clear records a witness clearance for a sealed page id. Necessary, NOT sufficient:
// Examine/WorkingSet still re-screen the bytes, so clearing a still-poisoned page does
// not release it.
func (im *Image) Clear(qid string) { im.sess.Clear(qid) }

// RequestContextChange applies a safe, model-visible context-control request to
// the attached image. The only shipped mutation is a tombstone: future working-set
// assembly skips the page, but the original core-image bytes remain for audit.
func (im *Image) RequestContextChange(req recall.ContextChangeRequest) (recall.ContextChange, error) {
	return im.sess.RequestContextChange(req)
}

// Persist writes manifest-level context-control changes back to the attached core
// image directory. CAS bytes are preserved.
func (im *Image) Persist() error { return im.sess.Persist(im.dir) }

// Slice re-exports a paged-in benign page so callers need not import recall.
type Slice = recall.Slice

// WorkingSet is Denning's working set W(query): the set of pages a follow-up question
// actually references, demand-paged from the cold swap device — and the residency
// accounting that proves you touched only a slice of the image, not the whole address
// space.
//
// The "reference string" is the query's token set; a page is referenced iff its
// extractive descriptor overlaps it. We demand-page the top-k referenced BENIGN pages
// (sealed pages are never candidates — their bytes are gone from any index), then
// report how little of the resident image that working set actually was. The pages
// NOT in the working set are page faults AVOIDED: they stay cold on the device.
type WorkingSet struct {
	Query string `json:"query"`

	Slices []Slice `json:"-"` // the paged-in bytes (not serialized — may be large)

	PagesTouched      int `json:"pages_touched"` // |W(query)|
	PagesTotal        int `json:"pages_total"`
	PagesBenign       int `json:"pages_benign"`
	SealedSkipped     int `json:"sealed_skipped"`     // candidates the trust gate excluded outright
	TombstonedSkipped int `json:"tombstoned_skipped"` // pages suppressed by context-control tombstones

	BytesPagedIn  int64 `json:"bytes_paged_in"` // distinct bytes the working set faulted in
	ResidentBytes int64 `json:"resident_bytes"` // the demand-pageable universe (benign distinct bytes)
	CASBytes      int64 `json:"cas_bytes"`

	ResidencyPct  float64 `json:"residency_pct"`  // BytesPagedIn / ResidentBytes — "you touched X%"
	FaultsAvoided int     `json:"faults_avoided"` // PagesBenign - PagesTouched
	PoisonInSet   bool    `json:"poison_in_set"`  // false by construction; checked, not assumed
}

// WorkingSet assembles W(query). k bounds the working set; k<=0 means "every page the
// query references" (the unbounded working set).
//
// cdb owns the reference predicate (which pages the query touches) rather than
// delegating to recall's ranker: deciding the working set is the debugger's whole job,
// and a bag-of-words ranker that counts stopwords ("the", "set") would fault in pages a
// real follow-up never references. So we rank benign pages in two phases — phase 1 is
// the STOPWORD-FILTERED extractive overlap (a HARD filter: nonzero overlap to be a
// candidate); phase 2 (#540) re-ranks the surviving candidates by relevance plus the
// page's learned, witness-gated recall.Page.Utility, the SAME second phase recall.Recall
// applies — then we take the top-k and page each in through the SAME gated recall.Resolve.
// The selection is cdb's, the trust boundary is still recall's. Sealed pages are never
// candidates, and utility re-ranks only WITHIN the provenance-clean set.
func (im *Image) WorkingSet(ctx context.Context, query string, k int) WorkingSet {
	info := im.Info()
	pages := im.sess.Pages()
	if k <= 0 {
		k = len(pages)
	}

	qtok := contentTokens(query)
	type scored struct {
		p     recall.Page
		score int     // phase 1: stopword-filtered lexical overlap (a HARD filter)
		rank  float64 // phase 2: relevance + learned outcome-utility (#540)
	}
	var cand []scored
	for _, p := range pages {
		if p.Quarantined {
			continue // sealed bytes are not a demand-paging candidate
		}
		if im.sess.Tombstoned(p.Step) {
			continue // agent-requested tombstones are absent from model-visible recall
		}
		// Phase 1 (unchanged): a score-0 page is never a candidate, so the phase-2
		// re-rank below sits BEHIND this hard filter — utility re-orders WITHIN the
		// relevant set and can never widen it (quarantined/tombstoned pages stay out).
		s := overlap(qtok, contentTokens(p.Descriptor))
		if s == 0 {
			continue
		}
		// Phase 2 (#540): re-rank the relevant candidates by relevance PLUS the page's
		// learned outcome-utility — the SAME second phase recall.Recall (recall.go)
		// applies, so the working set this debugger demand-pages and recall's own
		// assembled set order consistently. recall.Page.Utility is the witness-gated,
		// [0,recall.UtilityMax]-clamped scalar persisted in manifest.json: a quarantined
		// or witness-refuted page can never carry it (recall.Session.Credit skips it and
		// the dream seal path zeroes it), so this re-rank stays WITHIN the
		// provenance-clean set and never resurrects a sealed page. At default-neutral
		// utility (0) rank == float64(score), so the order is byte-identical to the
		// pre-#540 lexical-only ranking for every never-credited session.
		cand = append(cand, scored{p, s, float64(s) + p.Utility})
	}
	sort.SliceStable(cand, func(i, j int) bool { return cand[i].rank > cand[j].rank })

	ws := WorkingSet{
		Query: query, PagesTotal: info.Pages, PagesBenign: info.Benign,
		SealedSkipped: info.Sealed, TombstonedSkipped: info.Tombstoned,
		ResidentBytes: info.ResidentBytes, CASBytes: info.CASBytes,
	}
	pagedDigests := map[string]bool{}
	for i := 0; i < len(cand) && i < k; i++ {
		p := cand[i].p
		b, err := im.sess.Resolve(ctx, p.Step) // gated page-in (re-screened)
		if err != nil {
			continue
		}
		ws.Slices = append(ws.Slices, Slice{Step: p.Step, Role: p.Role, Descriptor: p.Descriptor, Bytes: b})
		ws.PagesTouched++
		if !pagedDigests[p.Digest] { // content addressing: a page faulted in twice costs the device once
			pagedDigests[p.Digest] = true
			ws.BytesPagedIn += int64(len(b))
		}
		if isInjection(b) { // belt-and-suspenders: assert the working set stayed clean
			ws.PoisonInSet = true
		}
	}
	ws.FaultsAvoided = info.Benign - info.Tombstoned - ws.PagesTouched
	if ws.ResidentBytes > 0 {
		ws.ResidencyPct = 100 * float64(ws.BytesPagedIn) / float64(ws.ResidentBytes)
	}
	return ws
}

// Grep searches the page table's descriptors — a read-only `search` over the memory
// map that pages in NOTHING. Returns matching frames in step order. Because a sealed
// page's descriptor is sealed-metadata only, grep can never echo poison bytes. pattern
// is a regexp; an invalid pattern degrades to a literal substring match.
func (im *Image) Grep(pattern string) []Frame {
	re, err := regexp.Compile(pattern)
	var match func(string) bool
	if err != nil {
		needle := strings.ToLower(pattern)
		match = func(s string) bool { return strings.Contains(strings.ToLower(s), needle) }
	} else {
		match = re.MatchString
	}
	var out []Frame
	for _, p := range im.sess.Pages() {
		if match(p.Descriptor) || match(p.Role) {
			out = append(out, frameOf(p, im.sess.Tombstoned(p.Step)))
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Step < out[j].Step })
	return out
}

// Pages is the count of page-table entries (the frozen world-version).
func (im *Image) Pages() int { return len(im.sess.Pages()) }

// stopwords are dropped from both the query and a page's descriptor before scoring
// overlap, so a natural-language follow-up ("what refund fee did the user's account
// show?") references a page on its CONTENT words, not on "the"/"did"/"show". This is
// the one place cdb improves on recall's naive ranker — and it is selection, never the
// trust boundary.
var stopwords = map[string]bool{
	"the": true, "and": true, "for": true, "did": true, "show": true, "was": true,
	"what": true, "with": true, "you": true, "your": true, "are": true, "this": true,
	"that": true, "from": true, "has": true, "have": true, "had": true, "set": true,
	"get": true, "use": true, "all": true, "any": true, "out": true, "via": true,
	"its": true, "our": true, "not": true, "can": true, "how": true, "why": true,
	"who": true, "does": true, "into": true, "over": true, "user": true, "users": true,
}

// contentTokens lowercases, splits on non-alphanumerics, and drops stopwords + tokens
// of length <=2 — the content-word reference set of a string.
func contentTokens(s string) []string {
	raw := strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return !isAlnum(r)
	})
	out := raw[:0]
	for _, t := range raw {
		if len(t) > 2 && !stopwords[t] {
			out = append(out, t)
		}
	}
	return out
}

func isAlnum(r rune) bool {
	return (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9')
}

// overlap is the count of DISTINCT query content-tokens that appear in doc.
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

// isInjection is a belt-and-suspenders check that a paged-in slice carries none of the
// canonical injection tell. It is NOT the gate (the gate is ctxmmu.Admit, run on every
// page-in); it only asserts the working set stayed clean, so the assertion is the
// debugger's, not a re-implementation of detection.
func isInjection(b []byte) bool {
	s := strings.ToLower(string(b))
	return strings.Contains(s, "ignore previous instructions") ||
		strings.Contains(s, "ignore all previous") ||
		strings.Contains(s, "system override")
}
