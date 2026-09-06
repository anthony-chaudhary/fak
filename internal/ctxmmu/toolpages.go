package ctxmmu

// toolpages.go — tool schemas as content-hashed read-only pages (#2440, part of
// the harness-native program #2387/#2398). The inspiring harness pages tool
// schemas in on demand, but the faulted schema lives in the TRANSCRIPT: history
// compaction lost schemas (type errors), resume hung on oversized deferred
// input, and every re-injection churned cache bytes. The fix is ownership: the
// ToolPageTable — not the transcript — is the tool catalog's home.
//
// Each schema is a read-only page keyed by the hex(sha256) of its bytes. Dedup
// is BY CONTENT HASH, never by tool name, so two versions of the same tool name
// are two distinct pages that can never collide (and identical schemas dedupe
// across sessions for free). Eviction reuses the C3 capability-body pager
// (capbody.go PageOutBody → the abi.PageOut codec seam), so an evicted schema is
// re-faultable through the same witness gate every paged-out body travels —
// evicted, never lost. The per-turn resident set is chosen by DETERMINISTIC
// lexical intent ranking (the internal/selfquery rankCards discipline behind
// fak_capabilities — never a model guess), and PrefixHash commits to a CANONICAL
// splice order so a paging event can never move the outbound prompt prefix and
// tax the provider cache.
//
// Honest bound: a paged-out schema's handle lives in the MMU's FIFO-bounded held
// ledger (DefaultMaxHeld). If churn ages a handle out, Fault returns an ERROR —
// safe and visible, never a silent byte loss; Evict itself refuses
// (keeps the page resident) when no durable CAS handle resolves.

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
)

// ToolPage is the catalog view of one content-hashed read-only tool-schema page.
type ToolPage struct {
	Name string // advertised tool name (ranking input; NOT the page identity)
	Hash string // hex(sha256) of the schema bytes — the page identity and dedup key
	Len  int    // schema length in bytes
}

// toolPageState is the table-internal residency record for one page. body is the
// resident copy; nil means the page is evicted and heldID names the PageOutBody
// handle a Fault re-faults it through.
type toolPageState struct {
	page   ToolPage
	body   []byte
	heldID string
}

// ToolPageTable is the tool catalog's home: a content-addressed page table over
// tool schemas whose eviction path is the MMU's capability-body pager. It is the
// source of truth the transcript is not — compaction can only EVICT a schema
// re-faultably, never lose it.
type ToolPageTable struct {
	mu        sync.Mutex
	mmu       *MMU
	pages     map[string]*toolPageState // hash -> page (identity = content hash)
	dedupHits int64
}

// NewToolPageTable builds a table paging through m (nil falls back to Default).
func NewToolPageTable(m *MMU) *ToolPageTable {
	if m == nil {
		m = Default
	}
	return &ToolPageTable{mmu: m, pages: map[string]*toolPageState{}}
}

// ToolSchemaHash is the page identity: hex(sha256) over the schema bytes. It is
// exported so a caller can address a page it registered without re-registering.
func ToolSchemaHash(schema []byte) string {
	sum := sha256.Sum256(schema)
	return hex.EncodeToString(sum[:])
}

// Register admits a tool schema as a read-only page and returns its content
// hash. A schema whose bytes are already in the table DEDUPES — keyed by hash,
// never by name, so two versions of one tool name are two pages and identical
// bytes under any name are one page (dedup=true, the dedup-hit counter moves,
// the first-seen name is kept). Empty bytes are refused (a page must commit to
// content). The registered copy is private: a caller mutating its slice after
// Register cannot corrupt the read-only page.
func (t *ToolPageTable) Register(name string, schema []byte) (hash string, dedup bool) {
	if t == nil || len(schema) == 0 {
		return "", false
	}
	hash = ToolSchemaHash(schema)
	t.mu.Lock()
	defer t.mu.Unlock()
	if _, ok := t.pages[hash]; ok {
		atomic.AddInt64(&t.dedupHits, 1)
		return hash, true
	}
	body := make([]byte, len(schema))
	copy(body, schema)
	t.pages[hash] = &toolPageState{
		page: ToolPage{Name: name, Hash: hash, Len: len(schema)},
		body: body,
	}
	return hash, false
}

// Evict pages a resident schema out through the MMU's capability-body pager
// (PageOutBody → the abi.PageOut codec seam): the bytes leave the table for the
// shared CAS and the page's entry records the held handle a Fault re-faults
// through. The PAGE STAYS IN THE CATALOG — eviction changes residency, never
// membership, which preserves the rule that compaction can only evict, never lose bytes.
// An unknown hash, an already-evicted page, or a pager
// that cannot mint a durable handle returns false and the page stays resident.
func (t *ToolPageTable) Evict(ctx context.Context, hash string) bool {
	if t == nil {
		return false
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	st, ok := t.pages[hash]
	if !ok || st.body == nil {
		return false
	}
	id, paged := t.mmu.PageOutBody(ctx, st.body)
	if !paged {
		return false // no durable handle — keep the bytes resident, never drop them
	}
	st.heldID = id
	st.body = nil
	return true
}

// Fault returns a page's schema bytes. A resident page returns a copy directly;
// an evicted page RE-FAULTS through the witness gate (Clear + PageInBody — the
// table is the residency owner, so its own eviction is its own witness) and the
// restored bytes are re-verified against the content hash before re-admission,
// so a read-only page can never come back different. The re-faulted page becomes
// resident again. Unknown hashes and un-restorable handles error — fail-closed
// and visible, never a silent loss.
func (t *ToolPageTable) Fault(ctx context.Context, hash string) ([]byte, error) {
	if t == nil {
		return nil, fmt.Errorf("ctxmmu: nil tool page table")
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	st, ok := t.pages[hash]
	if !ok {
		return nil, fmt.Errorf("ctxmmu: no tool page %s", hash)
	}
	if st.body == nil {
		t.mmu.Clear(st.heldID)
		body, err := t.mmu.PageInBody(ctx, st.heldID)
		if err != nil {
			return nil, fmt.Errorf("ctxmmu: tool page %s re-fault: %w", hash, err)
		}
		if ToolSchemaHash(body) != hash {
			return nil, fmt.Errorf("ctxmmu: tool page %s re-faulted with mismatched content", hash)
		}
		st.body = body
		st.heldID = ""
	}
	out := make([]byte, len(st.body))
	copy(out, st.body)
	return out, nil
}

// Pages returns the whole catalog — resident AND evicted pages — in canonical
// (Name, Hash) order. Membership, not residency: an evicted page is still here.
func (t *ToolPageTable) Pages() []ToolPage {
	if t == nil {
		return nil
	}
	t.mu.Lock()
	out := make([]ToolPage, 0, len(t.pages))
	for _, st := range t.pages {
		out = append(out, st.page)
	}
	t.mu.Unlock()
	sortToolPages(out)
	return out
}

// ResidentSet chooses the per-turn resident set: the catalog ranked against the
// turn's intent by DETERMINISTIC lexical scoring (the internal/selfquery
// rankCards discipline — token hits on the tool name, exact > part > substring —
// never a model guess), truncated to max (max <= 0 keeps all). Ties break on
// canonical (Name, Hash) order over a canonically pre-sorted catalog, so the
// same inputs always yield the same set in the same order, regardless of
// registration order or paging history. When no page scores (or the intent is
// empty) it falls back to the full canonical catalog — a turn never loses its
// whole tool surface to an off-vocabulary intent. Pure selection: it reads the
// catalog and moves no residency.
func (t *ToolPageTable) ResidentSet(intent string, max int) []ToolPage {
	pages := t.Pages() // canonical pre-order → deterministic ties
	toks := intentTokens(intent)
	if len(toks) > 0 {
		type hit struct {
			page  ToolPage
			score int
		}
		hits := make([]hit, 0, len(pages))
		for _, p := range pages {
			if s := toolPageScore(p, toks); s > 0 {
				hits = append(hits, hit{p, s})
			}
		}
		if len(hits) > 0 {
			sort.SliceStable(hits, func(i, j int) bool { return hits[i].score > hits[j].score })
			pages = make([]ToolPage, len(hits))
			for i, h := range hits {
				pages[i] = h.page
			}
		}
	}
	if max > 0 && len(pages) > max {
		pages = pages[:max]
	}
	return pages
}

// PrefixHash commits to the outbound tool-def splice for a resident set: the
// pages' (name, content-hash) pairs folded in CANONICAL splice order — ascending
// (Name, Hash), independent of both ranking order and paging history. Because
// each page is read-only (Fault re-verifies content against the hash) and the
// order is canonical, an evict/re-fault cycle between turns can never move this
// hash — the prompt-prefix the provider cached survives every paging event. The
// input slice is not mutated.
func PrefixHash(pages []ToolPage) string {
	ordered := append([]ToolPage(nil), pages...)
	sortToolPages(ordered)
	h := sha256.New()
	for _, p := range ordered {
		h.Write([]byte(p.Name))
		h.Write([]byte{0})
		h.Write([]byte(p.Hash))
		h.Write([]byte{'\n'})
	}
	return hex.EncodeToString(h.Sum(nil))
}

// ResidentBytes reports the bytes currently RESIDENT in the table (evicted pages
// contribute 0 — their bytes live in the CAS). The /metrics gauge behind
// tool_schema_resident_bytes.
func (t *ToolPageTable) ResidentBytes() int64 {
	if t == nil {
		return 0
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	var n int64
	for _, st := range t.pages {
		n += int64(len(st.body))
	}
	return n
}

// DedupHits reports the lifetime count of Register calls that deduped to an
// existing content-hashed page. The /metrics counter behind
// tool_page_dedup_hits_total.
func (t *ToolPageTable) DedupHits() int64 {
	if t == nil {
		return 0
	}
	return atomic.LoadInt64(&t.dedupHits)
}

// Len reports the catalog size (resident + evicted pages).
func (t *ToolPageTable) Len() int {
	if t == nil {
		return 0
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	return len(t.pages)
}

// Resident reports whether the page for hash is currently resident (false for
// an evicted page and for an unknown hash — membership is Pages()'s question).
func (t *ToolPageTable) Resident(hash string) bool {
	if t == nil {
		return false
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	st, ok := t.pages[hash]
	return ok && st.body != nil
}

// sortToolPages orders pages canonically: ascending Name, then Hash. This is the
// one splice/tie order every deterministic surface here shares.
func sortToolPages(pages []ToolPage) {
	sort.Slice(pages, func(i, j int) bool {
		if pages[i].Name != pages[j].Name {
			return pages[i].Name < pages[j].Name
		}
		return pages[i].Hash < pages[j].Hash
	})
}

// intentTokens lowercases and splits an intent on non-alphanumerics — the same
// cheap lexical tokenization the selfquery ranking discipline uses.
func intentTokens(intent string) []string {
	return strings.FieldsFunc(strings.ToLower(intent), func(r rune) bool {
		return (r < 'a' || r > 'z') && (r < '0' || r > '9')
	})
}

// toolPageScore scores one page against the intent tokens: exact-name hit 12,
// name-part hit 8 (tool names split on '_'/'-', the toolTags convention),
// substring hit 5. Pure integer lexical scoring — deterministic by construction.
func toolPageScore(p ToolPage, toks []string) int {
	name := strings.ToLower(p.Name)
	parts := strings.FieldsFunc(name, func(r rune) bool { return r == '_' || r == '-' })
	total := 0
	for _, tk := range toks {
		if name == tk {
			total += 12
		}
		for _, part := range parts {
			if part == tk {
				total += 8
				break
			}
		}
		if strings.Contains(name, tk) {
			total += 5
		}
	}
	return total
}
