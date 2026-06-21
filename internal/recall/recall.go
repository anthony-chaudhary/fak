// Package recall makes a COMPLETED agent session queryable without replaying it —
// the "treat a finished session as a core dump" leaf (../../session-recall-design.md).
//
// A finished session is not a flat transcript: the context-MMU already paged every
// heavy/poisoned tool result out to a content-addressed store AT WRITE TIME. So the
// session is a small page table (roles + digests + descriptors + quarantine state)
// over a frozen content-addressed swap device — a core dump. Answering a follow-up
// should DEMAND-PAGE only the working set the question touches, not execv the whole
// history back in.
//
// recall is a pure CONSUMER of three SHIPPED primitives — the context-MMU's
// write-time quarantine gate (held + witness-Clear), abi.Ref taint provenance, and
// content addressing — wired into a DURABLE, cross-process path the in-process
// primitives did not have. It adds NOTHING to the frozen ABI.
//
// The load-bearing, defensible property (rung 4 — the moat): a slice the gate
// QUARANTINED in the live session stays sealed across the PROCESS boundary. A
// reloaded core image refuses to page a quarantined slice into a NEW context unless
// (a) a witness Clear() ran AND (b) the bytes pass a fresh CONTENT re-screen through
// the same gate. The two gates are independent, so a still-poisoned page can never
// be laundered back in by a clearance alone. No memory system that re-pastes
// transcript bytes has this: naive RAG-over-history re-injects ungated.
package recall

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/canon"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/vdso"

	// Blank import so the context-MMU's page-out backend is registered and the gate
	// operates in its full production configuration during a recall recording, not a
	// degraded fallback. recall uses none of blob's API directly (it carries its own
	// durable CAS); this import only wires the shared backend.
	_ "github.com/anthony-chaudhary/fak/internal/blob"
)

// ManifestVersion is the on-disk core-image format tag.
const ManifestVersion = "recall.v1"

// ErrSealed is returned when a quarantined page is refused page-in. It wraps via
// errors.Is so a caller can branch on "the gate held" vs a plain lookup error.
var ErrSealed = errors.New("recall: page sealed by the trust gate")

// Page is one entry in the persisted page table. For a quarantined page the
// Descriptor carries ONLY safe metadata (tool, reason, length) — never any of the
// sealed bytes — so even the recall index cannot smuggle the poison back into a
// preview/ranker (design-note open question #5, the extractive answer).
type Page struct {
	Step        int    `json:"step"`
	Role        string `json:"role"`       // the tool that produced the result
	Descriptor  string `json:"descriptor"` // a REAL content descriptor (fixes the constant-"oversize"-hint, fact 1)
	Digest      string `json:"digest"`     // CAS address of the page's raw bytes
	Len         int64  `json:"len"`
	Taint       uint8  `json:"taint"` // abi.TaintLabel
	Quarantined bool   `json:"quarantined"`
	QID         string `json:"qid,omitempty"`
	Reason      string `json:"reason,omitempty"`
	Durability  string `json:"durability,omitempty"`  // rung-1 write-time class (#499/#496): turn|session|durable; auditable in manifest.json
	Witness     string `json:"witness,omitempty"`     // external trust witness this page was admitted under
	TrustEpoch  uint64 `json:"trust_epoch,omitempty"` // vdso trust epoch observed at record time
}

// Manifest is the persisted core image of a finished session: the page table, the
// frozen quarantine clearance state, and a frozen world-version marker. The bytes
// themselves live in a sibling cas.json (the swap device).
type Manifest struct {
	Version        string          `json:"version"`
	SessionID      string          `json:"session_id"`
	WorldVer       uint64          `json:"world_ver"` // frozen at persist: a finished session never writes again
	Pages          []Page          `json:"pages"`
	Cleared        map[string]bool `json:"cleared"`
	ContextChanges []ContextChange `json:"context_changes,omitempty"`
}

// ---------------------------------------------------------------------------
// Recording side — drive the SHIPPED gate over a finished session's results and
// capture a durable manifest + CAS.
// ---------------------------------------------------------------------------

// Recorder builds a durable core image from a completed session's tool results. It
// drives a real ctxmmu.MMU so the quarantine decision is the shipped gate's, then
// persists that decision + the raw bytes into its OWN content-addressed store so the
// image survives the process.
type Recorder struct {
	id    string
	mmu   *ctxmmu.MMU
	cas   map[string][]byte
	pages []Page
	qn    int // recall-owned quarantine counter (uniform QID space across both gates)

	promotion         PromotionMode // rung-1 default-expire gate posture (#499/#496)
	refusedPromotions int           // count of non-durable benign pages the gate would/did refuse
}

// PromotionMode is the rung-1 default-expire promotion gate's posture (S7, epic #496).
// The headline inversion — expire by default, promotion is the earned exception — is a
// CODE GATE, not a comment, but it ships in the #497 two-commit honesty split so every
// recall.Recorder caller can be audited before the boundary actually bites.
type PromotionMode uint8

const (
	// PromotionWarn is the audit-only, NON-BEHAVIOR-CHANGING default: classify and
	// STAMP Page.Durability, count a would-refuse for each non-durable benign page,
	// but still persist every page exactly as before. The safe posture while callers
	// (cdb ingest, the gateway memory path, the recall/dream tests) are audited for the
	// eventual enforce flip — see the caller-audit in CONTEXT-IS-NOT-MEMORY.md §S7.
	PromotionWarn PromotionMode = iota
	// PromotionEnforce makes the inversion bite: a benign page is promoted into the
	// durable core image ONLY if it classified `durable`. A turn/session/unknown page
	// never reaches manifest.json/cas.json, so it cannot be recalled in a later
	// process — the security floor against OWASP Memory-Poisoning T1 (a transient
	// injected observation can no longer silently become a persistent behavioral bias).
	PromotionEnforce
)

// NewRecorder starts recording a session by id. The promotion gate defaults to
// PromotionWarn (audit-only) so recording stays non-behavior-changing; opt into the
// enforced default-expire boundary with WithPromotion(PromotionEnforce).
func NewRecorder(sessionID string) *Recorder {
	return &Recorder{id: sessionID, mmu: ctxmmu.New(), cas: map[string][]byte{}}
}

// WithPromotion sets the rung-1 promotion gate posture and returns the recorder for
// chaining: NewRecorder(id).WithPromotion(PromotionEnforce).
func (r *Recorder) WithPromotion(m PromotionMode) *Recorder { r.promotion = m; return r }

// RefusedPromotions reports how many non-durable benign pages the gate refused (under
// PromotionEnforce) or would have refused (the would-refuse audit count under
// PromotionWarn) — the auditable signal of the default-expire inversion.
func (r *Recorder) RefusedPromotions() int { return r.refusedPromotions }

// promotionClass normalizes a raw write-time durability class to the rung-1 vocabulary,
// failing closed to turn for a missing/unknown/reserved value (e.g. an empty key from a
// gate that does not classify, or the reserved `bounded` with no validity home until
// rung 2). This is the reader-side fail-closed default, mirroring abi.FallbackDeny.
func promotionClass(raw string) string {
	switch raw {
	case ctxmmu.DurabilityDurable, ctxmmu.DurabilitySession, ctxmmu.DurabilityTurn:
		return raw
	default:
		return ctxmmu.DurabilityTurn
	}
}

// Record runs one tool result through the write-time gate and appends a page.
// rawBody is exactly what the tool produced (pre any MMU rewrite). The gate is the
// de-obfuscating canon.Scan composed with the shipped ctxmmu MMU, so an obfuscated
// payload (homoglyph/base64/zero-width/...) is sealed at write time with a SAFE
// descriptor — its bytes AND its obfuscated text never reach the recall index. It
// returns the effective verdict so the caller can account for it.
func (r *Recorder) Record(ctx context.Context, tool string, rawBody []byte) abi.Verdict {
	return r.RecordWithWitness(ctx, tool, rawBody, "")
}

// RecordWithWitness records a tool result admitted under an external trust witness.
// The witness is immutable page metadata, but its validity is not: Resolve checks
// the current vDSO revocation ledger before any page-in, so a source refuted after
// persist strands the old CAS bytes without mutating the CAS itself.
func (r *Recorder) RecordWithWitness(ctx context.Context, tool string, rawBody []byte, witness string) abi.Verdict {
	body := append([]byte(nil), rawBody...)
	digest := Digest(body)

	call := &abi.ToolCall{Tool: tool}
	if witness != "" {
		call.Meta = map[string]string{"witness": witness}
	}
	res := &abi.Result{
		Call:    call,
		Payload: abi.Ref{Kind: abi.RefInline, Inline: append([]byte(nil), body...), Len: int64(len(body))},
		Status:  abi.StatusOK,
	}
	v := r.mmu.Admit(ctx, res.Call, res)
	// Defense in depth: the de-obfuscating detector catches what the raw ctxmmu
	// substring/regex match misses. A canon hit is fail-closed to Quarantine.
	if v.Kind != abi.VerdictQuarantine {
		if f := canon.Scan(body); f.Any() {
			reason := abi.ReasonTrustViolation
			if f.Secret {
				reason = abi.ReasonSecretExfil
			}
			v = abi.Verdict{Kind: abi.VerdictQuarantine, Reason: reason, By: "recall/canon",
				Payload: abi.QuarantinePayload{PageOut: true}}
		}
	}

	p := Page{
		Step:   len(r.pages),
		Role:   tool,
		Digest: digest,
		Len:    int64(len(body)),
		Taint:  uint8(abi.TaintTainted),
	}
	if witness != "" {
		p.Witness = witness
		p.TrustEpoch = vdso.Default.TrustEpoch()
	}
	if v.Kind == abi.VerdictQuarantine {
		r.qn++
		p.Quarantined = true
		p.QID = fmt.Sprintf("q%d", r.qn) // recall-owned id, uniform across ctxmmu/canon catches
		p.Reason = abi.ReasonName(v.Reason)
		p.Taint = uint8(abi.TaintQuarantined)
		// SAFE descriptor only — neither the sealed bytes nor their obfuscated text
		// reach the index. A sealed page is ALWAYS recorded (the seal IS the audit
		// record) and its bytes live in CAS, refused on page-in. The durability gate
		// does not apply — sealed bytes never promote.
		p.Descriptor = fmt.Sprintf("%s: [sealed: %s, %d bytes]", tool, p.Reason, len(body))
		r.cas[digest] = body
		r.pages = append(r.pages, p)
		return v
	}

	// Benign page: rung-1 default-expire promotion gate (#499, epic #496). Read the
	// write-time durability class the ctxmmu gate stamped on the verdict; fail-closed
	// to turn for a missing/unknown value (forward-compat, mirrors abi.FallbackDeny).
	class := promotionClass(v.Meta[ctxmmu.DurabilityKey])
	p.Durability = class
	p.Descriptor = descriptorOf(tool, body)

	// Expire by default; promotion is the earned exception — only a `durable`-classed
	// benign fact crosses the durable boundary into the persisted core image.
	if class != ctxmmu.DurabilityDurable {
		r.refusedPromotions++
		if r.promotion == PromotionEnforce {
			// ENFORCE: do NOT promote. The bytes never reach cas.json and the page
			// never reaches manifest.json, so the fact cannot be recalled in a later
			// process. The verdict is still returned for the caller's accounting.
			return v
		}
		// WARN (default): non-behavior-changing — the would-refuse is counted, the
		// page still persists so callers can be audited before the enforce flip.
	}
	r.cas[digest] = body
	r.pages = append(r.pages, p)
	return v
}

// Manifest snapshots the current page table + clearance state.
func (r *Recorder) Manifest() Manifest {
	return Manifest{
		Version:   ManifestVersion,
		SessionID: r.id,
		WorldVer:  uint64(len(r.pages)), // frozen marker; a finished session never writes again
		Pages:     append([]Page(nil), r.pages...),
		Cleared:   r.mmu.Cleared(),
	}
}

// Persist writes the core image (manifest.json + cas.json) under dir. The CAS holds
// every digest the page table references, so the image is fully self-contained and
// reloadable in a fresh process.
func (r *Recorder) Persist(dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	mb, err := json.MarshalIndent(r.Manifest(), "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "manifest.json"), mb, 0o644); err != nil {
		return err
	}
	cb, err := json.MarshalIndent(r.cas, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "cas.json"), cb, 0o644)
}

// ---------------------------------------------------------------------------
// Query side — a reloaded core image with the trust gate enforced on every page-in.
// ---------------------------------------------------------------------------

// Session is a reloaded core image: a frozen page table over a PRIVATE CAS, with the
// quarantine gate enforced on every page-in. It deliberately carries its own CAS
// populated from disk and a FRESH ctxmmu gate, so a Resolve provably does not depend
// on the process that produced the session — the "survives death" proof.
type Session struct {
	Manifest Manifest
	cas      map[string][]byte
	cleared  map[string]bool
	gate     *ctxmmu.MMU // a fresh gate for the rung-4 re-screen on page-in
}

// Load reads a persisted core image and verifies every CAS entry against its digest
// address (a tampered swap device fails closed). The returned Session resolves
// against its own loaded bytes, never the global store.
func Load(dir string) (*Session, error) {
	mb, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		return nil, err
	}
	var m Manifest
	if err := json.Unmarshal(mb, &m); err != nil {
		return nil, fmt.Errorf("recall: bad manifest: %w", err)
	}
	if m.Version != ManifestVersion {
		return nil, fmt.Errorf("recall: manifest version %q != %q", m.Version, ManifestVersion)
	}
	cb, err := os.ReadFile(filepath.Join(dir, "cas.json"))
	if err != nil {
		return nil, err
	}
	var cas map[string][]byte
	if err := json.Unmarshal(cb, &cas); err != nil {
		return nil, fmt.Errorf("recall: bad cas: %w", err)
	}
	// Content-address integrity: a persisted blob MUST hash to its key, or the swap
	// device was tampered with and we refuse to serve any of it.
	for d, b := range cas {
		if Digest(b) != d {
			return nil, fmt.Errorf("recall: corrupt CAS entry %s (digest mismatch)", d)
		}
	}
	cleared := map[string]bool{}
	for id, ok := range m.Cleared {
		cleared[id] = ok
	}
	return &Session{Manifest: m, cas: cas, cleared: cleared, gate: ctxmmu.New()}, nil
}

// Clear records a witness clearance for a quarantined id on this reloaded session.
// It is NECESSARY but NOT SUFFICIENT: Resolve still re-screens the bytes through the
// content gate, so clearing a still-poisoned page does not release it.
func (s *Session) Clear(qid string) { s.cleared[qid] = true }

// Resolve returns the bytes of page `step`, enforcing rung 0 (verbatim re-output)
// and rung 4 (the trust gate):
//   - benign page: re-screened (defense in depth — a tightened pattern set can catch
//     a page that looked benign at write time) and returned BYTE-IDENTICAL;
//   - quarantined page: refused unless a witness Clear() ran AND a fresh content
//     re-screen passes. Either gate failing keeps the page sealed.
func (s *Session) Resolve(ctx context.Context, step int) ([]byte, error) {
	if step < 0 || step >= len(s.Manifest.Pages) {
		return nil, fmt.Errorf("recall: no page %d", step)
	}
	p := s.Manifest.Pages[step]
	if err := s.trustGate(p, step); err != nil {
		return nil, err
	}
	if ch, ok := s.tombstoneFor(step); ok {
		return nil, fmt.Errorf("%w: page %d suppressed by %s (%s)", ErrTombstoned, step, ch.ID, ch.Reason)
	}
	body, ok := s.cas[p.Digest]
	if !ok {
		return nil, fmt.Errorf("recall: page %d bytes (%s) absent from CAS", step, short(p.Digest))
	}
	if p.Quarantined {
		if !s.cleared[p.QID] {
			return nil, fmt.Errorf("%w: page %d (%s) refused — no witness Clear(%q)", ErrSealed, step, p.Reason, p.QID)
		}
		if v := s.reScreen(ctx, p.Role, body); v.Kind == abi.VerdictQuarantine {
			return nil, fmt.Errorf("%w: page %d cleared by witness but the content re-screen RE-QUARANTINED it (%s) — clearance does not launder poison",
				ErrSealed, step, abi.ReasonName(v.Reason))
		}
		return append([]byte(nil), body...), nil
	}
	if v := s.reScreen(ctx, p.Role, body); v.Kind == abi.VerdictQuarantine {
		return nil, fmt.Errorf("%w: page %d was benign at write time but a tightened gate now flags it (%s)",
			ErrSealed, step, abi.ReasonName(v.Reason))
	}
	return append([]byte(nil), body...), nil
}

func (s *Session) trustGate(p Page, step int) error {
	if p.Witness == "" {
		return nil
	}
	if vdso.Default.Revoked(p.Witness) {
		return fmt.Errorf("%w: page %d witness %q revoked (recorded trust_epoch=%d current=%d)",
			ErrSealed, step, p.Witness, p.TrustEpoch, vdso.Default.TrustEpoch())
	}
	return nil
}

// reScreen runs bytes back through TWO independent detectors on page-in:
//
//  1. canon.Scan — the DE-OBFUSCATING canonical detector (the same primitive
//     normgate uses at write time). This is the fix for the "re-screens with the
//     evadable patterns" gap: the session was recorded under a write-time gate
//     (this leaf's bare ctxmmu, or any older kernel) whose RAW substring/regex
//     match is defeated by char-spacing, base64, homoglyph, zero-width, or bidi
//     obfuscation. On reload, today's canonical detector undoes that obfuscation
//     before matching, so a payload that read as "ignore previous instructions"
//     to a model but slipped past the recording gate is RE-CAUGHT at the boundary.
//  2. the ctxmmu gate — oversize / byte-pollution / raw-marker fallbacks.
//
// A canon hit is fail-closed to Quarantine. Otherwise the verdict is the ctxmmu
// gate's: Quarantine means the bytes may not enter; Allow or Transform/page-out
// means they may.
func (s *Session) reScreen(ctx context.Context, tool string, body []byte) abi.Verdict {
	if f := canon.Scan(body); f.Any() {
		reason := abi.ReasonTrustViolation
		if f.Secret {
			reason = abi.ReasonSecretExfil
		}
		return abi.Verdict{Kind: abi.VerdictQuarantine, Reason: reason, By: "recall/canon",
			Payload: abi.QuarantinePayload{PageOut: true}}
	}
	res := &abi.Result{
		Call:    &abi.ToolCall{Tool: tool},
		Payload: abi.Ref{Kind: abi.RefInline, Inline: append([]byte(nil), body...), Len: int64(len(body))},
		Status:  abi.StatusOK,
	}
	// Fold the kernel's REGISTERED ResultAdmitter chain, most-restrictive-wins —
	// the same fold kvmmu.FoldedGate (kvmmu.go) uses. This is the readmission-gate-
	// strength fix (GROWTH.md): a session recorded under a weak write-time gate is
	// re-screened on page-in by EVERY detector the fleet ships now (normgate rank-5,
	// and anything added later), not just this leaf's bare ctxmmu. Most-restrictive-
	// wins means any registered Quarantine seals the page; the bare s.gate stays in
	// the chain as the unconditional rank-10 fallback (oversize / byte-pollution).
	best := s.gate.Admit(ctx, res.Call, res)
	bestRank := abi.FoldRank(best.Kind)
	for _, ra := range abi.ResultAdmittersFor(res.Call) {
		if v := ra.Admit(ctx, res.Call, res); abi.FoldRank(v.Kind) > bestRank {
			bestRank, best = abi.FoldRank(v.Kind), v
		}
	}
	return best
}

// Slice is one paged-in benign page in an assembled working set.
type Slice struct {
	Step       int    `json:"step"`
	Role       string `json:"role"`
	Descriptor string `json:"descriptor"`
	Bytes      []byte `json:"-"`
}

// Recall assembles a small working set for a follow-up question: the top-k BENIGN
// pages ranked by extractive descriptor overlap with the query. Quarantined pages
// are NEVER candidates (their bytes are sealed and their descriptor carries none of
// their content), so the assembled window can never contain a poisoned slice. This
// is the "working set, not the transcript" payoff: a ~k-page window instead of the
// whole history.
func (s *Session) Recall(ctx context.Context, query string, k int) []Slice {
	type scored struct {
		p     Page
		score int
	}
	q := tokenize(query)
	var cand []scored
	for _, p := range s.Manifest.Pages {
		if p.Quarantined || s.Tombstoned(p.Step) {
			continue
		}
		cand = append(cand, scored{p, overlap(q, tokenize(p.Descriptor))})
	}
	sort.SliceStable(cand, func(i, j int) bool { return cand[i].score > cand[j].score })

	var out []Slice
	for i := 0; i < len(cand) && len(out) < k; i++ {
		if cand[i].score == 0 {
			break
		}
		b, err := s.Resolve(ctx, cand[i].p.Step)
		if err != nil {
			continue
		}
		out = append(out, Slice{cand[i].p.Step, cand[i].p.Role, cand[i].p.Descriptor, b})
	}
	return out
}

// Stats summarises a reloaded core image.
type Stats struct {
	Version     string `json:"version"`
	SessionID   string `json:"session_id"`
	Pages       int    `json:"pages"`
	Benign      int    `json:"benign"`
	Quarantined int    `json:"quarantined"`
	Tombstoned  int    `json:"tombstoned"`
	Cleared     int    `json:"cleared"`
	CASBytes    int64  `json:"cas_bytes"`
}

// Stats reports the page/quarantine/byte accounting of a loaded session.
func (s *Session) Stats() Stats {
	st := Stats{Version: s.Manifest.Version, SessionID: s.Manifest.SessionID, Pages: len(s.Manifest.Pages)}
	for _, p := range s.Manifest.Pages {
		if p.Quarantined {
			st.Quarantined++
		} else {
			st.Benign++
		}
		if s.Tombstoned(p.Step) {
			st.Tombstoned++
		}
	}
	for _, c := range s.cleared {
		if c {
			st.Cleared++
		}
	}
	for _, b := range s.cas {
		st.CASBytes += int64(len(b))
	}
	return st
}

// Pages exposes the (read-only) page table for a loaded session.
func (s *Session) Pages() []Page { return append([]Page(nil), s.Manifest.Pages...) }

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

// Digest is the canonical content address (sha256 hex) — the same scheme the blob
// store uses, so a recall digest and a blob digest are interchangeable.
func Digest(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// descriptorOf builds a REAL extractive content descriptor for a benign page: the
// tool plus the first non-empty line, bounded. (Quarantined pages never reach here —
// their descriptor is the safe sealed-metadata form.)
func descriptorOf(tool string, body []byte) string {
	line := firstLine(body, 120)
	if line == "" {
		return tool
	}
	return tool + ": " + line
}

func firstLine(b []byte, max int) string {
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

func tokenize(s string) []string {
	return strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
}

func overlap(query, doc []string) int {
	set := make(map[string]bool, len(query))
	for _, t := range query {
		if len(t) > 2 {
			set[t] = true
		}
	}
	n := 0
	seen := map[string]bool{}
	for _, t := range doc {
		if set[t] && !seen[t] {
			seen[t] = true
			n++
		}
	}
	return n
}

func short(d string) string {
	if len(d) > 12 {
		return d[:12]
	}
	return d
}
