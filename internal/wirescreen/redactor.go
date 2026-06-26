package wirescreen

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"regexp"
	"sort"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// redactor.go is the pre-send PII/secret redaction rung of the local-model-on-the-
// wire spine (doc.go RUNG 5; issue #572). It is a Screener-STYLE sibling proposer —
// same witnessed-lossy-proposer contract (bounded by the original pinned in CAS, so
// a wrong proposal costs one demand-page fault, never a lost fact; strictly
// additive and one-sided; default-inert) — but it emits SPANS rather than a
// quarantine bit, because a redaction is an in-place rewrite (the rest of the body
// stays; only the secret span is replaced with a placeholder), not a whole-result
// hold-out.
//
// What it does: a Redactor proposes [start,end) byte spans that should be redacted
// before bytes leave the box; Apply replaces each span with a placeholder and pins
// the UNREDACTED original in the shared CAS so an authorized Restore returns it
// byte-exact. The reference piiRedactor here is a deterministic, dependency-free
// COMPLIANCE FLOOR — high-precision regex + Luhn detection of common PII/secret
// shapes (the zero-model floor the rung handoff explicitly blesses). The real value
// is a model-backed Redactor registered under "model" in a gated follow-on.
//
// Default-inert: with FAK_WIRE_REDACT unset, ActiveRedactor() returns nil and no
// redaction ever runs. The leaf touches no ABI seam (it registers no SemanticScreen,
// no capability), so the default binary is unchanged and TestABIGoldenFreeze /
// TestDefaultInertRegistersNoABIScreen are unaffected.
//
// Honest scope (the rung handoff, docs/notes/PROMPTS-local-model-on-the-wire-next
// -agents-2026-06-23.md): this is a compliance floor, NOT a token saver. The
// flagship `fak guard -- claude` Anthropic passthrough sends req.Raw VERBATIM, so a
// byte-rewrite rung changes nothing the model reads there until the cache-prefix-
// preserving req.Raw transform (#555, ctxplan-owned) lands. The redactor is the
// deterministic floor READY to be wired into that transform (and into the
// non-passthrough QuarantineOutboundMessages-style re-marshal); it cannot yet affect
// the live wire. The model arm + the outbound wiring are the named, #555-gated
// follow-ons — exactly the fence the ctxplan forecast AUTHOR (CLAIMS.md) shipped
// under.

// Redactor is the extension point a concrete redaction proposer implements. It is a
// LOSSY proposer bounded by a witness (Apply pins the original in CAS): it proposes
// spans to redact, never a decision the system trusts. A Redactor is NEVER trusted
// to be correct — a false positive is recoverable (Restore returns the exact bytes)
// and a false negative degrades to the unredacted body. It is strictly one-sided:
// the spans it proposes may only cause bytes to be REMOVED (replaced with a
// placeholder), never injected, reordered, or weakened.
type Redactor interface {
	// Name identifies the redactor for FAK_WIRE_REDACT selection and audit.
	Name() string
	// Propose returns the disjoint [Start,End) byte spans in body that should be
	// redacted before bytes leave the box. tool is the producing tool / message
	// role context (may be empty). The spans MUST be within bounds; Apply coalesces
	// them to be disjoint. A redactor that finds nothing returns nil.
	Propose(ctx context.Context, body []byte, tool string) []Span
}

// Span is one proposed redaction: a half-open byte range [Start,End) plus a short
// audit Kind ("credit_card", "us_ssn", "api_key", ...).
type Span struct {
	Start int    `json:"start"`
	End   int    `json:"end"`
	Kind  string `json:"kind"`
}

var (
	rmu       sync.RWMutex
	rregistry = map[string]Redactor{}

	ractive         Redactor // the FAK_WIRE_REDACT-selected redactor (nil = inert)
	ractiveResolved bool

	redactions int64 // lifetime count of spans redacted (observability)
)

// RegisterRedactor adds a named Redactor to the catalog. A leaf — the pii reference
// here, or a model-backed redactor in a follow-on — registers itself from init();
// the operator selects one with FAK_WIRE_REDACT=<name>. Last-write-wins per name,
// matching the Screener / RegionBackend idiom.
func RegisterRedactor(name string, r Redactor) {
	rmu.Lock()
	defer rmu.Unlock()
	rregistry[name] = r
}

// ActiveRedactor returns the Redactor selected by FAK_WIRE_REDACT, or nil when
// unset/unknown (the inert default). Resolution is lazy and once-only so selection
// is robust to init() ordering across files. This is the redaction peer of
// Active() (the screener).
func ActiveRedactor() Redactor {
	rmu.Lock()
	defer rmu.Unlock()
	if !ractiveResolved {
		ractive = rregistry[strings.TrimSpace(os.Getenv("FAK_WIRE_REDACT"))] // nil if unset/unknown
		ractiveResolved = true
	}
	return ractive
}

// SetActiveRedactorForTest forces the FAK_WIRE_REDACT selection to the named
// registered redactor ("" resolves to inert/nil), bypassing the one-shot env
// resolution, and returns a restore func that puts the prior selection back. It
// lets a cross-package test (e.g. the agent outbound-wire path) exercise the
// active-redactor branch deterministically without depending on FAK_WIRE_REDACT
// or on init-resolution order. Test-support only — production code selects via
// ActiveRedactor; the package's own tests poke ractive/ractiveResolved directly.
func SetActiveRedactorForTest(name string) (restore func()) {
	rmu.Lock()
	defer rmu.Unlock()
	prev, prevResolved := ractive, ractiveResolved
	ractive, ractiveResolved = rregistry[strings.TrimSpace(name)], true
	return func() {
		rmu.Lock()
		defer rmu.Unlock()
		ractive, ractiveResolved = prev, prevResolved
	}
}

// Redactions reports how many spans this leaf has redacted over its lifetime — the
// redaction peer of Flags() (the screener) and ctxmmu.MMU.Screened().
func Redactions() int64 { return atomic.LoadInt64(&redactions) }

// PIIRedactor returns the deterministic reference redactor (the zero-model floor),
// independent of the FAK_WIRE_REDACT selection. It lets a caller apply the floor
// directly — e.g. a non-passthrough re-marshal path that wants compliance hygiene
// without opting the whole spine in — and gives tests a handle without env coupling.
func PIIRedactor() Redactor { return piiRedactor{} }

// Redaction is the result of Apply: the redacted body, a CAS handle to the
// unredacted original (byte-exact restore via Restore), the spans that were
// redacted, and the redactor that proposed them.
type Redaction struct {
	Redacted []byte  // body with each proposed span replaced by "[REDACTED:<kind>]"
	Original abi.Ref // CAS handle to the UNREDACTED original (Restore returns it byte-exact)
	Spans    []Span  // the disjoint spans that were redacted (audit)
	By       string  // the redactor name
}

// Apply runs a proposed redaction in place. It asks r for spans, and if r proposes
// any it (1) pins the UNREDACTED original in the shared CAS so an authorized Restore
// returns it byte-exact, then (2) replaces each span with a "[REDACTED:<kind>]"
// placeholder. It is strictly one-sided: it only REMOVES bytes (a span -> a short
// placeholder), never injects or rewrites anything outside the proposed spans, so
// the witness invariant holds — a wrong proposal costs one demand-page fault, never
// a lost fact.
//
// ok is false (and Redaction.Redacted == body, Original empty) when r proposed no
// spans OR when no CAS page-out backend is registered to witness the original: the
// spine's founding contract is that redaction MUST be reversible, so Apply refuses
// to redact when it cannot pin the original rather than silently dropping bytes. An
// operator who opted in (FAK_WIRE_REDACT) links the full defconfig, which registers
// the "blob" CAS backend, so the witness is present whenever a redactor is active.
func Apply(ctx context.Context, r Redactor, body []byte, tool string) (Redaction, bool) {
	if r == nil || len(body) == 0 {
		return Redaction{Redacted: body}, false
	}
	spans := coalesce(r.Propose(ctx, body, tool), len(body))
	if len(spans) == 0 {
		return Redaction{Redacted: body}, false
	}
	handle, ok := pinOriginal(ctx, body)
	if !ok {
		return Redaction{Redacted: body}, false // no witness -> refuse (never drop bytes unreverseibly)
	}
	var out bytes.Buffer
	out.Grow(len(body))
	prev := 0
	for _, s := range spans {
		out.Write(body[prev:s.Start])
		fmt.Fprintf(&out, "[REDACTED:%s]", s.Kind)
		prev = s.End
		atomic.AddInt64(&redactions, 1)
	}
	out.Write(body[prev:])
	return Redaction{Redacted: out.Bytes(), Original: handle, Spans: spans, By: r.Name()}, true
}

// Restore returns the UNREDACTED original for a handle Apply returned, byte-exact.
// It is the witness-layer reversal: the caller that holds the handle (an operator /
// audit path) decides authorization, exactly as the MMU's gated PageIn requires a
// witness Clear() before it resolves a quarantined result. The redactor provides the
// MECHANISM; the caller provides the POLICY.
func Restore(ctx context.Context, handle abi.Ref) ([]byte, error) {
	b, ok := abi.PageOut(redactCodecID())
	if !ok {
		return nil, fmt.Errorf("wirescreen: no page-out backend registered for redaction restore")
	}
	ref, err := b.PageIn(ctx, handle)
	if err != nil {
		return nil, err
	}
	return ref.Inline, nil
}

// pinOriginal pages the unredacted body out to the shared CAS and pins it, mirroring
// ctxmmu.MMU.quarantineResult's pageOut + abi.PinResolved so the original survives a
// bounded-backend eviction until Restore resolves it.
func pinOriginal(ctx context.Context, body []byte) (abi.Ref, bool) {
	b, ok := abi.PageOut(redactCodecID())
	if !ok {
		return abi.Ref{}, false
	}
	inline := abi.Ref{Kind: abi.RefInline, Inline: body, Len: int64(len(body))}
	handle, err := b.PageOut(ctx, inline)
	if err != nil {
		return abi.Ref{}, false
	}
	abi.PinResolved(handle)
	return handle, true
}

// redactCodecID is the CAS codec redaction originals page through. It defaults to
// "blob" (the MMU's default) and honors FAK_PAGEOUT_BACKEND, so an operator who
// spilled quarantine to a durable codec (e.g. "blobfs") spills redaction originals
// to the SAME store — the originals live alongside the quarantined bytes they are
// the dual of.
func redactCodecID() string {
	if id := os.Getenv("FAK_PAGEOUT_BACKEND"); id != "" {
		return id
	}
	return "blob"
}

// coalesce sorts spans by start (longer-first at a tie) and drops overlaps so Apply
// receives a disjoint set. Greedy by start keeps the earlier-starting span on a
// nested overlap (a PEM private-key block wins over an email embedded inside it),
// which is the conservative direction — it redacts the LARGER secret.
func coalesce(in []Span, bodyLen int) []Span {
	if len(in) == 0 {
		return nil
	}
	sort.Slice(in, func(i, j int) bool {
		if in[i].Start != in[j].Start {
			return in[i].Start < in[j].Start
		}
		return in[i].End > in[j].End
	})
	out := make([]Span, 0, len(in))
	lastEnd := -1
	for _, s := range in {
		if s.Start < 0 || s.End > bodyLen || s.Start >= s.End {
			continue
		}
		if s.Start >= lastEnd {
			out = append(out, s)
			lastEnd = s.End
		}
	}
	return out
}

func init() {
	// The deterministic reference redactor is always in the catalog so
	// FAK_WIRE_REDACT=pii works out of the box, but it is INERT unless selected.
	RegisterRedactor("pii", piiRedactor{})
}

// ---------------------------------------------------------------------------
// Reference floor: piiRedactor (deterministic, dependency-free).
// ---------------------------------------------------------------------------

// piiRedactor is the deterministic, dependency-free reference Redactor — the
// zero-model compliance floor. It finds common PII/secret shapes via anchored,
// high-precision regexes (and Luhn validation for credit cards) and returns their
// spans. It is deliberately PRECISION-biased: a compliance floor that redacts a
// false positive breaks legit content reversibly (Restore fixes it) but erodes
// trust, so the patterns require strong shape evidence (canonical key prefixes,
// strict digit groupings, Luhn) rather than loose digit runs.
//
// Relationship to ctxmmu's regex floor: the inbound ScreenBytes floor QUARANTINES
// (removes entirely) a tool RESULT bearing a secret shape (sk-/AKIA/PRIVATE KEY/
// ghp/xox). The redactor's distinct value is (a) the OUTBOUND pre-send path, where
// the same shapes appear in user/assistant content the inbound floor never sees,
// and (b) PII the inbound floor does not block (credit cards, SSNs, emails). It is
// the floor for the outbound surface, not a duplicate of the inbound quarantine.
type piiRedactor struct{}

// Name returns "pii", the reference redactor's selection/audit id.
func (piiRedactor) Name() string { return "pii" }

// piiPatterns are the anchored, high-precision PII/secret shapes. Each carries its
// audit Kind. Credit cards are handled separately (candidate run + Luhn).
var piiPatterns = []struct {
	kind string
	re   *regexp.Regexp
}{
	{"private_key", regexp.MustCompile(`-----BEGIN (?:[A-Z ]+)?PRIVATE KEY-----[\s\S]*?-----END (?:[A-Z ]+)?PRIVATE KEY-----`)},
	{"aws_access_key", regexp.MustCompile(`\bAKIA[0-9A-Z]{16}\b`)},
	{"github_token", regexp.MustCompile(`\bghp_[A-Za-z0-9]{36}\b`)},
	{"slack_token", regexp.MustCompile(`\bxox[baprs]-[0-9A-Za-z-]{10,}\b`)},
	{"stripe_key", regexp.MustCompile(`\bsk_(?:live|test)_[A-Za-z0-9]{16,}\b`)},
	{"google_api_key", regexp.MustCompile(`\bAIza[0-9A-Za-z_-]{35}\b`)},
	{"us_ssn", regexp.MustCompile(`\b\d{3}-\d{2}-\d{4}\b`)},
	{"email", regexp.MustCompile(`\b[A-Za-z0-9._%+-]+@[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)+\b`)},
}

// bearerRe matches an Authorization-header bearer token (the clear secret-in-header
// shape). Case-insensitive on the scheme.
var bearerRe = regexp.MustCompile(`(?i)\bbearer\s+[A-Za-z0-9._-]{20,}`)

// ccCandidate matches a 13-19 digit run with optional single space/dash separators
// between digits; luhnValid + boundary checks filter it to real card numbers.
var ccCandidate = regexp.MustCompile(`(?:\d[ -]?){12,18}\d`)

// Propose returns the disjoint spans in body matching the high-precision PII/secret patterns (keys, tokens, SSN, email, bearer, and Luhn-validated credit cards).
func (piiRedactor) Propose(_ context.Context, body []byte, _ string) []Span {
	var spans []Span
	for _, p := range piiPatterns {
		for _, idx := range p.re.FindAllIndex(body, -1) {
			spans = append(spans, Span{Start: idx[0], End: idx[1], Kind: p.kind})
		}
	}
	for _, idx := range bearerRe.FindAllIndex(body, -1) {
		spans = append(spans, Span{Start: idx[0], End: idx[1], Kind: "bearer_token"})
	}
	for _, m := range ccCandidate.FindAllIndex(body, -1) {
		if m[0] > 0 && isDigitByte(body[m[0]-1]) {
			continue // not a boundary: mid-number substring
		}
		if m[1] < len(body) && isDigitByte(body[m[1]]) {
			continue
		}
		digits := digitsOnly(body[m[0]:m[1]])
		if len(digits) < 13 || len(digits) > 19 || !luhnValid(digits) {
			continue
		}
		spans = append(spans, Span{Start: m[0], End: m[1], Kind: "credit_card"})
	}
	return coalesce(spans, len(body))
}

func isDigitByte(b byte) bool { return b >= '0' && b <= '9' }

func digitsOnly(b []byte) []byte {
	out := make([]byte, 0, len(b))
	for _, c := range b {
		if c >= '0' && c <= '9' {
			out = append(out, c)
		}
	}
	return out
}

// luhnValid reports whether digits is a valid Luhn checksum (the canonical
// credit-card precision check). digits must be all ASCII digits.
func luhnValid(digits []byte) bool {
	sum, double := 0, false
	for i := len(digits) - 1; i >= 0; i-- {
		d := int(digits[i] - '0')
		if d < 0 || d > 9 {
			return false
		}
		if double {
			d *= 2
			if d > 9 {
				d -= 9
			}
		}
		sum += d
		double = !double
	}
	return sum%10 == 0
}
