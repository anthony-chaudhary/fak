package wirescreen

import (
	"context"
	"os"
	"strings"
	"sync"
	"sync/atomic"
)

// digester.go is the rung-3 useful-page-out proposer of the local-model-on-the-wire
// spine (doc.go RUNG 2; issue #570). It is a Screener/Redactor-STYLE sibling — same
// witnessed-lossy-proposer contract (bounded by the original pinned in CAS, so a wrong
// proposal costs one demand-page fault, never a lost fact; strictly additive and
// one-sided; default-inert) — but it emits a DIGEST string rather than a quarantine bit
// or redaction spans, because useful page-out is a lossy display that REPLACES the
// context-MMU's opaque oversize pointer with a summary the model can read.
//
// What it does: a Digester authors a short (~200-token) summary of a body the regex
// floor admitted. The screenAdapter (wirescreen.go) maps a non-empty summary onto an
// abi.ScreenAdvice{Disposition: abi.ScreenDigest, Digest: ...}; ctxmmu.MMU.Admit then
// pages the oversize body out to a stub that carries the digest, with the original
// retained in CAS and a witness Clear + PageIn restoring it byte-exact. The reference
// heuristicDigester here is a deterministic, dependency-free stand-in (leading-lines
// truncation). The real value is a model-backed Digester on the native CPU path,
// registered under "model" in a gated follow-on (needs weights + a measured digest
// latency before it can default on) — see doc.go.
//
// Default-inert: with FAK_WIRE_SCREEN unset, ActiveDigester() returns nil and no digest
// is ever authored. The leaf touches no ABI seam itself (the screenAdapter, registered
// only when FAK_WIRE_SCREEN is set, is what consults it), so the default binary's
// abi.SemanticScreens() stays empty and TestDefaultInertRegistersNoABIScreen is
// unaffected.
//
// Honest scope (the rung handoff, docs/notes/PROMPTS-local-model-on-the-wire-next
// -agents-2026-06-23.md): the digest reaches the model only on the NON-passthrough
// re-marshal path; on the flagship `fak guard -- claude` passthrough the model reads
// req.Raw verbatim, so the digest is dead there until the cache-prefix-preserving
// req.Raw transform (#555, ctxplan-owned) lands. The ctxmmu behaviour + the witness
// test ship now; the flagship-route value is #555-gated.

// Digester is the extension point a concrete digest proposer implements. It is a LOSSY
// proposer bounded by the witness the context-MMU enforces (the original is pinned in
// CAS and PageIn-after-Clear restores it byte-exact): it proposes a summary, never a
// decision the system trusts. A Digester is NEVER trusted to be correct — a wrong
// summary is recoverable (the operator pages the original back in) and a miss degrades
// to the opaque oversize pointer. It is strictly one-sided: it may only ADD information
// to the stub (a digest), never weaken or drop the CAS ref the witness needs.
type Digester interface {
	// Name identifies the digester for audit (it rides the same FAK_WIRE_SCREEN gate as
	// the Screener, so the adapter composes the audit trail from both).
	Name() string
	// Summarize authors a short (~200-token) digest of body (which SURVIVED the regex
	// floor and was NOT flagged for quarantine). tool is the producing tool name (may be
	// empty). ok is false (and digest empty) when the digester declines (e.g. an empty
	// body, or a model digester that is not confident enough to summarize).
	Summarize(ctx context.Context, body []byte, tool string) (digest string, ok bool)
}

var (
	dmu             sync.RWMutex
	dregistry       = map[string]Digester{}
	dactive         Digester // the FAK_WIRE_SCREEN-selected digester (nil = inert)
	dactiveResolved bool

	digests int64 // lifetime count of digests authored (observability)
)

// RegisterDigester adds a named Digester to the catalog. A leaf — the heuristic
// reference here, or a model-backed digester in a follow-on — registers itself from
// init(); the operator selects one with FAK_WIRE_SCREEN=<name> (the SAME gate as the
// Screener, so one opt-in activates both rungs). Last-write-wins per name, matching the
// Screener / Redactor / RegionBackend idiom.
func RegisterDigester(name string, d Digester) {
	dmu.Lock()
	defer dmu.Unlock()
	dregistry[name] = d
}

// ActiveDigester returns the Digester selected by FAK_WIRE_SCREEN, or nil when
// unset/unknown (the inert default). It shares the Screener's selection env so a single
// opt-in (e.g. FAK_WIRE_SCREEN=heuristic) activates both the rung-1 semantic screen and
// the rung-3 useful-page-out digest. Resolution is lazy and once-only so selection is
// robust to init() ordering across files — the same fence Active() uses.
func ActiveDigester() Digester {
	dmu.Lock()
	defer dmu.Unlock()
	if !dactiveResolved {
		dactive = dregistry[strings.TrimSpace(os.Getenv("FAK_WIRE_SCREEN"))] // nil if unset/unknown
		dactiveResolved = true
	}
	return dactive
}

// Digests reports how many digests this leaf has authored over its lifetime — the
// digest peer of Flags() (the screener), Redactions() (the redactor), and
// ctxmmu.MMU.Digested() (the page-outs the MMU actually used).
func Digests() int64 { return atomic.LoadInt64(&digests) }

// ---------------------------------------------------------------------------
// Reference floor: heuristicDigester (deterministic, dependency-free).
// ---------------------------------------------------------------------------

// digestMaxBytes caps the authored digest. ~200 tokens at ≈4 BPE bytes/token (English)
// is the rung-3 envelope (PROMPTS doc); the context-MMU's PointerMax (2KB) is the hard
// ceiling the stub must fit under, so a digest plus the JSON framing stays well inside.
const digestMaxBytes = 800

// heuristicDigester is the deterministic, dependency-free reference Digester. It is NOT
// a model — it exists to PROVE the wiring end to end (a ScreenDigest advisory produces a
// digest-bearing stub instead of an opaque pointer) and to give an operator a
// zero-dependency opt-in via FAK_WIRE_SCREEN=heuristic. It authors the leading distinct
// non-blank lines of the body, truncated to the ~200-token cap, so the model reads the
// opening of the paged-out result without a demand-page fault. The real semantic digest
// is the model-backed Digester (a gated follow-on); this is the floor for the floor.
type heuristicDigester struct{}

func (heuristicDigester) Name() string { return "heuristic" }

func (heuristicDigester) Summarize(_ context.Context, body []byte, _ string) (string, bool) {
	if len(body) == 0 {
		return "", false
	}
	var sb strings.Builder
	for _, raw := range strings.Split(string(body), "\n") {
		line := strings.TrimSpace(raw)
		if line == "" {
			continue
		}
		if sb.Len() > 0 {
			sb.WriteByte('\n')
		}
		if rem := digestMaxBytes - sb.Len(); len(line) > rem {
			sb.WriteString(line[:rem])
			break
		}
		sb.WriteString(line)
		if sb.Len() >= digestMaxBytes {
			break
		}
	}
	if sb.Len() == 0 {
		return "", false
	}
	atomic.AddInt64(&digests, 1)
	return sb.String(), true
}
