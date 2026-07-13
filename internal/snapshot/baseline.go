package snapshot

// baseline.go — govern the promotion of a golden or statistical baseline.
//
// A baseline (a golden decode series, a statistical accuracy bar) is exactly the kind of
// frozen primitive this package already envelopes: a candidate run produces new bytes and
// wants to REPLACE the pinned golden. That replacement is the dangerous moment. Issue
// #4576 asks for the layer between primitive correctness and coarse end-benchmark scores:
// a promotion contract that a failing run cannot talk its way past.
//
// The one rule the whole file exists to enforce: a baseline is promoted ONLY on a passing
// run with COMPLETE provenance, an INDEPENDENT review, a reason, a diff, a rollback
// pointer, and an explicit CI tier + documented cost. Anything short of that — a failing
// run, a missing field, inconclusive evidence, an un-reviewed change — fails CLOSED with a
// reason drawn from a fixed vocabulary (like the kernel's refusal tokens). A failing case
// does not self-update the baseline; it emits a scrubbed, integrity-checked replay artifact
// that localizes the first actionable divergence so the regression can be independently
// replayed.
//
// This stays inside snapshot's floor: stdlib only, off the request path, adds nothing to
// the frozen ABI. Promotion decisions and replay artifacts are pure values; the replay
// artifact rides the same integrity-stamped envelope as every other dump.

import (
	"fmt"
	"math"
)

// KindBaseline is the registry kind for a governed baseline case; KindBaselineReplay is
// the scrubbed failure bundle a refused promotion emits. Both are registered at init so a
// restore can recognize them and a tool can enumerate them.
const (
	KindBaseline       = "baseline"        // a golden/statistical baseline case (governed promotion)
	KindBaselineReplay = "baseline-replay" // the scrubbed replay artifact a failing case emits
)

// Tier is the CI cost class a baseline case is assigned to — the acceptance criterion
// "assign the case to an explicit PR, nightly, or release tier". The three tiers mirror
// the epic's ladder: cheap deterministic per-PR checks, nightly statistical/accuracy
// suites, and release/hardware qualification.
type Tier string

const (
	TierPR      Tier = "pr"      // cheap, deterministic, per-PR
	TierNightly Tier = "nightly" // statistical / accuracy, nightly
	TierRelease Tier = "release" // hardware / release qualification
)

// validTier reports whether t is one of the three sanctioned tiers.
func validTier(t Tier) bool {
	switch t {
	case TierPR, TierNightly, TierRelease:
		return true
	default:
		return false
	}
}

// RunResult is the tri-state outcome of a baseline case's run. The third state,
// RunInconclusive, is load-bearing: "missing or inconclusive evidence is never pass", so
// an empty or inconclusive result fails closed just like an outright failure. A bool
// could not carry that — the absence of a pass must be distinguishable from a real pass.
type RunResult string

const (
	RunPass         RunResult = "pass"
	RunFail         RunResult = "fail"
	RunInconclusive RunResult = "inconclusive" // missing/ambiguous evidence — never promotes
)

// Provenance is the mandatory identity of a baseline case — every field the acceptance
// criteria require a case to record so a promotion is reproducible and a regression is
// localizable. Seed and Oracle are alternatives: a case pins EITHER a numeric seed OR a
// deterministic oracle id, and at least one must be present.
type Provenance struct {
	Model     string `json:"model"`     // the model under test
	Tokenizer string `json:"tokenizer"` // the tokenizer / vocab revision
	Engine    string `json:"engine"`    // engine/backend (mode)
	Seed      string `json:"seed"`      // sampling seed, OR ...
	Oracle    string `json:"oracle"`    // ... a deterministic oracle id (one of the two required)
	Revision  string `json:"revision"`  // code/module revision (prefer module@rev)
	Tolerance string `json:"tolerance"` // tolerance / baseline provenance (how the bar was set)
	Tier      Tier   `json:"tier"`      // PR / nightly / release
	CostNote  string `json:"cost_note"` // documented runtime / resource cost
}

// Missing returns the names of the provenance fields that are absent, empty if complete.
// Seed/Oracle count as one requirement satisfied by either. Tier and CostNote are checked
// by the promotion governor (they carry their own refusal reasons), so Missing covers only
// the identity fields.
func (p Provenance) Missing() []string {
	var out []string
	if p.Model == "" {
		out = append(out, "model")
	}
	if p.Tokenizer == "" {
		out = append(out, "tokenizer")
	}
	if p.Engine == "" {
		out = append(out, "engine")
	}
	if p.Seed == "" && p.Oracle == "" {
		out = append(out, "seed-or-oracle")
	}
	if p.Revision == "" {
		out = append(out, "revision")
	}
	if p.Tolerance == "" {
		out = append(out, "tolerance")
	}
	return out
}

// PromotionRequest bundles a candidate's run outcome with the governance inputs the Scope
// requires: evidence (the run result), independent review, reason, diff, and a rollback
// pointer. GovernPromotion is the sole judge of whether it may replace the golden.
type PromotionRequest struct {
	Case     Provenance `json:"case"`
	Run      RunResult  `json:"run"`      // the evidence: did the case pass?
	Author   string     `json:"author"`   // who is promoting
	Reviewer string     `json:"reviewer"` // independent review — must be non-empty and != Author
	Reason   string     `json:"reason"`   // why this promotion is correct
	DiffRef  string     `json:"diff_ref"` // the diff under review (SHA / PR / patch id)
	Rollback string     `json:"rollback"` // rollback pointer (prior baseline id / revert SHA)
}

// PromotionOutcome is the binary verdict; a REFUSE always carries a Reason token.
type PromotionOutcome string

const (
	Promote PromotionOutcome = "PROMOTE"
	Refuse  PromotionOutcome = "REFUSE"
)

// The closed vocabulary of refusal reasons. A governed promotion never refuses with
// free-text — a caller switches on these tokens the way the kernel switches on dos.toml
// reasons.
const (
	ReasonBadTier           = "BAD_TIER"           // tier not one of pr/nightly/release
	ReasonNoCost            = "NO_COST"            // runtime/resource cost undocumented
	ReasonMissingProvenance = "MISSING_PROVENANCE" // a required identity field absent
	ReasonUnreviewed        = "UNREVIEWED"         // no independent review (empty or == author)
	ReasonNoReason          = "NO_REASON"          // no stated reason
	ReasonNoDiff            = "NO_DIFF"            // no diff pointer
	ReasonNoRollback        = "NO_ROLLBACK"        // no rollback pointer
	ReasonMissingEvidence   = "MISSING_EVIDENCE"   // run inconclusive/empty — never pass
	ReasonRunFailed         = "RUN_FAILED"         // the case's run failed — cannot self-promote
)

// PromotionDecision is the governor's verdict: an outcome plus, on refusal, the token and
// a human-readable detail. Detail never widens the contract — the Reason token is the
// machine-checkable part.
type PromotionDecision struct {
	Outcome PromotionOutcome `json:"outcome"`
	Reason  string           `json:"reason,omitempty"`
	Detail  string           `json:"detail,omitempty"`
}

// Promoted reports whether the decision permits promotion.
func (d PromotionDecision) Promoted() bool { return d.Outcome == Promote }

// GovernPromotion is the single gate over baseline promotion. It fails CLOSED: the request
// must satisfy every clause — a valid tier, a documented cost, complete provenance, an
// independent review, a reason, a diff, a rollback pointer, and finally CONCLUSIVE PASSING
// evidence — or the first unmet clause refuses with its token. The evidence clause is
// checked LAST so a request that is otherwise complete but whose run failed reports the
// headline RUN_FAILED (criterion #1: "failing runs cannot self-update baselines") rather
// than being masked by an earlier gap.
func GovernPromotion(req PromotionRequest) PromotionDecision {
	if !validTier(req.Case.Tier) {
		return refuse(ReasonBadTier, fmt.Sprintf("tier %q is not one of pr/nightly/release", req.Case.Tier))
	}
	if req.Case.CostNote == "" {
		return refuse(ReasonNoCost, "runtime/resource cost is undocumented (cost_note empty)")
	}
	if miss := req.Case.Missing(); len(miss) > 0 {
		return refuse(ReasonMissingProvenance, fmt.Sprintf("provenance incomplete: missing %v", miss))
	}
	if req.Reviewer == "" || req.Reviewer == req.Author {
		return refuse(ReasonUnreviewed, "promotion needs an independent reviewer distinct from the author")
	}
	if req.Reason == "" {
		return refuse(ReasonNoReason, "promotion needs a stated reason")
	}
	if req.DiffRef == "" {
		return refuse(ReasonNoDiff, "promotion needs a diff pointer")
	}
	if req.Rollback == "" {
		return refuse(ReasonNoRollback, "promotion needs a rollback pointer")
	}
	switch req.Run {
	case RunPass:
		return PromotionDecision{Outcome: Promote}
	case RunFail:
		return refuse(ReasonRunFailed, "the case's run failed — a failing run cannot promote its baseline")
	default: // RunInconclusive or empty — fail closed
		return refuse(ReasonMissingEvidence, "run evidence is missing or inconclusive — never a pass")
	}
}

func refuse(reason, detail string) PromotionDecision {
	return PromotionDecision{Outcome: Refuse, Reason: reason, Detail: detail}
}

// ---------------------------------------------------------------------------
// the deterministic comparator — first actionable divergence
// ---------------------------------------------------------------------------

// Divergence is the first place a candidate series departs from its golden beyond
// tolerance. Index locates it (which token / sample / metric position), so a failure
// points at ONE actionable spot rather than "the run differs".
type Divergence struct {
	Index     int     `json:"index"`
	Golden    float64 `json:"golden"`
	Candidate float64 `json:"candidate"`
	AbsDelta  float64 `json:"abs_delta"`
	Tol       float64 `json:"tol"`
}

// FirstDivergence walks golden vs candidate element-wise and returns the FIRST index whose
// absolute difference exceeds tol. ok=false means the series match within tolerance (no
// divergence). A length mismatch is itself a divergence at the first missing index — a
// truncated/over-long candidate is a real defect, not a pass. NaN is never within
// tolerance (a NaN candidate diverges), so a silent NaN cannot masquerade as a match.
func FirstDivergence(golden, candidate []float64, tol float64) (Divergence, bool) {
	n := len(golden)
	if len(candidate) < n {
		n = len(candidate)
	}
	for i := 0; i < n; i++ {
		d := math.Abs(golden[i] - candidate[i])
		if math.IsNaN(d) || d > tol {
			return Divergence{Index: i, Golden: golden[i], Candidate: candidate[i], AbsDelta: d, Tol: tol}, true
		}
	}
	if len(golden) != len(candidate) {
		// Series match on the common prefix but differ in length — the first missing/extra
		// index is the actionable divergence.
		i := n
		var g, c float64
		if i < len(golden) {
			g = golden[i]
		}
		if i < len(candidate) {
			c = candidate[i]
		}
		return Divergence{Index: i, Golden: g, Candidate: c, AbsDelta: math.NaN(), Tol: tol}, true
	}
	return Divergence{}, false
}

// ---------------------------------------------------------------------------
// the scrubbed replay artifact
// ---------------------------------------------------------------------------

// ReplayArtifact is the bundle a FAILING case emits so the divergence can be independently
// replayed. It is scrubbed BY CONSTRUCTION: it carries only the typed Provenance identity
// (identifiers, not secrets) and the numeric First divergence — never raw prompt or
// response bytes. Note is an operator-supplied, already-scrubbed one-liner; scrubReplay
// redacts anything in it that looks like a credential before the artifact is sealed.
type ReplayArtifact struct {
	Case  Provenance `json:"case"`
	First Divergence `json:"first_divergence"`
	Note  string     `json:"note,omitempty"`
}

// BuildReplay seals a scrubbed replay artifact for a failing case into an integrity-stamped
// snapshot envelope (kind baseline-replay), so the failure bundle is itself a verifiable,
// portable dump. div is the first actionable divergence (from FirstDivergence). It refuses
// to build a replay for a case with no divergence — a "replay" of a passing run would be a
// false artifact.
func BuildReplay(caseID string, prov Provenance, div Divergence, diverged bool, note string, now int64) (Snapshot, error) {
	if !diverged {
		return Snapshot{}, fmt.Errorf("snapshot: BuildReplay on a non-divergent case %q — nothing to replay", caseID)
	}
	art := ReplayArtifact{Case: prov, First: div, Note: scrubReplay(note)}
	return Marshal(KindBaselineReplay, caseID, art, map[string]string{"tier": string(prov.Tier)}, now)
}

// scrubReplay redacts credential-shaped tokens from an operator note. The artifact is
// already secret-free by construction (only typed provenance + numeric divergence); this
// is defense in depth over the one free-text field so a pasted key never rides along.
func scrubReplay(note string) string {
	const redacted = "[REDACTED]"
	out := make([]byte, 0, len(note))
	word := make([]byte, 0, 32)
	flush := func() {
		if looksSecret(string(word)) {
			out = append(out, redacted...)
		} else {
			out = append(out, word...)
		}
		word = word[:0]
	}
	for i := 0; i < len(note); i++ {
		c := note[i]
		if c == ' ' || c == '\t' || c == '\n' {
			flush()
			out = append(out, c)
			continue
		}
		word = append(word, c)
	}
	flush()
	return string(out)
}

// looksSecret flags a token that resembles an API key / bearer token: a long unbroken run
// of key-ish characters, or a known secret prefix. Deliberately conservative — it redacts
// the obvious leaks without mangling ordinary prose or module@rev revisions.
func looksSecret(w string) bool {
	switch {
	case len(w) >= 4 && (w[:3] == "sk-" || w[:3] == "pk-"):
		return true
	case len(w) >= 20 && (hasPrefix(w, "AKIA") || hasPrefix(w, "ghp_") || hasPrefix(w, "Bearer")):
		return true
	case len(w) >= 32 && isKeyish(w):
		return true
	default:
		return false
	}
}

func hasPrefix(s, p string) bool { return len(s) >= len(p) && s[:len(p)] == p }

// isKeyish reports whether every rune is a base64/hex-ish key character — no spaces, no
// punctuation that ordinary prose or a module@rev token would carry.
func isKeyish(s string) bool {
	for i := 0; i < len(s); i++ {
		c := s[i]
		ok := (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '+' || c == '/' || c == '=' || c == '_' || c == '-'
		if !ok {
			return false
		}
	}
	return true
}

// RestoreReplay thaws a baseline-replay snapshot back into its artifact, refusing a
// snapshot of the wrong kind.
func (s Snapshot) RestoreReplay() (ReplayArtifact, error) {
	if s.Kind != KindBaselineReplay {
		return ReplayArtifact{}, fmt.Errorf("snapshot: RestoreReplay on kind %q (want %q)", s.Kind, KindBaselineReplay)
	}
	var a ReplayArtifact
	if err := s.Into(&a); err != nil {
		return ReplayArtifact{}, err
	}
	return a, nil
}

func init() {
	Register(Kind{Name: KindBaseline, Level: 1, Desc: "a golden/statistical baseline case under governed promotion", Typed: true})
	Register(Kind{Name: KindBaselineReplay, Level: 1, Desc: "the scrubbed replay artifact a failing baseline case emits", Typed: true})
}
