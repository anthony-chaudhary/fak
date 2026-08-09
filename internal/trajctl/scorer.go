package trajctl

import (
	"errors"
	"fmt"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/sessionaudit"
)

// scorer.go — issue #2536, spine step 3 of the trajectory-control epic (#2533):
// the Scorer interface plus the versioned registry that is the extension seam.
//
// A Scorer folds an Objective and an evidence window into ScoreRows. It is a PURE
// data-plane fold: the window carries the claimed evidence and an INJECTED
// resolver (the same EvidenceResolver the audit fold uses), so the impure
// resolvers (git, transcript, verdict-hash) live at the call site and this
// package stays deterministic and tier-1. Registering a scorer is the one-step
// act that adds a method; the method name and version travel in every row it
// emits so a consumer can always tell which method and which version produced a
// value.

// EvidenceWindow is the evidence a scorer folds for one scoring pass. It is a
// plain data carrier assembled by the caller (phase→commit bindings from commit
// trailers or dos-verify) plus the resolver that confirms each pointer. Keeping
// the window pure mirrors the audit fold: the deterministic scoring lives here,
// the host-dependent resolution is injected.
type EvidenceWindow struct {
	// PhaseCommits maps a plan phase id to the candidate commit SHAs claimed to
	// resolve it. A phase with no entry — or whose candidate commits all fail to
	// verify — has made no witnessed progress.
	PhaseCommits map[string][]string
	// PriorScores is the existing curve for the objective set, in append order.
	// Behavioral scorers use it to detect high-activity/flat-progress divergence
	// without reading the ledger themselves.
	PriorScores []ScoreRow
	// Sessions are already-analyzed sessionaudit rows for the current evidence
	// window. The parser remains in internal/sessionaudit; trajctl only folds the
	// structured signals.
	Sessions []sessionaudit.Session
	// Resolve confirms one evidence pointer, reused from the audit fold. A nil
	// resolver treats every pointer as unknown, so a missing resolver scores 0
	// rather than silently crediting unverified work (fail-closed).
	Resolve EvidenceResolver
	// UnixMillis stamps the produced rows; 0 leaves the stamp unset so the ledger
	// append path can stamp instead. Injected so tests stay deterministic.
	UnixMillis int64
}

// Scorer computes progress rows for an objective from an evidence window. Every
// row it emits must carry Method()==row.Method and Version()==row.Version so a
// consumer can attribute the value to a method and version.
type Scorer interface {
	// Method is the stable identifier of the scoring method. It keys the registry
	// and travels in every emitted row.
	Method() string
	// Version is the method's implementation version. It travels in every emitted
	// row so a re-scored value can be told apart from an older one.
	Version() string
	// Score folds obj and win into zero or more ScoreRows. It must be pure: the
	// only host-dependent input is win.Resolve, injected by the caller.
	Score(obj Objective, win EvidenceWindow) []ScoreRow
}

// Registry is the versioned scorer registry — the documented one-step extension
// seam. Scorers are keyed by Method; a second scorer claiming a live method is a
// registration error, not a silent overwrite, so a method always resolves to one
// known implementation.
//
// THE QUALIFICATION GATE (#2573): a scorer ships with its backtest. Registering a
// method here is what lets it steer live sessions, so a NEW method — or a new
// Version of an existing one — earns that place by first passing
// `fak trajctl backtest --scorer X --corpus Y` on a recorded corpus: its readings
// must track the witnessed W3 outcome at the well-calibrated bar and must not
// materially regress the incumbent it replaces (backtest.go). The gate is offline,
// costs no model call, and is deliberately cheap enough that there is no excuse for
// skipping it — a method that has never read recorded history has no evidence it
// reads live history any better.
type Registry struct {
	byMethod map[string]Scorer
}

// NewRegistry returns an empty scorer registry.
func NewRegistry() *Registry {
	return &Registry{byMethod: map[string]Scorer{}}
}

// Register adds s under its Method. It errors on a nil scorer, an empty method or
// version, or a method already registered — registering is an explicit act, never
// an accidental overwrite.
func (r *Registry) Register(s Scorer) error {
	if s == nil {
		return errors.New("trajctl: nil scorer")
	}
	if s.Method() == "" {
		return errors.New("trajctl: scorer method is required")
	}
	if s.Version() == "" {
		return fmt.Errorf("trajctl: scorer %q version is required", s.Method())
	}
	if _, ok := r.byMethod[s.Method()]; ok {
		return fmt.Errorf("trajctl: scorer method %q already registered", s.Method())
	}
	r.byMethod[s.Method()] = s
	return nil
}

// Get returns the scorer registered under method, if any.
func (r *Registry) Get(method string) (Scorer, bool) {
	s, ok := r.byMethod[method]
	return s, ok
}

// Methods returns the registered method names in lexical order for deterministic
// rendering.
func (r *Registry) Methods() []string {
	out := make([]string, 0, len(r.byMethod))
	for m := range r.byMethod {
		out = append(out, m)
	}
	sort.Strings(out)
	return out
}
