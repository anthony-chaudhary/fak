// Package trajctlhook is the impure call-site assembly that binds the pure
// trajctl turn-boundary fold (trajctl.Sample / trajctl.CompactionBoundary /
// trajctl.AppendSample) to a running session's host evidence. #2539, epic
// #2533: the score-at-turn-end wiring.
//
// trajctl stays tier-1 and clock-free: its scorers are session-agnostic pure
// folds over an injected trajctl.EvidenceWindow. Everything a scorer cannot
// invent for itself — whether a claimed commit SHA still resolves, the analyzed
// sessionaudit rows for the window, the wall-clock stamp — is assembled HERE, at
// the boundary, and handed in. This package owns that assembly and the two
// fail-open drivers a hook calls:
//
//	RunTurnEnd     — the Stop-hook cadence: run the cheap W3/W2 scorers over the
//	                 session's open objectives and append their rows.
//	RunCompaction  — the PreCompact twin: append one context-reset boundary
//	                 marker per open objective so a curve reader can tell a flat
//	                 stretch across a reset from a flat stretch within one context.
//
// Both are fail-open by contract (the session and the hook must run on even when
// the ledger is unreadable or git is absent): a driver returns its Result with
// any error captured for observability, and NEVER panics. The cmd/fak Stop-hook
// verb that calls these is deliberately deferred — the trajctl dispatch case is
// parked behind a peer lane — so this package is the fully-testable unit of the
// wiring, exercised end-to-end by its own tests against a temp ledger and a
// fixture git repo.
package trajctlhook

import (
	"os/exec"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/sessionaudit"
	"github.com/anthony-chaudhary/fak/internal/trajctl"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// CheapScorers is the turn-cadence scorer set: the W3 witnessed-commit progress
// scorer and the W2 activity/progress divergence (stall) scorer. These are the
// two scorers cheap enough to run at every turn end — zero model calls, bounded
// pure folds. Judge-tier scorers are excluded by this choice, not by the fold.
// Returned fresh each call so a caller can append its own without mutating a
// shared slice.
func CheapScorers() []trajctl.Scorer {
	return []trajctl.Scorer{
		trajctl.CommitProgressScorer{},
		trajctl.ActivityDivergenceScorer{},
	}
}

// GitEvidenceResolver returns a trajctl.EvidenceResolver backed by the git repo
// at root. It resolves a "commit" evidence pointer by asking git whether the SHA
// still names a commit object:
//
//	git -C <root> cat-file -e <sha>^{commit}
//
// A zero exit means the object resolves (EvidenceVerified); a non-zero exit means
// the SHA is dangling — rewritten, pruned, or never real — (EvidenceDangling). A
// pointer of any other kind, or an empty ref, is EvidenceUnknown: this resolver
// speaks only for commits and must never demote a row whose evidence it cannot
// judge (the conservative rung the audit fold relies on).
//
// The returned closure shells to git off the hot path (this is turn-boundary
// wiring, never an adjudication decide). If root is empty git runs in the process
// cwd, matching plain `git` behavior.
func GitEvidenceResolver(root string) trajctl.EvidenceResolver {
	return func(ref trajctl.EvidenceRef) trajctl.EvidenceStatus {
		if ref.Kind != "commit" {
			return trajctl.EvidenceUnknown
		}
		sha := strings.TrimSpace(ref.Ref)
		if sha == "" {
			return trajctl.EvidenceUnknown
		}
		args := []string{}
		if root != "" {
			args = append(args, "-C", root)
		}
		args = append(args, "cat-file", "-e", sha+"^{commit}")
		cmd := exec.Command("git", args...)
		windowgate.ConfigureBackgroundCommand(cmd)
		if err := cmd.Run(); err != nil {
			return trajctl.EvidenceDangling
		}
		return trajctl.EvidenceVerified
	}
}

// LoadSessions analyzes each transcript path into a sessionaudit.Session, in
// input order, so the caller can feed EvidenceWindow.Sessions. Analyze itself is
// fail-soft — a missing or malformed file yields a Session with its Error set
// rather than an exception — so an unreadable transcript costs its own row, not
// the pass. An empty or nil paths slice yields nil (no sessions to fold).
func LoadSessions(paths []string) []sessionaudit.Session {
	if len(paths) == 0 {
		return nil
	}
	out := make([]sessionaudit.Session, 0, len(paths))
	for _, p := range paths {
		if strings.TrimSpace(p) == "" {
			continue
		}
		out = append(out, sessionaudit.Analyze(p))
	}
	return out
}

// WindowInput is the host evidence a turn-boundary pass folds. It names every
// impure input a scorer cannot invent — the resolver, the analyzed sessions, the
// phase→commit bindings, the wall-clock stamp — so BuildWindow is a pure
// assembly and the tests stay deterministic.
type WindowInput struct {
	// PhaseCommits maps a plan phase id to the candidate commit SHAs claimed to
	// resolve it (from commit trailers or a verify pass at the call site). A phase
	// with no entry, or whose candidates all fail to resolve, has made no
	// witnessed progress.
	PhaseCommits map[string][]string
	// SessionPaths are the transcript files to analyze for this window; each is
	// folded through sessionaudit.Analyze into EvidenceWindow.Sessions.
	SessionPaths []string
	// Resolve confirms each evidence pointer. Typically GitEvidenceResolver(root);
	// a nil resolver leaves every pointer unknown, so an un-resolvable window
	// scores 0 rather than crediting unverified work (fail-closed scoring).
	Resolve trajctl.EvidenceResolver
	// UnixMillis stamps the produced rows. 0 leaves the stamp unset so the append
	// path can stamp instead; injected so tests are deterministic.
	UnixMillis int64
}

// BuildWindow assembles a trajctl.EvidenceWindow from the folded ledger state and
// the host evidence. It is a pure fold: PriorScores comes from state (the
// existing curve the stall scorer reads without touching the ledger), Sessions
// are analyzed from in.SessionPaths, and the resolver / phase bindings / stamp
// pass through from in. No I/O beyond the transcript analysis LoadSessions does.
func BuildWindow(state trajctl.State, in WindowInput) trajctl.EvidenceWindow {
	return trajctl.EvidenceWindow{
		PhaseCommits: in.PhaseCommits,
		PriorScores:  state.Scores,
		Sessions:     LoadSessions(in.SessionPaths),
		Resolve:      in.Resolve,
		UnixMillis:   in.UnixMillis,
	}
}

// Result is the outcome of a turn-boundary driver: the pure sample it produced,
// how many rows it appended to the ledger, and any error it swallowed fail-open.
// A caller may surface Err for observability but must treat the pass as advisory:
// the hook and the session run on regardless.
type Result struct {
	// Sample is the pure fold the driver produced (rows + swallowed scorer
	// failures). Present even when the append short-circuits, so a caller can see
	// what would have been written.
	Sample trajctl.TurnSample
	// Appended is the number of rows successfully written to the ledger. It may be
	// fewer than len(Sample.Rows) if a row failed validation mid-append.
	Appended int
	// Err is the first ledger-append error, or nil. Captured, not raised: the
	// turn-boundary contract is fail-open, but a poisoned row is never silent.
	Err error
}

// RunTurnEnd is the Stop-hook driver: it folds the ledger at path, builds the
// evidence window, runs the cheap scorers (trajctl.Sample) over the session's
// open objectives, and appends the produced rows. It is fail-open end to end — an
// unreadable ledger folds to an empty state (zero objectives → an empty sample →
// nothing appended), a nil resolver scores 0 rather than crediting unverified
// work, and a panicking scorer costs its own row via the guarded Sample fold. The
// only surfaced error is a mid-append validation failure, returned in Result.Err
// with the count written so far.
func RunTurnEnd(path string, in WindowInput, stamp trajctl.Stamp) Result {
	state := trajctl.Fold(trajctl.ReadLedgerFile(path))
	win := BuildWindow(state, in)
	sample := trajctl.Sample(state.Objectives, CheapScorers(), win, stamp)
	n, err := trajctl.AppendSample(path, sample)
	return Result{Sample: sample, Appended: n, Err: err}
}

// RunCompaction is the PreCompact twin of RunTurnEnd: it appends one
// context-reset boundary marker per open objective (trajctl.CompactionBoundary)
// so a curve reader can distinguish a flat stretch spanning a reset from a flat
// stretch within one context. It runs no scorers — a boundary marker is a W0,
// value-0, no-progress row on its own method series — so it needs no resolver or
// sessions; only the folded objective set and the stamp. Fail-open on the same
// contract as RunTurnEnd.
func RunCompaction(path string, unixMillis int64, stamp trajctl.Stamp) Result {
	state := trajctl.Fold(trajctl.ReadLedgerFile(path))
	win := trajctl.EvidenceWindow{UnixMillis: unixMillis}
	sample := trajctl.CompactionBoundary(state.Objectives, win, stamp)
	n, err := trajctl.AppendSample(path, sample)
	return Result{Sample: sample, Appended: n, Err: err}
}
