package dispatchtick

import (
	"regexp"
	"sort"
	"strings"
)

// SelfSourceTreePrefixes are the repository sub-trees that compile into the running
// fak orchestrator binary -- the Go module's own source. A worker spawned under
// `fak guard` cannot SHIP an edit to fak's own running code: it is editing the very
// binary (and the guard) adjudicating it, the structural self-modification hazard
// #1397 names. The dispatcher must not route such work to a self-guarded worker; the
// observed cost when it does is real (10- and 42-turn investigations that both
// terminally blocked with 0 commit on #1338's cmd/fak work). cmd/** is the CLI and the
// orchestrator verbs; internal/** is every kernel subsystem. A lane rooted in either is
// operator-gated for a guarded worker -- it belongs on an unguarded/operator or
// worktree-isolated path (#1334), not a self-guarded worker.
//
// This is a PRUDENTIAL pre-route over the whole Go module, deliberately broader than
// the guard's literal self_modify_globs default (.git/, .env, id_rsa, ...): the
// dispatcher refuses to spawn into fak's own source AT ALL rather than discover the
// block tool-call by tool-call after a worker has already burned turns.
var SelfSourceTreePrefixes = []string{"cmd/", "internal/"}

// IsSelfSourceTree reports whether one lane-tree glob is rooted in fak's own running
// source (cmd/** or internal/**). A leading "./" or "fak/" module prefix is tolerated
// so a tree written as fak/internal/... still matches, and backslashes are normalized
// so a Windows-authored glob matches the same as a POSIX one.
func IsSelfSourceTree(glob string) bool {
	g := strings.ReplaceAll(strings.TrimSpace(glob), "\\", "/")
	g = strings.TrimPrefix(g, "./")
	g = strings.TrimPrefix(g, "fak/")
	for _, prefix := range SelfSourceTreePrefixes {
		if strings.HasPrefix(g, prefix) {
			return true
		}
	}
	return false
}

// SelfModifyHold is the pure pre-route verdict for one dispatch pick: a GUARDED worker
// aimed at a lane whose tree is part of fak's own running source can do real
// investigation but can never SHIP -- the only safe ships for a self-guarded worker are
// the non-self-modify lanes (docs, tools/*.py, .github, examples). Rather than spawn a
// doomed worker that burns turns and lands 0 commits, the tick HOLDS the pick.
//
// It returns held=true only when guarded AND at least one lane tree is self-source, and
// names the first offending tree as the witness. An UNGUARDED worker (the guard disabled,
// or a worktree-isolated/operator path) is never held -- that is exactly the escape #1397
// points operators toward.
func SelfModifyHold(guarded bool, laneTree []string) (held bool, tree string) {
	if !guarded {
		return false, ""
	}
	for _, t := range laneTree {
		if IsSelfSourceTree(t) {
			return true, t
		}
	}
	return false, ""
}

// selfSourceTextRE matches a reference, in an issue's title or body, to fak's own
// running Go-module source: a cmd/ or internal/ rooted path or glob, with an optional
// ./ or fak/ module prefix (cmd/**, internal/agent, ./cmd/fak/dispatch_tick.go,
// fak/internal/gateway/http.go). A leading boundary (start, or a non-path char) keeps it
// from matching cmd/ inside a longer word (subcommand/, internals/). It deliberately
// catches the BARE cmd/ and internal/ forms the router's path extractor (pathRE) misses
// -- pathRE only recognizes fak/-prefixed or tools|docs paths, which is exactly why a
// self-source issue (title `fix(dispatch):`, body "lives in cmd/** + internal/**")
// mis-routes to a safe lane (tools) carrying ZERO extracted paths (#1397).
var selfSourceTextRE = regexp.MustCompile(`(?:^|[^\w./-])((?:\./|fak/)?(?:cmd|internal)/[\w*][\w*./-]*)`)

// IssueTextTargetsSelfSource reports whether an issue's text (title + body) references
// fak's own running source -- a cmd/ or internal/ path or glob -- returning the first
// matched reference as the witness. It is the MIS-ROUTE arm of the #1397 pre-route: the
// router can send a self-source issue to a SAFE lane by scope/label/keyword alias (a
// `fix(dispatch):` title aliases to the tools lane) while its real work lives in cmd/**
// or internal/**, so the lane tree alone never reveals the self-modify hazard -- the
// exact #1338/#1397 failure (real work in cmd/fak, lane reported as tools).
func IssueTextTargetsSelfSource(text string) (held bool, tree string) {
	m := selfSourceTextRE.FindStringSubmatch(text)
	if m == nil {
		return false, ""
	}
	return true, m[1]
}

// SelfModifyHoldForPick is the full #1397 pre-route verdict for one dispatch pick: a
// GUARDED worker is held when EITHER the lane tree is fak's own source (a correctly-
// routed cmd/internal lane) OR the target issue's own text references cmd/** or
// internal/** (a MIS-ROUTED issue whose scope/label alias sent it to a safe lane). The
// lane-tree arm is checked first so a correctly-routed lane names its glob as the
// witness; the issue-text arm then catches the mis-route the lane tree hides. An
// UNGUARDED worker is never held (the operator/worktree-isolated escape #1334).
func SelfModifyHoldForPick(guarded bool, laneTree []string, issueText string) (held bool, tree string) {
	if !guarded {
		return false, ""
	}
	if held, t := SelfModifyHold(guarded, laneTree); held {
		return true, t
	}
	return IssueTextTargetsSelfSource(issueText)
}

// LaneDispatchableUnderGuard reports whether a lane whose canonical tree is laneTree
// can be the target of a GUARDED issue-resolution worker -- i.e. the lane is NOT rooted
// in fak's own running source (cmd/** or internal/**). It is the SELECTION-TIME twin of
// SelfModifyHold: SelfModifyHold answers "must I HOLD this already-chosen lane?" AFTER a
// pick, while this answers "should the picker even CONSIDER this lane?" BEFORE one.
//
// The motivating failure (#1397/#1338, the empty-dispatch-surface stall): on a guarded
// trunk the open-issue backlog is dominated by internal/** lanes (compute, gateway,
// promptmmu, metrics, engine, model, ...), so the busiest-by-step-budget lane is almost
// always self-source. A picker that chooses the single busiest lane and only THEN runs
// SelfModifyHoldForPick picks a held lane and refuses with SELF_MODIFY_HOLD every tick --
// reporting an EMPTY plan surface even though guard-shippable lanes (docs, tools, .github,
// examples, visuals, .claude) carry abundant work. Filtering the candidate set through
// this predicate first lets the picker skip the held lanes and surface a shippable one.
//
// An UNGUARDED worker can ship anywhere (the operator/worktree-isolated escape #1334), so
// for guarded=false every lane is dispatchable. A lane with NO declared tree is treated as
// dispatchable: the picker's own fallback names a tree only when one is chosen, and an
// empty tree carries no self-source witness to hold on -- failing OPEN here keeps a lane
// the taxonomy under-declares from silently vanishing from the surface.
func LaneDispatchableUnderGuard(guarded bool, laneTree []string) bool {
	if !guarded {
		return true
	}
	for _, t := range laneTree {
		if IsSelfSourceTree(t) {
			return false
		}
	}
	return true
}

// DispatchableLanesUnderGuard partitions a lane->tree map into the lanes a GUARDED worker
// can SHIP to (dispatchable, sorted) and the lanes it would be HELD on (held, sorted). It
// is the set-level form of LaneDispatchableUnderGuard the lane picker uses to drop
// self-source lanes from the busiest-lane search BEFORE choosing, so a guarded tick lands
// on a shippable lane instead of refusing on the busiest self-source one (#1397).
//
// For guarded=false every lane is dispatchable and held is empty (the operator escape
// #1334). The returned slices are independent copies so a caller can mutate them freely.
func DispatchableLanesUnderGuard(guarded bool, trees map[string][]string) (dispatchable, held []string) {
	dispatchable = make([]string, 0, len(trees))
	for lane, tree := range trees {
		if LaneDispatchableUnderGuard(guarded, tree) {
			dispatchable = append(dispatchable, lane)
		} else {
			held = append(held, lane)
		}
	}
	sort.Strings(dispatchable)
	sort.Strings(held)
	return dispatchable, held
}
