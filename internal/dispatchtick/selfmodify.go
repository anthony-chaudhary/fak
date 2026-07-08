package dispatchtick

import (
	"regexp"
	"sort"
	"strings"
)

// SelfSourceTreePrefixes are the repository sub-trees that compile into the running
// fak orchestrator binary -- the Go module's own source (cmd/** is the CLI/verbs,
// internal/** is every kernel subsystem). This is the BROAD "edits fak's own build"
// predicate: it is what a build-ISOLATION decision keys on (a self-source edit
// benefits from a detached worktree so N concurrent workers don't red each other's
// `go build`, #1334), NOT the narrower "a guarded worker may not SHIP this" hold.
//
// The HOLD is TrustCriticalTreePrefixes below -- a strict subset. The two were once
// conflated: the pre-route held the WHOLE module on the theory that "the guard
// refuses cmd/internal with reason=SELF_MODIFY", but that is false for the floor a
// dispatch worker actually runs under. `fak guard` (no --policy) loads the embedded
// guard-default-policy.json, whose self_modify_globs are SECRETS/dotfiles only
// (.git/, .env, id_rsa, .kube/config, ...) -- it names NO cmd/ or internal/ tree.
// So a guarded worker CAN ship internal/gateway, internal/agent, cmd/fak, ... ; the
// only self-edits it must never make are to the WITNESS MACHINERY that would let an
// RSI loop grade its own homework (the trust-critical set). Holding the whole module
// starved the guarded surface of the ~85% of the backlog the guard actually permits.
var SelfSourceTreePrefixes = []string{"cmd/", "internal/"}

// TrustCriticalTreePrefixes are the trust-substrate / witness-machinery sub-trees a
// GUARDED, self-improving worker must never SHIP an edit to: the ABI/kernel/adjudicator
// (the referee itself), the policy loader + registrations (what the referee enforces),
// and the architest/shipgate witness gates (what proves a ship is real). A guarded
// worker that could rewrite these could quietly grade its own homework -- the exact RSI
// hazard #1397 protects against. This set MIRRORS the coding-agent floor's
// SelfModifyGlobs (internal/adjudicator.DevAgentPolicy): it is the authoritative "an
// autonomous loop may not self-edit this" list, so the pre-route and the live guard
// agree on WHICH trees are load-bearing.
//
// Everything ELSE under cmd/**/internal/** is guard-shippable and is deliberately NOT
// held: a guarded worker resolves it normally.
var TrustCriticalTreePrefixes = []string{
	"internal/abi/",
	"internal/kernel/",
	"internal/adjudicator/",
	"internal/policy/",
	"internal/registrations/",
	"internal/architest/",
	"internal/shipgate/",
}

// TrustCriticalFileGlobs are the FILE-level members of the same trust set: the lane
// taxonomy + stamp grammar the referee binds to (dos.toml, .dos/), the on-disk policy
// manifest (policy.json), and the release version stamp (VERSION). A lane whose tree
// names one of these is held for the same reason as the directory prefixes. (Pure
// secrets like .git/ or id_rsa are NOT here: no dispatch lane routes to them, and the
// LIVE guard floor already refuses a write to them tool-call by tool-call.)
var TrustCriticalFileGlobs = []string{"dos.toml", ".dos/", "policy.json", "VERSION"}

// normalizeTree canonicalizes one lane-tree glob for prefix matching: a leading "./"
// or "fak/" module prefix is stripped so a tree written as fak/internal/... matches,
// backslashes are normalized so a Windows-authored glob matches the same as a POSIX
// one, and surrounding whitespace is trimmed.
func normalizeTree(glob string) string {
	g := strings.ReplaceAll(strings.TrimSpace(glob), "\\", "/")
	g = strings.TrimPrefix(g, "./")
	g = strings.TrimPrefix(g, "fak/")
	return g
}

// IsSelfSourceTree reports whether one lane-tree glob is rooted in fak's own running
// source (cmd/** or internal/**) -- the BROAD "compiles into the binary" predicate a
// build-isolation/worktree decision keys on. It is NOT the ship-hold predicate; see
// IsTrustCriticalTree for that.
func IsSelfSourceTree(glob string) bool {
	g := normalizeTree(glob)
	for _, prefix := range SelfSourceTreePrefixes {
		if strings.HasPrefix(g, prefix) {
			return true
		}
	}
	return false
}

// IsTrustCriticalTree reports whether one lane-tree glob is rooted in the trust-critical
// witness machinery a guarded worker must never SHIP an edit to (TrustCriticalTreePrefixes
// or TrustCriticalFileGlobs). This is the predicate the SELF_MODIFY hold keys on -- a
// strict subset of IsSelfSourceTree.
func IsTrustCriticalTree(glob string) bool {
	g := normalizeTree(glob)
	if g == "" {
		return false
	}
	for _, prefix := range TrustCriticalTreePrefixes {
		if strings.HasPrefix(g, prefix) {
			return true
		}
	}
	for _, f := range TrustCriticalFileGlobs {
		if g == f || strings.HasPrefix(g, f) {
			return true
		}
	}
	return false
}

// IsCoreSourceLaneTree reports whether EVERY glob in a lane (or issue) tree is fak's own
// GUARD-SHIPPABLE core engineering: self-source (cmd/** or internal/**) AND not the
// trust-critical referee a guarded worker can never ship. It is the "default the wave
// toward core forward progress" predicate: the operator wants the unattended loop to
// prefer core lanes over the coarse docs/tools buckets, so the dispatch ordering ranks a
// lane this returns true for ahead of a non-core lane of equal-or-greater step budget. An
// empty tree, any non-self-source glob, or any trust-critical glob yields false -- so a
// mixed lane never masquerades as core, and the held referee set is never PREFERRED (a
// guarded worker aimed there only wastes a slot on a doomed SELF_MODIFY-held pick).
func IsCoreSourceLaneTree(tree []string) bool {
	if len(tree) == 0 {
		return false
	}
	for _, g := range tree {
		if strings.TrimSpace(g) == "" || !IsSelfSourceTree(g) || IsTrustCriticalTree(g) {
			return false
		}
	}
	return true
}

// SelfModifyHold is the pure pre-route verdict for one dispatch pick: a GUARDED worker
// aimed at a lane whose tree is TRUST-CRITICAL (the ABI/kernel/adjudicator/policy/
// witness machinery) can do real investigation but must never SHIP -- letting an
// autonomous loop rewrite its own referee is the RSI hazard the hold prevents. Rather
// than spawn a doomed worker for that narrow set, the tick HOLDS the pick.
//
// It returns held=true only when guarded AND at least one lane tree is trust-critical,
// naming the first offending tree as the witness. A guarded worker aimed at any OTHER
// self-source lane (internal/gateway, internal/agent, cmd/fak, ...) is NOT held -- the
// worker guard permits those ships. An UNGUARDED worker (the guard disabled, or a
// worktree-isolated/operator path) is never held -- the escape #1334 points at.
func SelfModifyHold(guarded bool, laneTree []string) (held bool, tree string) {
	if !guarded {
		return false, ""
	}
	for _, t := range laneTree {
		if IsTrustCriticalTree(t) {
			return true, t
		}
	}
	return false, ""
}

// trustCriticalTextRE matches a reference, in an issue's title or body, to fak's
// trust-critical witness machinery: an internal/{abi,kernel,adjudicator,policy,
// registrations,architest,shipgate} rooted path or glob, with an optional ./ or fak/
// module prefix. A leading boundary (start, or a non-path char) keeps it from matching
// inside a longer word. It is the MIS-ROUTE arm of the hold: the router can send a
// trust-critical issue to a SAFE lane by scope/label/keyword alias (a `fix(policy):`
// title aliases to the tools lane) while its real work lives in internal/policy, so the
// lane tree alone never reveals the hazard.
var trustCriticalTextRE = regexp.MustCompile(
	`(?:^|[^\w./-])((?:\./|fak/)?internal/(?:abi|kernel|adjudicator|policy|registrations|architest|shipgate)[\w*./-]*)`)

// IssueTextTargetsTrustCritical reports whether an issue's text (title + body) references
// fak's trust-critical witness machinery, returning the first matched reference as the
// witness. It is the MIS-ROUTE arm of the pre-route: the router can send a trust-critical
// issue to a SAFE lane by scope/label/keyword alias while its real work lives in the
// referee's own trees, so the lane tree alone never reveals the hold hazard. A reference
// to a merely-self-source tree (internal/gateway, cmd/fak) is deliberately NOT matched --
// the guard permits shipping those, so holding them would starve the surface.
func IssueTextTargetsTrustCritical(text string) (held bool, tree string) {
	m := trustCriticalTextRE.FindStringSubmatch(text)
	if m == nil {
		return false, ""
	}
	return true, m[1]
}

// SelfModifyHoldForPick is the full pre-route verdict for one dispatch pick: a GUARDED
// worker is held when EITHER the lane tree is trust-critical (a correctly-routed
// adjudicator/policy/... lane) OR the target issue's own text references the trust-
// critical machinery (a MIS-ROUTED issue whose scope/label alias sent it to a safe lane).
// The lane-tree arm is checked first so a correctly-routed lane names its glob as the
// witness; the issue-text arm then catches the mis-route the lane tree hides. An
// UNGUARDED worker is never held (the operator/worktree-isolated escape #1334).
func SelfModifyHoldForPick(guarded bool, laneTree []string, issueText string) (held bool, tree string) {
	if !guarded {
		return false, ""
	}
	if held, t := SelfModifyHold(guarded, laneTree); held {
		return true, t
	}
	return IssueTextTargetsTrustCritical(issueText)
}

// LaneDispatchableUnderGuard reports whether a lane whose canonical tree is laneTree
// can be the target of a GUARDED issue-resolution worker -- i.e. the lane is NOT rooted
// in the trust-critical witness machinery. It is the SELECTION-TIME twin of
// SelfModifyHold: SelfModifyHold answers "must I HOLD this already-chosen lane?" AFTER a
// pick, while this answers "should the picker even CONSIDER this lane?" BEFORE one.
//
// The motivating failure (#1397/#1338, the empty-dispatch-surface stall): a picker that
// chose the single busiest lane and only THEN ran SelfModifyHoldForPick would refuse
// every tick whenever the busiest lane was held. Filtering the candidate set through
// this predicate first lets the picker skip the held lanes and surface a shippable one.
// With the hold narrowed to the trust-critical set, the vast majority of internal/**
// (gateway, agent, compute, ...) and all of cmd/** are now dispatchable, so the surface
// is starved only when the WHOLE backlog is the referee's own trees.
//
// An UNGUARDED worker can ship anywhere (the operator/worktree-isolated escape #1334), so
// for guarded=false every lane is dispatchable. A lane with NO declared tree is treated as
// dispatchable: the picker's own fallback names a tree only when one is chosen, and an
// empty tree carries no trust-critical witness to hold on -- failing OPEN here keeps a lane
// the taxonomy under-declares from silently vanishing from the surface.
func LaneDispatchableUnderGuard(guarded bool, laneTree []string) bool {
	if !guarded {
		return true
	}
	for _, t := range laneTree {
		if IsTrustCriticalTree(t) {
			return false
		}
	}
	return true
}

// DispatchableLanesUnderGuard partitions a lane->tree map into the lanes a GUARDED worker
// can SHIP to (dispatchable, sorted) and the lanes it would be HELD on (held, sorted). It
// is the set-level form of LaneDispatchableUnderGuard the lane picker uses to drop
// trust-critical lanes from the busiest-lane search BEFORE choosing, so a guarded tick
// lands on a shippable lane instead of refusing on the busiest held one (#1397).
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
