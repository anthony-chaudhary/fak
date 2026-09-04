package wipattr

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

// admit.go — the START-OF-TASK admission, the prospective mirror of SweepGuard.
//
// WHY IT EXISTS. Every WIP surface in this repo reads a tree that is ALREADY dirty:
// Attribute grades hunks that exist, SweepGuard warns about staging hunks that exist,
// the reconcile/census/tree-doctor family recovers hunks that exist, and the
// flowmetrics local_wip KPI scores a tree that has already accumulated. The only
// claim primitive is retroactive too — a checkpoint snapshots a delta, so a session's
// first edit is ORPHAN by construction and stays that way until its next checkpoint
// boundary. That leaves exactly one moment unguarded, and it is the cheapest one: the
// moment BEFORE the first edit, when the work has cost nothing and can still be aimed
// somewhere else.
//
// AdmitStart is that moment. Given the paths a task INTENDS to touch and the same
// evidence SweepGuard already folds — the attributed dirty set plus the live-session
// set — it answers whether starting here would (a) duplicate a live peer's in-flight
// work, (b) build on top of unattributed WIP a later sweep can destroy, or (c) widen a
// session that has not landed what it already holds.
//
// SCOPED TO INTENT, NOT TO THE TREE. The gate deliberately grades only the declared
// paths. A shared always-on checkout carries dirt no single session created or can
// fix; refusing on the tree's total dirty count would fire on every invocation forever
// and train every reader to pass --force. What a session CAN answer for is the small
// set of paths it is about to open, so that is what is judged.
//
// WARN-FIRST WHERE THE REMEDY IS SLOW. A collision on a declared path is a HARD hold —
// the remedy (pick another path, or adopt the orphan first) is immediate and local. The
// finish-before-start term is SOFT by default and promoted by Strict, the same posture
// the fleet's focus WIP cap (#3223) takes: breadth backpressure advises before it
// refuses, because "land your other work first" is not something a session can always
// do on the spot.
//
// The fold is pure — no git, no clock, no I/O. The cmd shell gathers the inputs.

// AdmitVerdict is the closed start-of-task admission vocabulary.
type AdmitVerdict string

const (
	// AdmitOK: no hard finding — start the work.
	AdmitOK AdmitVerdict = "ADMIT"
	// AdmitHold: at least one hard finding — do not start here yet.
	AdmitHold AdmitVerdict = "HOLD"
)

// AdmitReason is the closed reason vocabulary. Every finding carries exactly one, so a
// reader (and a caller keying on the JSON) never has to parse prose to route a remedy.
type AdmitReason string

const (
	// ReasonPathClaimedByPeer: a declared path already carries a delta owned by a LIVE
	// peer, or shared ambiguously across sessions. Starting here is duplicative work.
	ReasonPathClaimedByPeer AdmitReason = "PATH_CLAIMED_BY_PEER"
	// ReasonPathOrphaned: a declared path already carries a delta no live session owns —
	// unattributed, or checkpointed by a session that is gone. Recoverable work that
	// editing on top of makes unrecoverable.
	ReasonPathOrphaned AdmitReason = "PATH_ORPHANED"
	// ReasonPathUntrackedWIP: an untracked file already sits at a declared path. No
	// checkpoint can attribute it (a checkpoint captures the TRACKED delta), so the
	// attribution set reads it as absent — silence that must not be read as clean.
	ReasonPathUntrackedWIP AdmitReason = "PATH_UNTRACKED_WIP"
	// ReasonSelfWIPUnlanded: this session already holds more unlanded paths than the
	// ceiling. Soft by default: finish-before-start advice, not a refusal.
	ReasonSelfWIPUnlanded AdmitReason = "SELF_WIP_UNLANDED"
	// ReasonIntentUndeclared: no path was declared, so the per-path rules did not run.
	// Soft, and always reported, because "nothing checked" is not "nothing wrong".
	ReasonIntentUndeclared AdmitReason = "INTENT_UNDECLARED"
)

// DefaultSelfWIPCeiling is how many unlanded paths one session may already hold before
// opening more. Three is a whole small change plus its test and its doc — past that a
// session is accumulating rather than finishing, and each extra path is another thing a
// crash or a peer's broad add can lose.
const DefaultSelfWIPCeiling = 3

// AdmitInput is everything the admission needs, as plain data. Attrs and Live are the
// SAME two inputs SweepGuard takes, so the prospective and retrospective gates can never
// disagree about who owns what.
type AdmitInput struct {
	// Self is the session id asking to start; "" means an unidentified session, which
	// can own nothing, so every dirty declared path reads as someone else's.
	Self string
	// Intends is the declared path set — files or directories, in any spelling.
	Intends []string
	// Attrs is the attributed dirty set (Attribute's output).
	Attrs []Attribution
	// Untracked is the repo-relative untracked path set, which attribution cannot see.
	Untracked []string
	// Live is the set of session ids currently holding a lease.
	Live map[string]bool
	// SelfWIPCeiling overrides DefaultSelfWIPCeiling when > 0.
	SelfWIPCeiling int
	// Strict promotes every soft finding to a hard hold.
	Strict bool
	// Lane is the active lane name for Self.
	Lane string
	// LaneManifest is the declared scope patterns / globs for the active lane.
	LaneManifest []string
	// SessionScopes conveys per-session scope / lane declarations.
	SessionScopes []SessionScope
	// DOSToml provides dos.toml configuration bytes to resolve lane patterns.
	DOSToml []byte
}

// AdmitFinding is one reason a declared path (or the session as a whole) is not clear to
// start. Hard findings bind the verdict; soft ones advise.
type AdmitFinding struct {
	// Path is the declared path the finding is about, or "" for a session-wide finding.
	Path   string      `json:"path,omitempty"`
	Reason AdmitReason `json:"reason"`
	Detail string      `json:"detail"`
	Hard   bool        `json:"hard"`
	State  AttrState   `json:"state,omitempty"`
	Owner  string      `json:"owner,omitempty"`
}

// AdmitReport is the whole decision.
type AdmitReport struct {
	Schema  string       `json:"schema"`
	Verdict AdmitVerdict `json:"verdict"`
	Self    string       `json:"self,omitempty"`
	// Intends echoes the NORMALISED declared set, so a reader can see what was actually
	// matched rather than what was typed.
	Intends []string `json:"intends"`
	// SelfDirty is how many dirty paths this session already owns.
	SelfDirty int            `json:"self_dirty"`
	Ceiling   int            `json:"ceiling"`
	Findings  []AdmitFinding `json:"findings"`
	// Next is the single most useful command to run, chosen from the binding finding.
	Next string `json:"next,omitempty"`
}

// AdmitSchema tags the JSON payload so a consumer can version-check it.
const AdmitSchema = "fak-wip-admit/1"

// AdmitStart decides whether a task may begin on the declared paths.
//
// Total: every engaged declared path yields exactly one finding, and a clean declared
// path yields none. Deterministic: findings are sorted hard-first, then by reason, then
// by path, so two runs over the same input are byte-identical. Fail-safe: a path whose
// ownership cannot be resolved to Self holds, and an unmatched declaration is never
// silently treated as clear.
func AdmitStart(in AdmitInput) AdmitReport {
	intents := normalizeIntents(in.Intends)
	ceiling := in.SelfWIPCeiling
	if ceiling <= 0 {
		ceiling = DefaultSelfWIPCeiling
	}
	rep := AdmitReport{
		Schema:   AdmitSchema,
		Self:     in.Self,
		Intends:  intents,
		Ceiling:  ceiling,
		Findings: []AdmitFinding{},
	}

	// One finding per declared path: the FIRST engaged dirty file under it decides, and
	// the per-intent map keeps a directory that covers ten dirty files from emitting ten
	// rows for one decision.
	var selfPatterns []string
	if len(in.LaneManifest) > 0 {
		selfPatterns = in.LaneManifest
	} else if in.Lane != "" && len(in.DOSToml) > 0 {
		lanes := ParseDOSTomlLanes(in.DOSToml)
		selfPatterns = lanes[strings.ToLower(in.Lane)]
	}
	if len(selfPatterns) == 0 {
		for _, sc := range in.SessionScopes {
			if sc.Session == in.Self && sc.Active {
				if len(sc.Manifest) > 0 {
					selfPatterns = sc.Manifest
				} else if sc.Lane != "" && len(in.DOSToml) > 0 {
					lanes := ParseDOSTomlLanes(in.DOSToml)
					selfPatterns = lanes[strings.ToLower(sc.Lane)]
				}
				break
			}
		}
	}
	selfActive := in.Self != "" && (in.Live == nil || in.Live[in.Self] || len(in.Live) == 0)

	seen := map[string]bool{}
	for _, a := range in.Attrs {
		file := pathutil.NormalizeScope(a.File)

		// If Scope was empty and hunk was orphaned, but session/lane is active and dos.toml / manifest matches:
		// infer ownership (AttrOwned, owner = session).
		if a.State == AttrOrphan && selfActive && len(selfPatterns) > 0 && patternsMatch(selfPatterns, file) {
			a.State = AttrOwned
			a.Owner = in.Self
		}

		if a.State == AttrOwned && a.Owner == in.Self && in.Self != "" {
			rep.SelfDirty++
			continue
		}
		intent, ok := matchIntent(intents, file)
		if !ok || seen[intent] {
			continue
		}
		seen[intent] = true
		rep.Findings = append(rep.Findings, attrFinding(intent, file, a, in.Live))
	}
	for _, u := range in.Untracked {
		file := pathutil.NormalizeScope(u)
		intent, ok := matchIntent(intents, file)
		if !ok || seen[intent] {
			continue
		}
		seen[intent] = true
		rep.Findings = append(rep.Findings, AdmitFinding{
			Path:   intent,
			Reason: ReasonPathUntrackedWIP,
			Hard:   true,
			Detail: fmt.Sprintf("untracked file %s already exists here — no checkpoint can attribute an untracked file, so its owner is unknowable; confirm it is yours or move it aside before starting", file),
		})
	}

	if len(intents) == 0 {
		rep.Findings = append(rep.Findings, AdmitFinding{
			Reason: ReasonIntentUndeclared,
			Hard:   in.Strict,
			Detail: "no --path declared, so no per-path collision check ran — this is an UNCHECKED start, not a clear one",
		})
	}
	if rep.SelfDirty > ceiling {
		rep.Findings = append(rep.Findings, AdmitFinding{
			Reason: ReasonSelfWIPUnlanded,
			Hard:   in.Strict,
			Owner:  in.Self,
			Detail: fmt.Sprintf("this session already holds %d unlanded path(s), over the ceiling of %d — finish before you start: land or checkpoint what you hold", rep.SelfDirty, ceiling),
		})
	}

	sortFindings(rep.Findings)
	rep.Verdict = AdmitOK
	for _, f := range rep.Findings {
		if f.Hard {
			rep.Verdict = AdmitHold
			break
		}
	}
	rep.Next = nextAction(rep)
	return rep
}

// attrFinding turns one colliding attribution into the finding for its declared path.
// The live/dead split is the whole point: a LIVE peer means you are about to duplicate
// work in flight, while a dead or absent owner means you are about to bury recoverable
// work. Same hold, different remedy, so different reason.
func attrFinding(intent, file string, a Attribution, live map[string]bool) AdmitFinding {
	f := AdmitFinding{Path: intent, Hard: true, State: a.State, Owner: a.Owner}
	switch a.State {
	case AttrOwned:
		if live[a.Owner] {
			f.Reason = ReasonPathClaimedByPeer
			f.Detail = fmt.Sprintf("%s is already being edited by LIVE peer %s — starting here duplicates work in flight; pick another path or coordinate", file, a.Owner)
			return f
		}
		f.Reason = ReasonPathOrphaned
		f.Detail = fmt.Sprintf("%s carries a checkpointed delta from %s, which holds no live lease — recoverable work; adopt or discard it before editing over it", file, a.Owner)
		return f
	case AttrShared:
		f.Reason = ReasonPathClaimedByPeer
		f.Detail = fmt.Sprintf("%s is claimed by %s — ownership is ambiguous and no machine can resolve it to you", file, strings.Join(a.Owners, ","))
		return f
	default: // AttrOrphan
		f.Reason = ReasonPathOrphaned
		f.Detail = fmt.Sprintf("%s is dirty but NO checkpoint claims it — unattributed WIP that editing over it makes unrecoverable", file)
		return f
	}
}

// nextAction picks the one command worth printing, from the most binding finding. It
// reads the already-sorted list, so the row a reader sees first and the command they are
// told to run cannot name different problems.
func nextAction(rep AdmitReport) string {
	for _, f := range rep.Findings {
		switch f.Reason {
		case ReasonPathOrphaned:
			return "fak wip reconcile --reclaim   # adopt or discard the unowned delta, then re-run admit"
		case ReasonPathClaimedByPeer:
			return "fak wip attribute --json      # see who holds the path, then declare a disjoint --path set"
		case ReasonPathUntrackedWIP:
			return "fak tree-doctor               # identify the untracked file's owner before editing over it"
		case ReasonSelfWIPUnlanded:
			return "fak wip land                  # land what this session already holds, then start"
		case ReasonIntentUndeclared:
			return "fak wip admit --path <p>...   # declare the paths this task will touch"
		}
	}
	return ""
}

// sortFindings puts hard findings ahead of soft ones, then orders by reason and path, so
// the output is stable and the binding row is always first.
func sortFindings(fs []AdmitFinding) {
	sort.SliceStable(fs, func(i, j int) bool {
		if fs[i].Hard != fs[j].Hard {
			return fs[i].Hard
		}
		if fs[i].Reason != fs[j].Reason {
			return fs[i].Reason < fs[j].Reason
		}
		return fs[i].Path < fs[j].Path
	})
}

// normalizeIntents cleans, de-duplicates and sorts the declared set. Sorting matters for
// determinism: matchIntent returns the first covering declaration, so an unsorted input
// could attribute one collision to different declarations across runs.
func normalizeIntents(in []string) []string {
	seen := map[string]bool{}
	out := make([]string, 0, len(in))
	for _, p := range in {
		p = pathutil.NormalizeScope(p)
		if p == "" || seen[p] {
			continue
		}
		seen[p] = true
		out = append(out, p)
	}
	sort.Strings(out)
	return out
}

// normalizePath renders a declared or reported path in the one spelling the fold
// compares: forward slashes, no "./" prefix, no trailing slash. An operator on Windows
// matchIntent finds the declared path covering file — an exact match, or a declared
// directory the file sits beneath. It returns the declaration (not the file) so the
// finding is reported against what the caller asked for.
func matchIntent(intents []string, file string) (string, bool) {
	for _, in := range intents {
		if file == in || strings.HasPrefix(file, in+"/") {
			return in, true
		}
	}
	return "", false
}
