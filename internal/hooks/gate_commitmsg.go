package hooks

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// gate_commitmsg.go — the COMMIT_MSG gate, a port of tools/check_commit_msg.py. It nudges the
// subject toward `type(scope): <verb> <what>` so the DOS commit-audit witness can grade the
// commit. CommitMsgVerdict returns (ok, why): ok==true means gradeable.

var commitTypes = map[string]bool{
	"feat": true, "fix": true, "docs": true, "refactor": true, "perf": true, "test": true,
	"chore": true, "build": true, "ci": true, "style": true, "revert": true,
}

var commitTypesOrder = []string{"feat", "fix", "docs", "refactor", "perf", "test", "chore", "build", "ci", "style", "revert"}

// commitVerbs — the recognized-verb set, kept in lockstep with check_commit_msg.py
// VERBS (the differential parity harness asserts the two stay identical). setOf
// dedupes, so verbs that recur across groups are harmless.
var commitVerbs = setOf(
	"add", "implement", "create", "build", "introduce", "scaffold",
	"fix", "repair", "correct", "patch", "resolve", "address",
	"test", "verify", "validate", "assert", "cover",
	"refactor", "restructure", "rewrite", "reframe", "rework", "simplify",
	"remove", "delete", "drop", "strip", "prune", "purge",
	"redact", "scrub", "sanitize",
	"move", "rename", "repoint", "relocate", "migrate", "port",
	"update", "bump", "upgrade", "sync", "refresh", "regenerate",
	"wire", "gate", "enforce", "prevent", "guard", "bound", "cap", "limit",
	"restore", "recover", "reinstate",
	"document", "clarify", "annotate", "note",
	"optimize", "speed", "harden", "tune",
	"support", "enable", "disable", "deprecate",
	"revert", "merge", "split", "extract", "inline", "dedupe", "consolidate",
	"close", "land", "ship", "generalize", "normalize", "reconcile",
	"make", "use", "switch", "replace", "set", "allow", "ensure", "handle",
	"archive", "ignore", "back",
	// Concrete imperative verbs observed leading real commits the gate was
	// advisory-flagging despite naming a genuine action (28% -> ~1% false-flag
	// rate over 400 commits). Each describes a checkable change, not a noun.
	"define", "declare", "state", "explain", "describe",
	"record", "register", "log", "witness", "prove", "demonstrate",
	"fill", "populate", "seed", "stub",
	"standardize", "unify", "align", "tidy",
	"tighten", "loosen", "relax", "widen", "narrow", "scope",
	"default", "pin", "warm", "prewarm", "preload", "prefetch",
	"apply", "propagate", "thread", "plumb", "route", "dispatch", "feed",
	"acknowledge", "credit", "cite", "reference", "link", "anchor", "tie",
	"cross-ref", "index", "catalog",
	"hash", "checksum", "stamp", "tag", "label", "mark", "flag",
	"parallelize", "serialize", "batch", "stream", "buffer", "cache",
	"grant", "revoke", "authorize", "permit", "deny", "block", "reject",
	"idle", "reap", "drain", "flush", "evict", "expire", "retire",
	"fold", "unfold", "expand", "collapse",
	"emit", "surface", "expose", "publish", "export", "import",
	// Second harvest from the residual flags — more concrete imperative verbs
	// that name a real action (drove the false-flag rate from 11% toward ~3%).
	"file", "sort", "kill", "ground", "sample", "report", "frame", "rephrase",
	"grade", "trend", "calibrate", "recalibrate", "keep", "run", "name",
	"print", "lift", "prefer", "generate", "forward", "flip", "drive",
	"locate", "deepen", "pace", "lock", "onboard", "treat", "preserve",
	"quote", "fence", "gofmt",
	// Advisory-action verbs: a commit that ADDS a lint/gate which advises or nudges (the
	// commit-gardening surface itself, #1326) names a real, checkable change. The gate was
	// abstaining on "advise"/"nudge"/"recommend" despite each leading a concrete diff.
	"advise", "nudge", "recommend", "warn", "remind", "hint",
	// Imperative base forms the DOS commit-audit referee witnesses as a code effect
	// (dos_witness_verbs.go dosCodeEffectVerbs) that fak's gate was REJECTING as ungradeable —
	// the mirror image of the abstainHazard divergence (#2089). fak's accept-set must be a
	// SUPERSET of the referee's imperative code verbs so a subject that BINDS at the referee is
	// never red-flagged by `fak commit --preview`. Asserted by TestCommitVerbsSupersetOfRefereeCodeVerbs.
	"accumulate", "arm", "attribute", "author", "bind", "bridge", "carry",
	"consume", "dequant", "dequantize", "derive", "downgrade", "floor", "hook",
	"invert", "memoize", "optimise", "price", "refuse", "require", "reserve",
	"reset", "show", "splice", "synthesize",
	// A concrete imperative verb that names a checkable change but was absent from the harvest,
	// so `fak commit --preview` red-flagged a real subject the mutating `fak commit` (which never
	// runs this gate) accepted and scored 100/A — the preview/mutation grade divergence of #3912.
	// "isolate" leads a genuine action (isolate a code path / behavior under test); accepting it
	// makes both commands grade the one subject the one way.
	"isolate",
)

var subjectRE = regexp.MustCompile(`^([a-z]+)(\([^)]+\))?(!)?:\s+(.+)$`)

var exemptSubjectPrefixes = []string{"Merge ", "Revert ", "fixup! ", "squash! ", "amend! "}

// isGitDirectory reports whether root contains a .git directory or file (worktree pointer).
func isGitDirectory(root string) bool {
	if root == "" {
		return false
	}
	_, err := os.Stat(filepath.Join(root, ".git"))
	return err == nil
}

// HasMultipleParentsRef checks whether a commit has >= 2 topological parents (#10882).
// When ref is empty (e.g. at commit-msg hook time during an in-flight commit), it checks
// if MERGE_HEAD exists and is valid, or if HEAD has >= 2 parents. When ref is provided,
// it checks if ref^2 exists.
func HasMultipleParentsRef(root string, ref string) bool {
	if root == "" {
		return false
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	if ref != "" {
		cmd := exec.CommandContext(ctx, "git", "-C", root, "rev-parse", "-q", "--verify", ref+"^2")
		windowgate.ConfigureBackgroundCommand(cmd)
		return cmd.Run() == nil
	}

	// 1. In-flight merge: MERGE_HEAD exists
	cmd := exec.CommandContext(ctx, "git", "-C", root, "rev-parse", "-q", "--verify", "MERGE_HEAD")
	windowgate.ConfigureBackgroundCommand(cmd)
	if err := cmd.Run(); err == nil {
		return true
	}

	// 2. Existing commit at HEAD has >= 2 parents
	cmd = exec.CommandContext(ctx, "git", "-C", root, "rev-parse", "-q", "--verify", "HEAD^2")
	windowgate.ConfigureBackgroundCommand(cmd)
	return cmd.Run() == nil
}

// checkConflictBanners reports whether the commit message contains git conflict template lines or markers (#11306).
func checkConflictBanners(msg string) (ok bool, why string) {
	for _, line := range strings.Split(msg, "\n") {
		if strings.Contains(line, "# Conflicts:") {
			return false, "MERGE_CONFLICT_TEMPLATE_FORBIDDEN: commit message contains unedited git conflict template ('# Conflicts:')"
		}
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "<<<<<<<") || strings.HasPrefix(trimmed, "=======") || strings.HasPrefix(trimmed, ">>>>>>>") {
			return false, "MERGE_CONFLICT_MARKERS_FORBIDDEN: commit message contains git conflict markers ('<<<<<<<', '=======', or '>>>>>>>')"
		}
	}
	return true, ""
}

var silentMergeTrailerRE = regexp.MustCompile(`(?im)^\s*(?:merge-strategy\s*:\s*ours|silent-merge\s*:\s*(?:intentional|true|yes|allow))\b`)

func hasSilentMergeTrailer(msg string) bool {
	return silentMergeTrailerRE.MatchString(msg)
}

func checkSilentMergeOverride(msg string) bool {
	if os.Getenv("ALLOW_SILENT_MERGE") == "1" || os.Getenv("FLEET_ALLOW_SILENT_MERGE") == "1" {
		return true
	}
	return hasSilentMergeTrailer(msg)
}

func hasMergeHead(root string) bool {
	if root == "" {
		return false
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "git", "-C", root, "rev-parse", "-q", "--verify", "MERGE_HEAD")
	windowgate.ConfigureBackgroundCommand(cmd)
	return cmd.Run() == nil
}

// CheckSilentDropMerge verifies that a merge commit does not silently drop incoming commits
// by producing a tree identical to parent 1 while parent 2 contains non-empty unique commits (#11306).
func CheckSilentDropMerge(root string, msg string, ref string) (ok bool, why string) {
	if root == "" || !isGitDirectory(root) {
		return true, ""
	}
	if checkSilentMergeOverride(msg) {
		return true, ""
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Case 1: In-flight merge (ref is empty and MERGE_HEAD exists)
	if ref == "" {
		cmd := exec.CommandContext(ctx, "git", "-C", root, "rev-parse", "-q", "--verify", "MERGE_HEAD")
		windowgate.ConfigureBackgroundCommand(cmd)
		if cmd.Run() == nil {
			cmdStaged := exec.CommandContext(ctx, "git", "-C", root, "write-tree")
			windowgate.ConfigureBackgroundCommand(cmdStaged)
			outStaged, err1 := cmdStaged.Output()

			cmdP1 := exec.CommandContext(ctx, "git", "-C", root, "rev-parse", "-q", "--verify", "HEAD^{tree}")
			windowgate.ConfigureBackgroundCommand(cmdP1)
			outP1, err2 := cmdP1.Output()

			if err1 == nil && err2 == nil && strings.TrimSpace(string(outStaged)) == strings.TrimSpace(string(outP1)) {
				cmdCnt := exec.CommandContext(ctx, "git", "-C", root, "rev-list", "--count", "HEAD..MERGE_HEAD")
				windowgate.ConfigureBackgroundCommand(cmdCnt)
				outCnt, err3 := cmdCnt.Output()
				if err3 == nil && strings.TrimSpace(string(outCnt)) != "0" {
					cmdDiff := exec.CommandContext(ctx, "git", "-C", root, "diff", "--name-only", "HEAD...MERGE_HEAD")
					windowgate.ConfigureBackgroundCommand(cmdDiff)
					outDiff, err4 := cmdDiff.Output()
					if err4 == nil && len(strings.TrimSpace(string(outDiff))) > 0 {
						return false, "SILENT_DROP_MERGE_FORBIDDEN: merge tree matches parent 1 exactly while parent 2 contains non-empty unique commits (silent drop merge); supply 'Merge-Strategy: ours' or 'Silent-Merge: intentional' trailer to allow"
					}
				}
			}
			return true, ""
		}
	}

	// Case 2: Existing commit (ref != "" or HEAD when ref is empty and HEAD^2 exists)
	target := ref
	if target == "" {
		cmd := exec.CommandContext(ctx, "git", "-C", root, "rev-parse", "-q", "--verify", "HEAD^2")
		windowgate.ConfigureBackgroundCommand(cmd)
		if cmd.Run() == nil {
			target = "HEAD"
		}
	}
	if target != "" {
		cmd := exec.CommandContext(ctx, "git", "-C", root, "rev-parse", "-q", "--verify", target+"^2")
		windowgate.ConfigureBackgroundCommand(cmd)
		if cmd.Run() == nil {
			cmdTree := exec.CommandContext(ctx, "git", "-C", root, "rev-parse", "-q", "--verify", target+"^{tree}")
			windowgate.ConfigureBackgroundCommand(cmdTree)
			outTree, err1 := cmdTree.Output()

			cmdP1 := exec.CommandContext(ctx, "git", "-C", root, "rev-parse", "-q", "--verify", target+"^1^{tree}")
			windowgate.ConfigureBackgroundCommand(cmdP1)
			outP1, err2 := cmdP1.Output()

			if err1 == nil && err2 == nil && strings.TrimSpace(string(outTree)) == strings.TrimSpace(string(outP1)) {
				cmdCnt := exec.CommandContext(ctx, "git", "-C", root, "rev-list", "--count", target+"^1.."+target+"^2")
				windowgate.ConfigureBackgroundCommand(cmdCnt)
				outCnt, err3 := cmdCnt.Output()
				if err3 == nil && strings.TrimSpace(string(outCnt)) != "0" {
					cmdDiff := exec.CommandContext(ctx, "git", "-C", root, "diff", "--name-only", target+"^1..."+target+"^2")
					windowgate.ConfigureBackgroundCommand(cmdDiff)
					outDiff, err4 := cmdDiff.Output()
					if err4 == nil && len(strings.TrimSpace(string(outDiff))) > 0 {
						return false, "SILENT_DROP_MERGE_FORBIDDEN: merge tree matches parent 1 exactly while parent 2 contains non-empty unique commits (silent drop merge); supply 'Merge-Strategy: ours' or 'Silent-Merge: intentional' trailer to allow"
					}
				}
			}
		}
	}
	return true, ""
}

// CommitMsgVerdict reports whether a commit message's subject is witness-gradeable, and if not,
// why. It mirrors check_commit_msg.py verdict() (L61-77).
func CommitMsgVerdict(msg string) (ok bool, why string) {
	return CommitMsgVerdictWithGit(msg, "")
}

// CommitMsgVerdictWithGit reports whether a commit message's subject is witness-gradeable,
// taking into account repository git topology for Merge subjects (#10882, #11306).
func CommitMsgVerdictWithGit(msg string, root string) (ok bool, why string) {
	return CommitMsgVerdictWithGitRef(msg, root, "")
}

// CommitMsgVerdictWithGitRef reports whether a commit message's subject is witness-gradeable,
// taking into account repository git topology and commit ref for Merge subjects (#10882, #11306).
// When subject starts with "Merge ", it verifies that >= 2 topological parents exist and
// that the merge does not silently drop incoming unique commits without an explicit trailer.
func CommitMsgVerdictWithGitRef(msg string, root string, ref string) (ok bool, why string) {
	if ok, why := checkConflictBanners(msg); !ok {
		return false, why
	}
	subject := firstSubjectLine(msg)
	if subject == "" {
		return false, "empty subject"
	}
	if strings.HasPrefix(subject, "Merge ") {
		if root != "" && isGitDirectory(root) {
			if !HasMultipleParentsRef(root, ref) {
				return false, "MERGE_WITNESS_FAIL: commit subject starts with 'Merge ' but has fewer than 2 topological parents; pseudo-merges cannot bypass Conventional Commits and DCO"
			}
			if ok, why := CheckSilentDropMerge(root, msg, ref); !ok {
				return false, why
			}
		}
		return true, ""
	}
	if root != "" && isGitDirectory(root) && hasMergeHead(root) {
		if ok, why := CheckSilentDropMerge(root, msg, ref); !ok {
			return false, why
		}
	}
	for _, p := range exemptSubjectPrefixes {
		if p != "Merge " && strings.HasPrefix(subject, p) {
			return true, ""
		}
	}
	m := subjectRE.FindStringSubmatch(subject)
	if m == nil {
		return false, "subject is not `type(scope): <verb> <what>` (types: " + strings.Join(commitTypesOrder, "/") + ")"
	}
	typ := m[1]
	if !commitTypes[typ] {
		return false, "unknown type '" + typ + "' (use one of: " + strings.Join(commitTypesOrder, "/") + ")"
	}
	rest := strings.TrimSpace(m[4])
	first := strings.ToLower(splitFirstWordOrColon(rest))
	first = strings.Trim(first, "`*\"'")
	if !commitVerbs[first] {
		return false, "description leads with '" + first + "', not a recognized verb — the witness ABSTAINs on a noun-led subject. Lead with a verb (add/fix/implement/…)."
	}
	return true, ""
}

// firstSubjectLine returns the first non-empty line that is not a comment (check_commit_msg.py L53-58).
func firstSubjectLine(msg string) string {
	for _, ln := range strings.Split(msg, "\n") {
		s := strings.TrimSpace(ln)
		if s == "" || strings.HasPrefix(s, "#") {
			continue
		}
		return s
	}
	return ""
}

// splitFirstWordOrColon returns the first token split on whitespace OR colon
// (re.split(r"[\s:]", rest, maxsplit=1)[0]).
func splitFirstWordOrColon(s string) string {
	i := strings.IndexFunc(s, func(r rune) bool {
		return r == ' ' || r == '\t' || r == '\n' || r == '\r' || r == '\f' || r == '\v' || r == ':'
	})
	if i < 0 {
		return s
	}
	return s[:i]
}

func setOf(xs ...string) map[string]bool {
	m := make(map[string]bool, len(xs))
	for _, x := range xs {
		m[x] = true
	}
	return m
}

// nearMissTypes maps the conventional-commit type mistakes an author actually makes — a plural,
// an inflected form, or a near-synonym — onto the canonical type. It is deliberately conservative:
// only entries whose correction is unambiguous appear, so a genuinely unknown type earns no guess.
var nearMissTypes = map[string]string{
	"feature": "feat", "features": "feat",
	"fixes": "fix", "fixed": "fix", "bugfix": "fix", "bugfixes": "fix", "hotfix": "fix",
	"doc": "docs", "documentation": "docs",
	"tests": "test", "testing": "test",
	"chores":      "chore",
	"refactoring": "refactor", "refactored": "refactor",
	"performance": "perf",
	"builds":      "build",
	"styling":     "style", "styles": "style",
	"reverts": "revert", "reverted": "revert",
}

// irregularVerbBases maps common irregular inflections onto the imperative base commitVerbs carries.
// Only bases that are members of commitVerbs are listed, so membership still decides.
var irregularVerbBases = map[string]string{
	"built": "build", "made": "make", "ran": "run", "kept": "keep",
	"drove": "drive", "driven": "drive", "fed": "feed", "showed": "show", "shown": "show",
}

// imperativeBase returns the recognized imperative verb an inflected word derives from, or "" when
// the word is not an inflection of any commitVerbs member. It over-generates candidate base forms
// (stripping -s/-es/-ies/-ed/-ied/-ing, with +e and de-doubled variants) and lets commitVerbs
// membership decide — the same over-generative discipline the referee-parity harness uses.
func imperativeBase(w string) string {
	for _, cand := range imperativeBaseForms(w) {
		if commitVerbs[cand] {
			return cand
		}
	}
	return ""
}

func imperativeBaseForms(w string) []string {
	out := []string{w}
	add := func(s string) {
		if s != "" && s != w {
			out = append(out, s)
		}
	}
	switch {
	case strings.HasSuffix(w, "ies"):
		add(strings.TrimSuffix(w, "ies") + "y")
	case strings.HasSuffix(w, "es"):
		add(strings.TrimSuffix(w, "es"))
		add(strings.TrimSuffix(w, "s"))
	case strings.HasSuffix(w, "s"):
		add(strings.TrimSuffix(w, "s"))
	}
	switch {
	case strings.HasSuffix(w, "ied"):
		add(strings.TrimSuffix(w, "ied") + "y")
	case strings.HasSuffix(w, "ed"):
		base := strings.TrimSuffix(w, "ed")
		add(base)                       // added -> add
		add(strings.TrimSuffix(w, "d")) // wired -> wire
		add(deDoubleConsonant(base))    // wrapped -> wrap
	}
	if strings.HasSuffix(w, "ing") {
		base := strings.TrimSuffix(w, "ing")
		add(base)                    // adding -> add
		add(base + "e")              // caching -> cache, wiring -> wire
		add(deDoubleConsonant(base)) // wrapping -> wrap
	}
	if b, ok := irregularVerbBases[w]; ok {
		add(b)
	}
	return out
}

// deDoubleConsonant collapses a trailing doubled consonant ("wrapp" -> "wrap") so a gerund/past of a
// consonant-doubling verb resolves to its base. It leaves a non-doubled tail untouched.
func deDoubleConsonant(s string) string {
	n := len(s)
	if n >= 2 && s[n-1] == s[n-2] {
		return s[:n-1]
	}
	return s
}

// suggestGradeableSubject proposes a concrete rewrite of a subject that CommitMsgVerdict rejected,
// for the two DETERMINISTIC failure modes where the author's intent is unambiguous:
//
//  1. a near-miss conventional type — `feature(x): add …` -> `feat(x): add …`, `fixes:` -> `fix:`;
//  2. a non-imperative leading verb — the description IS verb-led but inflected
//     (`added` -> `add`, `caching` -> `cache`, `wiring` -> `wire`), which the exact-match verb gate
//     rejects even though a real verb was clearly meant.
//
// It returns "" whenever a mechanical fix would be a guess — an empty subject, a subject with no
// `type(scope): …` shape at all, an unknown type with no known correction, or a genuinely noun-led
// description — so a caller never surfaces a wrong suggestion. The rebuilt subject is only returned
// after CommitMsgVerdict confirms it is now gradeable, so the suggestion is self-verified: an agent
// can adopt it verbatim and clear the gate.
func suggestGradeableSubject(subject string) string {
	subject = strings.TrimSpace(subject)
	if subject == "" {
		return ""
	}
	for _, p := range exemptSubjectPrefixes {
		if strings.HasPrefix(subject, p) {
			return ""
		}
	}
	m := subjectRE.FindStringSubmatch(subject)
	if m == nil {
		// No `type(scope): <what>` shape — reconstructing a type+scope would be a guess.
		return ""
	}
	typ, scope, bang, rest := m[1], m[2], m[3], strings.TrimSpace(m[4])
	if !commitTypes[typ] {
		canon, ok := nearMissTypes[typ]
		if !ok {
			return ""
		}
		typ = canon
	}
	firstWord := splitFirstWordOrColon(rest)
	if !commitVerbs[strings.ToLower(firstWord)] {
		// Only rewrite a bare leading token (no surrounding backticks/quotes/emphasis) so the swap
		// can never mangle formatting; a decorated lead is left for the human/agent to resolve.
		if strings.Trim(firstWord, "`*\"'") != firstWord {
			return ""
		}
		base := imperativeBase(strings.ToLower(firstWord))
		if base == "" {
			return ""
		}
		rest = base + rest[len(firstWord):]
	}
	rebuilt := typ + scope + bang + ": " + rest
	if rebuilt == subject {
		return ""
	}
	if ok, _ := CommitMsgVerdict(rebuilt); ok {
		return rebuilt
	}
	return ""
}
