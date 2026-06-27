package hooks

import (
	"regexp"
	"strings"
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
)

var subjectRE = regexp.MustCompile(`^([a-z]+)(\([^)]+\))?(!)?:\s+(.+)$`)

var exemptSubjectPrefixes = []string{"Merge ", "Revert ", "fixup! ", "squash! ", "amend! "}

// CommitMsgVerdict reports whether a commit message's subject is witness-gradeable, and if not,
// why. It mirrors check_commit_msg.py verdict() (L61-77).
func CommitMsgVerdict(msg string) (ok bool, why string) {
	subject := firstSubjectLine(msg)
	if subject == "" {
		return false, "empty subject"
	}
	for _, p := range exemptSubjectPrefixes {
		if strings.HasPrefix(subject, p) {
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
