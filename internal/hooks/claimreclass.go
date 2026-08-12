package hooks

import (
	"sort"
	"strings"
)

// claimreclass.go — the forward-only cure for an ALREADY-LANDED claim-honesty residual (#5434).
//
// The push gate (tools/githooks/pre-push, DOS reason CLAIM_UNWITNESSED) refuses a push when a
// commit in `origin/<trunk>..HEAD` makes a claim its own diff cannot witness — canonically a
// `fix(...)`/`feat(...)` subject over a diff that touches only prose. The detection is correct.
// The HOLE is the remedy: the hook's advice is "amend the subject", and on this shared trunk
// amending is forbidden (a rebase rewrites peers' commits and moves HEAD under every live
// session). The gate reviews the WHOLE range, so one mistyped subject ~30 commits deep wedges
// every later commit too, and the range only grows. That leaves `FLEET_ALLOW_RESIDUAL=1` as the
// only exit — an override spent on a TRUE positive, indistinguishable in the log from one spent
// on a known false positive. The escape hatch becomes the ordinary exit, which erodes exactly
// the signal the gate exists to protect.
//
// The cure this file implements is a forward-only RECLASSIFICATION: a later commit appends a
// record that DEMOTES the earlier commit's claim to one its own diff already witnesses. Nothing
// is rewritten; the retraction lands as new history the way every other fact in this repo does.
//
// The one property that matters is that the cure cannot LAUNDER a claim. A cure that waves any
// subject through is strictly worse than the wedge, so the grammar is deliberately one-way:
//
//	a reclassification may only DEMOTE a claim to a type the commit's REAL diff witnesses.
//
// Four structural refusals fall out of that sentence and are what VerifyReclass enforces:
//
//  1. the target type may not itself be a code-effect claim — you cannot annotate a `feat`
//     residual into "no really, it IS a feat"; `fix`/`feat`/`perf`/`refactor` are refused outright;
//  2. the target type must be witnessed by the commit's own recorded path set — `test` over a
//     prose-only diff, or `docs` over a diff carrying source, is refused;
//  3. every cited witness path must appear in that commit's real diff, so a record cannot cite
//     files the commit never touched;
//  4. the record must name a commit the gate itself reported as a residual and that git can
//     resolve — an unresolvable or unreported id clears nothing.
//
// So the strongest thing an accepted record can ever say is "this commit's landed subject
// overstated it; the evidence supports only a <docs|chore|test|...> claim". That is a retraction,
// not a pass: the overstated claim is recorded as NOT witnessed, in the same tree, forward-only.
//
// Everything here is a pure function over (the gate's own residual list, the annotation text, the
// commit facts git recorded). The push-seam wiring reads the file below and hands the three in.

// ReclassFile is the repo-relative path of the append-only reclassification ledger the push-seam
// rung reads. A missing file simply means "no annotations", which clears nothing — the relaxation
// is fail-CLOSED in every direction, so a lost or unreadable ledger can only keep the gate strict.
const ReclassFile = "docs/claim-reclass.txt"

// Reclass is one forward-only reclassification record.
type Reclass struct {
	Commit  string   `json:"commit"`  // the landed commit id, as written (>=7 hex chars)
	Type    string   `json:"type"`    // the demoted conventional type the diff can witness
	Witness []string `json:"witness"` // paths from that commit's REAL diff carrying the demoted claim
	Reason  string   `json:"reason"`  // why the landed subject overstated it (auditable prose)
	Line    int      `json:"line"`    // 1-based line of the record's `commit:` key, for diagnostics
}

// CommitFacts is what git itself recorded for one commit — the evidence side of the verdict. The
// caller resolves these; nothing in this file trusts the annotation for them.
type CommitFacts struct {
	SHA     string   `json:"sha"`
	Subject string   `json:"subject"`
	Paths   []string `json:"paths"`
}

// ReclassVerdict is the per-record decision. Accepted=false always carries the refusal reason.
type ReclassVerdict struct {
	Commit   string `json:"commit"`
	Type     string `json:"type"`
	Accepted bool   `json:"accepted"`
	Reason   string `json:"reason"`
}

// ReclassGateResult is the push-seam decision over one whole residual list.
type ReclassGateResult struct {
	Residuals []string         `json:"residuals"`
	Verdicts  []ReclassVerdict `json:"verdicts"`
	Cleared   []string         `json:"cleared"`
	Uncured   []string         `json:"uncured"`
	OK        bool             `json:"ok"`
}

// reclassCodeEffectTypes are the conventional types that ARE a code-effect claim. Demoting INTO
// one of these is the laundering move, so it is refused unconditionally — `fix` and `refactor`
// bind through their type token alone, and `feat`/`perf` bind through their description verb.
var reclassCodeEffectTypes = setOf("fix", "feat", "perf", "refactor", "revert")

// reclassDemotionTypes are the only targets a reclassification may name: types that make no
// code-effect claim at all. Each still has to be independently witnessed by the real diff.
var reclassDemotionTypes = setOf("docs", "chore", "style", "build", "ci", "test")

// claimSourceExts are the file suffixes that count as program SOURCE for the "the diff touches no
// SOURCE file" residual. Test files are handled separately (claimIsTestPath) because a test IS the
// witness for a demoted `test` claim while it is NOT the witness for a code-effect one.
var claimSourceExts = []string{
	".go", ".py", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".c", ".h", ".cc", ".cpp",
	".hpp", ".cu", ".cuh", ".rs", ".java", ".kt", ".swift", ".cs", ".rb", ".m", ".mm",
	".sh", ".ps1", ".bash", ".zsh", ".sql", ".proto", ".pl", ".php", ".scala", ".lua",
}

// claimProseExts are the suffixes that count as prose. A dependency manifest wearing a doc suffix
// (requirements.txt) is data, not prose, and is excluded via the shared dosDepManifestRE.
var claimProseExts = []string{".md", ".rst", ".txt", ".adoc", ".org"}

// claimIsTestPath reports whether p is a test or CI witness — the same predicate the pre-commit
// `test(...)` gate binds to, reused rather than re-derived so the two surfaces cannot drift.
func claimIsTestPath(p string) bool { return isTestOrCIWitnessPath(p) }

// claimIsSourcePath reports whether p is non-test program SOURCE: a code-suffixed file that is
// neither a test nor a testdata fixture. This is the "did the diff touch real code?" half of the
// residual rule.
func claimIsSourcePath(p string) bool {
	q := strings.ToLower(normPath(p))
	if q == "" || claimIsTestPath(q) {
		return false
	}
	for _, seg := range strings.Split(q, "/") {
		if seg == "testdata" {
			return false
		}
	}
	for _, ext := range claimSourceExts {
		if strings.HasSuffix(q, ext) {
			return true
		}
	}
	return false
}

// claimIsProsePath reports whether p is documentation: a prose-suffixed file (that is not a
// dependency manifest in disguise) or anything under a docs/ tree.
func claimIsProsePath(p string) bool {
	q := strings.ToLower(normPath(p))
	if q == "" {
		return false
	}
	if dosDepManifestRE.MatchString(baseName(q)) {
		return false
	}
	if strings.HasPrefix(q, "docs/") {
		return true
	}
	for _, ext := range claimProseExts {
		if strings.HasSuffix(q, ext) {
			return true
		}
	}
	return false
}

// ClaimCodeEffectWithoutSourceWitness returns a non-empty advisory when a subject makes a
// code-effect claim (`fix`/`feat`/`perf`/`refactor`) while the paths it commits contain NO program
// source at all — the exact shape the push gate later reports as a residual, caught at the ONE
// seam where the cure is still cheap (retype the subject before the commit exists). It names the
// demoted type the paths actually witness so the author can adopt it verbatim.
//
// Empty when there is nothing to say: a non-code-effect type, an empty path set, or a diff that
// does touch source.
func ClaimCodeEffectWithoutSourceWitness(subject string, paths []string) string {
	m := subjectRE.FindStringSubmatch(strings.TrimSpace(subject))
	if m == nil || len(paths) == 0 || !reclassCodeEffectTypes[strings.ToLower(m[1])] {
		return ""
	}
	for _, p := range paths {
		if claimIsSourcePath(p) {
			return ""
		}
	}
	typ := strings.ToLower(m[1])
	suggest := reclassSuggestedType(paths)
	if suggest == "" {
		return ""
	}
	return "`" + typ + "(...)` is a code-effect claim but the diff touches no program source (" +
		strings.Join(claimShortPathList(paths), ", ") + "): the push-seam claim-honesty gate reports this shape as a " +
		"CLAIM_UNWITNESSED residual for the WHOLE range, and a landed subject cannot be amended on the shared trunk. " +
		"Use `" + suggest + "(...)` instead, or include the source file that witnesses the claim. See #5434."
}

// reclassSuggestedType names the demoted type a path set witnesses, or "" when the paths witness
// none (so no suggestion is invented).
func reclassSuggestedType(paths []string) string {
	prose, test, other := 0, 0, 0
	for _, p := range paths {
		switch {
		case claimIsProsePath(p):
			prose++
		case claimIsTestPath(p):
			test++
		default:
			other++
		}
	}
	switch {
	case prose > 0 && test == 0:
		return "docs"
	case test > 0 && prose == 0:
		return "test"
	case prose > 0 || other > 0:
		return "chore"
	}
	return ""
}

// claimShortPathList renders up to four paths for a message, then an ellipsis count.
func claimShortPathList(paths []string) []string {
	var out []string
	for i, p := range paths {
		if i == 4 {
			out = append(out, "…")
			break
		}
		out = append(out, normPath(p))
	}
	return out
}

// ParseResidualCommits pulls the commit ids the claim-honesty review reported under its RESIDUAL
// band out of the review's own rendered output. It reads ONLY that band: a CLEARED row is a claim
// the kernel already witnessed and is never a candidate for a cure.
//
// It is deliberately conservative. An output it cannot read yields an EMPTY list, and an empty
// list clears nothing — so a rendering change in the reviewer degrades to "the gate stays
// blocked", never to "the gate waves the range through".
func ParseResidualCommits(reviewOutput string) []string {
	var out []string
	seen := map[string]bool{}
	inResidual := false
	for _, raw := range strings.Split(reviewOutput, "\n") {
		line := strings.TrimRight(raw, "\r")
		if line == "" {
			continue
		}
		if !strings.HasPrefix(line, " ") && !strings.HasPrefix(line, "\t") {
			// A column-0 line opens a band; only the RESIDUAL band holds curable rows.
			inResidual = strings.HasPrefix(strings.ToUpper(strings.TrimSpace(line)), "RESIDUAL")
			continue
		}
		if !inResidual {
			continue
		}
		t := strings.TrimSpace(line)
		if strings.HasPrefix(t, "|") || strings.HasPrefix(t, "(") || strings.HasPrefix(t, "-") {
			continue // a wrapped explanation line under a row, not a row
		}
		id := t
		if i := strings.IndexAny(id, " \t"); i >= 0 {
			id = id[:i]
		}
		if !isHexID(id) {
			continue
		}
		id = strings.ToLower(id)
		if !seen[id] {
			seen[id] = true
			out = append(out, id)
		}
	}
	return out
}

// isHexID reports whether s is a plausible abbreviated-or-full git object id (7..40 hex chars).
func isHexID(s string) bool {
	if len(s) < 7 || len(s) > 40 {
		return false
	}
	for _, r := range s {
		if (r < '0' || r > '9') && (r < 'a' || r > 'f') && (r < 'A' || r > 'F') {
			return false
		}
	}
	return true
}

// ParseReclassRecords reads the append-only ledger. Records are `key: value` blocks opened by a
// `commit:` key; `#` comments and blank lines are ignored, and `witness:` may repeat or carry a
// comma/space separated list. Malformed input is reported as problems rather than skipped
// silently, because a record the parser drops is a cure the operator believes they wrote.
func ParseReclassRecords(text string) (records []Reclass, problems []string) {
	var cur *Reclass
	flush := func() {
		if cur != nil {
			records = append(records, *cur)
			cur = nil
		}
	}
	for i, raw := range strings.Split(text, "\n") {
		lineNo := i + 1
		line := strings.TrimSpace(strings.TrimRight(raw, "\r"))
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, value, ok := strings.Cut(line, ":")
		if !ok {
			problems = append(problems, formatReclassProblem(lineNo, "not a `key: value` line: "+line))
			continue
		}
		key = strings.ToLower(strings.TrimSpace(key))
		value = strings.TrimSpace(value)
		switch key {
		case "commit":
			flush()
			cur = &Reclass{Commit: value, Line: lineNo}
		case "type", "reclass":
			if cur == nil {
				problems = append(problems, formatReclassProblem(lineNo, "`"+key+"` before any `commit:` key"))
				continue
			}
			cur.Type = value
		case "witness":
			if cur == nil {
				problems = append(problems, formatReclassProblem(lineNo, "`witness` before any `commit:` key"))
				continue
			}
			cur.Witness = append(cur.Witness, splitReclassPaths(value)...)
		case "reason":
			if cur == nil {
				problems = append(problems, formatReclassProblem(lineNo, "`reason` before any `commit:` key"))
				continue
			}
			cur.Reason = value
		default:
			problems = append(problems, formatReclassProblem(lineNo, "unknown key `"+key+"` (want commit/type/witness/reason)"))
		}
	}
	flush()
	return records, problems
}

func formatReclassProblem(line int, detail string) string {
	return ReclassFile + ":" + itoa(int64(line)) + ": " + detail
}

func splitReclassPaths(v string) []string {
	var out []string
	for _, f := range strings.FieldsFunc(v, func(r rune) bool { return r == ',' || r == ' ' || r == '\t' }) {
		if f = strings.TrimSpace(f); f != "" {
			out = append(out, normPath(f))
		}
	}
	return out
}

// VerifyReclass decides ONE record against the facts git recorded for the commit it names. It is
// the whole anti-laundering surface: every accept path requires the commit's own diff to witness
// the demoted claim, so the record can never assert something the evidence does not already show.
func VerifyReclass(rec Reclass, f CommitFacts) ReclassVerdict {
	typ := strings.ToLower(strings.TrimSpace(rec.Type))
	v := ReclassVerdict{Commit: strings.ToLower(strings.TrimSpace(rec.Commit)), Type: typ}

	// 1. Identity — the record must bind to a commit git itself resolved.
	if !isHexID(v.Commit) {
		v.Reason = "commit id `" + rec.Commit + "` is not a 7-40 character object id; an ambiguous id cannot bind a cure to a commit"
		return v
	}
	real := strings.ToLower(strings.TrimSpace(f.SHA))
	if real == "" || !strings.HasPrefix(real, v.Commit) {
		v.Reason = "no commit in the reported residual set resolves to `" + v.Commit + "`; a cure may only speak for a commit the gate itself named"
		return v
	}

	// 2. Direction — a reclassification DEMOTES. Naming a code-effect type is the laundering move.
	if reclassCodeEffectTypes[typ] {
		v.Reason = "`" + typ + "` is itself a code-effect claim: a reclassification may only DEMOTE a landed claim to one the commit's own diff witnesses, never restate the unwitnessed claim in other words"
		return v
	}
	if !reclassDemotionTypes[typ] {
		v.Reason = "`" + rec.Type + "` is not a reclassification target; use one of " + strings.Join(sortedKeys(reclassDemotionTypes), ", ")
		return v
	}

	// 3. Accountability — an unexplained record is not an audit trail.
	if strings.TrimSpace(rec.Reason) == "" {
		v.Reason = "record has no `reason:` — a cure that records no rationale is indistinguishable from a spent override"
		return v
	}

	// 4. Citation — every cited witness must be in the commit's real diff.
	if len(rec.Witness) == 0 {
		v.Reason = "record cites no `witness:` path from the commit's diff"
		return v
	}
	inDiff := map[string]bool{}
	for _, p := range f.Paths {
		inDiff[strings.ToLower(normPath(p))] = true
	}
	for _, w := range rec.Witness {
		if !inDiff[strings.ToLower(normPath(w))] {
			v.Reason = "witness path `" + w + "` is not in commit " + v.Commit + "'s diff; a cure may only cite files the commit actually touched"
			return v
		}
	}

	// 5. Evidence — the demoted type must itself be witnessed by that diff.
	if why := reclassTypeUnwitnessed(typ, f.Paths); why != "" {
		v.Reason = "the diff does not witness a `" + typ + "` claim either: " + why
		return v
	}

	v.Accepted = true
	v.Reason = "claim demoted to `" + typ + "`, witnessed by " + strings.Join(claimShortPathList(rec.Witness), ", ")
	return v
}

// reclassTypeUnwitnessed returns "" when the path set witnesses the demoted type, else why not.
func reclassTypeUnwitnessed(typ string, paths []string) string {
	if len(paths) == 0 {
		return "the commit records no changed path at all"
	}
	source, prose, test := 0, 0, 0
	for _, p := range paths {
		switch {
		case claimIsSourcePath(p):
			source++
		case claimIsProsePath(p):
			prose++
		}
		if claimIsTestPath(p) {
			test++
		}
	}
	switch typ {
	case "docs":
		if prose == 0 {
			return "the diff touches no prose file"
		}
		if source > 0 {
			return "the diff also touches program source, so its claim is not merely documentary"
		}
	case "test":
		if test == 0 {
			return "the diff touches no test or CI witness file"
		}
	default: // chore | style | build | ci — each asserts no behavioral effect
		if source > 0 {
			return "the diff touches program source, so `" + typ + "` understates what it did"
		}
	}
	return ""
}

// sortedKeys renders a map's keys deterministically, for a message that must not churn between
// runs. Generic in the VALUE type because the callers disagree about it — a set (map[string]bool)
// here, a constant-to-token table (map[string]string) in the fail-closed ledger surfaces — and one
// helper both can reach is what keeps the package from carrying two copies of the same four lines.
func sortedKeys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// ClearResiduals is the push-seam decision: does a verified forward-only reclassification exist
// for EVERY commit the claim-honesty review reported as a residual?
//
// residuals is the review's own list (see ParseResidualCommits); records is the ledger; lookup
// resolves one reported id to the facts git recorded, returning false when it cannot. OK is true
// only when at least one residual was reported and none is left uncured — an empty or unreadable
// residual list can never clear a push.
func ClearResiduals(residuals []string, records []Reclass, lookup func(string) (CommitFacts, bool)) ReclassGateResult {
	res := ReclassGateResult{Residuals: residuals}
	for _, sha := range residuals {
		id := strings.ToLower(strings.TrimSpace(sha))
		facts, ok := lookup(id)
		if !ok {
			res.Uncured = append(res.Uncured, id)
			res.Verdicts = append(res.Verdicts, ReclassVerdict{
				Commit: id,
				Reason: "git could not resolve the reported residual commit; nothing to verify a cure against",
			})
			continue
		}
		matched := false
		accepted := false
		var last ReclassVerdict
		for _, rec := range records {
			if !reclassIDsMatch(rec.Commit, id) {
				continue
			}
			matched = true
			last = VerifyReclass(rec, facts)
			res.Verdicts = append(res.Verdicts, last)
			if last.Accepted {
				accepted = true
				break
			}
		}
		switch {
		case accepted:
			res.Cleared = append(res.Cleared, id)
		case matched:
			res.Uncured = append(res.Uncured, id)
		default:
			res.Uncured = append(res.Uncured, id)
			res.Verdicts = append(res.Verdicts, ReclassVerdict{
				Commit: id,
				Reason: "no reclassification record names this commit in " + ReclassFile,
			})
		}
	}
	res.OK = len(residuals) > 0 && len(res.Uncured) == 0
	return res
}

// reclassIDsMatch reports whether two git object ids name the same commit, allowing either to be
// the shorter abbreviation. Both must be plausible ids, so a truncated or typo'd token matches
// nothing rather than matching broadly.
func reclassIDsMatch(a, b string) bool {
	x := strings.ToLower(strings.TrimSpace(a))
	y := strings.ToLower(strings.TrimSpace(b))
	if !isHexID(x) || !isHexID(y) {
		return false
	}
	if len(x) > len(y) {
		x, y = y, x
	}
	return strings.HasPrefix(y, x)
}

// ReclassTemplate renders the exact record an operator should append to clear one residual, with
// the demoted type and witness paths already derived from the commit's own diff. It is the whole
// point of the cure being reachable: the refusal hands back the text that resolves it.
func ReclassTemplate(f CommitFacts) string {
	typ := reclassSuggestedType(f.Paths)
	if typ == "" {
		typ = "chore"
	}
	witness := f.Paths
	if len(witness) > 4 {
		witness = witness[:4]
	}
	var b strings.Builder
	b.WriteString("commit: " + f.SHA + "\n")
	b.WriteString("type: " + typ + "\n")
	for _, w := range witness {
		b.WriteString("witness: " + normPath(w) + "\n")
	}
	b.WriteString("reason: <why the landed subject overstated what this diff does>\n")
	return b.String()
}
