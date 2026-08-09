package closureaudit

// acceptance.go — resolve an issue's closure by ACCEPTANCE-SYMBOL PRESENCE on the
// trunk plus a CALLER-COUNT, instead of by commit-subject grep (#5435).
//
// WHY. The only binding between a landing and an issue used to be the issue number
// in the commit subject, and the highest-value stale-opens do not carry it: a
// subject names a different issue, or names none, or the acceptance item lands
// inside a peer's unrelated commit. `git log --grep "#<N>"` is blind to all three.
// What actually finds them is searching the trunk for the symbols the acceptance
// criteria NAME. This file is the pure fold for that: extraction (prose -> needles),
// then a verdict over caller-supplied presence facts. All git/gh I/O is the shell's
// job (cmd/fak/dispatch_acceptance_resolve.go), exactly like the grader half above.
//
// The second half is the caller-count. The repo's "ship the pure primitive first"
// convention lands a fully table-tested internal/ symbol whose only referents are
// its own package, its own tests, and internal/architest. The acceptance sentence is
// literally satisfied while production behaviour is unchanged, so a symbol-presence
// check alone would report SHIPPED and the wiring — the half carrying the
// user-visible value — would be silently dropped. Such a symbol resolves PARTIAL and
// the report NAMES the remaining work as "wire it into <seam>".
//
// # EXTRACTION RULE (and how it declines rather than guesses)
//
// Symbol extraction from prose is heuristic and will be wrong sometimes, so the
// failure direction is made safe and explicit: anything not confidently extracted
// produces UNKNOWN — never a confident SHIPPED. Mis-closing an open issue is far
// worse than declining to resolve it. Concretely:
//
//  1. An acceptance REGION must be located first: a heading, bold line, or bare
//     label line whose text starts with "acceptance", "definition of done",
//     "done when", "done condition", or "dod". The region runs to the next heading
//     at the same or a higher level. No region => UNKNOWN(NO_ACCEPTANCE_SECTION).
//     Prose outside the region is never mined; a body that never states its
//     acceptance is not resolvable, and says so.
//  2. Inside the region, fenced code blocks are dropped (they are repro commands,
//     not commitments) and ONLY BACKTICKED SPANS are considered. Bare prose words
//     are never promoted to symbols — that is the single largest source of
//     false-confident matches.
//  3. A backticked span is classified, or DECLINED with a reason that stays in the
//     report: any whitespace => NOT_AN_IDENTIFIER (a command or a phrase); under 3
//     chars => TOO_SHORT; on the generic stoplist => GENERIC_WORD; a bare
//     all-lowercase or all-uppercase word with no underscore => NOT_A_SYMBOL_SHAPE.
//     Declines are per-span and do not by themselves void the issue.
//  4. What survives is one of three kinds: PATH (contains "/" or a known source
//     extension), TOKEN (SCREAMING_SNAKE refusal token, or a --flag), SYMBOL (a
//     mixedCaps Go identifier, optionally package-qualified as pkg.Sym — the
//     qualifier is then ENFORCED against the declaration site, so a common name
//     like Decide cannot match some other package's).
//  5. A region with zero surviving needles => UNKNOWN(NO_NAMED_SYMBOL).
//
// The verdict fold adds three more UNKNOWN paths: a needle with no presence fact
// (PROBE_MISSING), a probe that errored (PROBE_FAILED), and evidence that is only
// generic (GENERIC_PRESENCE_ONLY — every present needle is a bare identifier
// spread module-wide, #6002). A verdict is only ever SHIPPED when every needle was
// probed cleanly, every needle is present, no present internal/ symbol is
// caller-less, and at least one present needle is specific enough to bind the
// evidence to this issue: a path, a token, a package-qualified symbol, or a
// narrowly-spread bare symbol.

import (
	"fmt"
	"path"
	"regexp"
	"sort"
	"strings"
)

// Acceptance-resolution verdicts. UNKNOWN is a first-class outcome, not an error:
// it is what "I could not confidently extract this" must look like to a reader.
const (
	AcceptanceShipped = "SHIPPED"
	AcceptancePartial = "PARTIAL"
	AcceptanceOpen    = "OPEN"
	AcceptanceUnknown = "UNKNOWN"
)

// Reason tokens. The Reason* set explains an UNKNOWN; the Decline* set explains why
// one backticked span was not promoted to a needle.
const (
	ReasonNoAcceptanceSection = "NO_ACCEPTANCE_SECTION"
	ReasonNoNamedSymbol       = "NO_NAMED_SYMBOL"
	ReasonProbeMissing        = "PROBE_MISSING"
	ReasonProbeFailed         = "PROBE_FAILED"
	// ReasonNotResolved marks an issue that was in the population but was never
	// probed, because a caller's budget cap cut the run short. It exists so a cap
	// cannot silently shrink the population: an unprobed issue must be visible as
	// unprobed, not absent. Reporting only the issues that fit under the cap is the
	// same class of error this whole verb exists to remove.
	ReasonNotResolved = "NOT_RESOLVED"
	// ReasonGenericPresenceOnly marks a resolution whose every present needle is a
	// bare identifier spread module-wide (#6002): the behaviour-shaped clauses all
	// declined at extraction, so the only surviving evidence is vocabulary
	// presence, which cannot witness any particular landing. The verdict abstains
	// (UNKNOWN), never SHIPPED — the #5822 false positive resolved SHIPPED off the
	// generic `Affected` alone while every real done condition had declined.
	ReasonGenericPresenceOnly = "GENERIC_PRESENCE_ONLY"

	DeclineNotIdentifier  = "NOT_AN_IDENTIFIER"
	DeclineTooShort       = "TOO_SHORT"
	DeclineGeneric        = "GENERIC_WORD"
	DeclineNotSymbolShape = "NOT_A_SYMBOL_SHAPE"
)

// DefaultRef is the ref every presence probe resolves against. Closure is a
// property of the TRUNK, not of the local working tree (where a peer's uncommitted
// lane would read as landed) and not of a local HEAD (which lags the shared trunk).
const DefaultRef = "origin/main"

// DefaultSeam is the production surface named when the acceptance text itself names
// no wiring site. cmd/fak is the module's only user-reachable binary surface, so an
// internal/ primitive that reaches no production caller is, by elimination, missing
// its wire to a verb there. SeamNamed=false marks this as the fallback, not a quote.
const DefaultSeam = "cmd/fak"

// architestDir is the layering-pin package. It imports the whole module by design,
// so a referent there proves nothing about production wiring.
const architestDir = "internal/architest"

// maxBareSymbolSpread is the widest trunk presence a BARE, unqualified symbol
// needle may have and still corroborate a SHIPPED verdict on its own (#6002). A
// symbol one leaf landed is present in its declaration file plus a handful of
// referents; an identifier present across dozens of files is module vocabulary
// that predates the issue (`Affected`, the #5822 false positive, sits in 34
// tracked Go files on the trunk), so its presence witnesses nothing about any
// particular landing. Paths, refusal tokens, and package-qualified symbols are
// exempt: each is already pinned to a specific site.
const maxBareSymbolSpread = 8

// NeedleKind is what a surviving backticked span was classified as.
type NeedleKind string

const (
	NeedleSymbol NeedleKind = "symbol"
	NeedlePath   NeedleKind = "path"
	NeedleToken  NeedleKind = "token"
)

// Needle is one thing the acceptance text commits to, plus the string to search the
// trunk for. Pkg is the package qualifier when the text wrote pkg.Sym ("" otherwise);
// it is enforced against the declaration site so a bare common identifier cannot
// resolve against an unrelated package.
type Needle struct {
	Text string     `json:"text"`
	Grep string     `json:"grep"`
	Kind NeedleKind `json:"kind"`
	Pkg  string     `json:"pkg,omitempty"`
}

// Declined is one backticked span that was NOT promoted to a needle, with the rule
// that rejected it. Kept in the payload so a reader can see what the extractor
// refused to guess at rather than wondering what it silently swallowed.
type Declined struct {
	Text   string `json:"text"`
	Reason string `json:"reason"`
}

// Acceptance is the extraction result for one issue body.
type Acceptance struct {
	Found    bool       `json:"found"`
	Section  string     `json:"section,omitempty"`
	Needles  []Needle   `json:"needles"`
	Declined []Declined `json:"declined,omitempty"`
	Reason   string     `json:"reason,omitempty"`
}

// Presence is one caller-supplied probe result for one needle, read off the trunk
// ref. Files are repo-relative paths (forward slashes) that contain the needle.
// DefFile is the file DECLARING it as a Go symbol, or "" when the needle is not a
// Go declaration on the ref (a path, a refusal token, or a symbol only referenced).
// Err non-empty means the probe could not be run — which forces UNKNOWN.
type Presence struct {
	Needle  string   `json:"needle"`
	Found   bool     `json:"found"`
	Files   []string `json:"files,omitempty"`
	DefFile string   `json:"def_file,omitempty"`
	Err     string   `json:"err,omitempty"`
}

// CallerCount partitions a symbol's referents on the trunk. Production is the only
// bucket that proves the primitive is on a live path; Unwired is Production == 0.
type CallerCount struct {
	Needle     string `json:"needle"`
	DefFile    string `json:"def_file"`
	DefPackage string `json:"def_package"`
	Production int    `json:"production"`
	OwnPackage int    `json:"own_package"`
	Tests      int    `json:"tests"`
	Architest  int    `json:"architest"`
	Unwired    bool   `json:"unwired"`
}

// Resolution is one issue's acceptance-based closure verdict with all its evidence.
type Resolution struct {
	Verdict    string        `json:"verdict"`
	Reason     string        `json:"reason"`
	Remaining  string        `json:"remaining,omitempty"`
	Ref        string        `json:"ref"`
	Seam       string        `json:"seam,omitempty"`
	SeamNamed  bool          `json:"seam_named"`
	Acceptance Acceptance    `json:"acceptance"`
	Presence   []Presence    `json:"presence,omitempty"`
	Callers    []CallerCount `json:"callers,omitempty"`
}

var (
	backtickSpanRE = regexp.MustCompile("`([^`\n]+)`")
	headingRE      = regexp.MustCompile(`^\s{0,3}(#{1,6})\s+(.*\S)\s*$`)
	boldLineRE     = regexp.MustCompile(`^\s{0,3}\*\*(.+?)\*\*\s*:?\s*(.*)$`)
	bareLabelRE    = regexp.MustCompile(`^\s{0,3}([A-Za-z][A-Za-z ]{2,40})\s*:\s*(.*)$`)
	sourceExtRE    = regexp.MustCompile(`\.(go|md|toml|json|ya?ml|py|sh|txt|jsonl)$`)
	screamingRE    = regexp.MustCompile(`^[A-Z][A-Z0-9]*(_[A-Z0-9]+)+$`)
	flagRE         = regexp.MustCompile(`^--?[a-z0-9][a-z0-9-]*$`)
	identRE        = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)
	qualifiedRE    = regexp.MustCompile(`^([a-z][a-z0-9]*)\.([A-Za-z_][A-Za-z0-9_]*)$`)
)

// acceptanceLabels are the label prefixes that open an acceptance region. Kept
// deliberately short: a wider net drags in "Proposal"/"Plan" prose, which states
// intent rather than a commitment, and intent must never resolve an issue.
var acceptanceLabels = []string{
	"acceptance",
	"definition of done",
	"done when",
	"done condition",
	"dod",
}

// genericStop are backticked spans that are real code but name nothing specific
// enough to witness a landing. Matching one declines the span (GENERIC_WORD).
var genericStop = map[string]bool{
	"main": true, "nil": true, "true": true, "false": true, "err": true,
	"fak": true, "git": true, "gh": true, "go": true, "dos": true, "make": true,
	"json": true, "yaml": true, "toml": true, "head": true, "todo": true,
	"origin/main": true, "cmd/fak": true, "internal": true, "internal/": true,
	"n/a": true, "ok": true, "go.mod": true, "readme.md": true,
}

// isAcceptanceLabel reports whether a heading/bold/bare-label text opens an
// acceptance region.
func isAcceptanceLabel(text string) bool {
	s := strings.ToLower(strings.TrimSpace(text))
	s = strings.Trim(s, "#*_ :-")
	for _, lbl := range acceptanceLabels {
		if s == lbl || strings.HasPrefix(s, lbl+" ") || strings.HasPrefix(s, lbl+"/") ||
			strings.HasPrefix(s, lbl+":") || strings.HasPrefix(s, lbl+" criteria") {
			return true
		}
	}
	return false
}

// acceptanceRegion returns the acceptance text of a body plus the label that opened
// it, or ("", "") when the body states no acceptance. Fenced code blocks are dropped
// from the returned text: they carry repro commands, not commitments.
func acceptanceRegion(body string) (text, label string) {
	lines := strings.Split(strings.ReplaceAll(body, "\r\n", "\n"), "\n")
	start, level := -1, 0
	var tail string
	for i, ln := range lines {
		if m := headingRE.FindStringSubmatch(ln); m != nil {
			if isAcceptanceLabel(m[2]) {
				start, level, label, tail = i, len(m[1]), strings.TrimSpace(m[2]), ""
				break
			}
			continue
		}
		if m := boldLineRE.FindStringSubmatch(ln); m != nil && isAcceptanceLabel(m[1]) {
			start, level, label, tail = i, 7, strings.TrimSpace(m[1]), m[2]
			break
		}
		if m := bareLabelRE.FindStringSubmatch(ln); m != nil && isAcceptanceLabel(m[1]) {
			start, level, label, tail = i, 7, strings.TrimSpace(m[1]), m[2]
			break
		}
	}
	if start < 0 {
		return "", ""
	}
	var body2 []string
	if strings.TrimSpace(tail) != "" {
		body2 = append(body2, tail)
	}
	fenced := false
	for _, ln := range lines[start+1:] {
		if strings.HasPrefix(strings.TrimSpace(ln), "```") {
			fenced = !fenced
			continue
		}
		if fenced {
			continue
		}
		if m := headingRE.FindStringSubmatch(ln); m != nil && len(m[1]) <= level {
			break
		}
		body2 = append(body2, ln)
	}
	return strings.Join(body2, "\n"), label
}

// classifySpan promotes one backticked span to a Needle or declines it with a
// reason. See the EXTRACTION RULE in the file header — every decline path here is
// deliberate: declining costs an unresolved issue, guessing costs a mis-closed one.
func classifySpan(raw string) (Needle, string) {
	s := strings.TrimSpace(raw)
	s = strings.Trim(s, "`")
	s = strings.TrimSuffix(s, "()")
	if s == "" {
		return Needle{}, DeclineTooShort
	}
	if strings.ContainsAny(s, " \t") {
		return Needle{}, DeclineNotIdentifier
	}
	if len(s) < 3 {
		return Needle{}, DeclineTooShort
	}
	if genericStop[strings.ToLower(s)] {
		return Needle{}, DeclineGeneric
	}
	if strings.Contains(s, "/") || sourceExtRE.MatchString(s) {
		return Needle{Text: s, Grep: s, Kind: NeedlePath}, ""
	}
	if screamingRE.MatchString(s) || flagRE.MatchString(s) {
		return Needle{Text: s, Grep: s, Kind: NeedleToken}, ""
	}
	if m := qualifiedRE.FindStringSubmatch(s); m != nil {
		if !mixedCaps(m[2]) {
			return Needle{}, DeclineNotSymbolShape
		}
		return Needle{Text: s, Grep: m[2], Kind: NeedleSymbol, Pkg: m[1]}, ""
	}
	if identRE.MatchString(s) && mixedCaps(s) {
		return Needle{Text: s, Grep: s, Kind: NeedleSymbol}, ""
	}
	return Needle{}, DeclineNotSymbolShape
}

// mixedCaps reports whether an identifier carries BOTH an upper and a lower case
// letter — the shape of a real Go declaration (ChannelCentral, dispatchProbeCount).
// A bare all-lowercase word is usually prose in backticks; a bare all-caps word with
// no underscore is usually an acronym. Both decline.
func mixedCaps(s string) bool {
	var up, low bool
	for _, r := range s {
		switch {
		case r >= 'A' && r <= 'Z':
			up = true
		case r >= 'a' && r <= 'z':
			low = true
		}
	}
	return up && low
}

// ExtractAcceptance mines one issue body for the symbols, paths, and refusal tokens
// its acceptance commits to. It never mines prose outside the acceptance region and
// never promotes an un-backticked word; see the EXTRACTION RULE in the file header.
func ExtractAcceptance(body string) Acceptance {
	region, label := acceptanceRegion(body)
	if strings.TrimSpace(region) == "" {
		return Acceptance{Found: false, Reason: ReasonNoAcceptanceSection}
	}
	acc := Acceptance{Found: true, Section: label}
	seen := map[string]bool{}
	declined := map[string]bool{}
	for _, m := range backtickSpanRE.FindAllStringSubmatch(region, -1) {
		n, why := classifySpan(m[1])
		if why != "" {
			key := strings.TrimSpace(m[1])
			if !declined[key] {
				declined[key] = true
				acc.Declined = append(acc.Declined, Declined{Text: key, Reason: why})
			}
			continue
		}
		if seen[n.Grep] {
			continue
		}
		seen[n.Grep] = true
		acc.Needles = append(acc.Needles, n)
	}
	if len(acc.Needles) == 0 {
		acc.Reason = ReasonNoNamedSymbol
	}
	return acc
}

// CountCallers partitions a symbol's referent files into own-package, own-tests,
// architest, and PRODUCTION buckets. Only the production bucket proves the symbol is
// on a live path — the ticket's whole point is that the other three can all be
// non-zero while production behaviour is unchanged.
func CountCallers(needle, defFile string, files []string) CallerCount {
	cc := CallerCount{Needle: needle, DefFile: defFile, DefPackage: path.Dir(defFile)}
	for _, f := range files {
		f = strings.TrimSpace(strings.ReplaceAll(f, "\\", "/"))
		if f == "" {
			continue
		}
		dir := path.Dir(f)
		switch {
		case strings.HasSuffix(f, "_test.go"):
			cc.Tests++
		case dir == cc.DefPackage:
			cc.OwnPackage++
		case dir == architestDir:
			cc.Architest++
		default:
			cc.Production++
		}
	}
	cc.Unwired = cc.Production == 0
	return cc
}

// nameSeam picks the wiring site to name in a PARTIAL report. It prefers a
// production path the ACCEPTANCE ITSELF named (a non-test path needle outside the
// primitive's own package); only when the text names none does it fall back to
// DefaultSeam, and it reports which of the two happened so a reader can tell a
// quoted seam from an inferred one.
func nameSeam(acc Acceptance, defPackage string) (string, bool) {
	for _, n := range acc.Needles {
		if n.Kind != NeedlePath || strings.HasSuffix(n.Grep, "_test.go") {
			continue
		}
		p := strings.TrimSuffix(n.Grep, "/")
		if p == defPackage || strings.HasPrefix(p+"/", defPackage+"/") {
			continue
		}
		return p, true
	}
	return DefaultSeam, false
}

// Resolve folds an extraction plus caller-supplied trunk presence facts into one
// verdict. It is total and pure: same inputs, same verdict, no I/O.
//
// SHIPPED requires ALL of: an acceptance region with at least one needle, a clean
// probe for every needle, every needle present on ref, no present internal/
// symbol that reaches zero production callers, and at least one present needle
// that corroborates — binds the evidence to a specific site — rather than merely
// matching module vocabulary (#6002). Anything less is PARTIAL, OPEN, or —
// whenever the evidence itself is missing rather than negative — UNKNOWN.
func Resolve(ref string, acc Acceptance, presence []Presence) Resolution {
	if ref == "" {
		ref = DefaultRef
	}
	res := Resolution{Ref: ref, Acceptance: acc, Presence: presence}
	if !acc.Found || len(acc.Needles) == 0 {
		res.Verdict = AcceptanceUnknown
		res.Reason = acc.Reason
		if res.Reason == "" {
			res.Reason = ReasonNoNamedSymbol
		}
		res.Reason = unknownSentence(res.Reason)
		res.Remaining = "state the acceptance in an `Acceptance` section naming the symbol, path, or refusal token it commits to, then re-resolve"
		return res
	}

	byNeedle := make(map[string]Presence, len(presence))
	for _, p := range presence {
		byNeedle[p.Needle] = p
	}

	var missing, present []Needle
	for _, n := range acc.Needles {
		p, ok := byNeedle[n.Grep]
		if !ok {
			res.Verdict = AcceptanceUnknown
			res.Reason = unknownSentence(ReasonProbeMissing) + ": no presence fact for " + n.Text
			res.Remaining = "probe " + n.Text + " on " + ref + ", then re-resolve"
			return res
		}
		if strings.TrimSpace(p.Err) != "" {
			res.Verdict = AcceptanceUnknown
			res.Reason = unknownSentence(ReasonProbeFailed) + " for " + n.Text + ": " + strings.TrimSpace(p.Err)
			res.Remaining = "fix the trunk probe (is " + ref + " fetched?), then re-resolve"
			return res
		}
		if !p.Found || !qualifierMatches(n, p) {
			missing = append(missing, n)
			continue
		}
		present = append(present, n)
	}

	if len(present) == 0 {
		res.Verdict = AcceptanceOpen
		res.Reason = fmt.Sprintf("none of the %d acceptance symbol(s) named by this issue are present on %s", len(acc.Needles), ref)
		res.Remaining = "build it: " + joinNeedles(acc.Needles)
		return res
	}

	// Caller-count every present internal/ Go declaration. A primitive whose only
	// referents are its own package, its own tests, and architest is code-complete
	// and wired to nothing — PARTIAL, with the wiring named.
	var unwired []string
	defPackage := ""
	for _, n := range present {
		if n.Kind != NeedleSymbol {
			continue
		}
		p := byNeedle[n.Grep]
		def := strings.ReplaceAll(strings.TrimSpace(p.DefFile), "\\", "/")
		if def == "" || !strings.HasPrefix(def, "internal/") {
			continue
		}
		cc := CountCallers(n.Text, def, p.Files)
		res.Callers = append(res.Callers, cc)
		if cc.Unwired {
			unwired = append(unwired, n.Text)
			if defPackage == "" {
				defPackage = cc.DefPackage
			}
		}
	}
	sort.Strings(unwired)

	seam, named := nameSeam(acc, defPackage)
	var clauses []string
	if len(missing) > 0 {
		clauses = append(clauses, "land the missing acceptance item(s): "+joinNeedles(missing))
	}
	if len(unwired) > 0 {
		res.Seam, res.SeamNamed = seam, named
		clauses = append(clauses, fmt.Sprintf("wire it into %s (%s reaches zero production callers on %s — only its own package, its tests, and architest)",
			seam, strings.Join(unwired, ", "), ref))
	}
	if len(clauses) == 0 {
		if generic := genericOnly(present, byNeedle); len(generic) > 0 {
			res.Verdict = AcceptanceUnknown
			res.Reason = fmt.Sprintf("%s: every present needle (%s) is a bare identifier found in more than %d file(s) on %s — module vocabulary, not a witness for this issue's landing — NOT evidence that it shipped",
				unknownSentence(ReasonGenericPresenceOnly), strings.Join(generic, ", "), maxBareSymbolSpread, ref)
			res.Remaining = "corroborate the acceptance with a seam-specific witness — a package-qualified `pkg.Symbol`, a source path, or a refusal token from the landing — then re-resolve"
			return res
		}
		res.Verdict = AcceptanceShipped
		res.Reason = fmt.Sprintf("all %d acceptance symbol(s) are present on %s and every internal/ symbol among them reaches a production caller", len(acc.Needles), ref)
		return res
	}
	res.Verdict = AcceptancePartial
	res.Reason = fmt.Sprintf("%d of %d acceptance symbol(s) present on %s; %d unwired", len(present), len(acc.Needles), ref, len(unwired))
	res.Remaining = strings.Join(clauses, "; ")
	return res
}

// genericOnly reports whether the present evidence is ONLY generic — no needle
// specific enough to bind the trunk facts to this issue's landing (#6002). A path
// names a site, a refusal token is a declared greppable commitment, and a
// package-qualified symbol is pinned to its declaring package; each corroborates
// on its own. A BARE symbol corroborates only when its presence is narrow (at
// most maxBareSymbolSpread files): an identifier spread wider is module
// vocabulary that predates the issue, and its presence witnesses nothing.
// Returns the generic needle texts when nothing corroborates, nil when at least
// one needle does.
func genericOnly(present []Needle, byNeedle map[string]Presence) []string {
	var generic []string
	for _, n := range present {
		switch {
		case n.Kind == NeedlePath || n.Kind == NeedleToken:
			return nil
		case n.Pkg != "":
			return nil
		case len(byNeedle[n.Grep].Files) <= maxBareSymbolSpread:
			return nil
		}
		generic = append(generic, n.Text)
	}
	return generic
}

// qualifierMatches enforces a pkg.Sym qualifier against the declaration site, so a
// common bare identifier (Decide, Plan, Build) cannot resolve against an unrelated
// package's declaration and manufacture a false SHIPPED.
func qualifierMatches(n Needle, p Presence) bool {
	if n.Pkg == "" {
		return true
	}
	def := strings.ReplaceAll(strings.TrimSpace(p.DefFile), "\\", "/")
	if def == "" {
		return false
	}
	return path.Base(path.Dir(def)) == n.Pkg
}

func unknownSentence(token string) string {
	switch token {
	case ReasonNoAcceptanceSection:
		return "UNKNOWN (" + token + "): the body states no acceptance section, so nothing can be witnessed on trunk — NOT evidence that it shipped"
	case ReasonNoNamedSymbol:
		return "UNKNOWN (" + token + "): the acceptance section names no backticked symbol, path, or refusal token — NOT evidence that it shipped"
	default:
		return "UNKNOWN (" + token + ")"
	}
}

func joinNeedles(ns []Needle) string {
	out := make([]string, 0, len(ns))
	for _, n := range ns {
		out = append(out, n.Text)
	}
	return strings.Join(out, ", ")
}

// AttachResolutions binds per-issue acceptance resolutions onto a graded closure
// report and folds the per-verdict tally the renderers print. Issues with no
// resolution are left untouched (nil pointer, omitted from JSON).
func AttachResolutions(rep *Report, byIssue map[int]Resolution) {
	if rep == nil || len(byIssue) == 0 {
		return
	}
	counts := map[string]int{}
	for i := range rep.Issues {
		r, ok := byIssue[rep.Issues[i].Number]
		if !ok {
			continue
		}
		rc := r
		rep.Issues[i].Acceptance = &rc
		counts[r.Verdict]++
	}
	if len(counts) > 0 {
		rep.AcceptanceCounts = counts
	}
}

// StaleOpens are the issues the subject-grep binding MISSED: still OPEN on GitHub,
// no diff-witnessed resolving commit, yet their acceptance symbols are all present
// on trunk. These are exactly the dispatch slots a fleet wave would waste.
func StaleOpens(rep Report) []int {
	var out []int
	for _, g := range rep.Issues {
		if g.Acceptance == nil || g.Acceptance.Verdict != AcceptanceShipped {
			continue
		}
		if g.Bucket == Open || g.Bucket == OpenWitnessed {
			out = append(out, g.Number)
		}
	}
	sort.Ints(out)
	return out
}
