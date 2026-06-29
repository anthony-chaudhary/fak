// Package docfreshrsi is the RSI rung of the durable docs-freshness loop (epic
// #1278, issue #1284): it auto-applies the MECHANICAL doc-defect fixes — a missing
// orientation signpost, a missing `Read next` outbound link, a stale version pin —
// and keeps a candidate ONLY through internal/shipgate's non-forgeable keep-bit, on
// a witness the loop DERIVES itself. The three keep conditions are exactly the three
// shipgate ClassFull signals, mapped 1:1 so shipgate is reused UNCHANGED:
//
//	strict learning-debt gain  -> Witness gain (Metric=learning_debt, LowerBetter)
//	clean worktree git status  -> Witness.SuiteGreen (the candidate diff is confined
//	                              to its declared target doc and is well-formed)
//	all cited links resolve    -> Witness.TruthClean (the truth syscall)
//
// Any one failing REVERTs. The fence is anti-goodhart: a candidate that lowers the
// defect count by DELETING a cited section reverts, because a link that pointed at
// it no longer resolves — so you cannot "improve" the docs by deleting their
// content. No metric-only gain survives a dead link.
//
// It mirrors internal/rsiloop's proposer/measurer seam. The loop NEVER mutates the
// caller's corpus or `main`: every candidate is measured against an isolated Clone
// (the in-memory analog of shipgate.ApplyInWorktree's forked git worktree the real
// driver uses), kept fixes accumulate only in the corpus Refresh RETURNS, and
// landing them on main is the caller's separate, logged step — Refresh writes
// nothing outside memory and hands back the per-candidate verdict log to land from.
package docfreshrsi

import (
	"regexp"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/shipgate"
)

// Corpus is the in-memory document set the loop reasons over, keyed by a logical doc
// path (e.g. "docs/fak/edge-quickstart.md") with the doc content as the value. Docs
// cite each other by that same key (optionally with a `#anchor` fragment), so a
// cross-doc link resolves iff the key exists and the anchor slug is a heading in it.
type Corpus map[string]string

// Clone returns a deep copy. Every candidate is applied to a Clone so the caller's
// corpus (and, in the real driver, `main`) is never the tree being rewritten.
func (c Corpus) Clone() Corpus {
	out := make(Corpus, len(c))
	for k, v := range c {
		out[k] = v
	}
	return out
}

// Target is the freshness reference the mechanical detectors measure against.
type Target struct {
	// Version is the current release token (e.g. "v0.7.0"). A version pin naming any
	// OTHER vX.Y.Z token is a stale-pin defect; a pin already equal to it is clean.
	Version string
	// KnownCommands optionally names the programs a cited command may invoke (the
	// first token of a fenced command line, e.g. "./fak", "make", "go"). When empty,
	// command resolution is skipped (links remain the truth signal); when supplied, a
	// cited command whose program is not in the set makes the witness truth-dirty.
	KnownCommands map[string]bool
}

// DefectClass names the three MECHANICAL defect classes the loop auto-fixes.
type DefectClass uint8

const (
	OrientationSignpost DefectClass = iota // no top-of-doc orientation signpost line
	ReadNextLink                           // no `## Read next` section with an outbound link
	VersionPin                             // one or more stale version pins (vX.Y.Z != Target.Version)
)

// String renders the defect class as a stable token.
func (d DefectClass) String() string {
	switch d {
	case OrientationSignpost:
		return "orientation-signpost"
	case ReadNextLink:
		return "read-next-link"
	case VersionPin:
		return "version-pin"
	}
	return "?"
}

var (
	orientationRe = regexp.MustCompile(`(?m)^>\s*Orientation\b`)
	versionRe     = regexp.MustCompile(`v\d+\.\d+\.\d+`)
	headingRe     = regexp.MustCompile(`(?m)^#{1,6}\s+(.+?)\s*#*\s*$`)
	linkRe        = regexp.MustCompile(`\[[^\]]*\]\(([^)\s]+)\)`)
)

// hasOrientation reports whether the doc carries a top-of-doc orientation signpost.
func hasOrientation(body string) bool { return orientationRe.MatchString(body) }

// hasReadNext reports whether the doc has a `## Read next` section that carries at
// least one outbound markdown link (an empty heading does not count).
func hasReadNext(body string) bool {
	i := strings.Index(body, "## Read next")
	return i >= 0 && linkRe.MatchString(body[i:])
}

// stalePins counts version tokens in the doc that name a release OTHER than version
// — each is one stale-pin defect. A pin already equal to version (or no pin at all)
// adds nothing.
func stalePins(body, version string) int {
	n := 0
	for _, m := range versionRe.FindAllString(body, -1) {
		if m != version {
			n++
		}
	}
	return n
}

// Defects lists the mechanical defects present in one doc, most-fixable-first.
func Defects(body string, tgt Target) []DefectClass {
	var out []DefectClass
	if !hasOrientation(body) {
		out = append(out, OrientationSignpost)
	}
	if !hasReadNext(body) {
		out = append(out, ReadNextLink)
	}
	if stalePins(body, tgt.Version) > 0 {
		out = append(out, VersionPin)
	}
	return out
}

// Debt is the corpus learning-debt: the total count of concrete mechanical defects —
// missing orientation signpost (1/doc), missing read-next link (1/doc), and every
// stale version pin (N/doc). It is an integer you can drive toward zero, and the
// metric shipgate's keep-bit requires a STRICT decrease of before a fix is kept.
func Debt(c Corpus, tgt Target) int {
	total := 0
	for _, body := range c {
		if !hasOrientation(body) {
			total++
		}
		if !hasReadNext(body) {
			total++
		}
		total += stalePins(body, tgt.Version)
	}
	return total
}

// ----------------------------------------------------------------------------
// Truth syscall: every cited link resolves (and, when an oracle is supplied, every
// cited command). This is the anti-goodhart fence — a debt drop bought by deleting a
// cited section leaves a dangling link, so the witness is truth-dirty and reverts.
// ----------------------------------------------------------------------------

// LinksResolve reports whether every cited link in the corpus resolves, returning the
// dangling "<doc> -> <target>" pairs as a diagnostic. Same-doc `#anchor` links must
// hit a heading slug in the doc; cross-doc `file.md#anchor` links must hit an existing
// corpus key and (if a fragment is given) a heading in it. External URLs (http/mailto)
// are not reachable offline and are treated as resolving — they are out of this loop's
// scope, by design, so the fence stays a deterministic, no-network check.
func LinksResolve(c Corpus) (bool, []string) {
	ok := true
	var dangling []string
	for _, k := range sortedKeys(c) {
		for _, tgt := range linkTargets(c[k]) {
			if !resolves(c, k, tgt) {
				ok = false
				dangling = append(dangling, k+" -> "+tgt)
			}
		}
	}
	return ok, dangling
}

// truthClean folds link resolution with optional command resolution into the single
// TruthClean bit shipgate.Evaluate consumes.
func truthClean(c Corpus, tgt Target) (bool, []string) {
	ok, dangling := LinksResolve(c)
	if cok, bad := commandsResolve(c, tgt.KnownCommands); !cok {
		ok = false
		dangling = append(dangling, bad...)
	}
	return ok, dangling
}

func resolves(c Corpus, fromKey, target string) bool {
	if target == "" {
		return false
	}
	if isExternal(target) {
		return true
	}
	file, anchor := target, ""
	if i := strings.IndexByte(target, '#'); i >= 0 {
		file, anchor = target[:i], target[i+1:]
	}
	body := c[fromKey] // same-doc anchor (file == "")
	if file != "" {
		b, found := lookupDoc(c, file)
		if !found {
			return false
		}
		body = b
	}
	if anchor == "" {
		return true
	}
	return headingSlugs(body)[anchor]
}

func isExternal(target string) bool {
	return strings.Contains(target, "://") ||
		strings.HasPrefix(target, "mailto:") ||
		strings.HasPrefix(target, "tel:")
}

// lookupDoc resolves a link's file part to a corpus body: an exact key match first,
// then a base-name match (a relative `../foo.md` link to a `docs/x/foo.md` key).
func lookupDoc(c Corpus, file string) (string, bool) {
	if body, ok := c[file]; ok {
		return body, true
	}
	fb := baseName(file)
	for _, k := range sortedKeys(c) {
		if baseName(k) == fb {
			return c[k], true
		}
	}
	return "", false
}

func linkTargets(body string) []string {
	var out []string
	for _, m := range linkRe.FindAllStringSubmatch(body, -1) {
		out = append(out, m[1])
	}
	return out
}

func headingSlugs(body string) map[string]bool {
	out := map[string]bool{}
	for _, m := range headingRe.FindAllStringSubmatch(body, -1) {
		out[slug(m[1])] = true
	}
	return out
}

// slug renders heading text as a GitHub-style anchor: lowercased, non-alphanumerics
// dropped, runs of spaces/dashes/underscores folded to a single dash.
func slug(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	var b strings.Builder
	lastDash := false
	for _, r := range s {
		switch {
		case (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9'):
			b.WriteRune(r)
			lastDash = false
		case r == ' ' || r == '-' || r == '_':
			if !lastDash {
				b.WriteByte('-')
				lastDash = true
			}
		}
	}
	return strings.Trim(b.String(), "-")
}

func baseName(p string) string {
	if i := strings.LastIndexByte(p, '/'); i >= 0 {
		return p[i+1:]
	}
	return p
}

// commandsResolve checks every fenced command line against the supplied oracle. An
// empty oracle means "no command check" — links remain the truth signal.
func commandsResolve(c Corpus, known map[string]bool) (bool, []string) {
	if len(known) == 0 {
		return true, nil
	}
	ok := true
	var bad []string
	for _, k := range sortedKeys(c) {
		for _, cmd := range fencedCommands(c[k]) {
			prog := firstToken(cmd)
			if prog == "" {
				continue
			}
			if !known[prog] {
				ok = false
				bad = append(bad, k+" -> cmd:"+prog)
			}
		}
	}
	return ok, bad
}

// fencedCommands returns the non-empty command lines inside ``` code fences.
func fencedCommands(body string) []string {
	var out []string
	inFence := false
	for _, ln := range strings.Split(body, "\n") {
		if strings.HasPrefix(strings.TrimSpace(ln), "```") {
			inFence = !inFence
			continue
		}
		if inFence {
			if t := strings.TrimSpace(ln); t != "" {
				out = append(out, t)
			}
		}
	}
	return out
}

func firstToken(s string) string {
	s = strings.TrimSpace(s)
	if i := strings.IndexAny(s, " \t"); i >= 0 {
		return s[:i]
	}
	return s
}

// ----------------------------------------------------------------------------
// Propose: derive one mechanical-fix candidate per defect (the smallest step).
// ----------------------------------------------------------------------------

// Patch transforms a corpus. It is applied to a Clone and may mutate+return it; the
// caller's corpus is never touched.
type Patch func(Corpus) Corpus

// Candidate is one proposed mechanical fix targeting exactly one doc.
type Candidate struct {
	Label string
	Class DefectClass
	Doc   string // the single doc the fix is confined to
	Apply Patch
}

// Propose derives the auto-fix candidates for every mechanical defect in the corpus,
// in deterministic order (doc key, then orientation/read-next/version). Each fix ADDS
// or REWRITES resolving content — it never deletes a section — so a proposed fix is
// truth-clean by construction; the keep-bit still has to confirm the strict debt gain
// and the clean diff before it is kept.
func Propose(c Corpus, tgt Target) []Candidate {
	keys := sortedKeys(c)
	var out []Candidate
	for _, k := range keys {
		body := c[k]
		if !hasOrientation(body) {
			doc := k
			out = append(out, Candidate{
				Label: "orientation:" + k, Class: OrientationSignpost, Doc: k,
				Apply: func(cc Corpus) Corpus { cc[doc] = insertOrientation(cc[doc]); return cc },
			})
		}
		if !hasReadNext(body) {
			if target := firstOther(keys, k); target != "" {
				doc, t := k, target
				out = append(out, Candidate{
					Label: "read-next:" + k, Class: ReadNextLink, Doc: k,
					Apply: func(cc Corpus) Corpus { cc[doc] = appendReadNext(cc[doc], t); return cc },
				})
			}
		}
		if stalePins(body, tgt.Version) > 0 {
			doc, ver := k, tgt.Version
			out = append(out, Candidate{
				Label: "version-pin:" + k, Class: VersionPin, Doc: k,
				Apply: func(cc Corpus) Corpus { cc[doc] = rewriteStalePins(cc[doc], ver); return cc },
			})
		}
	}
	return out
}

// insertOrientation inserts the signpost line just after the H1 (or at the top when
// the doc has no H1). It only ADDS a line — no existing anchor is disturbed.
func insertOrientation(body string) string {
	const line = "> Orientation: who this is for, what you need first, and what you'll be able to do."
	lines := strings.Split(body, "\n")
	for i, ln := range lines {
		if strings.HasPrefix(ln, "# ") {
			out := append([]string{}, lines[:i+1]...)
			out = append(out, "", line)
			out = append(out, lines[i+1:]...)
			return strings.Join(out, "\n")
		}
	}
	return line + "\n\n" + body
}

// appendReadNext appends a `## Read next` section whose single link points at an
// existing corpus key, so it resolves the moment it is written.
func appendReadNext(body, target string) string {
	return strings.TrimRight(body, "\n") + "\n\n## Read next\n\n- [" + target + "](" + target + ")\n"
}

// rewriteStalePins rewrites every stale version token to the target release IN PLACE,
// fixing the pin without removing the heading/anchor that surrounds it.
func rewriteStalePins(body, version string) string {
	return versionRe.ReplaceAllStringFunc(body, func(m string) string {
		if m == version {
			return m
		}
		return version
	})
}

// ----------------------------------------------------------------------------
// Keep-or-revert via shipgate (reused UNCHANGED).
// ----------------------------------------------------------------------------

// Verdict is the per-candidate record the loop logs. Kept is the non-forgeable
// keep-bit copied straight from shipgate — the loop cannot fabricate a KEEP.
type Verdict struct {
	Candidate    string
	Class        DefectClass
	Doc          string
	DebtBefore   int
	DebtAfter    int
	Improved     bool     // a STRICT learning-debt decrease
	Clean        bool     // diff confined to the target doc + well-formed (clean-status analog)
	LinksResolve bool     // the truth syscall: every cited link (+command) resolves
	Dangling     []string // links/commands that no longer resolve (REVERT diagnostic)
	Decision     string   // KEEP | REVERT | ESCALATE
	Kept         bool     // the shipgate keep-bit
}

// EvaluateCandidate measures one candidate against an isolated clone of base and folds
// the three derived signals through shipgate.Evaluate. It returns the verdict and the
// corpus to carry forward: the candidate corpus on KEEP, the UNCHANGED base on REVERT.
// base is never mutated.
func EvaluateCandidate(base Corpus, c Candidate, tgt Target) (Verdict, Corpus) {
	before := Debt(base, tgt)
	cand := c.Apply(base.Clone())
	after := Debt(cand, tgt)
	clean := confined(base, cand, c.Doc) && wellFormed(cand)
	links, dangling := truthClean(cand, tgt)

	w := shipgate.Witness{
		Class:       shipgate.ClassFull, // all three signals must hold (the issue's "any one failing REVERTs")
		Metric:      "learning_debt",
		Before:      float64(before),
		After:       float64(after),
		LowerBetter: true, // a SMALLER debt is the gain
		SuiteGreen:  clean,
		TruthClean:  links,
	}
	d, ev := shipgate.Evaluate(w)
	v := Verdict{
		Candidate:    c.Label,
		Class:        c.Class,
		Doc:          c.Doc,
		DebtBefore:   before,
		DebtAfter:    after,
		Improved:     after < before,
		Clean:        clean,
		LinksResolve: links,
		Dangling:     dangling,
		Decision:     d.String(),
		Kept:         ev.Kept(),
	}
	if ev.Kept() {
		return v, cand
	}
	return v, base
}

// Refresh runs the whole rung over base: it derives the mechanical-fix candidates and
// folds each through the keep-bit against the CURRENT kept corpus (so each kept fix is
// the new bar the next candidate competes against — the recursion). It returns the
// final kept corpus and the per-candidate verdict log. It NEVER mutates base or `main`:
// kept fixes accumulate only in the returned copy, and landing them is the caller's
// separate, logged step (Kept rows of the returned log name exactly what to land).
func Refresh(base Corpus, tgt Target) (Corpus, []Verdict) {
	kept := base.Clone()
	var log []Verdict
	for _, c := range Propose(kept, tgt) {
		v, next := EvaluateCandidate(kept, c, tgt)
		log = append(log, v)
		kept = next
	}
	return kept, log
}

// confined reports whether the candidate touched ONLY its declared target doc — same
// key set, every other doc byte-identical. The in-memory analog of an otherwise-clean
// `git status`: the loop changed exactly what it intended, nothing stray.
func confined(base, cand Corpus, doc string) bool {
	if len(base) != len(cand) {
		return false
	}
	if _, ok := base[doc]; !ok {
		return false
	}
	for k, bv := range base {
		cv, ok := cand[k]
		if !ok {
			return false
		}
		if k != doc && bv != cv {
			return false
		}
	}
	return true
}

// wellFormed rejects a candidate that left a git conflict marker behind — a torn apply
// that a clean worktree would never carry.
func wellFormed(c Corpus) bool {
	for _, body := range c {
		if strings.Contains(body, "<<<<<<<") || strings.Contains(body, ">>>>>>>") {
			return false
		}
	}
	return true
}

func sortedKeys(c Corpus) []string {
	keys := make([]string, 0, len(c))
	for k := range c {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// firstOther returns the first sorted key that is not k (a deterministic, existing
// link target for an appended read-next section), or "" when k is the only doc.
func firstOther(keys []string, k string) string {
	for _, c := range keys {
		if c != k {
			return c
		}
	}
	return ""
}
