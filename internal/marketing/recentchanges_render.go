// recentchanges_render.go — the Markdown surface of the recent-changes fold (#6040).
//
// Split from recentchanges.go along the data/render seam: BuildRecentChanges decides WHAT
// the page says (and is what --json emits), this file decides how it reads. Everything here
// is a pure function of the page value, so regenerating the same range twice produces the
// same bytes — the property the freshness check relies on to detect hand edits and drift.
package marketing

import (
	"fmt"
	"strings"
)

const repoIssueURL = "https://github.com/anthony-chaudhary/fak/issues/"
const claimsURL = repoBlobURL + "CLAIMS.md"

// releaseNotesURL points at the release-note authority. It is an ABSOLUTE url on purpose:
// docs/releases/ is excluded from the published Jekyll site, so a relative link there
// resolves in the repo but 404s for a reader who arrived from search.
const releaseNotesURL = repoBlobURL + "docs/releases/README.md"

// recentShortSHA is the citation form: long enough to be unambiguous in this repository,
// short enough to scan. The link always carries the full sha.
func recentShortSHA(sha string) string {
	if len(sha) > 9 {
		return sha[:9]
	}
	return sha
}

// escapeRecentText neutralizes the markdown-active characters a commit subject can carry so
// a stray bracket or angle bracket cannot break the rendered line (or smuggle raw HTML into
// the published page).
func escapeRecentText(s string) string {
	r := strings.NewReplacer("[", "\\[", "]", "\\]", "<", "&lt;", ">", "&gt;", "|", "\\|")
	return r.Replace(s)
}

// Markdown renders the whole page, front matter included.
func (p RecentChangesPage) Markdown() string {
	var b strings.Builder
	b.WriteString("---\n")
	b.WriteString("title: \"What's new in fak — recent changes witnessed by commit\"\n")
	b.WriteString("description: \"A generated, commit-witnessed summary of recent fak changes: grouped themes, an explicit freshness boundary, and a link that verifies every line.\"\n")
	b.WriteString("---\n\n")
	b.WriteString("# What's new in fak\n\n")
	b.WriteString(p.opener())
	b.WriteString(p.coverage())
	b.WriteString(p.freshness())
	b.WriteString(p.themes())
	b.WriteString(p.verification())
	b.WriteString(p.limits())
	b.WriteString(p.nextRoutes())
	return b.String()
}

func (p RecentChangesPage) opener() string {
	return fmt.Sprintf(`> **In one breath:** This page groups recent witnessed fak changes into human-readable themes.
> Each item names its shipped or planned scope and links to proof. It is current through
> commit `+"`%s`"+` on %s.

Every entry below is one commit that carries a per-leaf ship stamp, so each line is traceable
to repository evidence rather than to a hand-written claim. The page is refreshed by one
committed command, and it summarizes — it is not a complete changelog and it promises no
release cadence.

`, recentShortSHA(p.AnchorSHA), p.AnchorDate.Format("2006-01-02"))
}

func (p RecentChangesPage) coverage() string {
	var b strings.Builder
	b.WriteString("## What does this page cover?\n\n")
	version := ""
	if strings.TrimSpace(p.Version) != "" {
		version = fmt.Sprintf(" The repository release marker at that commit is `%s`.", strings.TrimSpace(p.Version))
	}
	b.WriteString(fmt.Sprintf("It covers %s: **%d stamped changes** across **%d non-merge commits**, grouped\ninto the themes below.%s\n\n",
		p.RangeLabel, p.Ships, p.Commits, version))
	b.WriteString("A stamped change is a commit whose subject carries a `(fak <leaf>)` trailer — the same\n")
	b.WriteString("witness the commit-message gate and the marketing feeds bind to. Every entry carries the\n")
	b.WriteString("scope it may be claimed at:\n\n")
	b.WriteString("| Scope | What it means | How it is decided |\n|---|---|---|\n")
	b.WriteString(fmt.Sprintf("| `%s` | The change landed on trunk and its subsystem is claimed as shipped. | Stamped commit whose leaf and issue pass the [`CLAIMS.md`](%s) honesty ledger. |\n", RecentScopeShipped, claimsURL))
	b.WriteString(fmt.Sprintf("| `%s` | A plan, note, or experiment — not a capability you can use. | Every path the commit touched is a research, planning, or notes surface. |\n", RecentScopeResearch))
	b.WriteString(fmt.Sprintf("| `%s` | Real work landed, but the capability is still `[SIMULATED]`/`[STUB]`. | [`CLAIMS.md`](%s) still tags the commit's leaf or issue as unshipped. |\n\n", RecentScopeUnclaimed, claimsURL))
	return b.String()
}

func (p RecentChangesPage) freshness() string {
	var b strings.Builder
	b.WriteString("## How fresh is this page, and how is it refreshed?\n\n")
	b.WriteString("The freshness boundary is explicit and machine-checkable — the page carries the commit it\n")
	b.WriteString("was folded from, so staleness is a computation rather than a judgement call.\n\n")
	b.WriteString("| Field | Value |\n|---|---|\n")
	b.WriteString(fmt.Sprintf("| Current through | `%s` |\n", p.AnchorSHA))
	b.WriteString(fmt.Sprintf("| Commit date | %s |\n", p.AnchorDate.Format("2006-01-02")))
	b.WriteString(fmt.Sprintf("| Range folded | `%s` |\n", p.RangeSpec))
	b.WriteString(fmt.Sprintf("| Counted | %d stamped changes / %d non-merge commits |\n", p.Ships, p.Commits))
	if strings.TrimSpace(p.GeneratorModule) != "" {
		b.WriteString(fmt.Sprintf("| Generator module | `%s` |\n", p.GeneratorModule))
	}
	b.WriteString(fmt.Sprintf("| Source of truth | `git log` + linked GitHub issues + [`CLAIMS.md`](%s), folded by the derived generator module above |\n", claimsURL))
	b.WriteString(fmt.Sprintf("| Refresh | `%s --write` |\n", RecentChangesVerb))
	b.WriteString(fmt.Sprintf("| Staleness check | `%s --check --json` (default ceiling: 7 commit-date days or 250 non-merge commits behind) |\n\n", RecentChangesVerb))
	b.WriteString(p.AnchorComment() + "\n\n")
	b.WriteString(fmt.Sprintf("`%s --check` reads that anchor, measures how far `HEAD` has moved past it, and\n", RecentChangesVerb))
	b.WriteString("regenerates the page over the recorded range. It exits non-zero when the page is older than\n")
	b.WriteString("the allowed window or no longer matches what the repository says, so a hand edit or a\n")
	b.WriteString("rotted snapshot is reported instead of trusted. Nothing on this page is maintained by\n")
	b.WriteString("hand: the only curated content in the generator is the theme titles and their \"why it\n")
	b.WriteString("matters\" lines, which describe subsystems rather than individual changes.\n\n")
	return b.String()
}

func (p RecentChangesPage) themes() string {
	var b strings.Builder
	b.WriteString("## What changed recently?\n\n")
	if len(p.Groups) == 0 {
		b.WriteString("No stamped changes landed in this window. That is an honest empty result, not a\n")
		b.WriteString("rendering failure — widen the window with `" + RecentChangesVerb + " --days 30 --write`.\n\n")
		return b.String()
	}
	b.WriteString("Each theme names why its subsystems matter, then lists its most user-facing changes\n")
	b.WriteString("newest-first. Features and fixes are listed before supporting work, and every theme states\n")
	b.WriteString("how many changes it counted so the summary never hides its own truncation.\n\n")
	for _, g := range p.Groups {
		b.WriteString(g.markdown(p.PerTheme, p.RangeSpec))
	}
	return b.String()
}

func (g RecentGroup) markdown(perTheme int, rangeSpec string) string {
	var b strings.Builder
	b.WriteString("### " + g.Title + "\n\n")
	b.WriteString("**Why it matters:** " + g.Matters + "\n\n")
	b.WriteString(fmt.Sprintf("**In this window:** %s across %s — %s.\n\n",
		plural(g.Total, "stamped change", "stamped changes"), leafList(g.Leaves), g.mix()))
	for _, it := range g.Items {
		b.WriteString(it.markdown())
	}
	if g.Total > len(g.Items) {
		b.WriteString(fmt.Sprintf("\n%d further change(s) in this theme are counted but not listed. To read the complete stamped history for this range and filter by the subsystems above:\n```bash\ngit log --no-merges --oneline %s --grep \"(fak \"\n```\n\n",
			g.Total-len(g.Items), rangeSpec))
	} else {
		b.WriteString("\n")
	}
	return b.String()
}

// mix is the honest composition line: what kind of work this theme actually held.
func (g RecentGroup) mix() string {
	parts := []string{
		fmt.Sprintf("%d feature(s)", g.Features),
		fmt.Sprintf("%d fix(es)", g.Fixes),
		fmt.Sprintf("%d supporting change(s)", g.Supporting),
	}
	if g.Research > 0 {
		parts = append(parts, fmt.Sprintf("%d research/plan", g.Research))
	}
	if g.Unclaimed > 0 {
		parts = append(parts, fmt.Sprintf("%d landed but not claimed shipped", g.Unclaimed))
	}
	return strings.Join(parts, ", ")
}

func (it RecentItem) markdown() string {
	line := fmt.Sprintf("- **[`%s`](%s%s)** %s — %s — subsystem `%s`, scope `%s`",
		recentShortSHA(it.SHA), repoCommitURL, it.SHA, it.Date.Format("2006-01-02"),
		escapeRecentText(it.Subject), it.Leaf, it.Scope)
	if it.Issue > 0 {
		line += fmt.Sprintf(", issue [#%d](%s%d)", it.Issue, repoIssueURL, it.Issue)
	}
	return line + "\n"
}

func (p RecentChangesPage) verification() string {
	return fmt.Sprintf(`## Where can I verify each item?

Every line above is a commit sha, and the sha is the verification. From a checkout:

`+"```bash"+`
git show <sha>                                   # the exact diff behind any line above
git log --no-merges --oneline %s --grep "(fak <leaf>)"   # every counted change in one subsystem
fak version modules --only internal/marketing    # current derived module@rev identity
%s --check --json                # this page's own freshness, as JSON
`+"```"+`

Each entry also links its commit on GitHub and, when the subject names one, the issue that
scoped it. Scope labels come from [`+"`CLAIMS.md`"+`](%s), which records what is shipped,
simulated, or stubbed; tagged release notes live in the
[release-notes index](%s). The generator identity above uses the repository's derived
[`+"`module@rev`"+` contract](notes/VERSION-EVERYTHING-SPINE-2026-07-03.md), pinned to the page's
anchor rather than borrowed from whatever binary performs the refresh.

`, p.RangeSpec, RecentChangesVerb, claimsURL, releaseNotesURL)
}

func (p RecentChangesPage) limits() string {
	return `## What this page does not claim

- **No release cadence.** A window with many changes and a window with few are both honest;
  nothing here promises when the next tagged release lands.
- **No unwitnessed claims.** A commit without a per-leaf ship stamp is counted in the
  non-merge commit total but is never listed as a change, and a capability still tagged
  ` + "`[SIMULATED]`/`[STUB]`" + ` in ` + "`CLAIMS.md`" + ` is downgraded rather than advertised.
- **Not a complete changelog.** Each theme shows a bounded slice and states how many more it
  counted; the ` + "`git log`" + ` recipe above is the complete list.
- **Not a concept explanation.** The themes name subsystems and route onward; they do not
  re-explain what the kernel is.

`
}

func (p RecentChangesPage) nextRoutes() string {
	return fmt.Sprintf(`## Where to go next

This page is a recent-changes view, not a general entry point. For anything other than
"what changed lately", use the authority that owns it:

- **Tagged releases and upgrade notes:** the [release-notes index](%s).
- **What is shipped, simulated, or stubbed:** [`+"`CLAIMS.md`"+`](%s).
- **Plans and research, before they ship:** [`+"`docs/research/`"+`](research/README.md).
- **The machine-readable version of this feed:** [`+"`docs/marketing/`"+`](marketing/README.md),
  which publishes the same witnessed ships as JSON-LD and plain text for answer engines.
`, releaseNotesURL, claimsURL)
}

func plural(n int, one, many string) string {
	if n == 1 {
		return fmt.Sprintf("%d %s", n, one)
	}
	return fmt.Sprintf("%d %s", n, many)
}

// leafList renders the subsystems a theme actually touched, bounded so a broad theme does
// not print thirty leaf names.
func leafList(leaves []string) string {
	const max = 6
	shown := leaves
	suffix := ""
	if len(shown) > max {
		shown = shown[:max]
		suffix = fmt.Sprintf(" and %d more", len(leaves)-max)
	}
	quoted := make([]string, 0, len(shown))
	for _, l := range shown {
		quoted = append(quoted, "`"+l+"`")
	}
	if len(quoted) == 0 {
		return "no subsystem"
	}
	return strings.Join(quoted, ", ") + suffix
}
