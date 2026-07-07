package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/devindex"
)

// The compact-overview taste gate. `fak --help` regressed into a ~650-line wall
// once; these tests hold the new shape: the overview stays under a screen, every
// verb it advertises is real, the per-verb carver finds the deep documentation,
// and a mistyped verb gets a suggestion instead of the wall.

// TestUsageCompactStaysCompact is the anti-wall ratchet: the curated overview
// must fit comfortably on one screen. If this reds, cut overview entries — do
// not widen the budget; depth belongs in `fak help <verb>` and `fak help --all`.
func TestUsageCompactStaysCompact(t *testing.T) {
	var b strings.Builder
	usageCompact(&b)
	lines := strings.Split(strings.TrimRight(b.String(), "\n"), "\n")
	if len(lines) > 45 {
		t.Fatalf("compact overview is %d lines; the budget is 45 — trim overviewGroups, don't grow the wall back", len(lines))
	}
	for _, ln := range lines {
		if n := len([]rune(ln)); n > 100 {
			t.Errorf("overview line %d runes wide (max 100): %q", n, ln)
		}
	}
}

// TestOverviewVerbsAreLive pins every curated overview verb to the live verb
// catalog (dispatch-derived when the repo is readable), so the overview can
// never advertise a verb the binary does not route.
func TestOverviewVerbsAreLive(t *testing.T) {
	cat := helpCatalog()
	if cat == nil {
		t.Skip("devindex catalog unavailable (no repo root); overview membership is only checkable in-repo")
	}
	live := map[string]bool{}
	for _, v := range cat.Verbs() {
		for _, sp := range v.Spellings() {
			live[strings.ToLower(sp)] = true
		}
	}
	for _, g := range overviewGroups {
		for _, e := range g.entries {
			if !live[e.name] {
				t.Errorf("overview advertises %q under %q but no dispatched verb has that spelling", e.name, g.title)
			}
		}
	}
}

// TestOverviewIsExactlyFrontdoor is the C3 (#2232) set-equality gate: the compact
// overview presents the FRONTDOOR tier and only the frontdoor tier, so C1's
// classification (internal/devindex) and the visible front door can never drift.
// It reds two ways: a dev/hidden verb sneaking into the overview, or a frontdoor
// verb missing from it (not even as a declared companion fold).
func TestOverviewIsExactlyFrontdoor(t *testing.T) {
	cat := helpCatalog()
	if cat == nil {
		t.Skip("devindex catalog unavailable (no repo root); tier equality is only checkable in-repo")
	}
	// 1. Nothing but frontdoor appears in the overview.
	inOverview := map[string]bool{}
	for _, g := range overviewGroups {
		for _, e := range g.entries {
			inOverview[e.name] = true
			tier, ok := devindex.TierOf(e.name)
			if !ok || tier != devindex.TierFrontdoor {
				t.Errorf("overview lists %q (tier %q, ok=%v) — the overview is frontdoor-ONLY; dev/repo tooling lives under 'fak dev'", e.name, tier, ok)
			}
		}
	}
	// 2. Every frontdoor verb is covered — its own line, or a declared companion
	//    (replay/top/pull/ls) whose primary is present AND names it in the blurb.
	blurbOf := map[string]string{}
	for _, g := range overviewGroups {
		for _, e := range g.entries {
			blurbOf[e.name] = e.blurb
		}
	}
	covered := func(v string) bool {
		if inOverview[v] {
			return true
		}
		p, ok := frontdoorCompanions[v]
		if !ok || !inOverview[p] {
			return false
		}
		// The fold must keep the spelling visible in its primary's blurb.
		return strings.Contains(blurbOf[p], v)
	}
	for _, fv := range cat.Verbs() {
		if fv.Tier != devindex.TierFrontdoor {
			continue
		}
		if !covered(fv.Name) {
			t.Errorf("frontdoor verb %q is missing from the compact overview (add a line, or fold it via frontdoorCompanions and name it in the primary's blurb)", fv.Name)
		}
	}
}

// TestUsageAllVerbsShowsTier is the C3 (#2232) tiered-catalog gate: `fak help --all`
// carries each verb's tier, and hidden re-exec seams stay unlisted.
func TestUsageAllVerbsShowsTier(t *testing.T) {
	if helpCatalog() == nil {
		t.Skip("devindex catalog unavailable (no repo root); --all tier column is only checkable in-repo")
	}
	var b strings.Builder
	usageAllVerbs(&b)
	out := b.String()
	for _, want := range []string{"guard", "[frontdoor]", "commit", "[dev]"} {
		if !strings.Contains(out, want) {
			t.Errorf("fak help --all missing %q; got:\n%s", want, out)
		}
	}
	// The frontdoor 'guard' and the dev 'commit' must be tagged with their tiers.
	for _, ln := range strings.Split(out, "\n") {
		f := strings.Fields(ln)
		if len(f) >= 2 && f[0] == "guard" && f[1] != "[frontdoor]" {
			t.Errorf("guard line not tagged frontdoor: %q", ln)
		}
		if len(f) >= 2 && f[0] == "commit" && f[1] != "[dev]" {
			t.Errorf("commit line not tagged dev: %q", ln)
		}
		// Hidden seams are never listed.
		if len(f) > 0 && (f[0] == "guard-stophook" || f[0] == "ablate-arm" || f[0] == "hook") {
			t.Errorf("fak help --all leaked a hidden re-exec seam: %q", ln)
		}
	}
}

// TestDevVerbHelpNamesDevSpelling is the C3 (#2232) header gate: `fak help <devverb>`
// still works (help is never gated) and its header names the canonical `fak dev <verb>`
// spelling, while a frontdoor verb keeps the bare `fak <verb>` spelling.
func TestDevVerbHelpNamesDevSpelling(t *testing.T) {
	if helpCatalog() == nil {
		t.Skip("devindex catalog unavailable (no repo root); dev-spelling header is only checkable in-repo")
	}
	var dev strings.Builder
	if !verbDeepHelpBody(&dev, "sweep") {
		t.Fatal("verbDeepHelpBody found nothing for the dev verb 'sweep'")
	}
	if !strings.Contains(dev.String(), "fak dev sweep") {
		t.Errorf("`fak help sweep` header must name the canonical 'fak dev sweep' spelling; got:\n%s", dev.String())
	}
	var front strings.Builder
	if !verbDeepHelpBody(&front, "guard") {
		t.Fatal("verbDeepHelpBody found nothing for the frontdoor verb 'guard'")
	}
	if got := front.String(); !strings.Contains(got, "fak guard ") || strings.Contains(got, "fak dev guard") {
		t.Errorf("`fak help guard` header must keep the bare 'fak guard' spelling; got:\n%s", got)
	}
}

// TestSuggestVerbSpellingIsTierAware is the C3 (#2232) did-you-mean gate: a
// top-level near-miss of a DEV verb yields the dev-prefixed spelling (so the C5
// enforcement flip is a one-line change), while a frontdoor near-miss stays bare.
func TestSuggestVerbSpellingIsTierAware(t *testing.T) {
	if got := suggestVerbSpelling("swep"); got != "dev sweep" {
		t.Errorf("suggestVerbSpelling(swep) = %q, want \"dev sweep\" (tier-aware)", got)
	}
	if got := suggestVerbSpelling("guardd"); got != "guard" {
		t.Errorf("suggestVerbSpelling(guardd) = %q, want \"guard\" (frontdoor stays bare)", got)
	}
	if got := suggestVerbSpelling("zzqx"); got != "" {
		t.Errorf("suggestVerbSpelling(zzqx) = %q, want no suggestion", got)
	}
}

// TestVerbWallSectionsCarvesDepth proves the per-verb carver recovers the deep
// documentation from the wall constants — the depth `fak --help` used to dump
// must remain reachable, verb by verb.
func TestVerbWallSectionsCarvesDepth(t *testing.T) {
	sections := verbWallSections([]string{"commit"})
	if len(sections) == 0 {
		t.Fatal("verbWallSections found no wall block for 'commit'")
	}
	joined := strings.Join(sections, "")
	if !strings.Contains(joined, "SAFE SHARED-TRUNK COMMIT") {
		t.Errorf("commit wall block lost its paragraph; got:\n%s", joined)
	}
	if strings.Contains(joined, "fak edit-tx") {
		t.Errorf("commit wall block leaked the next verb's section:\n%s", joined)
	}
	// A verb documented mid-wall with a one-line entry still carves cleanly.
	if s := verbWallSections([]string{"version"}); len(s) == 0 {
		t.Error("verbWallSections found no wall block for 'version'")
	}
	if s := verbWallSections([]string{"no-such-verb-ever"}); len(s) != 0 {
		t.Errorf("carver invented a section for an unknown verb: %q", s)
	}
}

func TestSyncHelpNamesPushAndDirtyGuidance(t *testing.T) {
	if cat := helpCatalog(); cat != nil {
		var synopsis string
		for _, v := range cat.Verbs() {
			if v.Name == "sync" {
				synopsis = v.Synopsis
				break
			}
		}
		if !strings.Contains(synopsis, "sync/push") {
			t.Fatalf("sync catalog synopsis = %q, want it to name sync/push", synopsis)
		}
	}

	sections := verbWallSections([]string{"sync"})
	if len(sections) == 0 {
		t.Fatal("verbWallSections found no wall block for 'sync'")
	}
	joined := strings.Join(sections, "")
	for _, want := range []string{
		"[check|apply|push]",
		"SAFE SYNC/PUSH",
		"dirty-tree sweep next action",
		"safe push path",
	} {
		if !strings.Contains(joined, want) {
			t.Fatalf("sync wall block missing %q:\n%s", want, joined)
		}
	}
}

// TestSuggestVerb pins did-you-mean: a near-miss typo maps to the real verb, and
// garbage maps to nothing rather than a random suggestion.
func TestSuggestVerb(t *testing.T) {
	if got := suggestVerb("comit"); got != "commit" {
		t.Errorf("suggestVerb(comit) = %q, want commit", got)
	}
	if got := suggestVerb("swep"); got != "sweep" {
		t.Errorf("suggestVerb(swep) = %q, want sweep", got)
	}
	if got := suggestVerb("zzqx"); got != "" {
		t.Errorf("suggestVerb(zzqx) = %q, want no suggestion", got)
	}
}

// TestCommitHelpShowsDeepHelpAboveFlagDump pins the #2246 acceptance example
// verbatim: `fak commit --help` used to print only the bare `flag` package
// dump ("Usage of commit: ..."); it must now show the carved wall block
// (verbFlagUsage, wired in runCommit) above the flag defaults.
func TestCommitHelpShowsDeepHelpAboveFlagDump(t *testing.T) {
	var out, errOut bytes.Buffer
	if rc := runCommitCommand(&out, &errOut, []string{"--help"}); rc != 2 {
		t.Fatalf("runCommitCommand(--help) = %d, want 2 (flag.ErrHelp stops the parse)", rc)
	}
	got := errOut.String()
	wallIdx := strings.Index(got, "SAFE SHARED-TRUNK COMMIT")
	flagIdx := strings.Index(got, "-m string")
	if wallIdx < 0 {
		t.Fatalf("fak commit --help lost its deep-help wall block; got:\n%s", got)
	}
	if flagIdx < 0 {
		t.Fatalf("fak commit --help lost its flag dump; got:\n%s", got)
	}
	if wallIdx > flagIdx {
		t.Errorf("fak commit --help printed the flag dump before the deep help; want the wall block first:\n%s", got)
	}
}

// TestOverviewVerbsAdoptVerbFlagUsage is the #2246 adoption ratchet: every
// compact-overview verb backed by a single top-level flag.FlagSet must wire
// verbFlagUsage right after constructing it, so `fak <verb> --help` shows the
// carved wall block instead of regressing to the bare `flag` package dump.
//
// Exempt are the verbs whose `fak <verb> --help` already shows a hand-curated
// usage, never the bare dump — the same rationale as the original "ps" exemption:
//   - "version"/"help" have no top-level flag.FlagSet at all (custom arg parse);
//   - "ps" sets fs.Usage to its own psUsage;
//   - "audit"/"egress"/"model"/"signal" are subcommand dispatchers whose no-arg /
//     --help path prints a bespoke usage (auditUsage/egressUsage/modelUsage/
//     signalUsage), and "codex" sets its own fs.Usage — none regress to the bare
//     dump, so wiring verbFlagUsage would be dead code behind the custom usage.
func TestOverviewVerbsAdoptVerbFlagUsage(t *testing.T) {
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatalf("glob *.go: %v", err)
	}
	var all strings.Builder
	for _, f := range files {
		if strings.HasSuffix(f, "_test.go") {
			continue
		}
		b, err := os.ReadFile(f)
		if err != nil {
			t.Fatalf("read %s: %v", f, err)
		}
		all.Write(b)
		all.WriteByte('\n')
	}
	src := all.String()
	exempt := map[string]bool{
		"version": true, "help": true, "ps": true,
		"audit": true, "egress": true, "model": true, "signal": true, "codex": true,
	}
	for _, g := range overviewGroups {
		for _, e := range g.entries {
			if exempt[e.name] {
				continue
			}
			want := `verbFlagUsage(fs, "` + e.name + `")`
			if !strings.Contains(src, want) {
				t.Errorf("overview verb %q: no %s call found in cmd/fak — its --help would regress to the bare flag.FlagSet dump", e.name, want)
			}
		}
	}
}
