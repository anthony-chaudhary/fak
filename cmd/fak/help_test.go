package main

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
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

// TestRuntimeHelpHasNoDevelopmentCatalog proves --all remains useful while
// exposing only runtime-owned help text. The development inventory belongs to
// the separately built fak-dev artifact.

// TestUsageCompactLeadsWithBaselineWorkflows is the front-door priority
// witness. A new operator should see the primary ways to use fak before
// diagnostics, measurement tools, and the supporting capability floor.
func TestUsageCompactLeadsWithBaselineWorkflows(t *testing.T) {
	var b strings.Builder
	usageCompact(&b)
	got := b.String()

	sections := []string{"start here", "save tokens + turns", "observe + operate", "capability floor", "models + housekeeping"}
	last := -1
	for _, section := range sections {
		pos := strings.Index(got, section)
		if pos < 0 {
			t.Fatalf("compact overview missing %q section:\n%s", section, got)
		}
		if pos < last {
			t.Fatalf("compact overview sections are not priority ordered at %q:\n%s", section, got)
		}
		last = pos
	}

	start := strings.Index(got, "start here")
	next := strings.Index(got, "save tokens + turns")
	baseline := got[start:next]
	commands := []string{"manage", "serve", "agent", "run", "codex", "build"}
	last = -1
	for _, command := range commands {
		pos := strings.Index(baseline, "  "+command)
		if pos < 0 {
			t.Fatalf("start-here section missing baseline command %q:\n%s", command, got)
		}
		if pos < last {
			t.Fatalf("baseline commands are not priority ordered at %q:\n%s", command, got)
		}
		last = pos
	}
}

func TestRuntimeHelpHasNoDevelopmentCatalog(t *testing.T) {
	var b strings.Builder
	usageAllVerbs(&b)
	out := b.String()
	if !strings.Contains(out, "fak guard") {
		t.Fatalf("runtime --all help omitted guard:\n%s", out)
	}
	for _, forbidden := range []string{"[dev]", "fak dev sweep", "internal/devindex"} {
		if strings.Contains(out, forbidden) {
			t.Errorf("runtime help leaked development catalog marker %q", forbidden)
		}
	}
}

func TestSuggestVerbSpellingUsesRuntimeSurface(t *testing.T) {
	if got := suggestVerbSpelling("guardd"); got != "manage" {
		t.Errorf("suggestVerbSpelling(guardd) = %q, want canonical manage", got)
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
	var b strings.Builder
	if !verbDeepHelpBody(&b, "sync") {
		t.Fatal("runtime usage wall has no sync section")
	}
	for _, want := range []string{"push", "dirty"} {
		if !strings.Contains(strings.ToLower(b.String()), want) {
			t.Errorf("sync runtime help omitted %q:\n%s", want, b.String())
		}
	}
}

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
//   - "up" delegates to serve's compatible parser and hand-curated usage;
//   - "manage" shares guard's hand-curated usage and fully compatible flag parser;
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
		"version": true, "help": true, "ps": true, "up": true,
		"audit": true, "egress": true, "model": true, "signal": true, "codex": true, "manage": true,
		"progress": true, "ultracode": true, "build": true, "self-update": true,
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

func TestOverviewBespokeHelpExemptionsRenderUsage(t *testing.T) {
	var progressOut bytes.Buffer
	if code := runProgress(io.Discard, &progressOut, []string{"--help"}); code != 2 {
		t.Fatalf("progress --help code=%d, want 2 from flag.ErrHelp", code)
	}
	if got := progressOut.String(); !strings.Contains(got, "Usage: fak progress") {
		t.Fatalf("progress --help lost bespoke usage: %q", got)
	}

	var ultracodeOut bytes.Buffer
	if code := runUltracode(&ultracodeOut, io.Discard, []string{"--help"}); code != 0 {
		t.Fatalf("ultracode --help code=%d, want 0", code)
	}
	if got := ultracodeOut.String(); !strings.Contains(got, "usage: fak ultracode") {
		t.Fatalf("ultracode --help lost bespoke usage: %q", got)
	}
}

// TestCapturedManageHelpSurfaces is the rendered-byte witness for #6538. It
// captures the same writers used by `fak help`, `fak help manage`, `fak help m`,
// and `fak help guard`; assertions are deliberately on user-visible output, not
// catalog internals.
func TestCapturedManageHelpSurfaces(t *testing.T) {
	var compact bytes.Buffer
	usageCompact(&compact)
	root := compact.String()
	if !strings.Contains(root, "coordinates the whole agent path: context + routing + typed effects + evidence") ||
		!strings.Contains(root, "start here:") ||
		!strings.Contains(root, "  manage") ||
		!strings.Contains(root, "'fak m'; legacy: guard") {
		t.Fatalf("captured `fak help` does not lead with canonical manage and its migration spellings:\n%s", root)
	}
	if strings.Contains(root, "\n  guard        ") {
		t.Fatalf("captured `fak help` still presents guard as canonical:\n%s", root)
	}

	for _, spelling := range []string{"manage", "m"} {
		var out bytes.Buffer
		if !printVerbHelp(&out, spelling) {
			t.Fatalf("captured `fak help %s` did not resolve", spelling)
		}
		got := out.String()
		if !strings.HasPrefix(got, "  fak manage    [flags]") {
			t.Fatalf("captured `fak help %s` does not lead with canonical manage:\n%s", spelling, got)
		}
		if !strings.Contains(got, "aliases: m (preferred short form), guard (deprecated compatibility name)") ||
			!strings.Contains(got, "flags: fak manage -h") {
			t.Fatalf("captured `fak help %s` lost alias resolution or canonical flag guidance:\n%s", spelling, got)
		}
	}

	var guard bytes.Buffer
	if !printVerbHelp(&guard, "guard") {
		t.Fatal("captured `fak help guard` did not resolve")
	}
	gotGuard := guard.String()
	manageAt := strings.Index(gotGuard, "  fak manage    [flags]")
	guardAt := strings.Index(gotGuard, "  fak guard     [--provider")
	if manageAt < 0 || guardAt < 0 || manageAt > guardAt {
		t.Fatalf("captured `fak help guard` does not lead with manage while retaining full guard help:\n%s", gotGuard)
	}
	for _, notice := range []string{
		"DEPRECATED: fak guard is the legacy compatibility spelling; migrate to fak manage (or fak m).",
		"Sunset notice: guard remains fully compatible during migration; no removal date is set.",
		"flags: fak manage -h",
	} {
		if !strings.Contains(gotGuard, notice) {
			t.Fatalf("captured `fak help guard` missing %q:\n%s", notice, gotGuard)
		}
	}
}

func TestManageSuggestionPrefersCanonicalSpelling(t *testing.T) {
	for _, typo := range []string{"guardd", "manag", "mange"} {
		if got := suggestVerbSpelling(typo); got != "manage" {
			t.Errorf("suggestVerbSpelling(%q) = %q, want canonical manage", typo, got)
		}
	}
}
