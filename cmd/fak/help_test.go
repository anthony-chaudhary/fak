package main

import (
	"bytes"
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
	if got := suggestVerbSpelling("guardd"); got != "guard" {
		t.Errorf("suggestVerbSpelling(guardd) = %q, want guard", got)
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
