package main

import (
	"context"
	"fmt"
	"path/filepath"
	"runtime/debug"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/modver"
)

type guardInfoRuntimeIdentity struct {
	Source         string `json:"source"`
	SourceModule   string `json:"source_module"`
	RunningBuild   string `json:"running_build"`
	ConfigDigest   string `json:"config_digest"`
	SessionStarted string `json:"session_started"`
	Verdict        string `json:"verdict"`
	Action         string `json:"action"`
}

// guardInfoRuntimeIdentityOf joins the source checkout, the binary executing this pane, and
// the active gateway receipt. It fails closed: only an exact clean build/source match is MATCH;
// an absent stamp or source revision is UNKNOWN. ConfigDigest is a digest of the gateway's
// guard-published effective-policy digest, not a path or configuration body, so the pane can bind later
// dogfood evidence without disclosing operator configuration.
func guardInfoRuntimeIdentityOf(source, sourceModule, runningBuild, configDigest, sessionStarted string) guardInfoRuntimeIdentity {
	id := guardInfoRuntimeIdentity{
		Source:         compactIdentityValue(source),
		SourceModule:   compactIdentityValue(sourceModule),
		RunningBuild:   compactIdentityValue(runningBuild),
		ConfigDigest:   compactIdentityValue(configDigest),
		SessionStarted: compactIdentityValue(sessionStarted),
		Verdict:        "UNKNOWN",
		Action:         "check with fak version --json; run go install ./cmd/fak, then relaunch with fak guard -- claude if identities differ",
	}
	if id.Source == "unknown" || id.RunningBuild == "unknown" {
		return id
	}
	running := strings.TrimSuffix(id.RunningBuild, "+dirty")
	if strings.HasPrefix(id.Source, running) || strings.HasPrefix(running, id.Source) {
		if strings.HasSuffix(id.RunningBuild, "+dirty") {
			return id
		}
		id.Verdict = "MATCH"
		id.Action = "none"
		return id
	}
	id.Verdict = "STALE"
	id.Action = "run go install ./cmd/fak, then relaunch with fak guard -- claude"
	return id
}

func compactIdentityValue(v string) string {
	v = strings.TrimSpace(v)
	if v == "" {
		return "unknown"
	}
	return v
}

func guardInfoStartupDigest(report string) string {
	const marker = "fak guard: active config digest "
	for _, line := range strings.Split(report, "\n") {
		if strings.HasPrefix(line, marker) {
			return compactIdentityValue(strings.TrimPrefix(line, marker))
		}
	}
	return "unknown"
}

func guardInfoSourceHeadFrom(run func(...string) ([]byte, error)) string {
	out, err := run("rev-parse", "--verify", "HEAD")
	if err != nil {
		return "unknown"
	}
	head := strings.TrimSpace(string(out))
	if len(head) > 12 {
		head = head[:12]
	}
	return compactIdentityValue(head)
}

func guardInfoCurrentRuntimeIdentity(v guardInfoVars) guardInfoRuntimeIdentity {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	root := repoRoot()
	source := guardInfoSourceHeadFrom(func(args ...string) ([]byte, error) {
		return modver.RealRunner(ctx, root, args...)
	})
	sourceModule := "unknown"
	if rep, err := modver.Snapshot(ctx, root, modver.RealRunner); err == nil {
		for _, m := range rep.Modules {
			if m.Name == "cmd/fak" {
				sourceModule = fmt.Sprintf("cmd/fak@r%d", m.Rev)
				break
			}
		}
	}
	var bi *debug.BuildInfo
	if got, ok := debug.ReadBuildInfo(); ok {
		bi = got
	}
	running := guardShortBuildIDOf(buildIdentity(bi))
	started := ""
	if v.Startup != nil {
		started = v.Startup.StartedAt
	}
	return guardInfoRuntimeIdentityOf(source, sourceModule, running, guardInfoStartupDigest(v.StartupReport), started)
}

func guardInfoRuntimeIdentityRow(id guardInfoRuntimeIdentity) string {
	return fmt.Sprintf("identity %s · source %s (%s) · running %s · config %s · session %s · action: %s",
		id.Verdict, id.Source, id.SourceModule, id.RunningBuild, id.ConfigDigest, id.SessionStarted, id.Action)
}

// guardInfoNarrowCols is the pane-width threshold below which the verbose multi-line legend
// is replaced by a single compact line: a narrow split pane (e.g. the --split right column)
// cannot show the full per-term legend without wrapping it, which crowds out the live status row.
const guardInfoNarrowCols = 80

// fitGuardInfoStatus formats the live status line for the in-place TTY redraw, capped so it
// can NEVER wrap the pane. It prefixes the two-space indent and trims the line to the
// remaining cell budget on a known pane width (width > 0); width <= 0 (size unknown) leaves
// the line whole — trimTUI returns its input for a non-positive budget. A wrapped status
// line is the classic single-line-redraw corruptor: once the text overflows, the terminal
// wraps it to a second row and the next tick's \r returns only to the start of that wrapped
// row, so the overflow is never cleared and the pane scrolls.
func fitGuardInfoStatus(line string, width int) string {
	return "  " + trimTUI(line, width-2)
}

// guardInfoStartupHeader is the one-time header + legend block, sized to the pane. A wide or
// unknown pane (width <= 0 or width >= guardInfoNarrowCols) keeps the full multi-line legend;
// a NARROW split pane gets a single compact legend line so the legend never wraps and crowds
// out the live status row. The header line is trimmed only when the pane width is known.
func guardInfoStartupHeader(base, laneTag string, interval time.Duration, width int) string {
	var b strings.Builder
	// Lead with the running fak's identity (version + short build id) so the version is visible
	// in the pane for the whole session — the startup banner has scrolled off by then, and a
	// "+"-marked build id is the staleness tell the version alone cannot give. Putting it first
	// means a narrow-pane width-trim drops the interval hint, not the version.
	stamp := guardInfoVersionTag()
	if laneTag != "" {
		// The active lane's module rev sits immediately BESIDE the binary stamp (#2491): the
		// version tag is the BINARY staleness tell, laneTag is the WORKING-SET one. Keeping them
		// adjacent (and ahead of base/interval) means a narrow-pane width-trim drops the interval
		// hint first, not either staleness tell the pane exists to keep visible.
		stamp += " · " + laneTag
	}
	header := fmt.Sprintf("fak info · %s · %s  (every %s, Ctrl-C to stop)", stamp, base, interval)
	if width > 0 {
		header = trimTUI(header, width)
	}
	b.WriteString(header)
	b.WriteByte('\n')
	if width > 0 && width < guardInfoNarrowCols {
		// A narrow split pane stays intentionally compact (header + one compact guide line). The
		// unattested-build warning is carried by the startup banner and the full-width pane; adding
		// a row here would crowd the live status line the narrow path exists to protect.
		b.WriteString(trimTUI(guardInfoCompactLegend(), width))
		b.WriteByte('\n')
		return b.String()
	}
	// Wide/unknown pane: room for the persistent unattested-build warning under the header. A build
	// with no VCS stamp cannot show a "+"-marked build id in the version tag above, so without this
	// the pane would give an operator NO staleness tell at all — and the startup banner that first
	// carried it has long since scrolled off. Reuses the banner's predicate and stamp source
	// (guardBuildStampUnattested over guardBannerBuildStamp); attested builds return "" and the pane
	// stays uncluttered.
	appendGuardInfoNote(&b, guardInfoStalenessNote(guardBannerBuildStamp()), width)
	// The ATTESTED-but-behind twin: guardInfoStalenessNote above fires only for an UNSTAMPED
	// binary (staleness UNVERIFIABLE); this fires for a STAMPED binary that git ancestry proves is
	// behind (Skewed) or off (Diverged) origin/main — the pane-persistent twin of the banner's
	// guardSkewBuildWarning (guard_startup.go). The assessment is a per-process sync.Once, so the
	// pane re-reads a cached verdict every frame and git never runs on the render path. The two
	// notes are mutually exclusive per binary — an unstamped build is never classified Skewed — so
	// emitting both unconditionally cannot double-warn.
	appendGuardInfoNote(&b, guardInfoSkewNote(guardBuildSkewAssessment()), width)
	b.WriteString(guardInfoLegend())
	return b.String()
}

// appendGuardInfoNote emits one optional pane note on its own line, trimmed to the pane
// width when a width is known. An empty note writes NOTHING -- not even the newline -- so a
// build with nothing to warn about leaves the pane uncluttered. Both staleness tells above
// are emitted through this one rule, so neither can pick up a different trim or spacing.
func appendGuardInfoNote(b *strings.Builder, note string, width int) {
	if note == "" {
		return
	}
	if width > 0 {
		note = trimTUI(note, width)
	}
	b.WriteString(note)
	b.WriteByte('\n')
}

// guardInfoActiveLaneTag resolves the pane's active-lane tell live: one modver.Snapshot for the
// module revs and one `git status --porcelain` for the working set, folded by guardInfoLaneRevTag
// into "lane <module>@r<N>". Best-effort and non-fatal — any failure (not a git repo, git absent,
// a snapshot or status error, or no cleanly resolvable lane) yields "" and the header simply omits
// the tag: the lane tell is ADDITIVE, never a reason to degrade or fail the pane. Computed once at
// pane startup because the working set is a session-level fact, and time-boxed so a slow history
// walk cannot stall the pane's first paint.
func guardInfoActiveLaneTag() string {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	root := repoRoot()
	rep, err := modver.Snapshot(ctx, root, modver.RealRunner)
	if err != nil {
		return ""
	}
	changed, err := guardInfoWorkingSet(ctx, root)
	if err != nil {
		return ""
	}
	return guardInfoLaneRevTag(rep, changed)
}

// guardInfoWorkingSet returns the repo-relative paths of the current working set — the tracked
// files with uncommitted edits — from `git status --porcelain -z`. The NUL-terminated form is
// parsed rather than the newline form so paths with spaces or unusual bytes survive intact. In
// -z rename order a rename/copy entry is "XY <new>\0<old>\0"; the new path (the one a lane
// resolves against) is on the status field, so the trailing origin field is skipped.
func guardInfoWorkingSet(ctx context.Context, root string) ([]string, error) {
	return guardInfoWorkingSetFrom(func(args ...string) ([]byte, error) {
		return modver.RealRunner(ctx, root, args...)
	})
}

// guardInfoWorkingSetFrom is the pure core of guardInfoWorkingSet: it runs `status --porcelain -z`
// through an injected git runner and parses the NUL-terminated entries, so the -z parse (rename
// origin-follower dropping in particular) is unit-tested without touching a real repo.
func guardInfoWorkingSetFrom(run func(...string) ([]byte, error)) ([]string, error) {
	out, err := run("status", "--porcelain", "-z")
	if err != nil {
		return nil, err
	}
	fields := strings.Split(string(out), "\x00")
	var paths []string
	for i := 0; i < len(fields); i++ {
		e := fields[i]
		if len(e) < 4 {
			continue // "" tail field or a malformed short entry
		}
		status, path := e[:2], e[3:]
		if strings.ContainsAny(status, "RC") && i+1 < len(fields) {
			i++ // consume (and drop) the origin-path follower; keep the new path
		}
		if path != "" {
			paths = append(paths, path)
		}
	}
	return paths, nil
}

// guardInfoLaneRevTag renders the active lane's module rev as "lane <module>@r<N>" — the
// working-set staleness tell for the session (#2491), the module-level twin of the binary
// build-id tell in guardInfoVersionTag. The "active lane" is the single tracked module the
// working set touches: every changed path that maps to a tracked module must agree on ONE
// module, or the lane is left unresolved (returns ""). An empty working set, changes only under
// paths no module tracks, or a set spanning more than one module are all reported as "no
// resolvable lane" rather than guessed — the tag names ONE lane, so ambiguity stays silent.
// Pure over its inputs (a modver.Report + the changed path set) so the pane-render witness
// exercises it without touching git.
func guardInfoLaneRevTag(rep modver.Report, changed []string) string {
	name := ""
	for _, p := range changed {
		m := guardInfoModuleForPath(rep, p)
		switch {
		case m == "":
			continue
		case name == "":
			name = m
		case name != m:
			return "" // working set spans >1 module — ambiguous, do not guess
		}
	}
	if name == "" {
		return ""
	}
	for _, m := range rep.Modules {
		if m.Name == name {
			return fmt.Sprintf("lane %s@r%d", m.Name, m.Rev)
		}
	}
	return ""
}

// guardInfoModuleForPath maps one repo-relative path to the tracked module that owns it, using
// only the module names the report already enumerates: the module whose Name is the longest
// path-segment prefix of the path (so "internal/modver/x.go" resolves to "internal/modver", and
// the more specific of two nested modules wins). Returns "" when no tracked module contains the
// path. Report-only — it needs nothing from modver's internals, so cmd/fak carries no new
// coupling to the version-everything spine beyond the exported Snapshot/Report surface.
func guardInfoModuleForPath(rep modver.Report, path string) string {
	// Normalize Windows-style separators cross-platform before prefix-matching:
	// filepath.ToSlash only rewrites the OS separator (a no-op for a backslash on a
	// Linux runner), so a backslash working-set path would never prefix-match a
	// slash-keyed module name. Replace backslashes explicitly on every platform.
	path = strings.ReplaceAll(filepath.ToSlash(path), `\`, "/")
	best := ""
	for _, m := range rep.Modules {
		n := m.Name
		if path == n || strings.HasPrefix(path, n+"/") {
			if len(n) > len(best) {
				best = n
			}
		}
	}
	return best
}

// guardInfoCompactLegend is the one-line guide for a narrow pane — the same plain words as the
// full guide, shortened so it fits beside the live status row instead of wrapping over it.
func guardInfoCompactLegend() string {
	return "what this means: cache = is re-using text saving money · safety = what fak blocked/fixed/set aside · assumptions = facts/source/confidence/expiry"
}

// guardInfoLegend — the full multi-line guide this header prints — lives beside the line it
// explains, in info.go next to renderGuardInfoLine. A legend that drifts from the line is worse
// than no legend, so the two are kept in one file: change the line, change the legend, in the
// same diff.
