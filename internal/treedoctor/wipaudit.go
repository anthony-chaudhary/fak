package treedoctor

// Untracked-WIP inventory — the land-or-park surface for the shared-trunk build poison.
//
// In an always-on fleet the trunk is never quiescent, and a worker that crashes mid-edit
// leaves untracked source behind. Two failure classes matter:
//
//   - BUILD POISON: an untracked .go file whose package no longer compiles. Because the
//     whole tree is built from the dirty working set (HEAD is not buildable standalone),
//     one such file crash-loops every peer that rebuilds — the §3.2 root gap. It must be
//     LANDED (commit the fix) or PARKED (move it aside) so the fleet stops crashing.
//   - ABANDONMENT: untracked source aged well past the live-refactor window with no live
//     owner — a crashed worker's residue that no janitor reclaims, or finished work that
//     was never landed.
//
// This inventory SURFACES both, read-only. It never moves or removes a file: the doctrine
// treedoctor enforces — never delete a peer's live work — forbids acting on WIP, because an
// untracked file that is load-bearing-but-unlanded is indistinguishable at the byte level
// from abandoned cruft. The disambiguation the surface CAN make safely is age + liveness +
// owner-liveness + build health, and the only automatic consequence is a listing a human
// (or a separate, explicit park step) acts on. A LIVE file — touched within the live window
// — is never classified abandoned or park-worthy, even when its package does not build yet
// (a mid-refactor edit), which is the load-bearing safety property this surface guarantees.

import (
	"context"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// DefaultAbandonAfter: an untracked source file not touched within this long, and not held
// by a live owner, is surfaced as an abandonment candidate for land-or-park. Generous on
// purpose — this drives a READ-ONLY surface (never a delete), so a false "abandoned" costs a
// human glance while a false "live" would hide real cruft.
const DefaultAbandonAfter = 60 * time.Minute

// WIPFile classifies one untracked durable artifact for the land-or-park inventory.
type WIPFile struct {
	Path        string `json:"path"`                  // repo-relative, forward slashes (as git emits)
	Kind        string `json:"kind"`                  // source | claude-control | test-fixture
	Action      string `json:"action"`                // land-or-park | park-or-delete | land-or-delete
	AgeSeconds  int64  `json:"age_seconds"`           // now - mtime, floored at 0
	Live        bool   `json:"live"`                  // touched within LiveWindow — an active edit, keep
	Owner       string `json:"owner,omitempty"`       // owning session/pid, where discoverable
	OwnerAlive  bool   `json:"owner_alive,omitempty"` // the owner process is still alive
	BuildProbed bool   `json:"build_probed"`          // a build probe ran for this file's package
	Builds      bool   `json:"builds"`                // package compiles (meaningful only when BuildProbed)
	Poison      bool   `json:"poison"`                // BuildProbed && !Builds — package won't compile
	Abandoned   bool   `json:"abandoned"`             // aged past AbandonAfter, not live, no live owner
	Class       string `json:"class"`                 // live | poison | abandoned | resident
	LandOrPark  bool   `json:"land_or_park"`          // surfaced for a human to land or park; never auto-acted
}

// WIPOptions configures the untracked-WIP inventory. Its zero value inventories age only:
// no build probe (Poison stays false/unprobed) and no owner (undiscoverable).
type WIPOptions struct {
	// AbandonAfter tunes the abandonment threshold; zero => DefaultAbandonAfter.
	AbandonAfter time.Duration
	// BuildProbe reports whether pkgDir (repo-relative, forward slashes) currently builds.
	// Probed at most once per package. nil => build health is not probed.
	BuildProbe func(pkgDir string) bool
	// OwnerOf resolves the owning session/pid for a repo-relative path and whether that owner
	// is still alive. nil, or ("", false), => owner undiscoverable (treated as not-alive, so
	// only age decides abandonment — the conservative read for a read-only surface).
	OwnerOf func(relPath string) (owner string, alive bool)
}

// classifyWIP fills the derived fields (Poison/Abandoned/Class/LandOrPark) from the raw
// signals already on f. Pure: no clock, no I/O — the whole land-or-park decision is a fold
// over (age, liveness, owner-liveness, build health), so it is exhaustively unit-testable.
//
// Live wins over everything: a file touched within the live window is an active edit, so it
// is never called abandoned nor flagged for park, even if its package does not build yet
// (typing a refactor). That ordering is the safety guarantee — the surface cannot point a
// janitor at a peer's in-flight file.
func classifyWIP(f *WIPFile, abandonAfterSec int64) {
	f.Poison = f.BuildProbed && !f.Builds
	aged := f.AgeSeconds >= abandonAfterSec && !f.OwnerAlive
	switch {
	case f.Live:
		f.Class = "live"
		f.Abandoned = false
		f.LandOrPark = false
	case f.Poison:
		// Not live and its package won't compile: this is the shared-trunk build poison
		// that crash-loops the fleet. Highest land-or-park priority — a human lands the fix
		// or parks the file so peers stop crashing on the rebuild.
		f.Class = "poison"
		f.Abandoned = aged
		f.LandOrPark = true
	case aged:
		f.Class = "abandoned"
		f.Abandoned = true
		f.LandOrPark = true
	default:
		f.Class = "resident"
		f.Abandoned = false
		f.LandOrPark = false
	}
}

// diagnoseWIP inventories untracked durable artifacts under repoRoot and classifies each for the
// land-or-park surface. Durable artifacts include source plus control-plane files under .claude/
// and fixtures under testdata/; these two trees otherwise evade a source-extension-only audit. Read-only: it lists (git ls-files --others --exclude-standard), stats
// mtimes, and — only when the corresponding probe is supplied — checks package build health
// and owner liveness. It never moves or removes a file. A missing/failed git read yields an
// empty inventory (the fail-safe: surface nothing rather than guess).
func diagnoseWIP(ctx context.Context, run Runner, repoRoot string, window time.Duration, now time.Time, wopts WIPOptions) []WIPFile {
	abandonAfter := wopts.AbandonAfter
	if abandonAfter <= 0 {
		abandonAfter = DefaultAbandonAfter
	}
	abandonSec := int64(abandonAfter / time.Second)

	out, code, err := run(ctx, repoRoot, "ls-files", "--others", "--exclude-standard")
	if err != nil || code != 0 {
		return nil
	}

	buildCache := map[string]bool{} // pkgDir => builds (probe once per package)
	var files []WIPFile
	for _, rel := range strings.Split(out, "\n") {
		rel = strings.TrimSpace(strings.TrimRight(rel, "\r"))
		if rel == "" {
			continue
		}
		kind, action, durable := classifyDurableArtifact(rel)
		if !durable {
			continue
		}
		abs := filepath.Join(repoRoot, filepath.FromSlash(rel))
		info, serr := os.Stat(abs)
		if serr != nil || info.IsDir() {
			continue
		}
		f := WIPFile{Path: rel, Kind: kind, Action: action}
		age := now.Sub(info.ModTime())
		if age > 0 {
			f.AgeSeconds = int64(age / time.Second)
		}
		f.Live = age < window
		if wopts.OwnerOf != nil {
			f.Owner, f.OwnerAlive = wopts.OwnerOf(rel)
		}
		if wopts.BuildProbe != nil {
			pkgDir := filepath.ToSlash(filepath.Dir(rel))
			builds, ok := buildCache[pkgDir]
			if !ok {
				builds = wopts.BuildProbe(pkgDir)
				buildCache[pkgDir] = builds
			}
			f.BuildProbed = true
			f.Builds = builds
		}
		classifyWIP(&f, abandonSec)
		files = append(files, f)
	}

	// Most-actionable first: land-or-park candidates, then oldest, then path — a stable order
	// that puts the fleet-crash-loop culprits at the top of the surface.
	sort.Slice(files, func(i, j int) bool {
		if files[i].LandOrPark != files[j].LandOrPark {
			return files[i].LandOrPark
		}
		if files[i].AgeSeconds != files[j].AgeSeconds {
			return files[i].AgeSeconds > files[j].AgeSeconds
		}
		return files[i].Path < files[j].Path
	})
	return files
}

// classifyDurableArtifact keeps operational control files and fixtures on the same aging
// surface as source. Tracked files never reach this fold because git supplies only untracked
// paths; a generated goal prompt should be parked or deleted, while a fixture must be either
// landed with its test or deleted rather than lingering as invisible local state.
func classifyDurableArtifact(rel string) (kind, action string, ok bool) {
	norm := strings.ToLower(filepath.ToSlash(rel))
	if strings.HasPrefix(norm, ".claude/") {
		return "claude-control", "park-or-delete", true
	}
	if strings.Contains("/"+norm, "/testdata/") {
		return "test-fixture", "land-or-delete", true
	}
	if isSourceFile(rel) {
		return "source", "land-or-park", true
	}
	return "", "", false
}

// isSourceFile reports whether rel names a source file the inventory tracks. Kept to code
// extensions so the surface targets build-relevant WIP, not scratch notes or data dumps.
func isSourceFile(rel string) bool {
	switch strings.ToLower(filepath.Ext(rel)) {
	case ".go", ".py", ".rs", ".ts", ".tsx", ".js", ".jsx", ".c", ".h", ".cc", ".cpp", ".java", ".rb":
		return true
	}
	return false
}
