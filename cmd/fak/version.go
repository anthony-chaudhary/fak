package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"runtime"
	"runtime/debug"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/appversion"
)

// cmdVersion prints the application version AND the build provenance of THIS binary.
//
// Why the second part matters: appversion.Current() resolves the friendly application
// version from $FAK_APP_VERSION, a release -ldflags BuildVersion, the release tag Go embeds
// in the module build info, or a repo-bounded VERSION file found by walking up from the
// EXECUTABLE's own directory. So a STALE dev `fak` binary that still sits inside an
// up-to-date checkout reports the TREE's version, not the one it was built from — it cannot
// reveal that the binary itself is old. That is exactly the "it still seems like an old
// `fak guard` is running" confusion: the version line looks current even when the running
// binary is not. (Current() does not consult the process working directory, so a binary
// installed OUTSIDE a checkout is safe from this; one built in place is not.)
//
// The embedded build stamp can tell them apart. The Go toolchain records the VCS revision,
// commit time, and a dirty flag in the binary at build (default for `go build`/`go install`
// of a VCS-tracked main package); ReadBuildInfo reads them back out. That stamp travels
// WITH the binary, so it reflects what the running `fak` was actually built from — making
// `fak version` a reliable "is the fak/guard I'm running current?" check. The first line is
// still appversion.Current() verbatim, so anything parsing line 1 is unaffected.
func cmdVersion(w io.Writer) {
	// The `fak version <sub>` subcommands. They are dispatched HERE rather than in main.go
	// so the contested dispatch table stays untouched, and they are gathered into one switch
	// so the full set of them is visible in one place. An unrecognized word is deliberately
	// NOT an error: it falls through to the flag-shaped handling below.
	if len(os.Args) > 2 {
		switch os.Args[2] {
		case "modules":
			// The per-module version report (internal/modver). The bare `fak version`
			// output below is unchanged — the self-update "build:" line parser depends
			// on it.
			os.Exit(runVersionModules(os.Stdout, os.Stderr, os.Args[3:]))
		case "score-adapter":
			// Folds per-file scorecard rows into the flat module score map consumed by
			// `fak version modules --scores` (#2466).
			os.Exit(runVersionScoreAdapter(os.Stdout, os.Stderr, os.Args[3:]))
		case "trend":
			// The historical companion to `modules`: fold the append-only
			// module-versions ledger into a per-module movement summary
			// (internal/modver.Trend). Reads only what was stamped; no git, no tree.
			os.Exit(runVersionTrend(os.Stdout, os.Stderr, os.Args[3:]))
		}
	}

	// `fak version --json` — the machine-readable binary identity (commit + dirty
	// bit + a "stamped" flag). Unlike the human line 1 (appversion.Current(), which
	// reads the TREE's VERSION file and so cannot reveal a stale binary), this is
	// the running binary's OWN provenance, folded from the embedded VCS stamp. It
	// is what a fleet monitor or a wave-admission skew witness reads to tell one
	// long-lived running copy from another (epic #2218 gap G2 / risk R2): a
	// detached worker cannot forge the commit it was built from, because the stamp
	// travels with the binary, not with the tree it runs against.
	if versionJSONRequested(os.Args[2:]) {
		bi, _ := debug.ReadBuildInfo()
		if err := writeVersionJSON(w, buildIdentity(bi)); err != nil {
			fmt.Fprintf(os.Stderr, "fak version --json: %v\n", err)
			os.Exit(1)
		}
		return
	}

	fmt.Fprintln(w, appversion.Current())

	bi, ok := debug.ReadBuildInfo()
	if !ok {
		fmt.Fprintf(w, "build: (no embedded build info)   go: %s  %s/%s\n",
			runtime.Version(), runtime.GOOS, runtime.GOARCH)
		return
	}
	fmt.Fprintln(w, buildProvenanceLine(bi))
	fmt.Fprintf(w, "go: %s  %s/%s\n", runtime.Version(), runtime.GOOS, runtime.GOARCH)
}

// guardBannerVersion is the friendly version string for the `fak guard` banner headline. It is
// appversion.Current() verbatim (the same first line `fak version` prints). On its own it can
// look current even when the running binary is stale — see guardBannerBuildStamp for why the
// banner also shows the embedded build stamp.
func guardBannerVersion() string {
	return appversion.Current()
}

// guardBannerBuildStamp is the embedded build provenance for the `fak guard` banner — the
// reliable "is the fak/guard I'm running actually current?" signal. appversion.Current() reads
// the TREE's VERSION file, so a STALE binary run from inside an up-to-date checkout still reports
// the tree's version; the VCS stamp baked into the binary at build does not lie about its own
// age (a +uncommitted marker even reveals a binary built from a dirty tree). It reuses
// buildProvenanceLine and strips its "build: " prefix so the banner can label the row itself.
func guardBannerBuildStamp() string {
	bi, ok := debug.ReadBuildInfo()
	if !ok {
		return "(no embedded build info)"
	}
	return strings.TrimPrefix(buildProvenanceLine(bi), "build: ")
}

// guardShortBuildID is the compact build identity for space-constrained surfaces (the guard's
// compact/animated launch identity, the fak info pane header): the short commit the binary was
// built from plus a "+" marker when that build carried uncommitted changes, or "" when the
// binary carries no commit stamp at all. The FULL stamp (commit time included) is
// guardBannerBuildStamp — this is the abbreviated tell for places one line is all there is room
// for.
//
// It folds the SAME binaryIdentity `fak version` publishes rather than re-reading bi.Settings
// itself, and that is load-bearing rather than tidiness. A source-install / `fak self-update`
// build carries its commit in the -X-injected appversion.BuildCommit, NOT in vcs.revision (the
// toolchain stamps no VCS settings when it builds from an export rather than the checkout), so
// a hand-rolled vcs.revision-only read rendered "fak guard 0.43.0 (no stamp)" on a freshly
// self-updated binary whose `fak version` said `build: 0c96937b61ac` (#6537) — the compact
// banner's stale-binary tell firing on the one binary that had just been made current. Two
// readers of one fact drift; one reader cannot.
func guardShortBuildID() string {
	bi, _ := debug.ReadBuildInfo()
	return guardShortBuildIDOf(buildIdentity(bi))
}

// guardShortBuildIDOf renders the compact build id for an already-resolved identity, so a render
// witness can drive the guard identity from the same fixture provenance as the version surface.
// An unstamped identity yields "" — the surfaces spell that out as "(no stamp)" themselves. That
// includes the `go install …@vX` module path, which carries a module version but no commit to
// abbreviate; `fak version` renders it as "build: module vX", which has no short form either.
func guardShortBuildIDOf(id binaryIdentity) string {
	if !id.Stamped {
		return ""
	}
	rev := id.Commit
	if len(rev) > 8 {
		rev = rev[:8]
	}
	if id.Dirty {
		rev += "+"
	}
	return rev
}

// guardInfoVersionTag is the compact "which fak is this pane watching?" identity for the fak info
// header. The info pane and the guard it sits beside are the SAME fak binary (the split pane runs
// `fak info` from the same executable), so this is the running guard's identity, persistently
// visible in the pane for the whole session — where the startup banner has already scrolled off.
// It pairs the version with the short build id (a "+" flags a dirty-tree build) because the
// version alone reads as current even on a stale binary; the build id is the staleness tell.
func guardInfoVersionTag() string {
	tag := "fak " + appversion.Current()
	if id := guardShortBuildID(); id != "" {
		tag += " (" + id + ")"
	}
	return tag
}

// buildProvenanceLine renders the one-line build stamp from a BuildInfo: the VCS revision
// (short), commit time, and an explicit dirty marker when the binary was built from a tree
// with uncommitted changes. Falls back to the module version (the `go install …@vX` path,
// where the proxy stamps a module version but no vcs.* settings), then to a clear "no VCS
// stamp" note so the absence of provenance is itself legible rather than a blank line.
func buildProvenanceLine(bi *debug.BuildInfo) string {
	var rev, when string
	dirty := false
	for _, s := range bi.Settings {
		switch s.Key {
		case "vcs.revision":
			rev = s.Value
		case "vcs.time":
			when = s.Value
		case "vcs.modified":
			dirty = s.Value == "true"
		}
	}
	if rev == "" {
		rev = strings.TrimSpace(appversion.BuildCommit)
	}
	if rev != "" {
		short := rev
		if len(short) > 12 {
			short = short[:12]
		}
		dirtyNote := ""
		if dirty {
			dirtyNote = " +uncommitted"
		}
		if when != "" {
			return fmt.Sprintf("build: %s%s  (committed %s)", short, dirtyNote, when)
		}
		return fmt.Sprintf("build: %s%s", short, dirtyNote)
	}
	if v := bi.Main.Version; v != "" && v != "(devel)" {
		return "build: module " + v
	}
	return "build: (no VCS stamp — built without module/VCS provenance; cannot confirm the commit)"
}

// binaryIdentity is the machine-readable provenance of THIS running fak binary: the
// VCS commit it was built from, whether that build carried uncommitted changes, and
// whether any VCS stamp is present at all. It is the identity a fleet monitor or a
// wave-admission skew witness reads to tell one long-lived running copy from another
// (epic #2218 gap G2 / risk R2). `stamped` is the load-bearing bit: when it is false
// the binary literally cannot attest which commit it is — the witnessed failure mode
// that made a mixed-version wave undetectable — so a consumer must treat commit as
// unknown rather than assume agreement.
type binaryIdentity struct {
	AppVersion    string `json:"app_version"`
	Commit        string `json:"commit"`                   // full vcs.revision; "" when unstamped
	CommitShort   string `json:"commit_short,omitempty"`   // first 12 hex of Commit
	Dirty         bool   `json:"dirty"`                    // built from a tree with uncommitted changes
	CommitTime    string `json:"commit_time,omitempty"`    // vcs.time (RFC3339)
	Stamped       bool   `json:"stamped"`                  // a real VCS revision is embedded
	ModuleVersion string `json:"module_version,omitempty"` // bi.Main.Version for the `go install …@vX` path
	Go            string `json:"go"`
	OS            string `json:"os"`
	Arch          string `json:"arch"`
}

// buildIdentity folds a BuildInfo into the machine-readable identity. It is the pure
// core of `fak version --json`: same extraction as buildProvenanceLine (vcs.revision
// / vcs.modified / vcs.time), but structured for a consumer instead of a human line.
// A nil BuildInfo (ReadBuildInfo reported no embedded info) yields an unstamped
// identity — Stamped=false, Commit="" — rather than a lie about the commit.
func buildIdentityFromRuntime() binaryIdentity {
	bi, _ := debug.ReadBuildInfo()
	return buildIdentity(bi)
}

func buildIdentity(bi *debug.BuildInfo) binaryIdentity {
	id := binaryIdentity{
		AppVersion: appversion.Current(),
		Go:         runtime.Version(),
		OS:         runtime.GOOS,
		Arch:       runtime.GOARCH,
	}
	if bi == nil {
		return id
	}
	if v := bi.Main.Version; v != "" && v != "(devel)" {
		id.ModuleVersion = v
	}
	for _, s := range bi.Settings {
		switch s.Key {
		case "vcs.revision":
			id.Commit = s.Value
		case "vcs.modified":
			id.Dirty = s.Value == "true"
		case "vcs.time":
			id.CommitTime = s.Value
		}
	}
	if id.Commit == "" {
		id.Commit = strings.TrimSpace(appversion.BuildCommit)
	}
	if id.Commit != "" {
		id.Stamped = true
		id.CommitShort = id.Commit
		if len(id.CommitShort) > 12 {
			id.CommitShort = id.CommitShort[:12]
		}
	}
	return id
}

// versionJSONRequested reports whether the version args ask for JSON output.
func versionJSONRequested(args []string) bool {
	for _, a := range args {
		if a == "--json" || a == "-json" {
			return true
		}
	}
	return false
}

// writeVersionJSON emits the identity as indented JSON with a trailing newline.
func writeVersionJSON(w io.Writer, id binaryIdentity) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(id)
}
