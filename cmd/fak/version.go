package main

import (
	"fmt"
	"io"
	"runtime"
	"runtime/debug"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/appversion"
)

// cmdVersion prints the application version AND the build provenance of THIS binary.
//
// Why the second part matters: appversion.Current() resolves the version by walking up
// from the current directory to the nearest VERSION file (then $FAK_APP_VERSION, then the
// -ldflags BuildVersion). So a STALE `fak` binary run from inside an up-to-date checkout
// reports the TREE's version, not its own — it cannot reveal that the binary itself is
// old. That is exactly the "it still seems like an old `fak guard` is running" confusion:
// the version line looks current even when the running binary is not.
//
// The embedded build stamp can tell them apart. The Go toolchain records the VCS revision,
// commit time, and a dirty flag in the binary at build (default for `go build`/`go install`
// of a VCS-tracked main package); ReadBuildInfo reads them back out. That stamp travels
// WITH the binary, so it reflects what the running `fak` was actually built from — making
// `fak version` a reliable "is the fak/guard I'm running current?" check. The first line is
// still appversion.Current() verbatim, so anything parsing line 1 is unaffected.
func cmdVersion(w io.Writer) {
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

// guardShortBuildID is the compact build identity for space-constrained surfaces (the fak info
// pane header): the short VCS revision plus a "+" marker when the binary was built from a dirty
// tree, or "" when no VCS stamp is embedded. The FULL stamp (commit time included) is
// guardBannerBuildStamp — this is the abbreviated tell for places one line is all there is room
// for.
func guardShortBuildID() string {
	bi, ok := debug.ReadBuildInfo()
	if !ok {
		return ""
	}
	var rev string
	dirty := false
	for _, s := range bi.Settings {
		switch s.Key {
		case "vcs.revision":
			rev = s.Value
		case "vcs.modified":
			dirty = s.Value == "true"
		}
	}
	if rev == "" {
		return ""
	}
	if len(rev) > 8 {
		rev = rev[:8]
	}
	if dirty {
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
