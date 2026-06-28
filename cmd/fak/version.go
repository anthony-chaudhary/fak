package main

import (
	"fmt"
	"io"
	"runtime"
	"runtime/debug"

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
