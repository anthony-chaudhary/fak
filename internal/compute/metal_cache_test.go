package compute

import (
	"go/build"
	"os"
	"slices"
	"strings"
	"testing"
)

// TestMetalCgoFlagsArePathPortable pins the cross-root cache contract for the default
// Apple-Silicon Metal build. cgo already puts the source directory on its compiler include
// path, so repeating -I${SRCDIR} in a directive only puts the checkout's absolute path in
// Go's build-action key. That turns an otherwise identical isolated checkout into a cache
// miss even under -trimpath.
func TestMetalCgoFlagsArePathPortable(t *testing.T) {
	ctx := build.Default
	ctx.GOOS = "darwin"
	ctx.GOARCH = "arm64"
	ctx.CgoEnabled = true

	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("resolve compute package directory: %v", err)
	}
	pkg, err := ctx.ImportDir(dir, 0)
	if err != nil {
		t.Fatalf("import darwin/arm64+cgo compute package: %v", err)
	}
	if !slices.Contains(pkg.CgoFiles, "metal.go") || !slices.Contains(pkg.MFiles, "metal_shim.m") {
		t.Fatalf("native Metal inputs missing: cgo=%v objective-c=%v", pkg.CgoFiles, pkg.MFiles)
	}
	flagClasses := []struct {
		name  string
		flags []string
	}{
		{name: "CPPFLAGS", flags: pkg.CgoCPPFLAGS},
		{name: "CFLAGS", flags: pkg.CgoCFLAGS},
		{name: "CXXFLAGS", flags: pkg.CgoCXXFLAGS},
		{name: "FFLAGS", flags: pkg.CgoFFLAGS},
		{name: "LDFLAGS", flags: pkg.CgoLDFLAGS},
	}
	for _, class := range flagClasses {
		for _, flag := range class.flags {
			if strings.Contains(flag, pkg.Dir) {
				t.Fatalf("cgo %s flag %q embeds checkout path %q; keep native build directives path-portable", class.name, flag, pkg.Dir)
			}
		}
	}
}
