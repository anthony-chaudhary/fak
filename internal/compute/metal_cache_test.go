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
	if !slices.Contains(pkg.CgoFiles, "metal.go") || !slices.Contains(pkg.CFiles, "metal_shim.c") {
		t.Fatalf("native Metal inputs missing: cgo=%v c=%v", pkg.CgoFiles, pkg.CFiles)
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

// TestComputeHoldsNoObjectiveCMFiles pins "one libobjc per binary" (see metal.go). cmd/go
// appends -lobjc to the link of every package that holds .m files; internal/metalgemm is
// always linked beside this package and already holds them, so a .m file here puts -lobjc
// on the link line twice and Apple's ld prints "ld: warning: ignoring duplicate libraries:
// '-lobjc'" on every cgo link (every `go run ./cmd/fak-dev`). Objective-C in this package
// uses a .c extension compiled with `#cgo CFLAGS: -x objective-c` instead.
func TestComputeHoldsNoObjectiveCMFiles(t *testing.T) {
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("resolve compute package directory: %v", err)
	}
	for _, arch := range []string{"arm64", "amd64"} {
		t.Run(arch, func(t *testing.T) {
			ctx := build.Default
			ctx.GOOS = "darwin"
			ctx.GOARCH = arch
			ctx.CgoEnabled = true
			pkg, err := ctx.ImportDir(dir, 0)
			if err != nil {
				t.Fatalf("import darwin/%s+cgo compute package: %v", arch, err)
			}
			if len(pkg.MFiles) != 0 {
				t.Fatalf("internal/compute holds .m files %v: each adds a duplicate -lobjc to every binary linking internal/metalgemm", pkg.MFiles)
			}
			if !slices.Contains(pkg.CgoCFLAGS, "objective-c") {
				t.Fatalf("internal/compute CFLAGS %v lack -x objective-c for its Objective-C .c sources", pkg.CgoCFLAGS)
			}
		})
	}
}
