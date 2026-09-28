package harnessinit

import (
	"reflect"
	"slices"
	"testing"
)

func TestOfflineGoEnvPinsLocalToolchainAndDropsGOROOT(t *testing.T) {
	got := offlineGoEnv([]string{
		"PATH=/usr/local/go/bin", "GOROOT=/sdk/go1.26.6", "GOTOOLCHAIN=auto", "GOPROXY=https://proxy.golang.org,direct",
		"GOSUMDB=sum.golang.org", "GOWORK=/w/go.work", "GOTOOLCHAIN_INTERNAL_SWITCH_COUNT=1", "GOFLAGS=-mod=mod", "HOME=/h",
	})
	want := []string{"PATH=/usr/local/go/bin", "GOFLAGS=-mod=mod", "HOME=/h", "GOPROXY=off", "GOSUMDB=off", "GOWORK=off", "GOTOOLCHAIN=local"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("offlineGoEnv=%q\nwant %q", got, want)
	}
}

func TestRequiredGoVersionTakesHighestDirective(t *testing.T) {
	product := []byte("// generated\nmodule example.test/p\n\ngo 1.26\n\nrequire github.com/anthony-chaudhary/fak v0.0.0\n")
	repo := []byte("module github.com/anthony-chaudhary/fak\n\ngo 1.26.0 // minimum\n\ntoolchain go1.26.6\n")
	if got := requiredGoVersion(product, repo); got != "1.26.0" {
		t.Fatalf("requiredGoVersion=%q, want 1.26.0", got)
	}
	if got := goModDirective(repo, "toolchain"); got != "go1.26.6" {
		t.Fatalf("toolchain directive=%q", got)
	}
	if got := requiredGoVersion([]byte("module x\n")); got != "" {
		t.Fatalf("missing directive=%q", got)
	}
}

func TestCompareGoVersionsMatchesToolchainOrdering(t *testing.T) {
	ordered := []string{"1.23.1", "1.26", "1.26beta1", "1.26rc1", "1.26rc2", "go1.26.0", "go1.26.6 X:nocoverageredesign", "devel go1.26.6-abc", "1.27"}
	for i := range ordered {
		for j := range ordered {
			got := compareGoVersions(ordered[i], ordered[j])
			want := 0
			if i < j {
				want = -1
			} else if i > j {
				want = 1
			}
			if got != want {
				t.Fatalf("compare(%q,%q)=%d want %d", ordered[i], ordered[j], got, want)
			}
		}
	}
	for _, bad := range []string{"", "go", "1", "1.x", "1.26.x", "1.26gamma1"} {
		if compareGoVersions(bad, "1.0") >= 0 {
			t.Fatalf("unparseable %q must sort below any real version", bad)
		}
	}
	// The failing host: PATH go1.23.1 cannot satisfy the generated product, cached go1.26.6 can.
	if compareGoVersions("go1.23.1", "1.26.0") >= 0 || compareGoVersions("go1.26.6", "1.26.0") < 0 || compareGoVersions("go1.26.0", "1.26") < 0 {
		t.Fatal("satisfaction ordering broken")
	}
}

func TestOfflineGoCandidatesOrder(t *testing.T) {
	cached := []cachedToolchain{
		{version: "go1.24.0", bin: "/mod/golang.org/toolchain@v0.0.1-go1.24.0.darwin-arm64/bin/go"},
		{version: "go1.26.8", bin: "/mod/golang.org/toolchain@v0.0.1-go1.26.8.darwin-arm64/bin/go"},
		{version: "go1.26.0", bin: "/mod/golang.org/toolchain@v0.0.1-go1.26.0.darwin-arm64/bin/go"},
		{version: "go1.26.6", bin: "/mod/golang.org/toolchain@v0.0.1-go1.26.6.darwin-arm64/bin/go"},
	}
	got := offlineGoCandidates("", "/usr/local/go/bin/go", "go1.26.6", "go1.26.0", cached)
	want := []string{
		"/usr/local/go/bin/go",
		"/mod/golang.org/toolchain@v0.0.1-go1.26.6.darwin-arm64/bin/go",
		"/mod/golang.org/toolchain@v0.0.1-go1.26.0.darwin-arm64/bin/go",
		"/mod/golang.org/toolchain@v0.0.1-go1.26.8.darwin-arm64/bin/go",
		"/mod/golang.org/toolchain@v0.0.1-go1.24.0.darwin-arm64/bin/go",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("candidates=%q\nwant %q", got, want)
	}
	// A GOROOT exported by a `go run` parent comes first and duplicates collapse.
	got = offlineGoCandidates(cached[3].bin, cached[3].bin, "go1.26.6", "", cached)
	if got[0] != cached[3].bin || slices.Index(got[1:], cached[3].bin) >= 0 || len(got) != len(cached) {
		t.Fatalf("goroot-first dedupe=%q", got)
	}
	if got := offlineGoCandidates("", "", "", "", nil); len(got) != 0 {
		t.Fatalf("empty host candidates=%q", got)
	}
}
