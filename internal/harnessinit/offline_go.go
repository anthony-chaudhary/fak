package harnessinit

import (
	"context"
	"fmt"
	"go/build"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// The generated-product selfcheck runs with GOPROXY=off and GOSUMDB=off. The go command
// verifies every golang.org/toolchain module against the checksum database -- even one
// already extracted in the module cache -- so under GOSUMDB=off any GOTOOLCHAIN=auto switch
// fails ("verifying module: checksum database disabled by GOSUMDB=off"). A bare `go` from
// PATH that is older than the generated go directive therefore can never build the product
// offline. The runner instead pins GOTOOLCHAIN=local and picks a locally installed toolchain
// that already satisfies the product's and the replaced fak module's go directives.

const goToolchainModulePrefix = "toolchain@v0.0.1-"

// offlineGoEnv returns environ with the offline module settings forced and GOROOT removed so
// the selected go binary derives its own root instead of inheriting a different toolchain's.
func offlineGoEnv(environ []string) []string {
	out := make([]string, 0, len(environ)+4)
	for _, kv := range environ {
		key, _, _ := strings.Cut(kv, "=")
		switch strings.ToUpper(key) {
		case "GOROOT", "GOTOOLCHAIN", "GOPROXY", "GOSUMDB", "GOWORK":
			continue
		}
		if strings.HasPrefix(strings.ToUpper(key), "GOTOOLCHAIN_INTERNAL_") {
			continue
		}
		out = append(out, kv)
	}
	return append(out, "GOPROXY=off", "GOSUMDB=off", "GOWORK=off", "GOTOOLCHAIN=local")
}

// goModDirective returns the argument of the first `go` or `toolchain` directive in a go.mod.
func goModDirective(gomod []byte, name string) string {
	for _, line := range strings.Split(string(gomod), "\n") {
		line, _, _ = strings.Cut(line, "//")
		fields := strings.Fields(line)
		if len(fields) == 2 && fields[0] == name {
			return fields[1]
		}
	}
	return ""
}

// requiredGoVersion is the highest go directive among the given go.mod files.
func requiredGoVersion(gomods ...[]byte) string {
	want := ""
	for _, raw := range gomods {
		if v := goModDirective(raw, "go"); v != "" && (want == "" || compareGoVersions(v, want) > 0) {
			want = v
		}
	}
	return want
}

type goVersion struct {
	major, minor int
	kind         int // 0 language version (1.26), 1 beta, 2 rc, 3 release (1.26.0)
	n            int // beta/rc number or patch release
	devel        bool
	ok           bool
}

// parseGoVersion accepts go directive values (1.26, 1.26.0, 1.26rc1), toolchain names
// (go1.26.6), and GOVERSION strings (go1.26.6 X:exp, devel go1.27-abcdef ...).
func parseGoVersion(s string) goVersion {
	s = strings.TrimSpace(s)
	if rest, found := strings.CutPrefix(s, "devel "); found {
		v := parseGoVersion(strings.SplitN(rest, "-", 2)[0])
		v.devel = true
		return v
	}
	if fields := strings.Fields(s); len(fields) > 0 {
		s = strings.TrimPrefix(fields[0], "go")
	}
	majorText, rest, found := strings.Cut(s, ".")
	if !found {
		return goVersion{}
	}
	major, err := strconv.Atoi(majorText)
	if err != nil {
		return goVersion{}
	}
	end := strings.IndexFunc(rest, func(r rune) bool { return r < '0' || r > '9' })
	if end < 0 {
		end = len(rest)
	}
	minor, err := strconv.Atoi(rest[:end])
	if err != nil {
		return goVersion{}
	}
	v := goVersion{major: major, minor: minor, ok: true}
	suffix := rest[end:]
	for kind, prefix := range []string{1: "beta", 2: "rc", 3: "."} {
		if tail, found := strings.CutPrefix(suffix, prefix); found && prefix != "" {
			if v.n, err = strconv.Atoi(tail); err != nil {
				return goVersion{}
			}
			v.kind = kind
			return v
		}
	}
	if suffix != "" {
		return goVersion{}
	}
	return v
}

// compareGoVersions orders Go versions like cmd/go/internal/gover:
// 1.26 < 1.26beta1 < 1.26rc1 < 1.26.0 < 1.26.6; a devel build sorts after its base release.
func compareGoVersions(a, b string) int {
	x, y := parseGoVersion(a), parseGoVersion(b)
	for _, d := range [][2]int{{boolInt(x.ok), boolInt(y.ok)}, {x.major, y.major}, {x.minor, y.minor}, {x.kind, y.kind}, {x.n, y.n}, {boolInt(x.devel), boolInt(y.devel)}} {
		if d[0] != d[1] {
			if d[0] < d[1] {
				return -1
			}
			return 1
		}
	}
	return 0
}

func boolInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

type cachedToolchain struct {
	version string // e.g. go1.26.6
	bin     string
}

// offlineGoCandidates orders the go binaries to try: the toolchain a `go run`/`go test`
// parent exported via GOROOT, the PATH go, then module-cache toolchains preferring the one
// that built this binary, then the fak module's toolchain directive, then newest first.
func offlineGoCandidates(gorootBin, pathGo, buildVersion, toolchainDirective string, cached []cachedToolchain) []string {
	sorted := append([]cachedToolchain(nil), cached...)
	rank := func(c cachedToolchain) int {
		switch c.version {
		case buildVersion:
			return 0
		case toolchainDirective:
			return 1
		}
		return 2
	}
	sort.SliceStable(sorted, func(i, j int) bool {
		if ri, rj := rank(sorted[i]), rank(sorted[j]); ri != rj {
			return ri < rj
		}
		return compareGoVersions(sorted[i].version, sorted[j].version) > 0
	})
	var out []string
	seen := map[string]bool{}
	add := func(bin string) {
		if bin == "" {
			return
		}
		key := filepath.Clean(bin)
		if resolved, err := filepath.EvalSymlinks(bin); err == nil {
			key = resolved
		}
		if !seen[key] {
			seen[key] = true
			out = append(out, bin)
		}
	}
	add(gorootBin)
	add(pathGo)
	for _, c := range sorted {
		add(c.bin)
	}
	return out
}

func goExecutable(root string) string {
	name := "go"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	return filepath.Join(root, "bin", name)
}

func fileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

func goEnvValue(ctx context.Context, bin, dir, key string) (string, error) {
	cmd := exec.CommandContext(ctx, bin, "env", key)
	cmd.Dir = dir
	cmd.Env = offlineGoEnv(os.Environ())
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}

func cachedToolchains(modcache string) []cachedToolchain {
	suffix := "." + runtime.GOOS + "-" + runtime.GOARCH
	matches, _ := filepath.Glob(filepath.Join(modcache, "golang.org", goToolchainModulePrefix+"go*"+suffix))
	var out []cachedToolchain
	for _, dir := range matches {
		version := strings.TrimSuffix(strings.TrimPrefix(filepath.Base(dir), goToolchainModulePrefix), suffix)
		if bin := goExecutable(dir); parseGoVersion(version).ok && fileExists(bin) {
			out = append(out, cachedToolchain{version: version, bin: bin})
		}
	}
	return out
}

// resolveOfflineGo picks a local go binary whose version satisfies the product and repository
// go directives, so the offline selfcheck never needs a toolchain download.
func resolveOfflineGo(ctx context.Context, productDir, repoRoot string) (string, error) {
	productMod, err := os.ReadFile(filepath.Join(productDir, "go.mod"))
	if err != nil {
		return "", err
	}
	repoMod, err := os.ReadFile(filepath.Join(repoRoot, "go.mod"))
	if err != nil {
		return "", err
	}
	want := requiredGoVersion(productMod, repoMod)
	probeDir := filepath.Dir(productDir)

	gorootBin := ""
	if root := os.Getenv("GOROOT"); root != "" && fileExists(goExecutable(root)) {
		gorootBin = goExecutable(root)
	}
	pathGo, _ := exec.LookPath("go")
	modcache := os.Getenv("GOMODCACHE")
	for _, bin := range []string{gorootBin, pathGo} {
		if modcache == "" && bin != "" {
			modcache, _ = goEnvValue(ctx, bin, probeDir, "GOMODCACHE")
		}
	}
	if gopath := filepath.SplitList(build.Default.GOPATH); modcache == "" && len(gopath) > 0 {
		modcache = filepath.Join(gopath[0], "pkg", "mod")
	}

	candidates := offlineGoCandidates(gorootBin, pathGo, runtime.Version(), goModDirective(repoMod, "toolchain"), cachedToolchains(modcache))
	var tried []string
	for _, bin := range candidates {
		have, err := goEnvValue(ctx, bin, probeDir, "GOVERSION")
		if err != nil {
			tried = append(tried, bin+": "+err.Error())
			continue
		}
		if want == "" || compareGoVersions(have, want) >= 0 {
			return bin, nil
		}
		tried = append(tried, bin+"="+have)
	}
	if len(tried) == 0 {
		tried = append(tried, "no go binary found")
	}
	return "", fmt.Errorf("no local Go toolchain satisfies go %s; the offline selfcheck pins GOTOOLCHAIN=local because GOSUMDB=off forbids toolchain switches (tried %s)", want, strings.Join(tried, ", "))
}
