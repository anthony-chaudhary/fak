package artifact

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func requireTools(t *testing.T, tools ...string) {
	t.Helper()
	for _, tool := range tools {
		if _, err := exec.LookPath(tool); err != nil {
			t.Skipf("%s is not on PATH: %v", tool, err)
		}
	}
}

func git(t *testing.T, dir string, args ...string) string {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %s: %v\n%s", strings.Join(args, " "), err, out)
	}
	return string(out)
}

// gitRepo commits files into a fresh temp repository and returns it with its HEAD. Tests
// never touch the live checkout.
func gitRepo(t *testing.T, files map[string]string) (string, string) {
	t.Helper()
	requireTools(t, "git")
	dir := t.TempDir()
	for name, body := range files {
		path := filepath.Join(dir, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	git(t, dir, "-c", "init.defaultBranch=main", "init", "-q")
	git(t, dir, "-c", "core.autocrlf=false", "add", "-A")
	git(t, dir, "-c", "user.name=deploykit-test", "-c", "user.email=deploykit-test@example.invalid",
		"-c", "commit.gpgsign=false", "-c", "core.autocrlf=false", "commit", "-q", "--no-verify", "-m", "pin")
	return dir, strings.TrimSpace(git(t, dir, "rev-parse", "HEAD"))
}

// tinyTuple is a two-repository tuple: an app module whose main package imports a library
// module that only the pinned go.work can resolve. The app prints `version --json` with the
// commit its -ldflags stamp.
func tinyTuple(t *testing.T) (app, appHead, lib, libHead string) {
	t.Helper()
	lib, libHead = gitRepo(t, map[string]string{
		"go.mod": "module example.com/tinylib\n\ngo 1.21\n",
		"lib.go": "package tinylib\n\nconst Version = \"1.2.3\"\n",
	})
	app, appHead = gitRepo(t, map[string]string{
		"go.mod": "module example.com/tinyapp\n\ngo 1.21\n\nrequire example.com/tinylib v0.0.0\n",
		"main.go": `package main

import (
	"encoding/json"
	"os"

	"example.com/tinylib"
)

var commit string

func main() {
	if len(os.Args) == 3 && os.Args[1] == "version" && os.Args[2] == "--json" {
		_ = json.NewEncoder(os.Stdout).Encode(map[string]any{
			"app_version": tinylib.Version, "commit": commit, "dirty": false, "stamped": commit != "",
		})
		return
	}
	os.Exit(2)
}
`,
	})
	return app, appHead, lib, libHead
}

// fakeToolchain scripts go (build writes payload to -o; vet passes unless vetFails) and the
// candidate smoke, and runs git for real against the temp repositories.
func fakeToolchain(smokeOut string, onBuild func(out string) error, vetFails bool) func(context.Context, call) (string, bool) {
	return func(ctx context.Context, c call) (string, bool) {
		switch {
		case c.name == "git":
			return realRun(ctx, c)
		case c.isGo() && len(c.args) == 2 && c.args[0] == "env" && c.args[1] == "GOVERSION":
			return "go1.26.7\n", true
		case c.isGo() && len(c.args) > 0 && c.args[0] == "build":
			out := ""
			for i, a := range c.args {
				if a == "-o" && i+1 < len(c.args) {
					out = c.args[i+1]
				}
			}
			if err := onBuild(out); err != nil {
				return err.Error(), false
			}
			return "", true
		case c.isGo() && len(c.args) > 0 && c.args[0] == "vet":
			if vetFails {
				return "vet: suspicious construct", false
			}
			return "", true
		case c.isSmoke():
			return smokeOut, true
		}
		return "unexpected command " + c.String(), false
	}
}

func writeCandidate(out string) error {
	return os.WriteFile(out, []byte("fake built candidate"), 0o700)
}

func envValue(env []string, key string) (string, bool) {
	val, found := "", false
	for _, kv := range env {
		if k, v, ok := strings.Cut(kv, "="); ok && k == key {
			val, found = v, true // the last entry wins, as for exec.Cmd
		}
	}
	return val, found
}

func TestBuildPinnedRefusesDirtyTree(t *testing.T) {
	cases := []struct {
		name  string
		dirty func(app, lib string) error
	}{
		{"untracked file in first repo", func(app, _ string) error {
			return os.WriteFile(filepath.Join(app, "stray.go"), []byte("package main\n"), 0o600)
		}},
		{"untracked file in second repo", func(_, lib string) error {
			return os.WriteFile(filepath.Join(lib, "stray.txt"), []byte("x"), 0o600)
		}},
		{"modified tracked file in second repo", func(_, lib string) error {
			return os.WriteFile(filepath.Join(lib, "lib.go"), []byte("package tinylib\n\nconst Version = \"6.6.6\"\n"), 0o600)
		}},
		{"staged change in first repo", func(app, _ string) error {
			if err := os.WriteFile(filepath.Join(app, "extra.go"), []byte("package main\n"), 0o600); err != nil {
				return err
			}
			cmd := exec.Command("git", "add", "extra.go")
			cmd.Dir = app
			return cmd.Run()
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tmp := isolateTemp(t)
			app, appHead, lib, libHead := tinyTuple(t)
			if err := tc.dirty(app, lib); err != nil {
				t.Fatal(err)
			}
			x := &scriptExec{handle: realRun}
			src := BuildPinned{ModuleDirs: []string{app, lib}, Commits: []string{appHead, libHead}, Package: ".", Ldflags: "-X main.commit=" + appHead, Generation: 1}
			c, err := Acquire(context.Background(), x, src)
			if !errors.Is(err, ErrDirtySource) {
				t.Fatalf("Acquire err = %v, want ErrDirtySource", err)
			}
			if c != (Candidate{}) {
				t.Fatalf("dirty tuple produced a candidate: %+v", c)
			}
			if n := x.count(call.isGo); n != 0 {
				t.Fatalf("dirty tuple reached the toolchain: %v", x.recorded())
			}
			if n := x.count(call.isSmoke); n != 0 {
				t.Fatalf("dirty tuple was smoked: %v", x.recorded())
			}
			assertNoLeftovers(t, tmp)
		})
	}
}

func TestBuildPinnedRefusesHeadNotPinned(t *testing.T) {
	for _, which := range []int{0, 1} {
		t.Run([]string{"first repo", "second repo"}[which], func(t *testing.T) {
			tmp := isolateTemp(t)
			app, appHead, lib, libHead := tinyTuple(t)
			commits := []string{appHead, libHead}
			commits[which] = commits[1-which] // a real commit, just not this repository's HEAD
			x := &scriptExec{handle: realRun}
			_, err := Acquire(context.Background(), x, BuildPinned{ModuleDirs: []string{app, lib}, Commits: commits, Package: ".", Generation: 1})
			if !errors.Is(err, ErrPinnedCommitMismatch) {
				t.Fatalf("Acquire err = %v, want ErrPinnedCommitMismatch", err)
			}
			if n := x.count(call.isGo); n != 0 {
				t.Fatalf("unpinned HEAD reached the toolchain: %v", x.recorded())
			}
			assertNoLeftovers(t, tmp)
		})
	}
	t.Run("not a repository", func(t *testing.T) {
		x := &scriptExec{handle: realRun}
		_, err := Acquire(context.Background(), x, BuildPinned{ModuleDirs: []string{t.TempDir()}, Commits: []string{commitA}, Package: ".", Generation: 1})
		if !errors.Is(err, ErrPinnedCommitMismatch) {
			t.Fatalf("Acquire err = %v, want ErrPinnedCommitMismatch", err)
		}
	})
}

// TestBuildPinnedBuildsUnderPinnedGoWork is the positive witness on the real toolchain: a
// two-repository tuple builds, vets, and smokes under the generated go.work while an ambient
// GOWORK and GOFLAGS that would break the build are in the environment.
func TestBuildPinnedBuildsUnderPinnedGoWork(t *testing.T) {
	requireTools(t, "go", "git")
	tmp := isolateTemp(t)
	app, appHead, lib, libHead := tinyTuple(t)
	ambient := filepath.Join(t.TempDir(), "go.work")
	if err := os.WriteFile(ambient, []byte("go 1.21\n\nuse ./does-not-exist\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("GOWORK", ambient)
	t.Setenv("GOFLAGS", "-mod=vendor")

	x := &scriptExec{handle: realRun}
	src := BuildPinned{
		ModuleDirs: []string{app, lib}, Commits: []string{strings.ToUpper(appHead), libHead},
		Package: ".", Ldflags: "-X main.commit=" + appHead, Generation: 7,
	}
	c, err := Acquire(context.Background(), x, src)
	if err != nil {
		t.Fatalf("Acquire: %v\ncalls: %v", err, x.recorded())
	}
	defer c.Remove()
	if c.Kind != KindBuild || c.Commit != appHead || c.AppVersion != "1.2.3" || c.Generation != 7 {
		t.Fatalf("candidate = %+v", c)
	}
	if err := VerifyBytes(c.Path, c.Digest, c.Size); err != nil {
		t.Fatalf("candidate bytes do not match the recorded identity: %v", err)
	}

	var goCalls int
	for _, cl := range x.recorded() {
		if !cl.isGo() {
			continue
		}
		goCalls++
		if !cl.withEnvSet {
			t.Fatalf("%s ran without the pinned environment", cl)
		}
		work, _ := envValue(cl.env, "GOWORK")
		if cl.args[0] == "env" {
			if work != "off" {
				t.Fatalf("toolchain probe GOWORK=%q, want off", work)
			}
		} else if work == ambient || filepath.Base(work) != "go.work" || !strings.HasPrefix(filepath.Base(filepath.Dir(work)), "fak-deploykit-gowork-") {
			t.Fatalf("%s ran with GOWORK=%q, want the generated workspace", cl, work)
		}
		if v, ok := envValue(cl.env, "GOFLAGS"); !ok || v != "" {
			t.Fatalf("%s ran with GOFLAGS=%q (set=%v), want cleared", cl, v, ok)
		}
		if v, _ := envValue(cl.env, "GOTOOLCHAIN"); v != "local" {
			t.Fatalf("%s ran with GOTOOLCHAIN=%q, want local", cl, v)
		}
		if v, _ := envValue(cl.env, "GOOS"); v != runtime.GOOS {
			t.Fatalf("%s ran with GOOS=%q", cl, v)
		}
	}
	if goCalls != 3 { // env GOVERSION, build, vet
		t.Fatalf("go calls = %d, want 3: %v", goCalls, x.recorded())
	}
	if n := x.count(call.isSmoke); n != 2 {
		t.Fatalf("smokes = %d, want 2 (gate and Verify)", n)
	}

	// The generated workspace is gone; only the candidate's staging dir remains until Remove.
	if err := c.Remove(); err != nil {
		t.Fatal(err)
	}
	assertNoLeftovers(t, tmp)
}

// TestBuildPinnedRefusesSmokedCommitMismatch builds for real with an ldflags stamp of a real,
// clean commit that is not the main module's pin.
func TestBuildPinnedRefusesSmokedCommitMismatch(t *testing.T) {
	requireTools(t, "go", "git")
	tmp := isolateTemp(t)
	app, appHead, lib, libHead := tinyTuple(t)
	x := &scriptExec{handle: realRun}
	c, err := Acquire(context.Background(), x, BuildPinned{
		ModuleDirs: []string{app, lib}, Commits: []string{appHead, libHead},
		Package: ".", Ldflags: "-X main.commit=" + libHead, Generation: 1,
	})
	if !errors.Is(err, ErrSmokeCommitMismatch) {
		t.Fatalf("Acquire err = %v, want ErrSmokeCommitMismatch\ncalls: %v", err, x.recorded())
	}
	if c != (Candidate{}) {
		t.Fatalf("mismatched smoke produced a candidate: %+v", c)
	}
	assertNoLeftovers(t, tmp)
}

func TestBuildPinnedSmokeAndLadderRefusals(t *testing.T) {
	cases := []struct {
		name     string
		smoke    func(appHead string) string
		onBuild  func(out string) error
		vetFails bool
		want     error
	}{
		{"smoked commit differs", func(string) string { return versionJSON(commitB, false, true, "1.0.0") }, writeCandidate, false, ErrSmokeCommitMismatch},
		{"smoked dirty", func(h string) string { return versionJSON(h, true, true, "1.0.0") }, writeCandidate, false, ErrProvenance},
		{"smoked unstamped", func(string) string { return versionJSON("", false, false, "1.0.0") }, writeCandidate, false, ErrProvenance},
		{"smoke is plain text", func(h string) string { return "fak " + h }, writeCandidate, false, ErrProvenance},
		{"build fails", func(h string) string { return versionJSON(h, false, true, "1.0.0") }, func(string) error { return errors.New("compile error") }, false, ErrBuildFailed},
		{"vet fails", func(h string) string { return versionJSON(h, false, true, "1.0.0") }, writeCandidate, true, ErrVetFailed},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tmp := isolateTemp(t)
			app, appHead, lib, libHead := tinyTuple(t)
			x := &scriptExec{handle: fakeToolchain(tc.smoke(appHead), tc.onBuild, tc.vetFails)}
			c, err := Acquire(context.Background(), x, BuildPinned{ModuleDirs: []string{app, lib}, Commits: []string{appHead, libHead}, Package: ".", Generation: 1})
			if !errors.Is(err, tc.want) {
				t.Fatalf("Acquire err = %v, want %v", err, tc.want)
			}
			if c != (Candidate{}) {
				t.Fatalf("refused Acquire returned a candidate: %+v", c)
			}
			assertNoLeftovers(t, tmp)
		})
	}
}

func TestBuildPinnedRefusesSourceChangedDuringBuild(t *testing.T) {
	tmp := isolateTemp(t)
	app, appHead, lib, libHead := tinyTuple(t)
	x := &scriptExec{handle: fakeToolchain(versionJSON(appHead, false, true, "1.0.0"), func(out string) error {
		// A peer writes into the second repository while the build runs.
		if err := os.WriteFile(filepath.Join(lib, "late.go"), []byte("package tinylib\n"), 0o600); err != nil {
			return err
		}
		return writeCandidate(out)
	}, false)}
	_, err := Acquire(context.Background(), x, BuildPinned{ModuleDirs: []string{app, lib}, Commits: []string{appHead, libHead}, Package: ".", Generation: 1})
	if !errors.Is(err, ErrDirtySource) || !strings.Contains(err.Error(), "after build") {
		t.Fatalf("Acquire err = %v, want ErrDirtySource after build", err)
	}
	assertNoLeftovers(t, tmp)
}

func TestBuildPinnedFakeLadderPasses(t *testing.T) {
	tmp := isolateTemp(t)
	app, appHead, lib, libHead := tinyTuple(t)
	fake := fakeToolchain(versionJSON(appHead, false, true, "1.0.0"), writeCandidate, false)
	var workDuringBuild string // the generated go.work, read while it exists
	x := &scriptExec{handle: func(ctx context.Context, c call) (string, bool) {
		if c.isGo() && c.args[0] == "build" {
			work, _ := envValue(c.env, "GOWORK")
			data, err := os.ReadFile(work)
			if err != nil {
				return "read generated go.work: " + err.Error(), false
			}
			workDuringBuild = string(data)
		}
		return fake(ctx, c)
	}}
	c, err := Acquire(context.Background(), x, BuildPinned{
		ModuleDirs: []string{app, lib}, Commits: []string{appHead, libHead}, GoWorkPin: "1.22",
		Package: "./cmd/x", Vet: []string{"./cmd/x", "./internal/..."}, Generation: 2,
	})
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	defer c.Remove()
	var sawVet, sawProbe bool
	for _, cl := range x.recorded() {
		if cl.isGo() && cl.args[0] == "vet" {
			sawVet = strings.Join(cl.args, " ") == "vet ./cmd/x ./internal/..."
		}
		if cl.isGo() && cl.args[0] == "env" {
			sawProbe = true
		}
		if cl.isGo() && cl.args[0] == "build" {
			if got := cl.args[len(cl.args)-1]; got != "./cmd/x" {
				t.Fatalf("built %q, want the descriptor package", got)
			}
		}
	}
	want := "go 1.22\n\nuse (\n\t\"" + filepath.ToSlash(app) + "\"\n\t\"" + filepath.ToSlash(lib) + "\"\n)\n"
	if workDuringBuild != want {
		t.Fatalf("generated go.work =\n%s\nwant\n%s", workDuringBuild, want)
	}
	if !sawVet {
		t.Fatalf("vet did not run the descriptor's targets: %v", x.recorded())
	}
	if sawProbe {
		t.Fatalf("an explicit GoWorkPin must not probe the toolchain version")
	}
	if err := c.Remove(); err != nil {
		t.Fatal(err)
	}
	assertNoLeftovers(t, tmp)
}

func TestBuildPinnedPreflight(t *testing.T) {
	other := "windows"
	if runtime.GOOS == "windows" {
		other = "linux"
	}
	good := BuildPinned{ModuleDirs: []string{"a", "b"}, Commits: []string{commitA, commitB}, Package: "./cmd/fak", Generation: 1}
	cases := []struct {
		name string
		edit func(*BuildPinned)
		want error
	}{
		{"no module dirs", func(b *BuildPinned) { b.ModuleDirs, b.Commits = nil, nil }, ErrInvalidSource},
		{"empty module dir", func(b *BuildPinned) { b.ModuleDirs = []string{"a", " "} }, ErrInvalidSource},
		{"duplicate module dir", func(b *BuildPinned) { b.ModuleDirs = []string{"a", "./a"} }, ErrInvalidSource},
		{"commit count", func(b *BuildPinned) { b.Commits = b.Commits[:1] }, ErrInvalidSource},
		{"short commit", func(b *BuildPinned) { b.Commits = []string{commitA, "bbbbbbb"} }, ErrInvalidSource},
		{"empty package", func(b *BuildPinned) { b.Package = "" }, ErrInvalidSource},
		{"flag as package", func(b *BuildPinned) { b.Package = "-toolexec=evil" }, ErrInvalidSource},
		{"flag as vet target", func(b *BuildPinned) { b.Vet = []string{"./...", "-vettool=evil"} }, ErrInvalidSource},
		{"bad go.work pin", func(b *BuildPinned) { b.GoWorkPin = "latest" }, ErrInvalidSource},
		{"zero generation", func(b *BuildPinned) { b.Generation = 0 }, ErrInvalidSource},
		{"foreign GOOS", func(b *BuildPinned) { b.GOOS = other }, ErrCrossBuildNotWired},
		{"foreign GOARCH", func(b *BuildPinned) { b.GOARCH = "riscv64"; b.GOOS = runtime.GOOS }, ErrCrossBuildNotWired},
	}
	if runtime.GOARCH == "riscv64" {
		cases = cases[:len(cases)-1]
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tmp := isolateTemp(t)
			src := good
			src.ModuleDirs = append([]string(nil), good.ModuleDirs...)
			src.Commits = append([]string(nil), good.Commits...)
			tc.edit(&src)
			x := &scriptExec{handle: realRun}
			if _, err := Acquire(context.Background(), x, src); !errors.Is(err, tc.want) {
				t.Fatalf("Acquire err = %v, want %v", err, tc.want)
			}
			if len(x.recorded()) != 0 {
				t.Fatalf("preflight refusal executed %v", x.recorded())
			}
			assertNoLeftovers(t, tmp)
		})
	}

	t.Run("executor without RunEnv", func(t *testing.T) {
		tmp := isolateTemp(t)
		inner := &scriptExec{handle: realRun}
		if _, err := Acquire(context.Background(), runOnly{inner}, good); !errors.Is(err, ErrEnvExecutorRequired) {
			t.Fatalf("Acquire err = %v, want ErrEnvExecutorRequired", err)
		}
		if len(inner.recorded()) != 0 {
			t.Fatalf("refusal executed %v", inner.recorded())
		}
		assertNoLeftovers(t, tmp)
	})
	t.Run("host platform spelled out is accepted", func(t *testing.T) {
		src := good
		src.GOOS, src.GOARCH = runtime.GOOS, runtime.GOARCH
		if err := src.preflight(&scriptExec{}); err != nil {
			t.Fatalf("preflight: %v", err)
		}
	})
}

func TestSourceKinds(t *testing.T) {
	if (BuildPinned{}).Kind() != KindBuild || (Fetch{}).Kind() != KindFetch {
		t.Fatal("source kinds drifted from the deploykit.Source.Kind tags")
	}
}
