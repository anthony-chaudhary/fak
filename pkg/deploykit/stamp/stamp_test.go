package stamp_test

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/binstamp"
	"github.com/anthony-chaudhary/fak/internal/versionskew"
	"github.com/anthony-chaudhary/fak/pkg/deploykit/stamp"
)

// head is the commit the real fixture in testdata/fak-version.json reports.
const head = "d5840f2e9ee30648b486b4284cb49f3cfccdac4a"

// otherHead is a clean full revision that differs from head.
const otherHead = "0123456789abcdef0123456789abcdef01234567"

func versionDoc(commit string, dirty, stamped bool) []byte {
	return []byte(fmt.Sprintf(`{"app_version":"0.55.0","commit":%q,"dirty":%t,"stamped":%t}`, commit, dirty, stamped))
}

// noGitRunner fails the test if AssessSkew shells out; an un-attestable stamp must be decided
// from the stamp alone.
func noGitRunner(t *testing.T) stamp.Runner {
	t.Helper()
	return func(_ context.Context, _, name string, args ...string) (string, bool) {
		t.Errorf("AssessSkew ran %s %s for an un-attestable stamp", name, strings.Join(args, " "))
		// Claim the tip equals head, so a skipped guard would surface as VerdictFresh.
		return head + "\n", true
	}
}

// TestDirtyOrUnstampedNeverCurrent is the negative witness: a dirty or unstamped identity whose
// commit equals HEAD must never be reported Fresh, by Compare, Explain, or AssessSkew.
func TestDirtyOrUnstampedNeverCurrent(t *testing.T) {
	cases := []struct {
		name        string
		doc         []byte
		wantCause   stamp.Cause
		wantVerdict stamp.Verdict
	}{
		{"dirty:true at HEAD", versionDoc(head, true, true), stamp.CauseDirty, stamp.VerdictDirty},
		{"stamped:false at HEAD", versionDoc(head, false, false), stamp.CauseUnstamped, stamp.VerdictUnstamped},
		{"stamped:false no commit", versionDoc("", false, false), stamp.CauseUnstamped, stamp.VerdictUnstamped},
		{"stamped:false dirty:true at HEAD", versionDoc(head, true, false), stamp.CauseUnstamped, stamp.VerdictUnstamped},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			id, err := stamp.ParseVersionJSON(tc.doc)
			if err != nil {
				t.Fatalf("ParseVersionJSON(%s) error = %v; a dirty or unstamped document is a classified identity, not a parse failure", tc.doc, err)
			}
			st := id.Stamp()
			for _, h := range []string{head, strings.ToUpper(head), head[:12]} {
				f, cause := stamp.Explain(st, h)
				if f == stamp.Fresh {
					t.Fatalf("Explain(%+v, %s) reported Fresh: a dirty/unstamped binary was reported current", st, h)
				}
				if f != stamp.Unknown || cause != tc.wantCause {
					t.Fatalf("Explain(%+v, %s) = (%v, %v), want (unknown, %v)", st, h, f, cause, tc.wantCause)
				}
				if got := stamp.Compare(st, h); got == stamp.Fresh {
					t.Fatalf("Compare(%+v, %s) reported Fresh: a dirty/unstamped binary was reported current", st, h)
				}
			}
			a := stamp.AssessSkew(context.Background(), noGitRunner(t), t.TempDir(), "HEAD", st)
			if a.Verdict == stamp.VerdictFresh {
				t.Fatalf("AssessSkew reported VerdictFresh for %s", tc.doc)
			}
			if a.Verdict != tc.wantVerdict || !a.Verdict.Refusable() {
				t.Fatalf("AssessSkew verdict = %v (refusable=%t), want refusable %v", a.Verdict, a.Verdict.Refusable(), tc.wantVerdict)
			}
		})
	}

	// The same rule holds for a Stamp read straight from build info.
	for _, st := range []stamp.Stamp{
		{Revision: head, Dirty: true, HasVCS: true},
		{Revision: head, HasVCS: false},
		{},
	} {
		if f := stamp.Compare(st, head); f == stamp.Fresh {
			t.Fatalf("Compare(%+v, HEAD) reported Fresh", st)
		}
	}

	// Control: the clean, stamped identity at the same HEAD IS Fresh, so the assertions above
	// cannot pass merely because nothing ever matches.
	id, err := stamp.ParseVersionJSON(versionDoc(head, false, true))
	if err != nil {
		t.Fatalf("ParseVersionJSON(clean) error = %v", err)
	}
	if f, cause := stamp.Explain(id.Stamp(), head); f != stamp.Fresh || cause != stamp.CauseMatched {
		t.Fatalf("control: Explain(clean identity, HEAD) = (%v, %v), want (fresh, matched)", f, cause)
	}
}

func TestParseVersionJSON(t *testing.T) {
	t.Run("real fak version --json fixture", func(t *testing.T) {
		data, err := os.ReadFile(filepath.Join("testdata", "fak-version.json"))
		if err != nil {
			t.Fatal(err)
		}
		id, err := stamp.ParseVersionJSON(data)
		if err != nil {
			t.Fatalf("ParseVersionJSON(fixture) error = %v", err)
		}
		want := stamp.Identity{AppVersion: "0.55.0", Commit: head, Dirty: false, Stamped: true}
		if id != want {
			t.Fatalf("ParseVersionJSON(fixture) = %+v, want %+v", id, want)
		}
		if st := id.Stamp(); st != (stamp.Stamp{Revision: head, HasVCS: true}) {
			t.Fatalf("Identity.Stamp() = %+v", st)
		}
		if f, cause := stamp.Explain(id.Stamp(), head); f != stamp.Fresh || cause != stamp.CauseMatched {
			t.Fatalf("Explain(fixture, own commit) = (%v, %v), want (fresh, matched)", f, cause)
		}
		if f, cause := stamp.Explain(id.Stamp(), otherHead); f != stamp.Stale || cause != stamp.CauseDiverged {
			t.Fatalf("Explain(fixture, other commit) = (%v, %v), want (stale, diverged)", f, cause)
		}
	})

	t.Run("tolerates BOM, CRLF and case", func(t *testing.T) {
		doc := "\xef\xbb\xbf \r\n" + `{"app_version":" 0.55.0 ","commit":"` + strings.ToUpper(head) + `","dirty":false,"stamped":true,"extra":{"x":1}}` + "\r\n"
		id, err := stamp.ParseVersionJSON([]byte(doc))
		if err != nil {
			t.Fatalf("ParseVersionJSON error = %v", err)
		}
		if id.Commit != head || id.AppVersion != "0.55.0" {
			t.Fatalf("ParseVersionJSON = %+v, want lower-cased commit and trimmed app_version", id)
		}
	})

	t.Run("app_version is optional", func(t *testing.T) {
		id, err := stamp.ParseVersionJSON([]byte(`{"commit":"` + head + `","dirty":false,"stamped":true}`))
		if err != nil || id.AppVersion != "" || id.Commit != head {
			t.Fatalf("ParseVersionJSON = (%+v, %v)", id, err)
		}
	})

	malformed := map[string]string{
		"empty":             "",
		"whitespace":        " \n\t",
		"not json":          "fak 0.55.0\nbuild: d5840f2e9ee3",
		"truncated":         `{"commit":"` + head,
		"array":             `[]`,
		"null":              `null`,
		"string":            `"` + head + `"`,
		"trailing data":     string(versionDoc(head, false, true)) + ` {}`,
		"commit wrong type": `{"commit":1,"dirty":false,"stamped":true}`,
		"dirty wrong type":  `{"commit":"` + head + `","dirty":"false","stamped":true}`,
		"missing all":       `{"app_version":"0.55.0"}`,
		"missing dirty":     `{"commit":"` + head + `","stamped":true}`,
		"missing stamped":   `{"commit":"` + head + `","dirty":false}`,
		"null commit":       `{"commit":null,"dirty":false,"stamped":true}`,
	}
	for name, doc := range malformed {
		t.Run("malformed/"+name, func(t *testing.T) {
			id, err := stamp.ParseVersionJSON([]byte(doc))
			if !errors.Is(err, stamp.ErrMalformedVersionJSON) {
				t.Fatalf("ParseVersionJSON(%q) error = %v, want ErrMalformedVersionJSON", doc, err)
			}
			if id != (stamp.Identity{}) {
				t.Fatalf("ParseVersionJSON(%q) identity = %+v, want zero", doc, id)
			}
			if f, cause := stamp.Explain(id.Stamp(), head); f != stamp.Unknown || cause != stamp.CauseUnstamped {
				t.Fatalf("malformed identity classified (%v, %v), want (unknown, unstamped)", f, cause)
			}
		})
	}

	badCommit := map[string][]byte{
		"short prefix of HEAD":   versionDoc(head[:12], false, true),
		"39 hex":                 versionDoc(head[:39], false, true),
		"41 hex":                 versionDoc(head+"a", false, true),
		"non-hex":                versionDoc(strings.Repeat("g", 40), false, true),
		"sha256 object id":       versionDoc(strings.Repeat("ab", 32), false, true),
		"stamped without commit": versionDoc("", false, true),
		"unstamped junk commit":  versionDoc("not-a-commit", false, false),
	}
	for name, doc := range badCommit {
		t.Run("invalid commit/"+name, func(t *testing.T) {
			id, err := stamp.ParseVersionJSON(doc)
			if !errors.Is(err, stamp.ErrInvalidCommit) || errors.Is(err, stamp.ErrMalformedVersionJSON) {
				t.Fatalf("ParseVersionJSON(%s) error = %v, want only ErrInvalidCommit", doc, err)
			}
			for _, h := range []string{head, head[:12]} {
				if f, cause := stamp.Explain(id.Stamp(), h); f != stamp.Unknown || cause != stamp.CauseUnstamped {
					t.Fatalf("Explain(%+v, %s) = (%v, %v), want (unknown, unstamped)", id, h, f, cause)
				}
			}
		})
	}
}

// TestFacadeForwardsVerdicts pins Compare/Explain/AssessSkew to the primitives they wrap.
func TestFacadeForwardsVerdicts(t *testing.T) {
	stamps := []stamp.Stamp{
		{},
		{Revision: head, HasVCS: true},
		{Revision: head[:12], HasVCS: true},
		{Revision: head, Dirty: true, HasVCS: true},
		{Revision: otherHead, HasVCS: true},
	}
	for _, st := range stamps {
		for _, h := range []string{"", head, " " + head + "\n", otherHead} {
			gotF, gotC := stamp.Explain(st, h)
			wantF, wantC := binstamp.Explain(st, h)
			if gotF != wantF || gotC != wantC || stamp.Compare(st, h) != binstamp.Compare(st, h) {
				t.Fatalf("Explain/Compare(%+v, %q) = (%v, %v), binstamp = (%v, %v)", st, h, gotF, gotC, wantF, wantC)
			}
		}
	}

	// A behind binary: trunk resolves to otherHead, and head is its strict ancestor.
	behind := func(_ context.Context, _, _ string, args ...string) (string, bool) {
		switch {
		case len(args) >= 4 && args[0] == "rev-parse" && args[3] == "trunk^{commit}":
			return otherHead + "\n", true
		case len(args) >= 1 && args[0] == "rev-parse":
			return "", true
		case len(args) == 4 && args[0] == "merge-base":
			return "", args[2] == head && args[3] == otherHead
		}
		return "", false
	}
	running := stamp.Stamp{Revision: head, HasVCS: true}
	got := stamp.AssessSkew(context.Background(), behind, "", "trunk", running)
	want := versionskew.AssessStamp(context.Background(), behind, "", "trunk", running)
	if got != want || got.Verdict != stamp.VerdictSkewed || got.Relation != stamp.RelBehind || !got.Verdict.Refusable() {
		t.Fatalf("AssessSkew = %+v, versionskew = %+v, want refusable SKEWED/behind", got, want)
	}

	// A nil runner defaults to RealRunner; an unstamped binary never reaches git.
	if a := stamp.AssessSkew(context.Background(), nil, t.TempDir(), "HEAD", stamp.Stamp{}); a.Verdict != stamp.VerdictUnstamped {
		t.Fatalf("AssessSkew(nil runner, unstamped) = %+v, want UNSTAMPED", a)
	}
}

func TestAppVersion(t *testing.T) {
	t.Setenv("FAK_APP_VERSION", "")
	if got, want := stamp.AppVersion(), appversion.Current(); got != want || got == "" {
		t.Fatalf("AppVersion() = %q, appversion.Current() = %q", got, want)
	}
	t.Setenv("FAK_APP_VERSION", "9.9.9-stamp-test")
	if got := stamp.AppVersion(); got != "9.9.9-stamp-test" {
		t.Fatalf("AppVersion() = %q, want the FAK_APP_VERSION override", got)
	}
}

func TestFromFileRejectsNonBinary(t *testing.T) {
	for _, path := range []string{filepath.Join("testdata", "fak-version.json"), filepath.Join(t.TempDir(), "missing")} {
		st, err := stamp.FromFile(path)
		if err == nil {
			t.Fatalf("FromFile(%s) error = nil, want a build-info read error", path)
		}
		if st != (stamp.Stamp{}) || stamp.Compare(st, head) == stamp.Fresh {
			t.Fatalf("FromFile(%s) = %+v, want the zero Stamp", path, st)
		}
	}
}

// TestFromFileMatchesSelf covers the test binary itself and a real, VCS-stamped binary built
// from a throwaway git repository: FromFile must read back the same revision the binary
// reports through Self, and the facade's verdicts must hold over real git.
func TestFromFileMatchesSelf(t *testing.T) {
	t.Run("test binary", func(t *testing.T) {
		exe, err := os.Executable()
		if err != nil {
			t.Fatal(err)
		}
		got, err := stamp.FromFile(exe)
		if err != nil {
			t.Fatalf("FromFile(test binary) error = %v", err)
		}
		if self := stamp.Self(); got != self {
			t.Fatalf("FromFile(test binary) = %+v, Self() = %+v", got, self)
		}
	})

	t.Run("vcs-stamped build", func(t *testing.T) {
		if testing.Short() {
			t.Skip("builds two probe binaries; skipped in -short mode")
		}
		goBin, err := exec.LookPath("go")
		if err != nil {
			t.Skip("go toolchain not on PATH")
		}
		if _, err := exec.LookPath("git"); err != nil {
			t.Skip("git not on PATH")
		}
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
		defer cancel()

		repo, outDir := t.TempDir(), t.TempDir()
		env := hermeticEnv()
		git := func(args ...string) string {
			t.Helper()
			cmd := exec.CommandContext(ctx, "git", args...)
			cmd.Dir, cmd.Env = repo, env
			out, err := cmd.CombinedOutput()
			if err != nil {
				t.Fatalf("git %s: %v\n%s", strings.Join(args, " "), err, out)
			}
			return strings.TrimSpace(string(out))
		}
		writeProbeModule(t, repo)
		git("init", "-q")
		git("config", "user.email", "stamp-test@example.invalid")
		git("config", "user.name", "stamp test")
		git("config", "commit.gpgsign", "false")
		git("config", "core.autocrlf", "false")
		git("config", "core.hooksPath", t.TempDir())
		git("add", "go.mod", "main.go")
		git("commit", "-q", "-m", "probe")
		rev := git("rev-parse", "HEAD")

		build := func(name string) (stamp.Stamp, stamp.Identity) {
			t.Helper()
			bin := filepath.Join(outDir, name)
			if runtime.GOOS == "windows" {
				bin += ".exe"
			}
			cmd := exec.CommandContext(ctx, goBin, "build", "-buildvcs=true", "-o", bin, ".")
			cmd.Dir, cmd.Env = repo, env
			if out, err := cmd.CombinedOutput(); err != nil {
				t.Fatalf("go build %s: %v\n%s", name, err, out)
			}
			fromFile, err := stamp.FromFile(bin)
			if err != nil {
				t.Fatalf("FromFile(%s) error = %v", name, err)
			}
			run := exec.CommandContext(ctx, bin)
			run.Env = env
			out, err := run.Output()
			if err != nil {
				t.Fatalf("run %s: %v\n%s", name, err, out)
			}
			id, err := stamp.ParseVersionJSON(out)
			if err != nil {
				t.Fatalf("ParseVersionJSON(%s output %q) error = %v", name, out, err)
			}
			if self := id.Stamp(); fromFile != self {
				t.Fatalf("%s: FromFile = %+v, but the binary's Self() reports %+v", name, fromFile, self)
			}
			return fromFile, id
		}

		clean, _ := build("probe-clean")
		if clean != (stamp.Stamp{Revision: rev, HasVCS: true}) {
			t.Fatalf("clean build stamp = %+v, want revision %s, clean", clean, rev)
		}
		if f, cause := stamp.Explain(clean, rev); f != stamp.Fresh || cause != stamp.CauseMatched {
			t.Fatalf("Explain(clean, HEAD) = (%v, %v), want (fresh, matched)", f, cause)
		}
		if a := stamp.AssessSkew(ctx, nil, repo, "HEAD", clean); a.Verdict != stamp.VerdictFresh || a.TrunkTip != rev {
			t.Fatalf("AssessSkew(clean, HEAD) = %+v, want FRESH at %s", a, rev)
		}

		if err := os.WriteFile(filepath.Join(repo, "uncommitted.txt"), []byte("dirty\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		dirty, dirtyID := build("probe-dirty")
		if dirty != (stamp.Stamp{Revision: rev, Dirty: true, HasVCS: true}) || !dirtyID.Dirty {
			t.Fatalf("dirty build stamp = %+v (identity %+v), want revision %s, dirty", dirty, dirtyID, rev)
		}
		if f, cause := stamp.Explain(dirty, rev); f != stamp.Unknown || cause != stamp.CauseDirty {
			t.Fatalf("Explain(dirty, HEAD) = (%v, %v), want (unknown, dirty)", f, cause)
		}
		if a := stamp.AssessSkew(ctx, nil, repo, "HEAD", dirty); a.Verdict != stamp.VerdictDirty || !a.Verdict.Refusable() {
			t.Fatalf("AssessSkew(dirty, HEAD) = %+v, want refusable DIRTY", a)
		}

		git("commit", "--allow-empty", "-q", "-m", "advance trunk")
		tip := git("rev-parse", "HEAD")
		if f, cause := stamp.Explain(clean, tip); f != stamp.Stale || cause != stamp.CauseDiverged {
			t.Fatalf("Explain(clean, new HEAD) = (%v, %v), want (stale, diverged)", f, cause)
		}
		if a := stamp.AssessSkew(ctx, nil, repo, "HEAD", clean); a.Verdict != stamp.VerdictSkewed || a.Relation != stamp.RelBehind || !a.Verdict.Refusable() {
			t.Fatalf("AssessSkew(clean, new HEAD) = %+v, want refusable SKEWED/behind", a)
		}
	})
}

// writeProbeModule writes a main module that depends on this checkout of fak and prints its
// own Self() stamp in the `version --json` shape.
func writeProbeModule(t *testing.T, dir string) {
	t.Helper()
	root, err := filepath.Abs(filepath.Join("..", "..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(root, "go.mod")); err != nil {
		t.Fatalf("module root %s: %v", root, err)
	}
	goMod := fmt.Sprintf("module example.invalid/stampprobe\n\ngo 1.26.0\n\nrequire github.com/anthony-chaudhary/fak v0.0.0\n\nreplace github.com/anthony-chaudhary/fak => %q\n", filepath.ToSlash(root))
	mainGo := `package main

import (
	"encoding/json"
	"os"

	"github.com/anthony-chaudhary/fak/pkg/deploykit/stamp"
)

func main() {
	s := stamp.Self()
	_ = json.NewEncoder(os.Stdout).Encode(map[string]any{
		"app_version": stamp.AppVersion(),
		"commit":      s.Revision,
		"dirty":       s.Dirty,
		"stamped":     s.HasVCS,
	})
}
`
	for name, body := range map[string]string{"go.mod": goMod, "main.go": mainGo} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

// hermeticEnv strips ambient git and module-mode settings (a hook's GIT_DIR, a go.work, a
// GOFLAGS -buildvcs=false) that would retarget the probe repository or its build.
func hermeticEnv() []string {
	var env []string
	for _, kv := range os.Environ() {
		key, _, _ := strings.Cut(kv, "=")
		switch upper := strings.ToUpper(key); {
		case strings.HasPrefix(upper, "GIT_"), upper == "GOWORK", upper == "GOFLAGS", upper == "GOPROXY", upper == "GOTOOLCHAIN":
			continue
		}
		env = append(env, kv)
	}
	return append(env, "GOWORK=off", "GOFLAGS=", "GOPROXY=off", "GOTOOLCHAIN=local")
}
