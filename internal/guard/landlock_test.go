package guard

import (
	"path/filepath"
	"testing"
)

func TestOptedIn(t *testing.T) {
	for _, tc := range []struct {
		val  string
		want bool
	}{
		{"", false}, {"0", false}, {"off", false}, {"no", false},
		{"1", true}, {"on", true}, {"true", true}, {"YES", true}, {" On ", true},
	} {
		got := OptedIn(func(string) string { return tc.val })
		if got != tc.want {
			t.Fatalf("OptedIn(%q)=%v, want %v", tc.val, got, tc.want)
		}
	}
}

func TestSpecEncodeDecodeRoundTrip(t *testing.T) {
	in := RulesetSpec{
		RepoRoot:     "/home/u/work/fak",
		GitDir:       "/home/u/work/fak/.git",
		ReadOnlyDirs: []string{"/home/u/work/fak/.git/hooks", "/etc/git hooks/with space"},
	}
	tok := in.Encode()
	out, err := DecodeSpec(tok)
	if err != nil {
		t.Fatalf("DecodeSpec: %v", err)
	}
	if out.RepoRoot != in.RepoRoot || out.GitDir != in.GitDir || len(out.ReadOnlyDirs) != len(in.ReadOnlyDirs) {
		t.Fatalf("round-trip mismatch: %+v vs %+v", out, in)
	}
	for i := range in.ReadOnlyDirs {
		if out.ReadOnlyDirs[i] != in.ReadOnlyDirs[i] {
			t.Fatalf("ReadOnlyDirs[%d]=%q, want %q", i, out.ReadOnlyDirs[i], in.ReadOnlyDirs[i])
		}
	}
}

func TestDecodeSpecMalformed(t *testing.T) {
	if _, err := DecodeSpec("!!!not base64!!!"); err == nil {
		t.Fatal("DecodeSpec of garbage must error (the trampoline turns this into fail-open)")
	}
}

func TestTrampolineArgvAndSplit(t *testing.T) {
	spec := RulesetSpec{GitDir: "/r/.git", ReadOnlyDirs: []string{"/r/.git/hooks"}}
	agent := []string{"claude", "--flag", "value with space"}
	argv := TrampolineArgv("/usr/bin/fak", spec, agent)

	if argv[0] != "/usr/bin/fak" || argv[1] != TrampolineVerb {
		t.Fatalf("argv prefix = %v, want [fak %s ...]", argv[:2], TrampolineVerb)
	}
	// The 4-token header is [fak, verb, spec, --]; the rest is the agent argv verbatim.
	if argv[3] != "--" {
		t.Fatalf("argv[3]=%q, want the -- separator", argv[3])
	}

	// SplitTrampolineArgs receives everything AFTER the verb (os.Args[2:] in the dispatcher).
	specTok, gotAgent, ok := SplitTrampolineArgs(argv[2:])
	if !ok {
		t.Fatal("SplitTrampolineArgs returned ok=false on a well-formed argv")
	}
	if specTok != spec.Encode() {
		t.Fatalf("recovered spec token mismatch")
	}
	if len(gotAgent) != len(agent) {
		t.Fatalf("recovered agent argv = %v, want %v", gotAgent, agent)
	}
	for i := range agent {
		if gotAgent[i] != agent[i] {
			t.Fatalf("agent argv[%d]=%q, want %q", i, gotAgent[i], agent[i])
		}
	}
}

func TestSplitTrampolineArgsRejectsMalformed(t *testing.T) {
	cases := [][]string{
		nil,                   // empty
		{"spectok"},           // no -- separator
		{"spectok", "--"},     // separator but no agent argv after it
		{"spectok", "x", "y"}, // no separator at all
	}
	for _, c := range cases {
		if _, _, ok := SplitTrampolineArgs(c); ok {
			t.Fatalf("SplitTrampolineArgs(%v) should be ok=false", c)
		}
	}
}

func TestResolveSpec(t *testing.T) {
	t.Run("default hooks under git dir", func(t *testing.T) {
		root := abs("/home/u/repo")
		gitDir := abs("/home/u/repo/.git")
		// git rev-parse --git-path hooks typically prints ".git/hooks" (relative to root).
		spec := ResolveSpec(root, gitDir, filepath.Join(".git", "hooks"), false)
		wantHook := filepath.Clean(filepath.Join(gitDir, "hooks"))
		if !containsPath(spec.ReadOnlyDirs, wantHook) {
			t.Fatalf("ReadOnlyDirs=%v, want it to include %q", spec.ReadOnlyDirs, wantHook)
		}
	})

	t.Run("absolute external core.hooksPath honored", func(t *testing.T) {
		root := abs("/home/u/repo")
		gitDir := abs("/home/u/repo/.git")
		external := abs("/etc/global-git-hooks")
		spec := ResolveSpec(root, gitDir, external, false)
		if !containsPath(spec.ReadOnlyDirs, filepath.Clean(external)) {
			t.Fatalf("an external core.hooksPath must be protected too; got %v", spec.ReadOnlyDirs)
		}
		// The canonical .git/hooks is still protected alongside it.
		if !containsPath(spec.ReadOnlyDirs, filepath.Clean(filepath.Join(gitDir, "hooks"))) {
			t.Fatalf("canonical .git/hooks must still be protected; got %v", spec.ReadOnlyDirs)
		}
	})

	t.Run("empty git dir yields empty spec (fail open)", func(t *testing.T) {
		spec := ResolveSpec("", "", "", false)
		if len(spec.ReadOnlyDirs) != 0 {
			t.Fatalf("a blank git dir must yield no protected dirs (fail open); got %v", spec.ReadOnlyDirs)
		}
	})

	t.Run("bare repo anchors relative hooks at git dir", func(t *testing.T) {
		gitDir := abs("/srv/repo.git")
		spec := ResolveSpec("", gitDir, "hooks", true)
		want := filepath.Clean(filepath.Join(gitDir, "hooks"))
		if !containsPath(spec.ReadOnlyDirs, want) {
			t.Fatalf("bare repo: ReadOnlyDirs=%v, want %q", spec.ReadOnlyDirs, want)
		}
	})
}

func TestDecideFailOpen(t *testing.T) {
	cases := []struct {
		name    string
		version int
		errno   int
		apply   bool
	}{
		{"supported v1", 1, 0, true},
		{"supported v5", 5, 0, true},
		{"ENOSYS", -1, errnoENOSYS, false},
		{"EOPNOTSUPP", -1, errnoEOPNOTSUPP, false},
		{"other errno", -1, 13, false},
		{"version 0", 0, 0, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			d := DecideFailOpen(tc.version, tc.errno)
			if d.Apply != tc.apply {
				t.Fatalf("DecideFailOpen(%d,%d).Apply=%v, want %v", tc.version, tc.errno, d.Apply, tc.apply)
			}
			if !d.Apply && d.Log == "" {
				t.Fatalf("a fail-open decision must carry a log line; got empty for %+v", tc)
			}
		})
	}
}

// abs makes a platform-correct absolute test path so the table runs identically on Windows
// (where filepath uses backslashes and a volume) and Linux.
func abs(unixPath string) string {
	p := filepath.FromSlash(unixPath)
	a, err := filepath.Abs(p)
	if err != nil {
		return p
	}
	return a
}

func containsPath(dirs []string, want string) bool {
	for _, d := range dirs {
		if filepath.Clean(d) == filepath.Clean(want) {
			return true
		}
	}
	return false
}
