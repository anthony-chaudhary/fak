package adjudicator

import "testing"

// TestEnvReadIsNotSelfModify pins the dotfile-boundary fix witnessed live on
// 2026-07-01: the inline-eval floor treats a `python -c` command as write-shaped
// and hands the WHOLE opaque segment to matchGlob as the write target, so the
// bare-substring ".env" glob matched the ".env" inside "os.environ" and denied a
// harmless environment READ as SELF_MODIFY (witness ".env"). A dotted name whose
// '.' merely punctuates an identifier is not a dotfile.
func TestEnvReadIsNotSelfModify(t *testing.T) {
	allowed := []string{
		// The exact live-denied shape: an env READ inside an inline program.
		`python -c "import os; print(len(os.environ))"`,
		`python -c "import os; print(os.environ.get('CI', ''))"`,
		// The same dotted-name trap via a pipe summarizer over scorecard JSON.
		`python tools/code_slop_scorecard.py --json | python -c "import json,sys,os; d=json.load(sys.stdin); print(d['corpus']['slop_debt'], os.environ.get('USERNAME'))"`,
		// node's spelling of the same read.
		`node -e "console.log(process.env.HOME)" > out.txt`,
		// A dotted FILE name is not the dotfile: config.env / prod.aws are words.
		`echo x > config.env`,
		`tee logs.aws/report.txt < in.txt`,
	}
	for _, c := range allowed {
		if g := commandSelfModify(map[string]any{"command": c}, guardCredentialGlobs); g != "" {
			t.Errorf("env read / dotted name wrongly refused SELF_MODIFY (glob %q):\n  %s", g, c)
		}
	}
}

// TestDotfileWriteStillDenied is the floor half the boundary fix must preserve:
// every genuine dotfile write form follows a separator (redirect, quote, '/',
// '=', start of token), so it still denies.
func TestDotfileWriteStillDenied(t *testing.T) {
	denied := []string{
		"echo AWS_KEY=x > .env",
		"echo AWS_KEY=x >> src/.env",
		"cp stolen src/.env",
		"tee ~/.aws/credentials < attacker",
		// The inline-eval write floor: the segment carries a QUOTED dotfile.
		`python -c "open('.env','w').write('pwn')"`,
		// A guarded-tree write via the .git/ glob after a plain space.
		"cp payload .git/hooks/pre-commit",
	}
	for _, c := range denied {
		if g := commandSelfModify(map[string]any{"command": c}, guardCredentialGlobs); g == "" {
			t.Errorf("a real dotfile/guarded write was NOT refused (should deny SELF_MODIFY):\n  %s", c)
		}
	}
}

// TestMatchGlobDotfileBoundary pins the matcher semantics directly: dot-prefixed
// fragments need a token boundary; non-dot fragments keep bare substring match.
func TestMatchGlobDotfileBoundary(t *testing.T) {
	globs := []string{".env", ".aws/", "id_rsa", "internal/abi/"}
	cases := []struct {
		path string
		want string
	}{
		{"os.environ", ""},                                 // dotted name, not a dotfile
		{"config.env", ""},                                 // suffixed word, not a dotfile
		{".env", ".env"},                                   // the dotfile itself, start of string
		{"src/.env", ".env"},                               // path segment
		{"--env-file=.env", ".env"},                        // '=' boundary
		{"'.env'", ".env"},                                 // quote boundary
		{"~/.aws/credentials", ".aws/"},                    // '/' boundary
		{"backup id_rsa now", "id_rsa"},                    // non-dot glob: substring as before
		{"david_rsa", "id_rsa"},                            // non-dot glob keeps its (coarse) substring reach
		{"x internal/abi/kernel.go", "internal/abi/"},      // non-dot glob unchanged
		{"shutil.rmtree('internal/abi')", "internal/abi/"}, // tree glob catches exact dir too
	}
	for _, tc := range cases {
		if got := matchGlob(tc.path, globs); got != tc.want {
			t.Errorf("matchGlob(%q) = %q, want %q", tc.path, got, tc.want)
		}
	}
}
