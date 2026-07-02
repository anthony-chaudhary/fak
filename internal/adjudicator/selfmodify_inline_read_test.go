package adjudicator

import "testing"

// TestInlineEvalReadNamingGuardedPathIsNotSelfModify pins the fix witnessed live on
// 2026-07-02: an interpreter inline-eval one-liner that only READS but happens to NAME a
// guarded path was denied SELF_MODIFY. The inline-eval floor treats the whole opaque program
// segment as the write target, and a non-dot credential glob (id_rsa / id_ed25519) matches it
// by bare substring, so a harmless env/key introspection tripped the WRITE floor. SELF_MODIFY
// is a write floor: a program with no file-mutation signal writes nothing.
func TestInlineEvalReadNamingGuardedPathIsNotSelfModify(t *testing.T) {
	allowed := []string{
		// The exact live-denied shapes (witness id_ed25519 / id_rsa, tool PowerShell).
		`python -c "import os; print(os.path.exists(os.path.expanduser('~/.ssh/id_ed25519')))"`,
		`python -c "print(open('log.txt').read().count('id_rsa'))"`,
		// Reading a dotfile's content by name inside an inline program (a read, not a write).
		`python -c "print(open('.env').read())"`,
		`node -e "console.log(require('fs').readFileSync(process.env.HOME + '/.aws/config','utf8'))"`,
		// Pure introspection that mentions a guarded tree by name.
		`python3 -c "import os; print(os.listdir('internal/abi'))"`,
		`ruby -e 'puts File.read("id_rsa.pub")'`,
	}
	for _, c := range allowed {
		if g := commandSelfModify(map[string]any{"command": c}, guardCredentialGlobs); g != "" {
			t.Errorf("read-only inline program naming a guarded path wrongly refused SELF_MODIFY (glob %q):\n  %s", g, c)
		}
		// Also exercise the DefaultPolicy globs (internal/abi/, dos.toml, …) for the tree case.
		if g := commandSelfModify(map[string]any{"command": c}, DefaultPolicy().SelfModifyGlobs); g != "" {
			t.Errorf("read-only inline program naming a guarded tree wrongly refused SELF_MODIFY (glob %q):\n  %s", g, c)
		}
	}
}

// TestInlineEvalWriteStillDenied is the floor half the read carve-out must preserve: an inline
// program that actually MUTATES a guarded path still denies — via a write/append/exclusive
// open, a .write/writeFile/File.write, a delete/rename/mkdir/truncate, or a subprocess that
// writes. This is the #172 Hole 1 residual the inline-eval floor exists to close.
func TestInlineEvalWriteStillDenied(t *testing.T) {
	cred := guardCredentialGlobs
	def := DefaultPolicy().SelfModifyGlobs
	denied := []struct {
		cmd   string
		globs []string
	}{
		// Direct write via open(mode)+.write (both signals present).
		{`python -c "open('.env','w').write('pwn')"`, cred},
		{`python -c "open('/home/me/.ssh/id_ed25519','w').write('pwn')"`, cred},
		{`node -e "require('fs').writeFileSync('dos.toml','x')"`, def},
		{`ruby -e 'File.write("fak/internal/adjudicator/decide.go", x)'`, def},
		// Truncate-on-open with NO .write call — caught by the comma-prefixed mode string alone.
		{`python3 -c "open('fak/internal/kernel/x.go','w')"`, def},
		{`python3 -c "open('.env', 'a')"`, cred},
		// Append to a key file.
		{`python -c "open('/home/me/.ssh/id_rsa','a').write('extra')"`, cred},
		// Deletion / rename of a guarded path.
		{`python -c "import os; os.remove('.env')"`, cred},
		{`python -c "import os; os.rename('a', '.env')"`, cred},
		{`python -c "import shutil; shutil.rmtree('internal/abi')"`, def},
		// Subprocess laundering a write.
		{`python -c "import os; os.system('cat pwn > .env')"`, cred},
		{`python -c "import subprocess; subprocess.run(['rm', '.env'])"`, cred},
	}
	for _, tc := range denied {
		if g := commandSelfModify(map[string]any{"command": tc.cmd}, tc.globs); g == "" {
			t.Errorf("a real inline write/mutate into a guarded path was NOT refused (should deny SELF_MODIFY):\n  %s", tc.cmd)
		}
	}
}
