package adjudicator

import "testing"

// TestHasFileWriteRedirect pins the redirect classifier: a `>`/`>>` to a real file
// is a write; an fd-duplication (`2>&1`, `>&2`, `1>&2`) or a redirect to the null
// device (`2>/dev/null`, `>/dev/null`) is NOT — it writes no named file. This is the
// #1569 fix: the prior `strings.Contains(cmd, ">")` counted the ubiquitous
// `… 2>/dev/null` / `… 2>&1` stderr idiom as a write, so any command naming a guarded
// glob that merely carried one was refused SELF_MODIFY.
func TestHasFileWriteRedirect(t *testing.T) {
	writes := []string{
		"echo x > out.txt",
		"echo x >> out.txt",
		"cat a > internal/kernel/x.go",
		"echo pwn > ~/.ssh/id_rsa",
		"foo 2>/dev/null > realfile",  // a real redirect survives alongside a null one
		"echo x >| clobber.txt",       // `>|` clobber-override still writes a file
		"echo x >>/tmp/log",           // glued `>>target`
		"printf y >internal/kernel/k", // glued `>target`
	}
	for _, c := range writes {
		if !hasFileWriteRedirect(c) {
			t.Errorf("hasFileWriteRedirect(%q) = false; want true (real file redirect)", c)
		}
	}
	noWrites := []string{
		"cat internal/kernel/x.go 2>/dev/null",
		"go test ./internal/adjudicator 2>&1",
		"grep -n foo file >&2",
		"ls 1>&2",
		"make ci > /dev/null 2>&1",
		"cat x 2>>/dev/null", // appending stderr to the null device is still null
		"echo hi",            // no redirect at all
		"diff a b",           // no redirect
	}
	for _, c := range noWrites {
		if hasFileWriteRedirect(c) {
			t.Errorf("hasFileWriteRedirect(%q) = true; want false (fd-dup or /dev/null)", c)
		}
	}
}

// TestStderrRedirectOverGuardedTreeIsNotSelfModify is the integration half of #1569:
// a READ of a guarded file whose stderr is suppressed (`2>/dev/null` / `2>&1`) must
// NOT be refused SELF_MODIFY — only a real WRITE into the guarded tree is. This is the
// floor's own documented "a read of a guarded file is allowed; only writes are
// refused" contract, which the `>`-anywhere check violated.
func TestStderrRedirectOverGuardedTreeIsNotSelfModify(t *testing.T) {
	globs := []string{"internal/kernel/", ".git/", "id_rsa"}

	allowed := []string{
		"cat internal/kernel/forward.go 2>/dev/null",
		"head -n 40 internal/kernel/forward.go 2>&1",
		"grep -rn TODO internal/kernel/ > /dev/null 2>&1",
		// the real-world #1569 shape: a remote build over ssh whose only redirect is a
		// stderr-suppress, naming a credential glob only as the -i identity it reads.
		"ssh -i ~/.ssh/id_rsa anthony@node 'go build ./cmd/fak 2>/dev/null'",
	}
	for _, c := range allowed {
		if g := commandSelfModify(map[string]any{"command": c}, globs); g != "" {
			t.Errorf("read-with-redirect over a guarded tree wrongly refused SELF_MODIFY (glob %q):\n  %s", g, c)
		}
	}

	denied := []string{
		"echo x > internal/kernel/x.go",
		"cat payload >> internal/kernel/forward.go",
		"echo pwn > ~/.ssh/id_rsa",
	}
	for _, c := range denied {
		if g := commandSelfModify(map[string]any{"command": c}, globs); g == "" {
			t.Errorf("a real write into a guarded tree was NOT refused (should deny SELF_MODIFY):\n  %s", c)
		}
	}
}

// TestCommandSelfModifyUsesWriteTargetsOnly pins #1917: the shell SELF_MODIFY floor
// matches guarded globs against the paths being written, not every guarded path the
// command happens to read or mention in another command segment.
func TestCommandSelfModifyUsesWriteTargetsOnly(t *testing.T) {
	globs := []string{"VERSION", ".dos/", "internal/abi/", "id_rsa"}

	allowed := []string{
		"cat VERSION > /tmp/v",
		"echo x > /tmp/y ; cat .dos/state",
		"cp VERSION /tmp/v",
		"tee /tmp/out < VERSION",
		"dd if=VERSION of=/tmp/v",
		"install VERSION /tmp/v",
		"ln VERSION /tmp/v",
	}
	for _, c := range allowed {
		if g := commandSelfModify(map[string]any{"command": c}, globs); g != "" {
			t.Errorf("guarded read/source wrongly refused SELF_MODIFY (glob %q):\n  %s", g, c)
		}
	}

	denied := []string{
		"echo x > VERSION",
		"echo x > /tmp/y ; echo z > .dos/state",
		"cp /tmp/v VERSION",
		"mv VERSION /tmp/v",
		"tee .dos/state",
		"dd if=/tmp/v of=internal/abi/x.go",
		"install /tmp/v internal/abi/x.go",
		"ln /tmp/v internal/abi/x.go",
		"rsync -a /tmp/src/ internal/abi/",
	}
	for _, c := range denied {
		if g := commandSelfModify(map[string]any{"command": c}, globs); g == "" {
			t.Errorf("real guarded write was NOT refused SELF_MODIFY:\n  %s", c)
		}
	}
}
