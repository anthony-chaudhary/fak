package adjudicator

import "testing"

// guardCredentialGlobs mirrors the credential entries of the shipped guard floor
// (cmd/fak/guard-default-policy.json self_modify_globs) so these tests exercise the real
// false-positive #1086 hit: an ssh/scp identity-file READ named alongside a write verb.
var guardCredentialGlobs = []string{".git/", ".env", ".aws/", ".npmrc", ".netrc", "id_rsa", "id_ed25519", "/etc/"}

// TestSSHIdentityReadIsNotSelfModify: a remote build+bench over `ssh -i <key>` (#1086) is
// allowed — the credential glob matched only the identity file the command READS for auth,
// not a write target. Covers both key types and the scp `cp ` substring trap.
func TestSSHIdentityReadIsNotSelfModify(t *testing.T) {
	allowed := []string{
		// ssh remote build with a write verb in the REMOTE command (cp/mv) + an -i key.
		"ssh -i ~/.ssh/id_rsa anthony@node 'cd ~/fak && cp build/out /tmp/x && go test -tags fakmetal ./internal/metalgemm'",
		"ssh -i ~/.ssh/id_ed25519 anthony@node 'go build -tags fakmetal ./... && mv a b'",
		// scp itself: `scp ` contains the `cp ` write verb, and the key is the -i operand.
		"scp -i ~/.ssh/id_ed25519 ./fak anthony@node:~/",
		"scp -i ~/.ssh/id_rsa anthony@node:~/result.json ./",
		// glued -i<path> spelling.
		"ssh -i~/.ssh/id_rsa anthony@node 'cp a b'",
	}
	for _, c := range allowed {
		if g := commandSelfModify(map[string]any{"command": c}, guardCredentialGlobs); g != "" {
			t.Errorf("ssh/scp identity read wrongly refused SELF_MODIFY (glob %q):\n  %s", g, c)
		}
	}
}

// TestRealKeyWriteStillDenied: the carve-out must NOT open a hole — a genuine WRITE that
// names a credential glob after the identity arg is stripped (or with no -i at all) still
// denies. This is the security floor the narrow #1086 fix must preserve.
func TestRealKeyWriteStillDenied(t *testing.T) {
	denied := []string{
		// A redirect/tee that WRITES a key — a real self-modify of a credential, no -i to strip.
		"tee ~/.ssh/id_rsa < attacker_key",
		"cp attacker_key ~/.ssh/id_ed25519",
		"echo pwn > ~/.ssh/id_rsa",
		// An ssh command that ALSO writes a guarded key in the REMOTE command — the glob
		// survives stripping the -i identity arg, so it stays denied.
		"ssh -i ~/.ssh/id_rsa anthony@node 'cp evil ~/.ssh/id_ed25519'",
		// Writing into another guarded tree over scp's remote path is still a write of a
		// guarded glob that is not the -i operand.
		"cp payload .git/hooks/pre-commit",
	}
	for _, c := range denied {
		if g := commandSelfModify(map[string]any{"command": c}, guardCredentialGlobs); g == "" {
			t.Errorf("a real credential/guarded-tree write was NOT refused (should deny SELF_MODIFY):\n  %s", c)
		}
	}
}

// TestStripSSHIdentityArg pins the helper: it strips `-i <key>` (spaced and glued) only for
// ssh/scp/sftp leading tokens, and leaves a non-ssh `-i` (or no -i) untouched.
func TestStripSSHIdentityArg(t *testing.T) {
	cases := []struct {
		in      string
		want    string
		wantDid bool
	}{
		{"ssh -i ~/.ssh/id_rsa host 'cmd'", "ssh host 'cmd'", true},
		{"scp -i key.pem f host:~/", "scp f host:~/", true},
		{"ssh -ikey host", "ssh host", true},
		{"ssh host 'go build'", "ssh host 'go build'", false}, // no -i
		// A non-ssh program with a -i flag must NOT be touched by this carve-out (it is
		// not an identity-file read; e.g. `sed -i` is a real in-place write).
		{"sed -i s/a/b/ id_rsa", "sed -i s/a/b/ id_rsa", false},
		{"docker run -i img", "docker run -i img", false},
	}
	for _, tc := range cases {
		got, did := stripSSHIdentityArg(tc.in)
		if did != tc.wantDid || (did && got != tc.want) {
			t.Errorf("stripSSHIdentityArg(%q) = (%q, %v); want (%q, %v)", tc.in, got, did, tc.want, tc.wantDid)
		}
	}
}
