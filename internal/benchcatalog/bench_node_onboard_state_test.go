package benchcatalog

import "testing"

// TestClassifyOnboardState pins the ssh-probe (exit, stderr) -> OnboardState
// mapping so the classifier cannot silently regress. It is the lock the bash
// mirror in tools/bench_node.sh points at.
func TestClassifyOnboardState(t *testing.T) {
	cases := []struct {
		name   string
		exit   int
		stderr string
		want   OnboardState
	}{
		{
			name:   "authorized: exit 0, no stderr",
			exit:   0,
			stderr: "",
			want:   OnboardAuthorized,
		},
		{
			name:   "authorized: exit 0 with incidental stderr",
			exit:   0,
			stderr: "Warning: Permanently added 'host' to the list of known hosts.",
			want:   OnboardAuthorized,
		},
		{
			name:   "sshd-only: exact OpenSSH publickey,password,kbd-interactive string",
			exit:   255,
			stderr: "user@host: Permission denied (publickey,password,keyboard-interactive).",
			want:   OnboardSSHDOnly,
		},
		{
			name:   "sshd-only: publickey-only denial",
			exit:   255,
			stderr: "user@host: Permission denied (publickey).",
			want:   OnboardSSHDOnly,
		},
		{
			name:   "sshd-only: too many authentication failures still means sshd answered",
			exit:   255,
			stderr: "Received disconnect from 198.51.100.20 port 22:2: Too many authentication failures",
			want:   OnboardSSHDOnly,
		},
		{
			name:   "sshd-only wins over a connect notice when both appear",
			exit:   255,
			stderr: "ssh: connect to host 198.51.100.20 port 22: Connection timed out\r\nPermission denied (publickey).",
			want:   OnboardSSHDOnly,
		},
		{
			name:   "no-sshd: connection refused",
			exit:   255,
			stderr: "ssh: connect to host 198.51.100.20 port 22: Connection refused",
			want:   OnboardNoSSHD,
		},
		{
			name:   "no-sshd: connection timed out",
			exit:   255,
			stderr: "ssh: connect to host 198.51.100.20 port 22: Connection timed out",
			want:   OnboardNoSSHD,
		},
		{
			name:   "no-sshd: banner never arrived",
			exit:   255,
			stderr: "kex_exchange_identification: Connection closed by remote host",
			want:   OnboardNoSSHD,
		},
		{
			name:   "no-sshd: no route to host",
			exit:   255,
			stderr: "ssh: connect to host 198.51.100.20 port 22: No route to host",
			want:   OnboardNoSSHD,
		},
		{
			name:   "unknown: bare permission denied is NOT an auth denial",
			exit:   1,
			stderr: "bash: /root/secret: Permission denied",
			want:   OnboardUnknown,
		},
		{
			name:   "unknown: host key changed",
			exit:   255,
			stderr: "WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!",
			want:   OnboardUnknown,
		},
		{
			name:   "unknown: nonzero exit with unclassifiable stderr",
			exit:   255,
			stderr: "some future ssh failure mode",
			want:   OnboardUnknown,
		},
		{
			name:   "unknown: exit 0 is authoritative only when it is 0",
			exit:   1,
			stderr: "",
			want:   OnboardUnknown,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ClassifyOnboardState(tc.exit, tc.stderr)
			if got != tc.want {
				t.Fatalf("ClassifyOnboardState(%d, %q) = %q, want %q", tc.exit, tc.stderr, got, tc.want)
			}
		})
	}
}

// TestOnboardStatesAreExactlyFour locks the closed vocabulary: no fifth state may
// be introduced silently, and every state carries a remedy.
func TestOnboardStatesAreExactlyFour(t *testing.T) {
	all := []OnboardState{OnboardNoSSHD, OnboardSSHDOnly, OnboardAuthorized, OnboardUnknown}
	if len(all) != 4 {
		t.Fatalf("OnboardState vocabulary drifted: %d states", len(all))
	}
	seen := map[OnboardState]bool{}
	for _, s := range all {
		if s == "" {
			t.Fatal("empty OnboardState token")
		}
		if seen[s] {
			t.Fatalf("duplicate OnboardState %q", s)
		}
		seen[s] = true
		if s.Remedy() == "" {
			t.Fatalf("state %q has no remedy line", s)
		}
	}
}

// TestOnboardStateLine pins the machine-greppable line format that callers rely
// on (`grep '^ONBOARD-STATE'`).
func TestOnboardStateLine(t *testing.T) {
	got := OnboardStateLine("desktop", OnboardSSHDOnly)
	if got != "ONBOARD-STATE desktop sshd-only" {
		t.Fatalf("OnboardStateLine = %q", got)
	}
}
