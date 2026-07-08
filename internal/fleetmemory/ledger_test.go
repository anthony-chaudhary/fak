package fleetmemory

import "testing"

func sampleLedger() *Ledger {
	return New([]Lesson{
		{
			ID:      "bash-git-hang",
			Fact:    "Bash git hangs on this host — use PowerShell",
			Trigger: Trigger{Host: "fleet-win", Tool: "Bash"},
			Witness: "memory:bash_git_gh_hang_use_powershell",
		},
		{
			ID:      "wsl-go-test",
			Fact:    "native go test is OS-blocked — route through WSL",
			Trigger: Trigger{PathGlobs: []string{"internal/*"}},
			Witness: "memory:wsl_go_test_capture_technique",
		},
		{
			ID:      "off-trunk",
			Fact:    "never open a feature branch — the guard refuses off-trunk commits",
			Trigger: Trigger{RefusalToken: "OFF_TRUNK"},
		},
	})
}

func TestFactKeyNormalizes(t *testing.T) {
	a := factKey("Bash git hangs on this host — use PowerShell")
	b := factKey("  bash GIT hangs on this host, use powershell!!  ")
	if a != b {
		t.Fatalf("expected equivalent facts to share a key:\n a=%q\n b=%q", a, b)
	}
	if factKey("   ") != "" {
		t.Fatalf("whitespace-only fact must have empty key")
	}
}

func TestMatchFindsCanonicalEntry(t *testing.T) {
	l := sampleLedger()
	// A casing/punctuation variant of an existing fact must match.
	got, ok := l.Match("BASH git hangs on this host, use PowerShell.")
	if !ok {
		t.Fatalf("expected a match for an equivalent fact")
	}
	if got.ID != "bash-git-hang" {
		t.Fatalf("matched the wrong lesson: %q", got.ID)
	}
	if _, ok := l.Match("something genuinely new and unseen"); ok {
		t.Fatalf("did not expect a match for a novel fact")
	}
}

func TestInjectByTriggerHostAndTool(t *testing.T) {
	l := sampleLedger()
	// A session on fleet-win running Bash gets the git-hang lesson.
	got := l.Inject(SessionContext{Host: "fleet-win", Tool: "Bash"})
	if !containsID(got, "bash-git-hang") {
		t.Fatalf("expected bash-git-hang injected for fleet-win/Bash; got %v", ids(got))
	}
	// A session on a DIFFERENT host must NOT get the host-scoped lesson.
	got = l.Inject(SessionContext{Host: "other-host", Tool: "Bash"})
	if containsID(got, "bash-git-hang") {
		t.Fatalf("host-scoped lesson leaked to the wrong host: %v", ids(got))
	}
}

func TestInjectByPathGlob(t *testing.T) {
	l := sampleLedger()
	got := l.Inject(SessionContext{Paths: []string{"internal/gateway/gateway.go"}})
	if !containsID(got, "wsl-go-test") {
		t.Fatalf("expected wsl-go-test injected for an internal/ path; got %v", ids(got))
	}
	got = l.Inject(SessionContext{Paths: []string{"docs/readme.md"}})
	if containsID(got, "wsl-go-test") {
		t.Fatalf("path-scoped lesson fired on a non-matching path: %v", ids(got))
	}
}

func TestInjectByRefusalToken(t *testing.T) {
	l := sampleLedger()
	got := l.Inject(SessionContext{RefusalToken: "OFF_TRUNK"})
	if !containsID(got, "off-trunk") {
		t.Fatalf("expected off-trunk injected for the OFF_TRUNK refusal; got %v", ids(got))
	}
	if containsID(l.Inject(SessionContext{RefusalToken: "STALE_CRED"}), "off-trunk") {
		t.Fatalf("refusal-scoped lesson fired on the wrong token")
	}
}

func TestUniversalTriggerAlwaysInjects(t *testing.T) {
	l := New([]Lesson{{ID: "always", Fact: "commit by explicit path", Trigger: Trigger{}}})
	if !containsID(l.Inject(SessionContext{Host: "anything"}), "always") {
		t.Fatalf("an all-empty trigger must inject into every session")
	}
}

func containsID(ls []Lesson, id string) bool {
	for _, l := range ls {
		if l.ID == id {
			return true
		}
	}
	return false
}

func ids(ls []Lesson) []string {
	out := make([]string, 0, len(ls))
	for _, l := range ls {
		out = append(out, l.ID)
	}
	return out
}
