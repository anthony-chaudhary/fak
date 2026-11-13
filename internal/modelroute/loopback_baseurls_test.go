package modelroute

import (
	"reflect"
	"testing"
)

// TestLoopbackBaseURLs pins the exported seam the serve startup check uses: it
// returns exactly the loopback accounts' EFFECTIVE base URLs (account base_url,
// else the kind default), and excludes remote accounts so a mixed local+remote
// roster stays legal.
func TestLoopbackBaseURLs(t *testing.T) {
	r := Roster{
		Version: RosterVersion,
		Accounts: []Account{
			{ID: "local", Kind: KindLocal, BaseURL: "http://127.0.0.1:8010/v1"},
			{ID: "openai", Kind: KindOpenAI, BaseURL: "https://api.openai.com/v1", CredEnv: "OPENAI_API_KEY"},
			{ID: "anthropic", Kind: KindAnthropic, CredEnv: "ANTHROPIC_API_KEY"},
		},
	}
	got := r.LoopbackBaseURLs()
	want := []string{"http://127.0.0.1:8010/v1"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("LoopbackBaseURLs() = %v, want %v", got, want)
	}

	// No loopback account: an empty roster of remote accounts returns nothing.
	remoteOnly := Roster{
		Version: RosterVersion,
		Accounts: []Account{
			{ID: "openai", Kind: KindOpenAI, BaseURL: "https://api.openai.com/v1", CredEnv: "OPENAI_API_KEY"},
		},
	}
	if got := remoteOnly.LoopbackBaseURLs(); len(got) != 0 {
		t.Fatalf("remote-only roster returned loopback URLs: %v", got)
	}

	// Two loopback accounts (two replicas on one box) both surface.
	twoLocal := Roster{
		Version: RosterVersion,
		Accounts: []Account{
			{ID: "replica1", Kind: KindLocal, BaseURL: "http://127.0.0.1:8000/v1"},
			{ID: "replica2", Kind: KindLocal, BaseURL: "http://127.0.0.1:8010/v1"},
		},
	}
	got = twoLocal.LoopbackBaseURLs()
	want = []string{"http://127.0.0.1:8000/v1", "http://127.0.0.1:8010/v1"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("two-replica roster LoopbackBaseURLs() = %v, want %v", got, want)
	}
}
