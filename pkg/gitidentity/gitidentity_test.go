package gitidentity

import "testing"

// TestFixtureIdentityClassify pins the refusal set against the fixture
// identities witnessed on trunk (git log origin/main --since=2026-09-01) and
// the real identities that must keep passing.
func TestFixtureIdentityClassify(t *testing.T) {
	cases := []struct {
		name, email string
		want        Reason
	}{
		// Witnessed fixture authors.
		{"fak-test", "test@fak.local", ReasonFixtureName},
		{"test", "test@example.invalid", ReasonFixtureName},
		{"DoctorTest", "doctor@fak.local", ReasonFixtureName},
		{"orphan-test", "orphan@fak.local", ReasonFixtureName},
		{"Witness", "witness@fak.local", ReasonFixtureName},
		{"t", "t@t", ReasonFixtureName},
		{"a", "a@b.c", ReasonFixtureName},
		{"fak-flow-tester", "flowtest@fak.local", ReasonFixtureName},
		{"  FAK-TEST ", "real@users.noreply.github.com", ReasonFixtureName},
		{"user.email", "ops.agent@eng-corp.ai", ReasonConfigKeyName},
		// Real-looking names with fixture addresses.
		{"audit", "audit@example.invalid", ReasonReservedDomain},
		{"Probe", "probe@fak.local", ReasonReservedDomain},
		{"Doc Writer", "docs@example.com", ReasonReservedDomain},
		{"Doc Writer", "docs@mail.example.org", ReasonReservedDomain},
		{"Runner", "ci@runner.test", ReasonReservedDomain},
		{"Runner", "ci@localhost", ReasonReservedDomain},
		{"Someone", "x@t", ReasonUnroutableDomain},
		{"Someone", "x@b.c", ReasonUnroutableDomain},
		{"Someone", "T@T", ReasonFixtureEmail},
	}
	for _, c := range cases {
		v := Classify(c.name, c.email)
		if !v.Fixture || v.Reason != c.want {
			t.Errorf("Classify(%q, %q) = %+v; want fixture with %s", c.name, c.email, v, c.want)
		}
		if v.String() == "ok" {
			t.Errorf("Classify(%q, %q).String() = ok for a fixture verdict", c.name, c.email)
		}
	}
}

func TestFixtureIdentityPassesRealIdentities(t *testing.T) {
	for _, c := range [][2]string{
		{"Claude", "12345678+fixture-user@users.noreply.github.com"},
		{"Fak Agent 3", "fak300000@users.noreply.github.com"},
		{"Jordan Fixtures", "j.fixtures@eng-corp.ai"},
		{"Testa Rossi", "testa@rossi.it"},         // "test" only as a prefix
		{"Ada", "ada@example-corp.com"},           // not example.com
		{"", ""},                                  // missing: git's own fail-closed case
		{"Claude", ""},                            // missing email
		{"Claude", "not-an-address"},              // no @: not this guard's call
		{"Claude", "claude@anthropic.com."},       // trailing root dot
		{"Witnessed Person", "w@witness.dev"},     // fixture name only as a word
		{"Tester McTestface", "tm@company.io"},    // not an exact fixture name
		{"contest", "contest@contest.example.co"}, // .co is a real TLD
	} {
		if v := Classify(c[0], c[1]); v.Fixture {
			t.Errorf("Classify(%q, %q) = %+v; want pass", c[0], c[1], v)
		}
	}
}

func TestFixtureIdentityParseIdent(t *testing.T) {
	name, email, ok := ParseIdent("fak-test <test@fak.local> 1727400000 -0700")
	if !ok || name != "fak-test" || email != "test@fak.local" {
		t.Fatalf("ParseIdent = %q %q %v", name, email, ok)
	}
	if _, _, ok := ParseIdent("no email here"); ok {
		t.Fatal("ParseIdent accepted a line without <email>")
	}
	if name, email, ok := ParseIdent("Fak Agent 3 <fak300000@users.noreply.github.com> 1 +0000\n"); !ok || name != "Fak Agent 3" || email != "fak300000@users.noreply.github.com" {
		t.Fatalf("ParseIdent multiword = %q %q %v", name, email, ok)
	}
}

func TestFixtureIdentityIsForgeRemote(t *testing.T) {
	for url, want := range map[string]bool{
		"https://github.com/anthony-chaudhary/fak-private.git": true,
		"git@github.com:anthony-chaudhary/fak-private.git":     true,
		"ssh://git@github.com/anthony-chaudhary/fak.git":       true,
		"git://example.org/repo.git":                           true,
		"github.com:owner/repo":                                true,
		"file:///C:/tmp/bare.git":                              false,
		`C:\Users\x\AppData\Local\Temp\bare.git`:               false,
		"C:/Users/x/AppData/Local/Temp/bare.git":               false,
		"/tmp/remote.git":                                      false,
		"../remote.git":                                        false,
		"origin":                                               false,
		"":                                                     false,
		"https://":                                             false,
	} {
		if got := IsForgeRemote(url); got != want {
			t.Errorf("IsForgeRemote(%q) = %v; want %v", url, got, want)
		}
	}
}
