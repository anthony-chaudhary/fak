package hooks

import "testing"

// gate_e2eovermocks_test.go — focused unit tests for the E2E_OVER_MOCKS advisory gate (#2901).
// Built on the in-package diffOf helper (hooks_test.go), the same fixture style the other gate
// tests use. The contract: touching a security-critical floor/quarantine/adjudicator package
// without a staged end-to-end witness emits one advisory finding per touched package; a staged
// "E2E-verified:" line silences it; non-security paths are quiet.

func TestMatchSecurityPrefix(t *testing.T) {
	cases := []struct {
		path       string
		wantPrefix string
		wantOK     bool
	}{
		{"internal/adjudicator/adjudicator.go", "internal/adjudicator/", true},
		{"internal/egressfloor/egressfloor.go", "internal/egressfloor/", true},
		{"internal/policy/policy.go", "internal/policy/", true},
		{"internal/normgate/normgate.go", "internal/normgate/", true},
		{"internal/gateway/gateway.go", "internal/gateway/", true},
		{"internal/repoguard/repoguard.go", "internal/repoguard/", true},
		{"cmd/fak/main.go", "cmd/fak/", true},
		{"internal/engine/engine.go", "internal/engine/", true},
		{"internal/dogfood/dogfood.go", "internal/dogfood/", true},
		{`internal\adjudicator\secretposture.go`, "internal/adjudicator/", true}, // backslashes normalized
		{"internal/cache/cache.go", "", false},                                   // not in the guarded set
		{"README.md", "", false},                                                 // non-security
		{"internal/adjudicatorx/x.go", "", false},                                // prefix must end at a dir boundary
	}
	for _, c := range cases {
		prefix, ok := matchSecurityPrefix(c.path)
		if ok != c.wantOK || prefix != c.wantPrefix {
			t.Errorf("matchSecurityPrefix(%q) = (%q, %v), want (%q, %v)", c.path, prefix, ok, c.wantPrefix, c.wantOK)
		}
	}
}

// (a) touching a security-critical file with NO staged witness yields exactly one advisory.
func TestE2EOverMocks_securityFileEmitsAdvisory(t *testing.T) {
	d := diffOf("/r", map[string][]string{
		"internal/adjudicator/adjudicator.go": {"package adjudicator", "// tweak a deny rule"},
	})
	f, err := gateE2EOverMocks(d)
	if err != nil {
		t.Fatal(err)
	}
	if len(f) != 1 {
		t.Fatalf("expected exactly one E2E_OVER_MOCKS finding, got %d: %+v", len(f), f)
	}
	if f[0].Gate != "E2E_OVER_MOCKS" {
		t.Errorf("gate name = %q, want E2E_OVER_MOCKS", f[0].Gate)
	}
	if f[0].File != "internal/adjudicator/adjudicator.go" {
		t.Errorf("finding File = %q, want the touched security path", f[0].File)
	}
	if !hasFindingFor(f, "E2E_OVER_MOCKS", "mocks hide integration bugs") {
		t.Errorf("advisory should cite the Hermes rule; got %+v", f)
	}
	if !hasFindingFor(f, "E2E_OVER_MOCKS", "/verify") {
		t.Errorf("advisory should point at the /verify skill; got %+v", f)
	}
}

// (b) touching a non-security file yields zero findings.
func TestE2EOverMocks_nonSecurityFileQuiet(t *testing.T) {
	d := diffOf("/r", map[string][]string{
		"internal/cache/cache.go": {"package cache"},
		"README.md":               {"some prose"},
	})
	f, err := gateE2EOverMocks(d)
	if err != nil {
		t.Fatal(err)
	}
	if len(f) != 0 {
		t.Fatalf("non-security files must yield no E2E_OVER_MOCKS finding, got %+v", f)
	}
}

// (c) a staged "E2E-verified:" or "Shift-left-verified:" line suppresses the gate entirely, even
// when a security-critical file is touched.
func TestE2EOverMocks_witnessTrailerSuppresses(t *testing.T) {
	for _, token := range []string{
		"// E2E-verified: drove `fak preflight` against a temp home; metadata egress DENY witnessed.",
		"// Shift-left-verified: executed dogfood probe against loopback server.",
		"// Smoke-verified: ran `fak validate --smoke` and verified hermetic execution.",
		"// smoke-test: executed CLI preflight smoke on compiled binary.",
		"// Real-world-verified: proved via live agent execution.",
	} {
		d := diffOf("/r", map[string][]string{
			"internal/egressfloor/egressfloor.go": {
				"package egressfloor",
				token,
			},
		})
		f, err := gateE2EOverMocks(d)
		if err != nil {
			t.Fatal(err)
		}
		if len(f) != 0 {
			t.Fatalf("a staged %q line must silence the gate, got %+v", token, f)
		}
	}
}

// (d) touching TWO files of the SAME package yields ONE finding (dedupe by prefix).
func TestE2EOverMocks_dedupeByPackage(t *testing.T) {
	d := diffOf("/r", map[string][]string{
		"internal/policy/policy.go": {"package policy"},
		"internal/policy/load.go":   {"package policy"},
	})
	f, err := gateE2EOverMocks(d)
	if err != nil {
		t.Fatal(err)
	}
	if len(f) != 1 {
		t.Fatalf("two files of one package must dedupe to a single finding, got %d: %+v", len(f), f)
	}
}

// (e) touching TWO distinct security packages yields two findings, sorted by prefix.
func TestE2EOverMocks_twoPackagesTwoFindings(t *testing.T) {
	d := diffOf("/r", map[string][]string{
		"internal/policy/policy.go":           {"package policy"},
		"internal/adjudicator/adjudicator.go": {"package adjudicator"},
	})
	f, err := gateE2EOverMocks(d)
	if err != nil {
		t.Fatal(err)
	}
	if len(f) != 2 {
		t.Fatalf("two distinct security packages must yield two findings, got %d: %+v", len(f), f)
	}
	if f[0].File != "internal/adjudicator/adjudicator.go" {
		t.Errorf("findings should be sorted by prefix; first = %q, want the adjudicator path", f[0].File)
	}
}
