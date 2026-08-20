package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/httptrust"
)

// stubDoctorTrustFacts swaps the fact collector for the duration of a test. The
// verdict itself is tested in internal/httptrust; what these tests hold is the CLI
// contract — exit code, what the human output names, and the JSON schema — against a
// synthetic intercepted host, which is the one thing that cannot be observed from
// whatever machine the suite runs on.
func stubDoctorTrustFacts(t *testing.T, facts httptrust.Facts) {
	t.Helper()
	prev := doctorTrustFactsFn
	doctorTrustFactsFn = func(bool, []string, time.Duration) httptrust.Facts { return facts }
	t.Cleanup(func() { doctorTrustFactsFn = prev })
}

// A clean host exits 0 and says so. `fak doctor trust` composes as a CI gate, so a
// green host must not return a finding exit code.
func TestDoctorTrustExitsZeroAndRaisesNothingOnACleanHost(t *testing.T) {
	stubDoctorTrustFacts(t, httptrust.Facts{
		Probes: []httptrust.ProbeResult{{Host: "api.anthropic.com:443", Reached: true, DefaultOK: true, RootLabel: "DigiCert Global Root CA"}},
	})
	var out, errout bytes.Buffer
	if rc := runDoctorTrust(&out, &errout, []string{"--probe=false"}); rc != 0 {
		t.Fatalf("rc=%d out=%s err=%s", rc, out.String(), errout.String())
	}
	if !strings.Contains(out.String(), "healthy (0 findings)") {
		t.Fatalf("output=%s", out.String())
	}
	if strings.Contains(out.String(), "WARN") {
		t.Fatalf("a clean host produced a WARN row: %s", out.String())
	}
}

// The acceptance from #8172: on an intercepted host the check names the interception,
// the CA subject observed, the trust source fak used, and which child runtimes would
// not inherit it — and exits nonzero so it gates.
func TestDoctorTrustNamesInterceptionTheCASubjectAndTheUninheritedRuntimes(t *testing.T) {
	stubDoctorTrustFacts(t, httptrust.Facts{
		Source: httptrust.Source{Path: `C:\corp\combined-ca-bundle.pem`, Bundle: &httptrust.Bundle{
			Path:           `C:\corp\combined-ca-bundle.pem`,
			Subjects:       []string{"ca.dsa-netskope.goskope.com"},
			EarliestExpiry: time.Date(2035, 1, 1, 0, 0, 0, 0, time.UTC),
		}},
		Probes: []httptrust.ProbeResult{{
			Host: "sts.amazonaws.com:443", Reached: true,
			VerifyErr: "x509: certificate signed by unknown authority",
			RootLabel: "SSI-ISSUINGCA", BundleTried: true,
		}},
		ChildEnv: []string{"PATH=/usr/bin"},
	})
	var out, errout bytes.Buffer
	if rc := runDoctorTrust(&out, &errout, []string{"--probe=false"}); rc != 1 {
		t.Fatalf("rc=%d, want 1 (findings) out=%s err=%s", rc, out.String(), errout.String())
	}
	text := out.String()
	for _, want := range []string{
		`C:\corp\combined-ca-bundle.pem`, // the trust source fak used
		"ca.dsa-netskope.goskope.com",    // the CA subjects it carries
		"SSI-ISSUINGCA",                  // the CA subject observed on the wire
		"sts.amazonaws.com:443",          // where
		"NODE_EXTRA_CA_CERTS",            // a child runtime that would not inherit it
		"interception: WITNESSED",        // the headline
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("output does not name %q:\n%s", want, text)
		}
	}
	if strings.Contains(text, "insecure") || strings.Contains(text, "skip-verify escape") {
		t.Fatalf("the report offered a verification escape:\n%s", text)
	}
}

// The sibling signal needs no network and is the one that fires on a real managed
// host: another runtime already points at a corporate bundle and fak does not.
func TestDoctorTrustReportsTheSiblingBundleWithNoProbe(t *testing.T) {
	stubDoctorTrustFacts(t, httptrust.Facts{
		Siblings: []httptrust.SiblingVar{{Name: "AWS_CA_BUNDLE", Path: `C:\corp\combined-ca-bundle.pem`}},
	})
	var out, errout bytes.Buffer
	if rc := runDoctorTrust(&out, &errout, []string{"--probe=false"}); rc != 1 {
		t.Fatalf("rc=%d, want 1 out=%s err=%s", rc, out.String(), errout.String())
	}
	if !strings.Contains(out.String(), "FAK_CA_BUNDLE") || !strings.Contains(out.String(), `C:\corp\combined-ca-bundle.pem`) {
		t.Fatalf("output=%s", out.String())
	}
}

func TestDoctorTrustJSONCarriesTheVersionedSchema(t *testing.T) {
	stubDoctorTrustFacts(t, httptrust.Facts{
		Probes: []httptrust.ProbeResult{{Host: "api.anthropic.com:443", Reached: true, DefaultOK: true, RootLabel: "ISRG Root X1"}},
	})
	var out, errout bytes.Buffer
	if rc := runDoctorTrust(&out, &errout, []string{"--json"}); rc != 0 {
		t.Fatalf("rc=%d err=%s", rc, errout.String())
	}
	var got doctorTrustReport
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("decode: %v raw=%s", err, out.String())
	}
	if got.Schema != doctorTrustSchema {
		t.Fatalf("schema=%q, want %q", got.Schema, doctorTrustSchema)
	}
	if !got.OK || got.Findings != 0 {
		t.Fatalf("ok=%v findings=%d, want a clean verdict", got.OK, got.Findings)
	}
	if len(got.Recommendations) == 0 {
		t.Fatal("no recommendations emitted; a check that reports nothing cannot be told from one that did not run")
	}
}

func TestDoctorTrustRejectsAPositionalArgument(t *testing.T) {
	var out, errout bytes.Buffer
	if rc := runDoctorTrust(&out, &errout, []string{"trust"}); rc != 2 {
		t.Fatalf("rc=%d, want 2 out=%s err=%s", rc, out.String(), errout.String())
	}
}

// The subcommand has to be reachable through `fak doctor` itself: both new config
// bails print `check: fak doctor trust`, and a check the bail names but the dispatch
// does not route is worse than no check at all.
func TestDoctorDispatchRoutesTrust(t *testing.T) {
	stubDoctorTrustFacts(t, httptrust.Facts{})
	var out, errout bytes.Buffer
	if rc := runDoctor(strings.NewReader(""), &out, &errout, []string{"trust", "--probe=false"}); rc != 0 {
		t.Fatalf("rc=%d out=%s err=%s", rc, out.String(), errout.String())
	}
	if !strings.Contains(out.String(), "== fak doctor: upstream trust ==") {
		t.Fatalf("`fak doctor trust` did not reach the trust check: %s", out.String())
	}
}
