package httptrust

import (
	"errors"
	"strings"
	"testing"
	"time"
)

// usableSource builds a Source that looks loaded without touching a filesystem or a
// certificate: Usable() is Bundle != nil && Err == nil, and every assessment reads
// only the described fields.
func usableSource(path string, subjects ...string) Source {
	return Source{Path: path, Bundle: &Bundle{
		Path:           path,
		Subjects:       subjects,
		EarliestExpiry: time.Date(2035, 1, 1, 0, 0, 0, 0, time.UTC),
	}}
}

func findingFor(rep Report, check string) (Finding, bool) {
	for _, f := range rep.Findings {
		if f.Check == check {
			return f, true
		}
	}
	return Finding{}, false
}

// The clean-host half of the acceptance in #8172: a host with no trust source
// declared, whose chain validates on the platform store, must raise NOTHING. A
// diagnostic that warns on every unintercepted laptop is one operators learn to
// ignore, and by the time they are on the box this check was written for they have
// already stopped reading it.
func TestAssessRaisesNothingOnACleanHost(t *testing.T) {
	rep := Assess(Facts{
		Probes: []ProbeResult{{Host: "api.anthropic.com:443", Reached: true, DefaultOK: true, RootLabel: "DigiCert Global Root CA"}},
	})
	if rep.Warnings != 0 {
		t.Fatalf("warnings=%d on a clean host, want 0: %+v", rep.Warnings, rep.Findings)
	}
	if rep.Intercepted {
		t.Fatal("interception reported on a chain that validated against the platform store with no bundle in play")
	}
	for _, f := range rep.Findings {
		if strings.Contains(f.Finding, "INTERCEPTED") {
			t.Fatalf("clean host names interception: %q", f.Finding)
		}
	}
}

// The intercepted half: name the interception, the CA subject observed, and the knob
// that fixes it. This is the finding the whole check exists to produce.
func TestAssessNamesTheInterceptionAndTheObservedCA(t *testing.T) {
	rep := Assess(Facts{
		Probes: []ProbeResult{{
			Host:      "api.anthropic.com:443",
			Reached:   true,
			VerifyErr: "x509: certificate signed by unknown authority",
			RootLabel: "ca.dsa-netskope.goskope.com",
		}},
	})
	if rep.Warnings == 0 {
		t.Fatal("an unverifiable intercepted chain raised no warning")
	}
	if !rep.Intercepted {
		t.Fatal("interception not reported for a chain that failed platform verification")
	}
	tls, ok := findingFor(rep, "upstream-tls")
	if !ok {
		t.Fatal("no upstream-tls row")
	}
	if !strings.Contains(tls.Finding, "ca.dsa-netskope.goskope.com") {
		t.Fatalf("the observed CA subject is not named: %q", tls.Finding)
	}
	if !strings.Contains(tls.Finding, "x509: certificate signed by unknown authority") {
		t.Fatalf("the platform's own error is not quoted: %q", tls.Finding)
	}
	src, _ := findingFor(rep, "trust-source")
	if !strings.Contains(src.Recommend, BundleKey) {
		t.Fatalf("the trust-source row does not name %s: %q", BundleKey, src.Recommend)
	}
}

// A managed Windows box usually looks like this: the interceptor's root is installed
// in the OS store, so the chain validates and nothing appears wrong — while every
// runtime that does NOT read the OS store still fails. The check has to witness the
// interception from the anchor name, and must not warn, because fak works here.
func TestAssessWitnessesInterceptionWhoseRootIsAlsoInThePlatformStore(t *testing.T) {
	rep := Assess(Facts{
		Source: usableSource(`C:\corp\combined-ca-bundle.pem`, "ca.dsa-netskope.goskope.com", "SSI-RootCA"),
		Probes: []ProbeResult{{Host: "api.anthropic.com:443", Reached: true, DefaultOK: true, RootLabel: "SSI-RootCA"}},
	})
	if !rep.Intercepted {
		t.Fatal("a chain terminating at a CA from the declared bundle is interception; it was not reported")
	}
	if rep.Warnings != 0 {
		t.Fatalf("warnings=%d on a working intercepted host, want 0: %+v", rep.Warnings, rep.Findings)
	}
	tls, _ := findingFor(rep, "upstream-tls")
	if !strings.Contains(tls.Finding, "SSI-RootCA") {
		t.Fatalf("the anchor is not named: %q", tls.Finding)
	}
}

// The multi-interceptor case that motivated widen-never-replace: the bundle covers
// the proxy in front of the model endpoint and not the one in front of the control
// plane. That is a warning, and the recommendation has to say ADD rather than
// REPLACE.
func TestAssessWarnsWhenTheBundleDoesNotCoverThisEndpoint(t *testing.T) {
	rep := Assess(Facts{
		Source: usableSource("/etc/corp/netskope.pem", "ca.dsa-netskope.goskope.com"),
		Probes: []ProbeResult{{
			Host:        "sts.amazonaws.com:443",
			Reached:     true,
			VerifyErr:   "x509: certificate signed by unknown authority",
			RootLabel:   "SSI-ISSUINGCA",
			BundleTried: true,
		}},
	})
	tls, _ := findingFor(rep, "upstream-tls")
	if tls.Severity != SeverityWarn {
		t.Fatalf("severity=%q for an endpoint fak cannot validate, want warn", tls.Severity)
	}
	if !strings.Contains(tls.Finding, "SSI-ISSUINGCA") || !strings.Contains(tls.Recommend, "Concatenate") {
		t.Fatalf("row does not name the uncovered CA or the add-not-replace fix: %+v", tls)
	}
}

// A bundle that rescues the connection is the configured posture WORKING. It is
// named at ok severity on purpose: a check that goes red on every correctly
// configured corporate host is a check nobody runs twice.
func TestAssessTreatsABundleRescuedChainAsHealthyButNamesIt(t *testing.T) {
	rep := Assess(Facts{
		Source: usableSource("/etc/corp/root.pem", "Corp Root CA"),
		Probes: []ProbeResult{{
			Host:        "api.anthropic.com:443",
			Reached:     true,
			VerifyErr:   "x509: certificate signed by unknown authority",
			RootLabel:   "Corp Root CA",
			BundleTried: true,
			BundleOK:    true,
		}},
		ChildEnv: []string{"NODE_EXTRA_CA_CERTS=/etc/corp/root.pem", "AWS_CA_BUNDLE=/etc/corp/root.pem",
			"CURL_CA_BUNDLE=/etc/corp/root.pem", "SSL_CERT_FILE=/etc/corp/root.pem",
			"REQUESTS_CA_BUNDLE=/etc/corp/root.pem", "GIT_SSL_CAINFO=/etc/corp/root.pem"},
	})
	if rep.Warnings != 0 {
		t.Fatalf("warnings=%d on a host whose declared bundle works, want 0: %+v", rep.Warnings, rep.Findings)
	}
	tls, _ := findingFor(rep, "upstream-tls")
	if !strings.Contains(tls.Finding, "TLS-INTERCEPTED") {
		t.Fatalf("the interception is not named: %q", tls.Finding)
	}
}

// An endpoint that cannot be reached says nothing about trust. An air-gapped or
// firewalled host must therefore produce no findings — otherwise the check is a false
// alarm on exactly the hosts that have no proxy at all.
func TestAssessDoesNotTurnAnUnreachableEndpointIntoATrustFinding(t *testing.T) {
	rep := Assess(Facts{
		Probes: []ProbeResult{{Host: "api.anthropic.com:443", Unreached: "dial tcp: i/o timeout"}},
	})
	if rep.Warnings != 0 {
		t.Fatalf("warnings=%d for an unreachable endpoint, want 0: %+v", rep.Warnings, rep.Findings)
	}
	if rep.Intercepted {
		t.Fatal("interception claimed without a handshake verdict")
	}
	tls, _ := findingFor(rep, "upstream-tls")
	if !strings.Contains(tls.Finding, "no trust verdict available") {
		t.Fatalf("the row does not say the verdict is unknown: %q", tls.Finding)
	}
}

// The no-network signal, and the one that fires on a real managed host today: another
// runtime here is already pointed at a corporate bundle and fak is not. It needs no
// handshake, cannot false-positive, and the recommendation carries the exact file.
func TestAssessWarnsWhenSiblingRuntimesCarryABundleAndFakDoesNot(t *testing.T) {
	rep := Assess(Facts{
		Siblings: []SiblingVar{{Name: "AWS_CA_BUNDLE", Path: `C:\corp\combined-ca-bundle.pem`}},
	})
	sib, ok := findingFor(rep, "sibling-trust-vars")
	if !ok {
		t.Fatal("no sibling-trust-vars row")
	}
	if sib.Severity != SeverityWarn {
		t.Fatalf("severity=%q, want warn", sib.Severity)
	}
	if !strings.Contains(sib.Recommend, `C:\corp\combined-ca-bundle.pem`) || !strings.Contains(sib.Recommend, BundleKey) {
		t.Fatalf("the recommendation is not actionable: %q", sib.Recommend)
	}
}

// With fak's own source declared the sibling row is informational — the host is
// consistent, and warning about it would be noise.
func TestAssessDoesNotWarnAboutSiblingsOnceFakHasItsOwnSource(t *testing.T) {
	rep := Assess(Facts{
		Source:   usableSource("/etc/corp/root.pem", "Corp Root CA"),
		Siblings: []SiblingVar{{Name: "AWS_CA_BUNDLE", Path: "/etc/corp/root.pem"}},
	})
	sib, _ := findingFor(rep, "sibling-trust-vars")
	if sib.Severity != SeverityOK {
		t.Fatalf("severity=%q, want ok: %+v", sib.Severity, sib)
	}
}

// "Which child runtimes would NOT inherit it" — the half of the finding that explains
// why the wrapped agent still fails after fak itself started working.
func TestAssessNamesTheChildRuntimesThatWouldNotInheritTrust(t *testing.T) {
	rep := Assess(Facts{
		Source:   usableSource("/etc/corp/root.pem", "Corp Root CA"),
		ChildEnv: []string{"PATH=/usr/bin", "AWS_CA_BUNDLE=/etc/corp/root.pem"},
	})
	child, ok := findingFor(rep, "child-runtime-trust")
	if !ok {
		t.Fatal("no child-runtime-trust row")
	}
	if child.Severity != SeverityWarn {
		t.Fatalf("severity=%q, want warn", child.Severity)
	}
	if !strings.Contains(child.Finding, "NODE_EXTRA_CA_CERTS") {
		t.Fatalf("the uninherited runtime is not named: %q", child.Finding)
	}
	if strings.Contains(child.Finding, "AWS_CA_BUNDLE") {
		t.Fatalf("a variable the child DOES receive is reported as missing: %q", child.Finding)
	}
}

// Trust and routing are different failures with the same symptom. A cloud-routed host
// whose certificates are perfect still has no adjudicated traffic, so the trust report
// must say so instead of letting a page of green rows imply otherwise.
func TestAssessWarnsOnACloudRouteEvenWhenTrustIsClean(t *testing.T) {
	rep := Assess(Facts{
		Probes:     []ProbeResult{{Host: "api.anthropic.com:443", Reached: true, DefaultOK: true, RootLabel: "DigiCert Global Root CA"}},
		CloudRoute: "CLAUDE_CODE_USE_BEDROCK",
	})
	route, ok := findingFor(rep, "cloud-route")
	if !ok {
		t.Fatal("no cloud-route row")
	}
	if route.Severity != SeverityWarn {
		t.Fatalf("severity=%q, want warn", route.Severity)
	}
	if !strings.Contains(route.Finding, "CLAUDE_CODE_USE_BEDROCK") || !strings.Contains(route.Finding, "UPSTREAM_UNSUPPORTED") {
		t.Fatalf("the row names neither the selector nor the bail token: %q", route.Finding)
	}
	if strings.Contains(route.Recommend, "--insecure") {
		t.Fatalf("a skip-verify escape leaked into a recommendation: %q", route.Recommend)
	}
}

// A waived route still warns, and the warning has to name what the session is NOT
// getting — otherwise a waiver reads as approval.
func TestAssessWaivedCloudRouteSaysTheTrafficIsUnadjudicated(t *testing.T) {
	rep := Assess(Facts{CloudRoute: "CLAUDE_CODE_USE_VERTEX", CloudRouteWaived: true})
	route, _ := findingFor(rep, "cloud-route")
	if route.Severity != SeverityWarn || !strings.Contains(route.Finding, "NONE of this session's model traffic") {
		t.Fatalf("waived route row does not name the gap: %+v", route)
	}
}

// A declared bundle that did not load is UPSTREAM_TRUST_UNVERIFIED, and the row has
// to hand over the pre-bound recovery rather than an adjective.
func TestAssessRoutesAnUnloadableBundleToItsRecovery(t *testing.T) {
	rep := Assess(Facts{Source: Source{Path: "/etc/corp/missing.pem", Err: errors.New("httptrust: read bundle /etc/corp/missing.pem: no such file")}})
	src, _ := findingFor(rep, "trust-source")
	if src.Severity != SeverityWarn {
		t.Fatalf("severity=%q, want warn", src.Severity)
	}
	if !strings.Contains(src.Recommend, "fak recover UPSTREAM_TRUST_UNVERIFIED --set path=/etc/corp/missing.pem") {
		t.Fatalf("recommendation is not the pre-bound recovery: %q", src.Recommend)
	}
	if _, ok := findingFor(rep, "bundle-expiry"); ok {
		t.Fatal("an expiry row was assessed for a bundle that never loaded")
	}
}

func TestAssessWarnsOnAnExpiredBundleAndOnOneAboutToExpire(t *testing.T) {
	now := time.Date(2026, 8, 19, 0, 0, 0, 0, time.UTC)
	expired := usableSource("/etc/corp/root.pem", "Corp Root CA")
	expired.Bundle.Expired = 1
	rep := Assess(Facts{Source: expired, Now: now})
	if f, _ := findingFor(rep, "bundle-expiry"); f.Severity != SeverityWarn {
		t.Fatalf("an expired root did not warn: %+v", f)
	}
	soon := usableSource("/etc/corp/root.pem", "Corp Root CA")
	soon.Bundle.EarliestExpiry = now.Add(72 * time.Hour)
	rep = Assess(Facts{Source: soon, Now: now})
	if f, _ := findingFor(rep, "bundle-expiry"); f.Severity != SeverityWarn {
		t.Fatalf("a root expiring in 3 days did not warn: %+v", f)
	}
	fine := usableSource("/etc/corp/root.pem", "Corp Root CA")
	rep = Assess(Facts{Source: fine, Now: now})
	if f, _ := findingFor(rep, "bundle-expiry"); f.Severity != SeverityOK {
		t.Fatalf("a root valid until 2035 warned: %+v", f)
	}
}

// The report is what an operator pastes into a ticket, so it must carry paths and CA
// names and no values from the environment beyond them.
func TestHostSiblingTrustVarsReadsOnlyTheKnownTrustNames(t *testing.T) {
	got := HostSiblingTrustVars([]string{
		"PATH=/usr/bin",
		"AWS_SECRET_ACCESS_KEY=shhh",
		"aws_ca_bundle=/etc/corp/root.pem", // Windows folds env-name case
		"REQUESTS_CA_BUNDLE=",              // set-but-empty is not a declaration
		"NODE_EXTRA_CA_CERTS=/etc/corp/node.pem",
	})
	if len(got) != 2 {
		t.Fatalf("got %+v, want exactly the two declared trust vars", got)
	}
	if got[0].Name != "NODE_EXTRA_CA_CERTS" || got[1].Name != "AWS_CA_BUNDLE" {
		t.Fatalf("order is not ChildTrustVars order: %+v", got)
	}
	if got[1].Path != "/etc/corp/root.pem" {
		t.Fatalf("case-folded name lost its value: %+v", got[1])
	}
	for _, s := range got {
		if strings.Contains(s.Path, "shhh") {
			t.Fatal("a credential value reached the report")
		}
	}
}
