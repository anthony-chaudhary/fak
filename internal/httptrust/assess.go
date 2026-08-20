// assess.go is the PURE verdict half of the trust seam: given a set of observed
// facts about a host — the declared trust source, what a handshake against each
// upstream actually did, which sibling runtimes already carry a corporate bundle,
// and what environment a child would receive — decide what to tell the operator.
//
// It is pure (no I/O, no clock beyond an injected Now) so `fak doctor trust` can be
// tested for the two properties that matter and are otherwise untestable on the
// machine you happen to be sitting at: on an INTERCEPTED host it names the
// interception, the CA subject observed, the trust source fak used, and which child
// runtimes would not inherit it; on a CLEAN host it raises nothing.
//
// "Raises nothing" means zero WARN findings, which is the doctor's own vocabulary
// for a finding (`doctorReport.Findings` counts warnings, and a nonzero count is
// what exits 1). The OK rows are still printed, exactly as every other `fak doctor`
// check prints its passing rows — a diagnostic that stays silent when healthy
// cannot be distinguished from one that failed to run.

package httptrust

import (
	"fmt"
	"sort"
	"strings"
	"time"
)

// Severity is the doctor's two-value vocabulary, mirrored here so the pure
// assessment can classify its own findings without importing cmd/fak.
const (
	SeverityOK   = "ok"
	SeverityWarn = "warn"
)

// expirySoon is how far ahead a bundle expiry is still worth a warning. A
// corporate root is typically re-issued years apart, so a root inside this window
// is either already being rotated or about to break every connection on the host —
// either way the operator wants it before the outage, not after.
const expirySoon = 30 * 24 * time.Hour

// Finding is one row of the trust report: what was checked, whether it needs
// action, what was observed, and the action. Field names match cmd/fak's
// Recommendation so the doctor can map one to the other without a translation
// table.
type Finding struct {
	Check     string `json:"check"`
	Severity  string `json:"severity"`
	Finding   string `json:"finding"`
	Recommend string `json:"recommend,omitempty"`
}

// SiblingVar is one ChildTrustVars name the HOST already sets, with the path it
// points at.
//
// This is the load-bearing zero-false-positive signal, and it needs no network: a
// host where someone has already set AWS_CA_BUNDLE or NODE_EXTRA_CA_CERTS is a host
// where a human has already discovered the interception and written down the answer
// for one runtime. fak reading that and staying on the platform default is the
// "works in the parent shell, degrades inside fak" failure exactly. The path is
// carried because it makes the recommendation concrete — it is a filesystem path to
// PUBLIC certificates, never a credential.
type SiblingVar struct {
	Name string `json:"name"`
	Path string `json:"path"`
}

// ProbeResult is one bounded TLS handshake observation. Every field is a string or
// a bool so a test can construct a whole intercepted host without a network, a
// certificate, or a fixture directory.
type ProbeResult struct {
	// Host is the "host:port" dialed.
	Host string `json:"host"`
	// Reached is true when the handshake produced a TRUST verdict — either it
	// validated, or it failed certificate verification. A DNS failure, a refused
	// connection, or a timeout leaves this false: those say nothing about trust and
	// must never be reported as a trust finding.
	Reached bool `json:"reached"`
	// DefaultOK is true when the chain validated against the platform trust store
	// alone, with no bundle appended.
	DefaultOK bool `json:"default_ok"`
	// VerifyErr is the verification failure as the platform reported it, empty when
	// the chain validated. It is quoted verbatim because the operator has almost
	// certainly already seen this exact string from some other tool.
	VerifyErr string `json:"verify_err,omitempty"`
	// Unreached is why no trust verdict is available (dial/DNS/timeout), empty when
	// Reached.
	Unreached string `json:"unreached,omitempty"`
	// RootLabel names the trust anchor the server's chain terminates at: the issuer
	// of the outermost certificate it presented. On an intercepted connection this is
	// the interceptor's CA — the single most useful string in the whole report,
	// because it is what the operator compares against the root their IT team
	// published.
	RootLabel string `json:"root_label,omitempty"`
	// LeafIssuer names the CA that signed the endpoint's own certificate.
	LeafIssuer string `json:"leaf_issuer,omitempty"`
	// BundleTried records that the handshake was retried with the declared bundle
	// appended, and BundleOK whether that retry validated.
	BundleTried bool `json:"bundle_tried"`
	BundleOK    bool `json:"bundle_ok"`
}

// Facts is everything Assess is allowed to look at. Collecting it is the impure
// half (probe.go dials, cmd/fak reads the environment); deciding what it means is
// this file.
type Facts struct {
	// Source is the resolved trust posture: what the config surface declared and
	// whether it loaded.
	Source Source
	// Probes are the per-host handshake observations, possibly empty (--probe=false,
	// or an air-gapped host).
	Probes []ProbeResult
	// Siblings are the ChildTrustVars the HOST already sets. See SiblingVar.
	Siblings []SiblingVar
	// ChildEnv is the environment a wrapped child would actually receive, used to
	// answer "which child runtimes would not inherit fak's trust source". Nil means
	// the caller did not model a child, and the inheritance row is skipped rather
	// than guessed.
	ChildEnv []string
	// CloudRoute is the selector name of a detected request-signed cloud model route
	// (e.g. CLAUDE_CODE_USE_BEDROCK), empty when none. Passed in as a plain string
	// rather than resolved here so this package stays free of any dependency on the
	// route detector.
	CloudRoute string
	// CloudRouteWaived records that the operator waived the UPSTREAM_UNSUPPORTED
	// refusal for that route.
	CloudRouteWaived bool
	// Now is the clock for expiry arithmetic; the zero value means time.Now().
	Now time.Time
}

// Report is the assessed trust posture.
type Report struct {
	Declared bool     `json:"declared"`
	Usable   bool     `json:"usable"`
	Path     string   `json:"path,omitempty"`
	Subjects []string `json:"subjects,omitempty"`
	// Intercepted is true when at least one probe WITNESSED interception: a chain
	// that failed platform verification, or one that validated but terminates at a
	// CA the declared bundle carries (i.e. the interceptor's root is installed in the
	// OS store, which is how a managed Windows box usually looks).
	Intercepted bool      `json:"intercepted"`
	Findings    []Finding `json:"findings"`
	// Warnings counts findings that need operator action. Zero on a clean host.
	Warnings int `json:"warnings"`
}

// Assess turns observed facts into the operator-facing report.
//
// Row order is fixed — trust source, expiry, per-probe, sibling runtimes, child
// inheritance, cloud route — so the output diffs cleanly across runs and a test can
// assert on position.
func Assess(f Facts) Report {
	now := f.Now
	if now.IsZero() {
		now = time.Now()
	}
	rep := Report{Declared: f.Source.Declared(), Usable: f.Source.Usable(), Path: f.Source.Path}
	if f.Source.Bundle != nil {
		rep.Subjects = f.Source.Bundle.Subjects
	}
	for _, p := range f.Probes {
		if p.Reached && (!p.DefaultOK || labelInBundle(p.RootLabel, rep.Subjects)) {
			rep.Intercepted = true
		}
	}

	rep.Findings = append(rep.Findings, assessSource(f, rep))
	if f.Source.Usable() {
		rep.Findings = append(rep.Findings, assessExpiry(f.Source.Bundle, now))
	}
	for _, p := range f.Probes {
		rep.Findings = append(rep.Findings, assessProbe(p, f.Source, rep.Subjects))
	}
	if len(f.Siblings) > 0 {
		rep.Findings = append(rep.Findings, assessSiblings(f.Siblings, f.Source))
	}
	if f.ChildEnv != nil && f.Source.Usable() {
		rep.Findings = append(rep.Findings, assessChildInheritance(f.ChildEnv))
	}
	if strings.TrimSpace(f.CloudRoute) != "" {
		rep.Findings = append(rep.Findings, assessCloudRoute(f))
	}
	for _, fd := range rep.Findings {
		if fd.Severity == SeverityWarn {
			rep.Warnings++
		}
	}
	return rep
}

// assessSource is the always-present row: which trust store fak is validating with.
func assessSource(f Facts, rep Report) Finding {
	const check = "trust-source"
	switch {
	case f.Source.Usable():
		return Finding{
			Check:    check,
			Severity: SeverityOK,
			Finding: fmt.Sprintf("validating against the platform trust store PLUS %s (%d certificate(s): %s)",
				f.Source.Path, len(rep.Subjects), SubjectSummary(rep.Subjects)),
			Recommend: "children inherit the same file via " + strings.Join(ChildTrustVars, ", "),
		}
	case f.Source.Declared():
		return Finding{
			Check:    check,
			Severity: SeverityWarn,
			Finding: fmt.Sprintf("%s declares %s and it did not load (%v), so fak is validating against the platform trust store only",
				BundleKey, f.Source.Path, f.Source.Err),
			Recommend: "fak recover UPSTREAM_TRUST_UNVERIFIED --set path=" + f.Source.Path,
		}
	case rep.Intercepted:
		return Finding{
			Check:    check,
			Severity: SeverityWarn,
			Finding: fmt.Sprintf("TLS interception is in play on this host and no trust source is declared — %s is unset, so fak validates against the platform trust store only",
				BundleKey),
			Recommend: fmt.Sprintf("export %s=/path/to/corporate-root.pem — fak then validates against the platform store PLUS that root (never the root alone) and derives %s for child runtimes. There is deliberately no skip-verify escape.",
				BundleKey, strings.Join(ChildTrustVars, "/")),
		}
	default:
		return Finding{
			Check:    check,
			Severity: SeverityOK,
			Finding:  "no corporate trust source declared; validating against the platform trust store — the correct posture on an unintercepted host",
		}
	}
}

// assessExpiry warns before the interception root breaks every connection on the
// host rather than after. A bundle can be perfectly loadable and still carry a root
// that expired last month.
func assessExpiry(b *Bundle, now time.Time) Finding {
	const check = "bundle-expiry"
	switch {
	case b.Expired > 0:
		return Finding{
			Check:     check,
			Severity:  SeverityWarn,
			Finding:   fmt.Sprintf("%d of %d certificate(s) in %s are past their expiry", b.Expired, len(b.Subjects), b.Path),
			Recommend: "re-export the current root from your IT trust source; an expired root cannot validate the chain it signed",
		}
	case b.EarliestExpiry.IsZero():
		return Finding{Check: check, Severity: SeverityOK, Finding: "no expiry could be read from the bundle"}
	case b.EarliestExpiry.Sub(now) < expirySoon:
		return Finding{
			Check:     check,
			Severity:  SeverityWarn,
			Finding:   fmt.Sprintf("the soonest certificate in %s expires %s", b.Path, b.EarliestExpiry.UTC().Format(time.RFC3339)),
			Recommend: "refresh the bundle before that date — every intercepted connection on this host fails the moment it passes",
		}
	default:
		return Finding{
			Check:    check,
			Severity: SeverityOK,
			Finding:  fmt.Sprintf("soonest certificate expiry %s", b.EarliestExpiry.UTC().Format(time.RFC3339)),
		}
	}
}

// assessProbe classifies ONE handshake observation.
//
// The severity discipline here is deliberate: severity answers "must the operator
// act", not "is this host unusual". An intercepted host whose interceptor fak
// already trusts is a WORKING host — reporting it as a warning would make `fak
// doctor trust` red on every correctly-configured corporate box, and a check that is
// always red is a check nobody reads. So interception is NAMED in the finding text
// at OK severity, and only an endpoint fak cannot currently validate warns.
func assessProbe(p ProbeResult, src Source, subjects []string) Finding {
	check := "upstream-tls"
	switch {
	case !p.Reached:
		return Finding{
			Check:    check,
			Severity: SeverityOK,
			Finding:  fmt.Sprintf("%s: not reached (%s) — no trust verdict available, so nothing is claimed either way", p.Host, p.Unreached),
		}
	case p.DefaultOK && labelInBundle(p.RootLabel, subjects):
		return Finding{
			Check:    check,
			Severity: SeverityOK,
			Finding: fmt.Sprintf("%s: TLS-INTERCEPTED — the chain terminates at %q, a certificate carried by %s, so the interceptor's root is also installed in this host's platform store and the chain validates either way",
				p.Host, p.RootLabel, src.Path),
		}
	case p.DefaultOK:
		return Finding{
			Check:    check,
			Severity: SeverityOK,
			Finding:  fmt.Sprintf("%s: the chain validated against the platform trust store (anchor %q)", p.Host, p.RootLabel),
		}
	case p.BundleOK:
		return Finding{
			Check:    check,
			Severity: SeverityOK,
			Finding: fmt.Sprintf("%s: TLS-INTERCEPTED — the platform trust store rejects this chain (%s) and it validates once %s is appended; the interceptor's CA is %q",
				p.Host, p.VerifyErr, src.Path, p.RootLabel),
			Recommend: "this is the configured posture working. Every OTHER tool on this host that does not read " +
				strings.Join(ChildTrustVars, "/") + " will still fail on this endpoint",
		}
	case p.BundleTried:
		return Finding{
			Check:    check,
			Severity: SeverityWarn,
			Finding: fmt.Sprintf("%s: the chain does not validate against the platform store (%s) NOR with %s appended; the CA presented is %q",
				p.Host, p.VerifyErr, src.Path, p.RootLabel),
			Recommend: "a site commonly runs more than one interceptor — the root in front of THIS endpoint is a different one than your bundle covers. " +
				"Concatenate its PEM onto the bundle; fak widens the platform store, so adding a root never removes one",
		}
	default:
		return Finding{
			Check:    check,
			Severity: SeverityWarn,
			Finding: fmt.Sprintf("%s: TLS-INTERCEPTED and unverifiable — %s. The CA that re-signed the connection is %q",
				p.Host, p.VerifyErr, p.RootLabel),
			Recommend: fmt.Sprintf("this is interception, not a firewall: nothing is blocked, a private root re-signed the connection. Export %s pointing at a PEM holding %q",
				BundleKey, p.RootLabel),
		}
	}
}

// assessSiblings reports the host's own evidence. It warns only in the case that is
// unambiguously wrong: another runtime on this host is pointed at a corporate
// bundle and fak is not.
func assessSiblings(siblings []SiblingVar, src Source) Finding {
	const check = "sibling-trust-vars"
	names := make([]string, 0, len(siblings))
	for _, s := range siblings {
		names = append(names, s.Name)
	}
	sort.Strings(names)
	if src.Declared() {
		return Finding{
			Check:    check,
			Severity: SeverityOK,
			Finding:  fmt.Sprintf("this host also declares a bundle for %s; fak has its own trust source declared", strings.Join(names, ", ")),
		}
	}
	return Finding{
		Check:    check,
		Severity: SeverityWarn,
		Finding: fmt.Sprintf("%s already point at a corporate CA bundle on this host, but %s is unset — fak is the only runtime here still on the platform default",
			strings.Join(names, ", "), BundleKey),
		Recommend: fmt.Sprintf("export %s=%s (the same file %s already uses)", BundleKey, siblings[0].Path, siblings[0].Name),
	}
}

// assessChildInheritance answers the "which child runtimes would NOT inherit it"
// half of the finding, from the environment the child would ACTUALLY receive rather
// than from the allow-list in the abstract.
func assessChildInheritance(childEnv []string) Finding {
	const check = "child-runtime-trust"
	missing := UninheritedRuntimes(childEnv)
	if len(missing) == 0 {
		return Finding{
			Check:    check,
			Severity: SeverityOK,
			Finding:  "every child runtime inherits fak's trust source (" + strings.Join(ChildTrustVars, ", ") + ")",
		}
	}
	return Finding{
		Check:    check,
		Severity: SeverityWarn,
		Finding:  "a wrapped child would NOT inherit fak's trust source for " + strings.Join(missing, ", "),
		Recommend: "these names are stripped or unset on the child path, so a runtime the agent shells out to will still fail TLS. " +
			"Check FAK_SANDBOX_INHERIT_ENV and the sandbox allow-list, or set them in the launching environment",
	}
}

// assessCloudRoute names the posture the trust seam cannot fix. Trust and routing
// are different failures with the same symptom ("it does not work in fak"), and an
// operator who fixes the certificate on a Bedrock-routed box still has no adjudicated
// traffic — so the trust report has to say so rather than let the green rows imply
// otherwise.
func assessCloudRoute(f Facts) Finding {
	const check = "cloud-route"
	if f.CloudRouteWaived {
		return Finding{
			Check:    check,
			Severity: SeverityWarn,
			Finding:  f.CloudRoute + " selects a request-signed cloud model route and the UPSTREAM_UNSUPPORTED refusal is waived — the hook floor, tool brokering, transcript, and sandbox apply, but fak sees NONE of this session's model traffic",
			Recommend: "for an adjudicated capability floor on this posture use `fak serve --stdio --policy FILE` (fak as an MCP server the agent calls), " +
				"which is provider-agnostic",
		}
	}
	return Finding{
		Check:     check,
		Severity:  SeverityWarn,
		Finding:   f.CloudRoute + " selects a request-signed cloud model route, so the child signs each request itself and never reads ANTHROPIC_BASE_URL — fak's gateway repoint is inert and `fak guard` refuses with UPSTREAM_UNSUPPORTED",
		Recommend: "fak recover UPSTREAM_UNSUPPORTED",
	}
}

// labelInBundle reports whether a chain anchor is one of the certificates the
// declared bundle contributed. Compared case-insensitively on the rendered label
// because that is the string an operator reads in both places; this drives a report
// line only, never a verification decision.
func labelInBundle(label string, subjects []string) bool {
	label = strings.TrimSpace(label)
	if label == "" {
		return false
	}
	for _, s := range subjects {
		if strings.EqualFold(strings.TrimSpace(s), label) {
			return true
		}
	}
	return false
}

// HostSiblingTrustVars extracts the ChildTrustVars the HOST already sets from an
// environ snapshot, in ChildTrustVars order.
//
// It takes the snapshot as a parameter rather than reading the process environment
// so it stays pure and testable — and so the doctor, which already holds the
// environ it is modelling a child from, reports on exactly that.
func HostSiblingTrustVars(environ []string) []SiblingVar {
	values := make(map[string]string, len(environ))
	for _, kv := range environ {
		if i := strings.IndexByte(kv, '='); i > 0 {
			if v := strings.TrimSpace(kv[i+1:]); v != "" {
				values[envName(kv[:i])] = v
			}
		}
	}
	var out []SiblingVar
	for _, name := range ChildTrustVars {
		if v, ok := values[envName(name)]; ok {
			out = append(out, SiblingVar{Name: name, Path: v})
		}
	}
	return out
}
