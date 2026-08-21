// httptrust.go is the trust seam itself: resolve ONE declared CA bundle, build a
// RootCAs pool that WIDENS the platform pool with it, and hand back an
// *http.Transport / *http.Client every fak-originated HTTPS call site can adopt
// with a single import. See doc.go for why the widen-never-replace rule is the
// whole point of the package.

package httptrust

import (
	"crypto/tls"
	"crypto/x509"
	"encoding/pem"
	"errors"
	"fmt"
	"net/http"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/secretload"
)

// BundleKey names the config/secretload key that declares the corporate trust
// source: a PEM file holding the root (and any intermediate) certificates the
// site's TLS-intercepting proxy re-signs with.
//
// It is read through secretload.Loader, not os.Getenv, for two reasons: the
// loader is the repo's declared config surface (so an encrypted-file or vault
// backend can supply it without changing this package), and a literal
// os.Getenv("FAK_CA_BUNDLE") would be a CONFIG_NOT_ENV offense —
// internal/envconfiglint classifies a non-credential name read straight from the
// process env as configuration that belongs on a config surface, and a
// filesystem path to a set of PUBLIC certificates is exactly that. There is
// deliberately no --insecure sibling: interception is a trust problem, and
// normalizing skipped verification inside a governance tool is the wrong fix.
const BundleKey = "FAK_CA_BUNDLE"

// ChildTrustVars are the per-runtime trust variables a fak child needs in order
// to honor the SAME bundle fak itself validates against. They exist because no
// two runtimes read the same variable: Go on Windows consults the OS store only
// (SSL_CERT_FILE is Unix-only in crypto/x509), Node reads NODE_EXTRA_CA_CERTS,
// the AWS SDKs read AWS_CA_BUNDLE, curl reads CURL_CA_BUNDLE, Python requests
// reads REQUESTS_CA_BUNDLE, and git reads GIT_SSL_CAINFO. An operator should not
// have to know that list — declaring the bundle once derives all of it.
//
// Order is fixed so a derived environment is byte-stable across runs.
var ChildTrustVars = []string{
	"NODE_EXTRA_CA_CERTS", // Node — i.e. Claude Code itself, the primary wrapped harness
	"AWS_CA_BUNDLE",       // AWS SDKs and CLI (Bedrock, S3, STS)
	"CURL_CA_BUNDLE",      // curl in any hook or adapter script
	"SSL_CERT_FILE",       // OpenSSL, and Go on non-Windows
	"REQUESTS_CA_BUNDLE",  // Python requests
	"GIT_SSL_CAINFO",      // git over HTTPS, which hooks shell out to constantly
}

// ErrNoCertificates is returned when the declared bundle parses to zero usable
// certificates. It is an error rather than a silent no-op on purpose: a bundle
// that contributes nothing means the operator believes trust is configured while
// fak validates against the platform default, which is the exact "works in the
// parent shell, degrades inside fak" failure this package exists to end.
var ErrNoCertificates = errors.New("httptrust: bundle contributed no usable certificates")

// Bundle is a loaded trust source: the pool fak validates with, plus the facts an
// operator-facing surface (fak doctor, a config bail) needs to name what changed.
type Bundle struct {
	// Path is the PEM file the pool was built from.
	Path string
	// Pool is the platform pool WIDENED with Path's certificates. It is never a
	// bare x509.NewCertPool: see Load.
	Pool *x509.CertPool
	// Subjects names each certificate the bundle contributed, most-specific field
	// first (CN, else the first O, else the raw RDN string), first-seen order.
	Subjects []string
	// Expired counts contributed certificates already past NotAfter. A bundle can
	// be usable and still carry an expired root — the operator wants to know.
	Expired int
	// EarliestExpiry is the soonest NotAfter among the contributed certificates,
	// zero when the bundle contributed none.
	EarliestExpiry time.Time
}

// subjectSummaryMax caps how many of a bundle's subjects a human-facing line
// names. A corporate bundle is routinely the whole public root set with the site's
// own roots appended — 123 subjects on the host this was dogfooded on — and one
// line carrying all of them is not information, it is a wall the operator scrolls
// past to reach the row that mattered. The complete list stays machine-readable in
// `fak doctor trust --json`, and the root that actually matters is named again, by
// itself, on the upstream-tls row that witnessed the interception.
const subjectSummaryMax = 6

// SubjectSummary renders bundle subjects for an operator-facing line: at most
// subjectSummaryMax of them in file order, then a count of the remainder and where
// to read the rest. A bundle at or under the cap renders in full, unchanged.
func SubjectSummary(subjects []string) string {
	if len(subjects) <= subjectSummaryMax {
		return strings.Join(subjects, ", ")
	}
	return fmt.Sprintf("%s, and %d more (fak doctor trust --json lists all)",
		strings.Join(subjects[:subjectSummaryMax], ", "), len(subjects)-subjectSummaryMax)
}

// Source is the resolved trust posture: what the config surface declared, and
// whether it loaded. A zero Source means "nothing declared", which is the clean
// box and the historical behavior byte-for-byte.
type Source struct {
	// Path is the declared bundle path, empty when nothing was declared.
	Path string
	// Bundle is the loaded bundle, nil when nothing was declared or the load failed.
	Bundle *Bundle
	// Err is why a DECLARED bundle did not load. A non-nil Err with a non-empty
	// Path is the UPSTREAM_TRUST_UNVERIFIED condition: the operator declared a
	// trust source and fak is not using it.
	Err error
}

// Declared reports that a trust source was configured at all.
func (s Source) Declared() bool { return strings.TrimSpace(s.Path) != "" }

// Usable reports that a declared trust source loaded and is what fak validates with.
func (s Source) Usable() bool { return s.Bundle != nil && s.Err == nil }

// Load reads a PEM bundle and returns the platform pool WIDENED with it.
//
// The widen-never-replace rule is the load-bearing decision here, and it is not
// cosmetic. A corporate site routinely runs more than one interceptor: a bundle
// that covers the proxy in front of the model endpoint typically does NOT cover
// the proxy in front of the cloud control plane (STS, the IdP, the package
// mirror). Building RootCAs from a fresh x509.NewCertPool plus the bundle
// therefore trades one broken endpoint for every OTHER endpoint the OS store was
// already validating — a regression the operator would read as "fak broke my
// network". Appending to x509.SystemCertPool() keeps the platform verifier in
// play, which is also what makes this work on Windows, where the OS store is
// Go's only default and SSL_CERT_FILE does nothing.
//
// A SystemCertPool failure is therefore FATAL rather than a fallback: silently
// narrowing trust to the bundle alone is a worse outcome than refusing, and the
// refusal is exactly what UPSTREAM_TRUST_UNVERIFIED reports.
func Load(path string) (*Bundle, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return nil, errors.New("httptrust: empty bundle path")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("httptrust: read bundle %s: %w", path, err)
	}
	pool, err := x509.SystemCertPool()
	if err != nil {
		// Never fall back to a fresh pool: that would REPLACE the platform trust
		// store rather than widen it, breaking every endpoint the bundle does not
		// cover. Refuse and let the caller name the posture.
		return nil, fmt.Errorf("httptrust: platform trust store unavailable, refusing to narrow trust to %s alone: %w", path, err)
	}
	if !pool.AppendCertsFromPEM(raw) {
		return nil, fmt.Errorf("%w: %s", ErrNoCertificates, path)
	}
	b := &Bundle{Path: path, Pool: pool}
	describeBundle(b, raw)
	return b, nil
}

// describeBundle fills the operator-facing fields by re-walking the PEM blocks.
// AppendCertsFromPEM already told us at least one certificate landed; this pass
// exists only to name them, so an unparsable block is skipped rather than fatal —
// the pool is authoritative, this description is reporting.
func describeBundle(b *Bundle, raw []byte) {
	now := time.Now()
	rest := raw
	for {
		var blk *pem.Block
		blk, rest = pem.Decode(rest)
		if blk == nil {
			return
		}
		if blk.Type != "CERTIFICATE" {
			continue
		}
		crt, err := x509.ParseCertificate(blk.Bytes)
		if err != nil {
			continue
		}
		b.Subjects = append(b.Subjects, SubjectLabel(crt))
		if crt.NotAfter.Before(now) {
			b.Expired++
		}
		if b.EarliestExpiry.IsZero() || crt.NotAfter.Before(b.EarliestExpiry) {
			b.EarliestExpiry = crt.NotAfter
		}
	}
}

// SubjectLabel renders the most operator-recognizable name for a certificate:
// the Common Name, else the first Organization, else the full RDN string. This is
// what an operator compares against the CA their IT team told them about, so it
// is exported and shared with the doctor check.
func SubjectLabel(crt *x509.Certificate) string {
	if crt == nil {
		return ""
	}
	return NameLabel(crt.Subject)
}

// Resolve resolves the declared trust source from the default config surface.
//
// It is deliberately NOT cached. Clients are built per planner/per command, not
// per request, so re-reading a small PEM costs nothing measurable, and a
// process-wide sync.Once would make the resolution order-dependent in tests and
// unable to see a bundle written after startup.
func Resolve() Source { return ResolveWith(secretload.Default()) }

// ResolveWith is Resolve against an injected loader, for tests and for hosts that
// source the bundle path from an encrypted file or vault-backed SecretSource.
func ResolveWith(loader *secretload.Loader) Source {
	if loader == nil {
		loader = secretload.Default()
	}
	raw, ok := loader.Lookup(BundleKey)
	if !ok || strings.TrimSpace(raw) == "" {
		return Source{}
	}
	path := strings.TrimSpace(raw)
	b, err := Load(path)
	if err != nil {
		return Source{Path: path, Err: err}
	}
	return Source{Path: path, Bundle: b}
}

// TransportFrom returns a transport that validates against the platform pool
// widened by the bundle at path. It clones http.DefaultTransport so every tuned
// default (proxy resolution from HTTP_PROXY/HTTPS_PROXY, HTTP/2, connection
// pooling, timeouts) is preserved — the ONLY change is RootCAs.
func TransportFrom(path string) (*http.Transport, error) {
	b, err := Load(path)
	if err != nil {
		return nil, err
	}
	return TransportForBundle(b), nil
}

// TransportForBundle is TransportFrom for an already-loaded bundle.
func TransportForBundle(b *Bundle) *http.Transport {
	if b == nil || b.Pool == nil {
		return nil
	}
	tr, ok := http.DefaultTransport.(*http.Transport)
	if !ok {
		// Something replaced the stdlib default. Build a bare transport rather than
		// panicking: a governance tool must not die because a host program swapped a
		// global, and RootCAs is still applied.
		return &http.Transport{TLSClientConfig: &tls.Config{RootCAs: b.Pool, MinVersion: tls.VersionTLS12}}
	}
	out := tr.Clone()
	if out.TLSClientConfig == nil {
		out.TLSClientConfig = &tls.Config{}
	}
	out.TLSClientConfig.RootCAs = b.Pool
	if out.TLSClientConfig.MinVersion == 0 {
		out.TLSClientConfig.MinVersion = tls.VersionTLS12
	}
	return out
}

// Client returns the client a fak-originated HTTPS call site should use.
//
// With no bundle declared it returns &http.Client{Timeout: timeout} — a nil
// Transport, byte-for-byte the literal every call site builds today, so an
// unconfigured box is unchanged and nothing downstream that inspects
// Client.Transport starts seeing a wrapper it did not before.
//
// With a declared bundle that FAILED to load it also returns the plain client:
// this constructor cannot return an error, and degrading is the only option left
// at this depth. The degrade is never silent in practice — Resolve().Err is what
// the launch path bails on (UPSTREAM_TRUST_UNVERIFIED) and what `fak doctor`
// reports, so the operator learns before a request is ever made.
func Client(timeout time.Duration) *http.Client {
	return ClientForSource(Resolve(), timeout)
}

// ClientForSource is Client against an already-resolved Source, for callers that
// resolved once and want every client they build to agree.
func ClientForSource(src Source, timeout time.Duration) *http.Client {
	c := &http.Client{Timeout: timeout}
	if !src.Usable() {
		return c
	}
	if tr := TransportForBundle(src.Bundle); tr != nil {
		c.Transport = tr
	}
	return c
}

// Wrap installs the declared trust source on an existing client, preserving its
// timeout, jar, and redirect policy. A client that already carries a custom
// Transport is left alone: that transport is the caller's deliberate choice
// (an observer, a recorder, a test round-tripper), and silently discarding it
// would break behavior far from here. Callers that build such a transport should
// take RootCAs from Pool instead.
func Wrap(c *http.Client) *http.Client {
	if c == nil || c.Transport != nil {
		return c
	}
	if tr := TransportForBundle(Resolve().Bundle); tr != nil {
		c.Transport = tr
	}
	return c
}

// RootCAs is the pool an already-resolved Source validates with, or nil when it
// declared nothing or did not load. A nil pool means "the platform default", which
// is what every crypto/tls call site does with a nil RootCAs — so a caller can pass
// this straight through without a branch, and without importing crypto/x509 itself.
func (s Source) RootCAs() *x509.CertPool {
	if s.Bundle == nil {
		return nil
	}
	return s.Bundle.Pool
}

// Pool returns the declared RootCAs pool, or nil when nothing is declared or the
// declared bundle did not load. Call sites that build their own tls.Config set
// RootCAs from this; a nil result means "leave RootCAs nil", i.e. the platform
// default, which is the correct unconfigured behavior.
func Pool() *x509.CertPool {
	if b := Resolve().Bundle; b != nil {
		return b.Pool
	}
	return nil
}

// ChildEnv derives the per-runtime trust variables for a child process from one
// declared bundle path, skipping every name the parent environment ALREADY sets.
//
// Skipping is the point of taking parent: an operator who set NODE_EXTRA_CA_CERTS
// by hand may well have pointed it at a fuller bundle than the one fak was given,
// and a derived value that overwrote it would make fak the reason the child's
// trust got narrower. Derivation fills gaps; it never overrides a human.
//
// Returns entries in "NAME=VALUE" form, in ChildTrustVars order, so the result
// can be appended directly to an exec environment. A nil result means every
// variable was already set (or no bundle was declared) and nothing needs adding.
func ChildEnv(path string, parent []string) []string {
	path = strings.TrimSpace(path)
	if path == "" {
		return nil
	}
	present := make(map[string]struct{}, len(parent))
	for _, kv := range parent {
		if i := strings.IndexByte(kv, '='); i > 0 && strings.TrimSpace(kv[i+1:]) != "" {
			present[envName(kv[:i])] = struct{}{}
		}
	}
	var out []string
	for _, name := range ChildTrustVars {
		if _, already := present[envName(name)]; already {
			continue
		}
		out = append(out, name+"="+path)
	}
	return out
}

// UninheritedRuntimes reports which ChildTrustVars a child would NOT receive
// given the environment that will actually be handed to it — the concrete answer
// to the doctor check's "which child runtimes would not inherit fak's trust".
// Sorted for a stable report.
func UninheritedRuntimes(childEnv []string) []string {
	present := make(map[string]struct{}, len(childEnv))
	for _, kv := range childEnv {
		if i := strings.IndexByte(kv, '='); i > 0 && strings.TrimSpace(kv[i+1:]) != "" {
			present[envName(kv[:i])] = struct{}{}
		}
	}
	var out []string
	for _, name := range ChildTrustVars {
		if _, ok := present[envName(name)]; !ok {
			out = append(out, name)
		}
	}
	sort.Strings(out)
	return out
}

// TLSTrustHint turns a transport error into the operator-facing fix, or "" when
// the error is not a trust failure.
//
// This exists because the diagnosis was already correct and already thrown away.
// internal/agent's deterministicTransportError has always matched
// *tls.CertificateVerificationError and tls.RecordHeaderError and declined to
// retry them — the right call, made for the right reason — but the verdict stayed
// inside the retry decision. The operator saw an opaque non-retried transport
// failure and read it as "the endpoint is firewalled". One string closes that gap:
// the same condition the retry layer identifies precisely now names the knob that
// fixes it, and names whether a trust source is already declared, because
// "declared and still failing" and "nothing declared" need different next steps.
func TLSTrustHint(err error) string {
	if err == nil {
		return ""
	}
	var certErr *tls.CertificateVerificationError
	if !errors.As(err, &certErr) {
		return ""
	}
	src := Resolve()
	switch {
	case src.Usable():
		return fmt.Sprintf("the certificate chain did not validate even with the CA bundle declared in %s (%s, %d root(s): %s) — the intercepting CA for THIS endpoint is probably a different one than the bundle covers",
			BundleKey, src.Path, len(src.Bundle.Subjects), SubjectSummary(src.Bundle.Subjects))
	case src.Declared():
		return fmt.Sprintf("%s is set to %s but it did not load (%v), so fak is validating against the platform trust store only", BundleKey, src.Path, src.Err)
	default:
		return fmt.Sprintf("this is TLS interception, not a firewall: a private CA re-signed the connection and no trust source is declared. Point %s at a PEM file holding your corporate root and fak will validate against the platform store PLUS that root (and derive %s for child runtimes)",
			BundleKey, strings.Join(ChildTrustVars, "/"))
	}
}

// envName canonicalizes an environment variable name for comparison. Windows env
// names are case-insensitive and Unix names are not; upper-casing everywhere is
// the safe direction here because every name in ChildTrustVars is already
// upper-case, so folding can only match the variable we mean.
func envName(k string) string { return strings.ToUpper(strings.TrimSpace(k)) }
