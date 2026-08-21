package httptrust

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"errors"
	"fmt"
	"math/big"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/secretload"
)

// writeBundle writes a self-signed CA PEM (the shape a corporate root arrives in)
// and returns its path plus the Subject CN it should be reported under.
func writeBundle(t *testing.T, cn string, notAfter time.Time) string {
	t.Helper()
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatalf("generate key: %v", err)
	}
	tmpl := &x509.Certificate{
		SerialNumber:          big.NewInt(1),
		Subject:               pkix.Name{CommonName: cn, Organization: []string{"Example Corp"}},
		NotBefore:             time.Now().Add(-time.Hour),
		NotAfter:              notAfter,
		IsCA:                  true,
		BasicConstraintsValid: true,
		KeyUsage:              x509.KeyUsageCertSign,
	}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &key.PublicKey, key)
	if err != nil {
		t.Fatalf("create cert: %v", err)
	}
	path := filepath.Join(t.TempDir(), "corp-root.pem")
	if err := os.WriteFile(path, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}), 0o600); err != nil {
		t.Fatalf("write bundle: %v", err)
	}
	return path
}

// TestLoadWidensThePlatformPool pins the load-bearing property: the pool fak
// validates with is the PLATFORM pool plus the bundle, never the bundle alone. It
// is asserted by subject count — a bundle of one root that produced a pool with
// exactly one subject would mean the platform roots were replaced, which is the
// regression that breaks every endpoint the bundle does not cover.
//
// Windows is the deliberate exception: x509.SystemCertPool() there returns an
// EMPTY pool with a nil error and delegates verification to the platform verifier,
// so subject counting cannot witness the widening. The append still preserves that
// delegation, which is why the check is skipped rather than inverted.
func TestLoadWidensThePlatformPool(t *testing.T) {
	path := writeBundle(t, "Example Corp Root CA", time.Now().Add(24*time.Hour))
	sys, err := x509.SystemCertPool()
	if err != nil {
		t.Skipf("no platform trust store on this host: %v", err)
	}
	b, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if b.Pool == nil {
		t.Fatal("Load returned a nil pool")
	}
	if len(sys.Subjects()) == 0 {
		return // Windows / delegating verifier: nothing to count.
	}
	if got, want := len(b.Pool.Subjects()), len(sys.Subjects())+1; got != want {
		t.Fatalf("pool subjects = %d, want %d (platform pool widened by one root, not replaced)", got, want)
	}
}

func TestLoadDescribesTheBundleForOperators(t *testing.T) {
	expiry := time.Now().Add(72 * time.Hour).Truncate(time.Second)
	b, err := Load(writeBundle(t, "Example Corp Root CA", expiry))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(b.Subjects) != 1 || b.Subjects[0] != "Example Corp Root CA" {
		t.Fatalf("Subjects = %v, want the CN so an operator can match it against their IT-supplied CA", b.Subjects)
	}
	if b.Expired != 0 {
		t.Fatalf("Expired = %d, want 0 for a live root", b.Expired)
	}
	if !b.EarliestExpiry.Equal(expiry) {
		t.Fatalf("EarliestExpiry = %s, want %s", b.EarliestExpiry, expiry)
	}
}

func TestLoadCountsAnExpiredRoot(t *testing.T) {
	b, err := Load(writeBundle(t, "Stale Corp Root CA", time.Now().Add(-time.Hour)))
	if err != nil {
		t.Fatalf("Load: %v (an expired root still loads — the operator is TOLD, not refused)", err)
	}
	if b.Expired != 1 {
		t.Fatalf("Expired = %d, want 1", b.Expired)
	}
}

func TestLoadRefusesAnEmptyOrNonCertificateBundle(t *testing.T) {
	path := filepath.Join(t.TempDir(), "notes.txt")
	if err := os.WriteFile(path, []byte("# the CA is attached to the ticket\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(path); !errors.Is(err, ErrNoCertificates) {
		t.Fatalf("Load error = %v, want ErrNoCertificates so a bundle that contributes nothing is never mistaken for configured trust", err)
	}
}

func TestLoadRefusesAMissingBundle(t *testing.T) {
	if _, err := Load(filepath.Join(t.TempDir(), "absent.pem")); err == nil {
		t.Fatal("Load of a missing bundle must fail, not silently fall back to the platform default")
	}
	if _, err := Load("   "); err == nil {
		t.Fatal("Load of an empty path must fail")
	}
}

// mapSource is a SecretSource backed by an in-memory map, mirroring
// internal/secretload's own test double (an absent value and an empty value are
// both misses, so a zero never shadows a later source).
type mapSource map[string]string

func (s mapSource) Name() string { return "test" }
func (s mapSource) Lookup(k string) (string, bool) {
	v, ok := s[k]
	if !ok || v == "" {
		return "", false
	}
	return v, true
}

// mapLoader builds a loader whose only source is an in-memory map, so the config
// key is exercised without touching the process environment.
func mapLoader(kv map[string]string) *secretload.Loader {
	return secretload.New(mapSource(kv))
}

func TestResolveWithReportsNothingDeclared(t *testing.T) {
	src := ResolveWith(mapLoader(nil))
	if src.Declared() || src.Usable() || src.Err != nil {
		t.Fatalf("unconfigured Source = %+v, want the zero value (the clean box is unchanged)", src)
	}
	c := ClientForSource(src, 7*time.Second)
	if c.Transport != nil {
		t.Fatalf("unconfigured client installed a %T transport; the historical literal has a nil Transport", c.Transport)
	}
	if c.Timeout != 7*time.Second {
		t.Fatalf("Timeout = %s, want 7s", c.Timeout)
	}
}

func TestResolveWithLoadsADeclaredBundleAndInstallsRootCAs(t *testing.T) {
	path := writeBundle(t, "Example Corp Root CA", time.Now().Add(24*time.Hour))
	src := ResolveWith(mapLoader(map[string]string{BundleKey: path}))
	if !src.Declared() || !src.Usable() {
		t.Fatalf("Source = %+v (err %v), want a declared, usable bundle", src, src.Err)
	}
	c := ClientForSource(src, time.Second)
	tr, ok := c.Transport.(*http.Transport)
	if !ok {
		t.Fatalf("Transport = %T, want *http.Transport so http.DefaultTransport's tuned defaults survive", c.Transport)
	}
	if tr.TLSClientConfig == nil || tr.TLSClientConfig.RootCAs != src.Bundle.Pool {
		t.Fatal("RootCAs is not the widened pool")
	}
	// Proxy resolution is the property most easily lost by building a bare
	// transport, and an intercepting site almost always also has a proxy.
	if tr.Proxy == nil {
		t.Fatal("cloned transport lost Proxy; HTTP_PROXY/HTTPS_PROXY would stop being honored")
	}
}

func TestResolveWithReportsADeclaredButBrokenBundle(t *testing.T) {
	src := ResolveWith(mapLoader(map[string]string{BundleKey: filepath.Join(t.TempDir(), "absent.pem")}))
	if !src.Declared() {
		t.Fatal("a declared-but-unloadable bundle must still report Declared() — that is the UPSTREAM_TRUST_UNVERIFIED condition")
	}
	if src.Usable() || src.Err == nil {
		t.Fatalf("Source = %+v, want Usable()==false with a non-nil Err", src)
	}
	if c := ClientForSource(src, time.Second); c.Transport != nil {
		t.Fatal("a broken bundle must not install a transport; the launch path bails instead")
	}
}

func TestChildEnvDerivesEveryRuntimeVariable(t *testing.T) {
	got := ChildEnv("/etc/corp/roots.pem", nil)
	if len(got) != len(ChildTrustVars) {
		t.Fatalf("ChildEnv = %v, want one entry per ChildTrustVars (%d)", got, len(ChildTrustVars))
	}
	for i, name := range ChildTrustVars {
		if want := name + "=/etc/corp/roots.pem"; got[i] != want {
			t.Fatalf("entry %d = %q, want %q (order must be stable)", i, got[i], want)
		}
	}
}

func TestChildEnvNeverOverridesAnOperatorsOwnValue(t *testing.T) {
	parent := []string{"NODE_EXTRA_CA_CERTS=/opt/site/full-chain.pem", "AWS_CA_BUNDLE="}
	got := ChildEnv("/etc/corp/roots.pem", parent)
	for _, kv := range got {
		if strings.HasPrefix(kv, "NODE_EXTRA_CA_CERTS=") {
			t.Fatalf("derived %q over the operator's own value; derivation fills gaps, it never overrides a human", kv)
		}
	}
	// An exported-but-EMPTY value is not a choice, so it IS filled — the same
	// "empty means absent" rule secretload's OSEnv.Lookup already uses.
	var sawAWS bool
	for _, kv := range got {
		if kv == "AWS_CA_BUNDLE=/etc/corp/roots.pem" {
			sawAWS = true
		}
	}
	if !sawAWS {
		t.Fatalf("ChildEnv = %v, want an empty AWS_CA_BUNDLE treated as absent and filled", got)
	}
}

func TestChildEnvIsInertWithoutABundle(t *testing.T) {
	if got := ChildEnv("  ", []string{"PATH=/usr/bin"}); got != nil {
		t.Fatalf("ChildEnv = %v, want nil when no bundle is declared", got)
	}
}

func TestUninheritedRuntimesNamesWhatAChildWouldNotGet(t *testing.T) {
	child := []string{"PATH=/usr/bin", "NODE_EXTRA_CA_CERTS=/etc/corp/roots.pem", "SSL_CERT_FILE=/etc/corp/roots.pem"}
	got := UninheritedRuntimes(child)
	want := []string{"AWS_CA_BUNDLE", "CURL_CA_BUNDLE", "GIT_SSL_CAINFO", "REQUESTS_CA_BUNDLE"}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("UninheritedRuntimes = %v, want %v", got, want)
	}
	if got := UninheritedRuntimes(ChildEnv("/etc/corp/roots.pem", nil)); len(got) != 0 {
		t.Fatalf("UninheritedRuntimes = %v, want empty once ChildEnv has been applied", got)
	}
}

func TestWrapLeavesADeliberateTransportAlone(t *testing.T) {
	existing := &http.Transport{}
	c := &http.Client{Transport: existing}
	if got := Wrap(c); got.Transport != existing {
		t.Fatalf("Wrap replaced a caller-installed %T; that transport is a deliberate choice (observer, recorder, test stub)", existing)
	}
	if Wrap(nil) != nil {
		t.Fatal("Wrap(nil) must be nil-safe")
	}
}

func TestSubjectLabelFallsBackFromCNToOrganization(t *testing.T) {
	if got := SubjectLabel(&x509.Certificate{Subject: pkix.Name{CommonName: "Root"}}); got != "Root" {
		t.Fatalf("SubjectLabel = %q, want the CN", got)
	}
	if got := SubjectLabel(&x509.Certificate{Subject: pkix.Name{Organization: []string{"Example Corp"}}}); got != "Example Corp" {
		t.Fatalf("SubjectLabel = %q, want the first O when the CN is empty", got)
	}
	if got := SubjectLabel(nil); got != "" {
		t.Fatalf("SubjectLabel(nil) = %q, want empty", got)
	}
}

// TestSubjectSummaryCapsARealCorporateBundle pins the dogfood defect: on the
// managed host this seam was built for, FAK_CA_BUNDLE held 123 certificates and
// the trust-source row rendered every subject on one line, burying the two roots
// the operator was actually looking for. A summary that names a few and counts the
// rest keeps the row readable; the full list stays in --json.
func TestSubjectSummaryCapsARealCorporateBundle(t *testing.T) {
	small := []string{"Example Corp Root CA", "Example Corp Issuing CA"}
	if got, want := SubjectSummary(small), strings.Join(small, ", "); got != want {
		t.Fatalf("SubjectSummary(%d) = %q, want the full list unchanged at or under the cap", len(small), got)
	}
	if got := SubjectSummary(nil); got != "" {
		t.Fatalf("SubjectSummary(nil) = %q, want empty", got)
	}

	big := make([]string, 123)
	for i := range big {
		big[i] = fmt.Sprintf("Root %d", i)
	}
	got := SubjectSummary(big)
	if strings.Contains(got, "Root 100") {
		t.Fatalf("SubjectSummary named a subject past the cap: %q", got)
	}
	if !strings.HasPrefix(got, "Root 0, ") || !strings.Contains(got, "Root 5") {
		t.Fatalf("SubjectSummary = %q, want the first %d subjects in file order", got, subjectSummaryMax)
	}
	if want := fmt.Sprintf("and %d more", len(big)-subjectSummaryMax); !strings.Contains(got, want) {
		t.Fatalf("SubjectSummary = %q, want it to count the remainder (%q) rather than truncate silently", got, want)
	}
	if !strings.Contains(got, "--json") {
		t.Fatalf("SubjectSummary = %q, want it to say where the complete list is readable", got)
	}
}
