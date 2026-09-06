package httptrust

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"errors"
	"fmt"
	"math/big"
	"net/http"
	"strings"
	"testing"
	"time"
)

var (
	benchSinkReport    Report
	benchSinkStrings   []string
	benchSinkString    string
	benchSinkSiblings  []SiblingVar
	benchSinkBool      bool
	benchSinkTransport *http.Transport
	benchSinkBundle    *Bundle
)

var (
	benchBaseEnv = []string{
		"PATH=/usr/local/bin:/usr/bin:/bin",
		"HOME=/home/agent",
		"USER=agent",
		"SHELL=/bin/bash",
		"TERM=xterm-256color",
		"LANG=en_US.UTF-8",
		"LC_ALL=en_US.UTF-8",
		"PWD=/work/fak",
		"GIT_AUTHOR_NAME=Agent",
		"GIT_AUTHOR_EMAIL=agent@fak.internal",
		"EDITOR=nano",
		"TMPDIR=/tmp",
		"HOSTNAME=devbox-01",
		"COLORTERM=truecolor",
		"NODE_ENV=production",
		"GOPATH=/home/agent/go",
		"GOROOT=/usr/local/go",
		"SHLVL=1",
	}

	benchHostSiblingEnv = append(append([]string(nil), benchBaseEnv...),
		"aws_ca_bundle=/etc/corp/bundle.pem",
		"NODE_EXTRA_CA_CERTS=/etc/corp/bundle.pem",
		"CURL_CA_BUNDLE=/etc/corp/bundle.pem",
		"REQUESTS_CA_BUNDLE=",
		"AWS_SECRET_ACCESS_KEY=sensitive-should-not-leak",
	)

	benchEnvFull = append(append([]string(nil), benchBaseEnv...),
		"NODE_EXTRA_CA_CERTS=/etc/corp/bundle.pem",
		"AWS_CA_BUNDLE=/etc/corp/bundle.pem",
		"CURL_CA_BUNDLE=/etc/corp/bundle.pem",
		"SSL_CERT_FILE=/etc/corp/bundle.pem",
		"REQUESTS_CA_BUNDLE=/etc/corp/bundle.pem",
		"GIT_SSL_CAINFO=/etc/corp/bundle.pem",
	)

	benchEnvPartial = append(append([]string(nil), benchBaseEnv...),
		"NODE_EXTRA_CA_CERTS=/etc/corp/bundle.pem",
		"AWS_CA_BUNDLE=/etc/corp/bundle.pem",
	)

	benchSubjectsSmall = []string{
		"Example Corp Root CA",
		"Example Corp Intermediate CA",
	}

	benchSubjectsLarge []string

	benchPEMSingle []byte
	benchPEMMulti  []byte

	benchCertWithCN  *x509.Certificate
	benchCertWithOrg *x509.Certificate
	benchPKIXName    pkix.Name

	benchBundleWithPool *Bundle

	benchFactsClean                   Facts
	benchFactsInterceptedUnverifiable Facts
	benchFactsRescued                 Facts
	benchFactsWindowsPlatform         Facts
	benchFactsMultiProbe              Facts
)

func generateBenchmarkCertPEM(cn, org string, notAfter time.Time) []byte {
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		panic(err)
	}
	tmpl := &x509.Certificate{
		SerialNumber:          big.NewInt(1),
		Subject:               pkix.Name{CommonName: cn, Organization: []string{org}},
		NotBefore:             time.Now().Add(-time.Hour),
		NotAfter:              notAfter,
		IsCA:                  true,
		BasicConstraintsValid: true,
		KeyUsage:              x509.KeyUsageCertSign,
	}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &key.PublicKey, key)
	if err != nil {
		panic(err)
	}
	return pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der})
}

func init() {
	benchSubjectsLarge = make([]string, 125)
	for i := 0; i < 125; i++ {
		benchSubjectsLarge[i] = fmt.Sprintf("Corp Root %d", i)
	}

	benchPEMSingle = generateBenchmarkCertPEM("Example Corp Root CA", "Example Corp", time.Now().Add(365*24*time.Hour))

	var multiBuilder []byte
	for i := 0; i < 5; i++ {
		multiBuilder = append(multiBuilder, generateBenchmarkCertPEM(fmt.Sprintf("Chain CA %d", i), "Example Corp", time.Now().Add(365*24*time.Hour))...)
	}
	benchPEMMulti = multiBuilder

	benchCertWithCN = &x509.Certificate{
		Subject: pkix.Name{CommonName: "corp-root.internal", Organization: []string{"Internal Corp"}},
	}
	benchCertWithOrg = &x509.Certificate{
		Subject: pkix.Name{Organization: []string{"Fallback Org"}},
	}
	benchPKIXName = pkix.Name{
		CommonName:   "api.proxy.corp",
		Organization: []string{"Global Corp"},
	}

	pool := x509.NewCertPool()
	benchBundleWithPool = &Bundle{
		Path:           "/etc/corp/bundle.pem",
		Pool:           pool,
		Subjects:       benchSubjectsSmall,
		EarliestExpiry: time.Date(2035, 1, 1, 0, 0, 0, 0, time.UTC),
	}

	benchFactsClean = Facts{
		Probes: []ProbeResult{
			{Host: "api.anthropic.com:443", Reached: true, DefaultOK: true, RootLabel: "DigiCert Global Root CA"},
		},
	}

	benchFactsInterceptedUnverifiable = Facts{
		Probes: []ProbeResult{
			{
				Host:      "api.anthropic.com:443",
				Reached:   true,
				DefaultOK: false,
				VerifyErr: "x509: certificate signed by unknown authority",
				RootLabel: "ca.corp-interceptor.goskope.com",
			},
		},
	}

	benchFactsRescued = Facts{
		Source: Source{
			Path:   "/etc/corp/root.pem",
			Bundle: benchBundleWithPool,
		},
		Probes: []ProbeResult{
			{
				Host:        "api.anthropic.com:443",
				Reached:     true,
				DefaultOK:   false,
				VerifyErr:   "x509: certificate signed by unknown authority",
				RootLabel:   "Example Corp Root CA",
				BundleTried: true,
				BundleOK:    true,
			},
		},
		ChildEnv: benchEnvFull,
	}

	benchFactsWindowsPlatform = Facts{
		Source: Source{
			Path:   `C:\corp\ca-bundle.pem`,
			Bundle: benchBundleWithPool,
		},
		Probes: []ProbeResult{
			{
				Host:      "api.anthropic.com:443",
				Reached:   true,
				DefaultOK: true,
				RootLabel: "Example Corp Root CA",
			},
		},
	}

	benchFactsMultiProbe = Facts{
		Source: Source{
			Path:   "/etc/corp/root.pem",
			Bundle: benchBundleWithPool,
		},
		Probes: []ProbeResult{
			{
				Host:        "api.anthropic.com:443",
				Reached:     true,
				DefaultOK:   false,
				VerifyErr:   "x509: certificate signed by unknown authority",
				RootLabel:   "Example Corp Root CA",
				BundleTried: true,
				BundleOK:    true,
			},
			{
				Host:        "sts.amazonaws.com:443",
				Reached:     true,
				DefaultOK:   false,
				VerifyErr:   "x509: certificate signed by unknown authority",
				RootLabel:   "Uncovered-Issuing-CA",
				BundleTried: true,
				BundleOK:    false,
			},
			{
				Host:      "storage.googleapis.com:443",
				Reached:   true,
				DefaultOK: true,
				RootLabel: "Google Trust Services",
			},
			{
				Host:      "internal-isolated.corp:443",
				Reached:   false,
				Unreached: "dial tcp: i/o timeout",
			},
		},
		Siblings: []SiblingVar{
			{Name: "AWS_CA_BUNDLE", Path: "/etc/corp/root.pem"},
			{Name: "NODE_EXTRA_CA_CERTS", Path: "/etc/corp/root.pem"},
		},
		ChildEnv:   benchEnvPartial,
		CloudRoute: "CLAUDE_CODE_USE_BEDROCK",
		Now:        time.Date(2026, 9, 1, 0, 0, 0, 0, time.UTC),
	}
}

// BenchmarkAssess measures pure verdict generation across host trust postures.
func BenchmarkAssess(b *testing.B) {
	b.Run("CleanHost", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkReport = Assess(benchFactsClean)
		}
	})

	b.Run("Intercepted_Unverifiable", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkReport = Assess(benchFactsInterceptedUnverifiable)
		}
	})

	b.Run("Intercepted_RescuedByBundle", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkReport = Assess(benchFactsRescued)
		}
	})

	b.Run("PlatformStore_WithAnchorMatch", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkReport = Assess(benchFactsWindowsPlatform)
		}
	})

	b.Run("MultiProbe_ComplexFacts", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkReport = Assess(benchFactsMultiProbe)
		}
	})
}

// BenchmarkAssessParallel measures concurrent assessment throughput across goroutines.
func BenchmarkAssessParallel(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			rep := Assess(benchFactsMultiProbe)
			if !rep.Intercepted || rep.Warnings == 0 {
				b.Fatalf("unexpected Assess verdict: intercepted=%v, warnings=%d", rep.Intercepted, rep.Warnings)
			}
		}
	})
}

// BenchmarkChildEnv measures per-runtime trust variable derivation against parent environments.
func BenchmarkChildEnv(b *testing.B) {
	b.Run("EmptyParent", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkStrings = ChildEnv("/etc/corp/root.pem", nil)
		}
	})

	b.Run("RealisticParent_NoOverlap", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkStrings = ChildEnv("/etc/corp/root.pem", benchBaseEnv)
		}
	})

	b.Run("RealisticParent_PartialOverlap", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkStrings = ChildEnv("/etc/corp/root.pem", benchEnvPartial)
		}
	})

	b.Run("RealisticParent_FullOverlap", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkStrings = ChildEnv("/etc/corp/root.pem", benchEnvFull)
		}
	})
}

// BenchmarkUninheritedRuntimes measures the inspection of child environment inheritance gaps.
func BenchmarkUninheritedRuntimes(b *testing.B) {
	b.Run("CompleteChildEnv", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkStrings = UninheritedRuntimes(benchEnvFull)
		}
	})

	b.Run("PartialChildEnv", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkStrings = UninheritedRuntimes(benchEnvPartial)
		}
	})

	b.Run("EmptyChildEnv", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkStrings = UninheritedRuntimes(benchBaseEnv)
		}
	})
}

// BenchmarkHostSiblingTrustVars measures scanning host environment snapshots for corporate trust vars.
func BenchmarkHostSiblingTrustVars(b *testing.B) {
	b.Run("CleanHost", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkSiblings = HostSiblingTrustVars(benchBaseEnv)
		}
	})

	b.Run("CorporateHost_MixedVars", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkSiblings = HostSiblingTrustVars(benchHostSiblingEnv)
		}
	})
}

// BenchmarkSubjectSummary measures operator-facing subject formatting with cap rules.
func BenchmarkSubjectSummary(b *testing.B) {
	b.Run("Small_UnderCap", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkString = SubjectSummary(benchSubjectsSmall)
		}
	})

	b.Run("LargeCorporate_OverCap", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkString = SubjectSummary(benchSubjectsLarge)
		}
	})
}

// BenchmarkSubjectLabel measures certificate name formatting and field priority fallback.
func BenchmarkSubjectLabel(b *testing.B) {
	b.Run("CommonName", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkString = SubjectLabel(benchCertWithCN)
		}
	})

	b.Run("OrganizationFallback", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkString = SubjectLabel(benchCertWithOrg)
		}
	})

	b.Run("NameLabel_PKIX", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkString = NameLabel(benchPKIXName)
		}
	})
}

// BenchmarkDescribeBundle measures PEM block decoding, cert parsing, and expiry analysis.
func BenchmarkDescribeBundle(b *testing.B) {
	b.Run("SingleCert", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			bundle := &Bundle{Path: "/etc/corp/single.pem"}
			describeBundle(bundle, benchPEMSingle)
			benchSinkBundle = bundle
		}
	})

	b.Run("MultiCert_5Certs", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			bundle := &Bundle{Path: "/etc/corp/chain.pem"}
			describeBundle(bundle, benchPEMMulti)
			benchSinkBundle = bundle
		}
	})
}

// BenchmarkTransportForBundle measures cloning http.DefaultTransport and installing RootCAs.
func BenchmarkTransportForBundle(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkTransport = TransportForBundle(benchBundleWithPool)
	}
}

// BenchmarkTLSTrustHint measures transport error matching and operator remedy generation.
func BenchmarkTLSTrustHint(b *testing.B) {
	b.Run("NilError", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkString = TLSTrustHint(nil)
		}
	})

	b.Run("NonTLSError", func(b *testing.B) {
		err := errors.New("connection reset by peer")
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkString = TLSTrustHint(err)
		}
	})

	b.Run("CertificateVerificationError", func(b *testing.B) {
		certErr := &tls.CertificateVerificationError{
			Err: errors.New("x509: certificate signed by unknown authority"),
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkString = TLSTrustHint(certErr)
		}
	})
}

// BenchmarkLabelInBundle measures case-insensitive anchor searching in bundle subjects.
func BenchmarkLabelInBundle(b *testing.B) {
	b.Run("Hit_First", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkBool = labelInBundle("Corp Root 0", benchSubjectsLarge)
		}
	})

	b.Run("Hit_Middle", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkBool = labelInBundle("Corp Root 60", benchSubjectsLarge)
		}
	})

	b.Run("Miss", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkBool = labelInBundle("Unknown Root CA", benchSubjectsLarge)
		}
	})
}

// TestBenchmarkOperationsSanity ensures benchmark fixtures and operations function correctly
// during standard test suite execution.
func TestBenchmarkOperationsSanity(t *testing.T) {
	repClean := Assess(benchFactsClean)
	if repClean.Warnings != 0 || repClean.Intercepted {
		t.Fatalf("repClean unexpected: warnings=%d intercepted=%v", repClean.Warnings, repClean.Intercepted)
	}

	repIntercepted := Assess(benchFactsInterceptedUnverifiable)
	if !repIntercepted.Intercepted || repIntercepted.Warnings == 0 {
		t.Fatalf("repIntercepted unexpected: warnings=%d intercepted=%v", repIntercepted.Warnings, repIntercepted.Intercepted)
	}

	repMulti := Assess(benchFactsMultiProbe)
	if !repMulti.Intercepted || repMulti.Warnings == 0 || len(repMulti.Findings) == 0 {
		t.Fatalf("repMulti unexpected: warnings=%d intercepted=%v findings=%d", repMulti.Warnings, repMulti.Intercepted, len(repMulti.Findings))
	}

	childDerived := ChildEnv("/etc/corp/root.pem", benchBaseEnv)
	if len(childDerived) != len(ChildTrustVars) {
		t.Fatalf("ChildEnv derived %d entries, want %d", len(childDerived), len(ChildTrustVars))
	}

	uninherited := UninheritedRuntimes(benchEnvPartial)
	if len(uninherited) == 0 {
		t.Fatal("expected uninherited runtimes for partial env")
	}

	siblings := HostSiblingTrustVars(benchHostSiblingEnv)
	if len(siblings) != 3 {
		t.Fatalf("HostSiblingTrustVars got %d siblings, want 3", len(siblings))
	}

	summarySmall := SubjectSummary(benchSubjectsSmall)
	if !strings.Contains(summarySmall, "Example Corp Root CA") {
		t.Fatalf("summarySmall missing expected root: %q", summarySmall)
	}

	summaryLarge := SubjectSummary(benchSubjectsLarge)
	if !strings.Contains(summaryLarge, "and 119 more") {
		t.Fatalf("summaryLarge missing remaining count: %q", summaryLarge)
	}

	singleBundle := &Bundle{Path: "/etc/corp/single.pem"}
	describeBundle(singleBundle, benchPEMSingle)
	if len(singleBundle.Subjects) != 1 || singleBundle.Subjects[0] != "Example Corp Root CA" {
		t.Fatalf("describeBundle single subjects: %v", singleBundle.Subjects)
	}

	multiBundle := &Bundle{Path: "/etc/corp/chain.pem"}
	describeBundle(multiBundle, benchPEMMulti)
	if len(multiBundle.Subjects) != 5 {
		t.Fatalf("describeBundle multi subjects len=%d, want 5", len(multiBundle.Subjects))
	}

	tr := TransportForBundle(benchBundleWithPool)
	if tr == nil || tr.TLSClientConfig == nil || tr.TLSClientConfig.RootCAs == nil {
		t.Fatal("TransportForBundle returned unconfigured transport")
	}

	certErr := &tls.CertificateVerificationError{
		Err: errors.New("x509: certificate signed by unknown authority"),
	}
	hint := TLSTrustHint(certErr)
	if !strings.Contains(hint, "TLS interception") {
		t.Fatalf("TLSTrustHint missing expected explanation: %q", hint)
	}

	if !labelInBundle("Corp Root 0", benchSubjectsLarge) {
		t.Fatal("expected labelInBundle true for Corp Root 0")
	}
	if labelInBundle("Unknown Root CA", benchSubjectsLarge) {
		t.Fatal("expected labelInBundle false for Unknown Root CA")
	}
}
