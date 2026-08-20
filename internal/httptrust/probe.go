// probe.go is the observing half of `fak doctor trust`: ONE bounded TLS handshake
// per endpoint, reported as facts assess.go can judge.
//
// The important thing about this file is what it does NOT contain. Naming the
// interceptor's CA is normally done with `openssl s_client` or an
// InsecureSkipVerify dial — and adding a skip-verify code path to a governance tool
// so it can print a nicer diagnostic is exactly the escape hatch #8172 argues
// against, because that path never stays diagnostic-only. It is unnecessary anyway:
// a failed verification in Go returns *tls.CertificateVerificationError, which
// carries UnverifiedCertificates — the chain the server actually presented. So the
// full diagnosis comes out of a handshake that verified normally and REFUSED the
// connection. There is no InsecureSkipVerify anywhere in this package.

package httptrust

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"errors"
	"net"
	"strings"
	"time"
)

// DefaultProbeTimeout bounds one handshake. A doctor check must terminate on a host
// whose proxy black-holes the connection, which is a thing intercepting proxies
// do — so every dial is deadlined and a timeout is reported as "no verdict", never
// as a trust failure.
const DefaultProbeTimeout = 4 * time.Second

// ProbeHost performs one bounded TLS handshake against host ("host:port") and
// reports what the platform trust store made of the chain, what CA the chain
// terminates at, and — when the platform store rejected it and pool is non-nil —
// whether the declared bundle rescues it.
//
// A non-verification failure (DNS, refused, timeout, no route) is reported as
// Unreached with Reached=false. That distinction is the whole reason an offline or
// air-gapped host produces no findings from this check: "I could not connect" is not
// evidence about trust, and reporting it as such would make the check cry wolf
// everywhere except the one environment it was written for.
func ProbeHost(ctx context.Context, host string, pool *x509.CertPool, timeout time.Duration) ProbeResult {
	if timeout <= 0 {
		timeout = DefaultProbeTimeout
	}
	res := ProbeResult{Host: host}
	st, err := handshake(ctx, host, nil, timeout)
	if err == nil {
		res.Reached, res.DefaultOK = true, true
		res.RootLabel, res.LeafIssuer = chainLabels(st.PeerCertificates)
		return res
	}
	var certErr *tls.CertificateVerificationError
	if !errors.As(err, &certErr) {
		res.Unreached = err.Error()
		return res
	}
	res.Reached = true
	res.VerifyErr = verificationDetail(certErr)
	res.RootLabel, res.LeafIssuer = chainLabels(certErr.UnverifiedCertificates)
	if pool == nil {
		return res
	}
	res.BundleTried = true
	if _, err := handshake(ctx, host, pool, timeout); err == nil {
		res.BundleOK = true
	}
	return res
}

// handshake dials host and completes the TLS handshake with RootCAs set to pool (nil
// = the platform default), returning the connection state. It closes the connection
// immediately: the handshake is the entire measurement, and a diagnostic must not
// leave sockets open against an upstream it is only inspecting.
func handshake(ctx context.Context, host string, pool *x509.CertPool, timeout time.Duration) (*tls.ConnectionState, error) {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	d := &tls.Dialer{
		NetDialer: &net.Dialer{Timeout: timeout},
		Config:    &tls.Config{RootCAs: pool, MinVersion: tls.VersionTLS12},
	}
	conn, err := d.DialContext(ctx, "tcp", host)
	if err != nil {
		return nil, err
	}
	defer conn.Close()
	tc, ok := conn.(*tls.Conn)
	if !ok {
		return nil, errors.New("httptrust: dialer returned a non-TLS connection")
	}
	st := tc.ConnectionState()
	return &st, nil
}

// verificationDetail renders the platform's own verification failure. The wrapped
// error is preferred over the wrapper's text because the wrapped string is the one
// the operator has already seen from curl, npm, or the AWS CLI — matching it is what
// lets them recognize their problem in fak's output.
func verificationDetail(err *tls.CertificateVerificationError) string {
	if err == nil {
		return ""
	}
	if err.Err != nil {
		return err.Err.Error()
	}
	return err.Error()
}

// chainLabels names the trust anchor the presented chain terminates at, and the CA
// that signed the leaf.
//
// The anchor is the ISSUER of the outermost certificate presented, not its subject:
// a server normally sends leaf+intermediates and omits the root, so the outermost
// certificate's issuer is the name of the certificate the verifier had to find in a
// trust store. A self-signed root that IS sent has Issuer == Subject, so the same
// read is correct in both shapes.
func chainLabels(chain []*x509.Certificate) (root, leafIssuer string) {
	if len(chain) == 0 {
		return "", ""
	}
	return NameLabel(chain[len(chain)-1].Issuer), NameLabel(chain[0].Issuer)
}

// NameLabel renders the most operator-recognizable form of an X.509 name: the Common
// Name, else the first Organization, else the full RDN string.
func NameLabel(n pkix.Name) string {
	if cn := strings.TrimSpace(n.CommonName); cn != "" {
		return cn
	}
	for _, o := range n.Organization {
		if o = strings.TrimSpace(o); o != "" {
			return o
		}
	}
	return n.String()
}
