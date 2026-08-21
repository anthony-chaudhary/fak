// Package httptrust is the corporate-trust seam: one declared CA bundle source
// honored by every fak-originated HTTPS client and derived for the child runtimes
// a fak session launches.
//
// # Why this exists
//
// On a TLS-intercepting corporate network a private root re-signs every HTTPS
// connection. Nothing is blocked — the chain simply does not validate against the
// default trust store — and every runtime reports it differently: Go says
// "x509: certificate signed by unknown authority", Node says
// SELF_SIGNED_CERT_IN_CHAIN, Windows curl says CRYPT_E_NO_REVOCATION_CHECK, and
// OpenSSL says "self-signed certificate in certificate chain". All four read as
// "the network is down". Before this package there was no RootCAs or
// x509.SystemCertPool anywhere in the tree and every outbound client was a bare
// &http.Client{Timeout: …}, so a box whose curl, pip, and npm were all working
// had no route to make fak work — least of all on Windows, where crypto/x509
// consults the OS store only and SSL_CERT_FILE is a Unix-only knob.
//
// # The two rules
//
// WIDEN, NEVER REPLACE. The pool is always x509.SystemCertPool() with the bundle
// APPENDED. A site typically runs more than one interceptor, so a pool built from
// the bundle alone would trade one broken endpoint for every endpoint the OS store
// was already validating. A platform pool that cannot be obtained is a hard error,
// not a fallback to a fresh pool.
//
// NO --insecure SIBLING. Interception is a trust problem. Skipping verification
// would "fix" it by removing the property a governance tool exists to assert, so
// the only supported input is a real certificate an operator can point at.
//
// # Tier
//
// Tier: foundation-composite (2) - see internal/architest. This package may
// import only packages whose tier is <= 2; an upward import fails the architest
// gate. It imports internal/secretload(2) for config resolution and stdlib.
// See AGENTS.md and internal/architest for the layering contract.
package httptrust
