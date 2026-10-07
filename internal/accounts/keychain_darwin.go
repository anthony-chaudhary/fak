//go:build darwin

package accounts

// keychain_darwin.go — the ONLY platform-specific piece of the keychain source
// (#5363): wire Apple's `security` CLI (and `sw_vers` for diagnostics) into the seams.
// Everything above the exec (service naming, parsing, expiry semantics, the TTL cache,
// and the typed missing/timeout/error classification of fak-private#3063) is
// platform-neutral in keychain.go and keychain_state.go.

func init() {
	claudeKeychainReadPassword = securityReadPassword(execKeychainRunner, claudeKeychainReadTimeout)
	keychainOSVersionFunc = func() string { return swVersProductVersion(execKeychainRunner) }
}
