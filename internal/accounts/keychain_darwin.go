//go:build darwin

package accounts

import (
	"context"
	"os/exec"
	"time"
)

// keychain_darwin.go — the ONLY platform-specific piece of the keychain source
// (#5363): exec Apple's `security` CLI to read the generic-password value Claude Code
// stored. Everything above the exec (service naming, parsing, expiry semantics, the TTL
// cache) is platform-neutral in keychain.go.

// claudeKeychainReadTimeout bounds one `security` exec. An item that does not exist
// returns in milliseconds with no prompt; the slow case is an item whose ACL makes
// securityd raise the GUI consent dialog (fak is not in the ACL Claude Code created, so
// the FIRST read on a box prompts until the operator clicks "Always Allow"). 15s gives
// an attended operator time to click; an unattended box times out and the probe misses —
// the same needs_login answer it gave before this source existed, never a wedge.
const claudeKeychainReadTimeout = 15 * time.Second

func init() {
	claudeKeychainReadPassword = func(service, account string) ([]byte, error) {
		ctx, cancel := context.WithTimeout(context.Background(), claudeKeychainReadTimeout)
		defer cancel()
		// -w prints the password only (never item metadata), so the value can flow
		// straight into the JSON parser; the absolute path pins Apple's binary.
		out, err := exec.CommandContext(ctx,
			"/usr/bin/security", "find-generic-password",
			"-a", account, "-w", "-s", service).Output()
		if err != nil {
			return nil, err
		}
		return out, nil
	}
}
