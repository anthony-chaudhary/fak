package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

// serve_upstream_key.go — durable upstream-key resolution for the always-on gateway
// (fak-private#3063). The launchd unit used to be told to read ANTHROPIC_API_KEY from
// `launchctl setenv`, which is wiped on reboot: the gateway then came back keyless and
// served upstream 401s until someone noticed. With --require-upstream-key, an empty
// --api-key-env variable falls back to the macOS login Keychain — the existing keychain
// reader in internal/accounts, so no new secret store — and a gateway that still has no
// key refuses to start with a "no gateway key" error. The key value is never written to
// the plist, never logged, and only its SOURCE is reported.
//
// Resolution order, first hit wins:
//  1. the --api-key-env variable itself (unchanged behavior);
//  2. a fak-owned generic-password item named fak-<ENV> (any account) — the durable
//     store the operator guidance points at;
//  3. for ANTHROPIC_API_KEY only, Claude Code's own saved API key for the default config
//     home (the item `fak guard` already adopts, #5363).
//
// Without --require-upstream-key, step 1 is the whole story — byte-for-byte the prior
// behavior, including an empty key that leaves the hop in client-credential passthrough.

// serveUpstreamKeychainService is the fak-owned Keychain service holding the durable copy
// of the key for envName, e.g. fak-ANTHROPIC_API_KEY.
func serveUpstreamKeychainService(envName string) string { return "fak-" + envName }

// serveUpstreamKeyStoreHint is the operator command that creates the durable copy. `-w`
// as the LAST argument with no value makes `security` prompt for the secret, so the key
// never lands in shell history or a process listing.
func serveUpstreamKeyStoreHint(envName string) string {
	return fmt.Sprintf("security add-generic-password -U -a \"$USER\" -s %s -w", serveUpstreamKeychainService(envName))
}

// serveUpstreamKeySources are the seams resolveServeUpstreamKey reads, so tests drive it
// with fakes and never touch the process env or the real Keychain.
type serveUpstreamKeySources struct {
	getenv       func(string) string
	keychainItem func(service string) (string, accounts.KeychainProbe)
	claudeAPIKey func(dir string) (string, accounts.KeychainProbe)
	claudeHome   func() string
}

func defaultServeUpstreamKeySources() serveUpstreamKeySources {
	return serveUpstreamKeySources{
		getenv:       os.Getenv,
		keychainItem: accounts.KeychainSecret,
		claudeAPIKey: accounts.ClaudeKeychainAPIKeyProbe,
		claudeHome: func() string {
			if d := strings.TrimSpace(os.Getenv("CLAUDE_CONFIG_DIR")); d != "" {
				return d
			}
			home, err := os.UserHomeDir()
			if err != nil || home == "" {
				return ""
			}
			return filepath.Join(home, ".claude")
		},
	}
}

// resolveServeUpstreamKey returns the upstream key for envName, the non-secret name of
// the source that supplied it, and — when nothing did — one line per durable source
// saying why it missed. durable=false consults only the env var.
func resolveServeUpstreamKey(envName string, durable bool, src serveUpstreamKeySources) (key, source string, misses []string) {
	if envName == "" {
		return "", "", nil
	}
	if v := src.getenv(envName); v != "" {
		return v, "env $" + envName, nil
	}
	if !durable {
		return "", "", nil
	}
	misses = append(misses, "$"+envName+" is empty")
	service := serveUpstreamKeychainService(envName)
	k, probe := src.keychainItem(service)
	if probe.OK() {
		return k, "macOS Keychain item " + service, nil
	}
	if probe.State == accounts.KeychainUnsupported {
		misses = append(misses, "no macOS login Keychain on this platform")
		return "", "", misses
	}
	misses = append(misses, probe.Reason())
	if envName == "ANTHROPIC_API_KEY" {
		if dir := src.claudeHome(); dir != "" {
			k, probe := src.claudeAPIKey(dir)
			if probe.OK() {
				return k, "Claude Code's saved API key in the macOS Keychain", nil
			}
			misses = append(misses, "Claude Code saved API key: "+probe.Reason())
		}
	}
	return "", "", misses
}

// resolveServeAPIKey is the serve-startup wrapper: resolve, report a durable source on
// stderr (name only), and fail fast with a "no gateway key" bail when
// --require-upstream-key found nothing.
func resolveServeAPIKey(envName string, required bool) string {
	if required && envName == "" {
		writeConfigBail(os.Stderr, configBail{
			Verb: "fak serve", Reason: bailKeyEnvUnset,
			Summary: "--require-upstream-key needs --api-key-env naming the upstream key variable",
			Knobs:   []bailKnob{bailFlag("api-key-env", "").want("e.g. ANTHROPIC_API_KEY")},
		})
		os.Exit(2)
	}
	key, source, misses := resolveServeUpstreamKey(envName, required, defaultServeUpstreamKeySources())
	if key != "" {
		if required && source != "env $"+envName {
			fmt.Fprintf(os.Stderr, "fak serve: upstream key from %s ($%s was empty)\n", source, envName)
		}
		return key
	}
	if required {
		writeConfigBail(os.Stderr, serveNoGatewayKeyBail(envName, misses))
		os.Exit(2)
	}
	return ""
}

// serveNoGatewayKeyBail renders the fail-fast refusal: what was tried, why each durable
// source missed, and the one command that makes the key survive a reboot.
func serveNoGatewayKeyBail(envName string, misses []string) configBail {
	return configBail{
		Verb:   "fak serve",
		Reason: bailKeyEnvUnset,
		Summary: "no gateway key: refusing to start a gateway that would answer every turn with an upstream 401 (" +
			strings.Join(misses, "; ") + ")",
		Knobs: []bailKnob{
			bailFlag("api-key-env", envName),
			bailEnv(envName, "").want("the upstream API key"),
		},
		Check: "store the key durably (survives reboot; never put it in the plist): " + serveUpstreamKeyStoreHint(envName),
		Bind:  []string{"env=" + envName},
	}
}
