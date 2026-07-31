package main

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

// parseSeatExtraEnv turns the repeatable `--env KEY=VALUE` list into the map a seat stores,
// then validates it. It is fail-loud on every ambiguity rather than guessing:
//
//   - a missing `=` is a usage error, not an empty value — `--env FOO` almost always means
//     the operator meant to pass a value and lost it to shell quoting
//   - a duplicate KEY is an error, since silently keeping the first or last would make the
//     resulting seat depend on flag order
//   - an empty VALUE is ALLOWED: exporting a variable as empty is a real posture, and it is
//     distinguishable from unset
//
// Validation is accounts.ValidateExtraEnv, so `add` and `launch` refuse on identical terms
// and there is exactly one definition of what may live in a seat's overlay.
func parseSeatExtraEnv(pairs []string) (map[string]string, error) {
	if len(pairs) == 0 {
		return nil, nil
	}
	out := make(map[string]string, len(pairs))
	for _, kv := range pairs {
		key, val, ok := strings.Cut(kv, "=")
		if !ok {
			return nil, fmt.Errorf("--env %q: want KEY=VALUE (no '=' found)", kv)
		}
		key = strings.TrimSpace(key)
		if _, dup := out[key]; dup {
			return nil, fmt.Errorf("--env %s given twice; the seat's value would depend on flag order", key)
		}
		out[key] = val
	}
	if err := accounts.ValidateExtraEnv(out); err != nil {
		return nil, err
	}
	return out, nil
}

// Launching a THIRD-PARTY-endpoint seat: the environment it gets, and the one posture it
// cannot have. Both live here as functions rather than inline in runAccountsLaunch so the
// decisions are testable without spawning an agent.

// thirdPartyGuardConflict reports whether this seat and this guard posture are
// incompatible, and why.
//
// A seat carrying base_url names a vendor gateway that authenticates its OWN tenant
// credential. `fak guard` binds an in-process gateway and fronts the child with its own
// ANTHROPIC_BASE_URL at that loopback, proxying upstream with the credential GUARD holds —
// so under guard the seat's endpoint is not merely unused, it is replaced, and the traffic
// bills a different account than the operator named. That failure is invisible from the
// outside: the agent starts, answers, and looks healthy.
//
// So this is a REFUSAL, not a warning. Silently routing somewhere the operator did not ask
// for is the outcome worth preventing, and the fix is one flag the message names.
func thirdPartyGuardConflict(home accounts.Home, useGuard bool) (string, bool) {
	if !home.ThirdParty() || !useGuard {
		return "", false
	}
	return fmt.Sprintf("seat %q carries base_url (%s), a third-party Anthropic-compatible endpoint, but guard fronts the child "+
		"with its OWN ANTHROPIC_BASE_URL at guard's loopback gateway and proxies with guard's credential — the seat's endpoint "+
		"would be replaced and a different upstream billed.\n  relaunch with --guard=false to reach the seat's endpoint directly "+
		"(no kernel adjudication, no vCache hop)", home.Name, home.BaseURL), true
}

// firstPartyCredentialEnv are the variables the agent binary will read as an Anthropic
// credential. They are the collision set for a third-party seat: the endpoint is overridden
// by the overlay, but an INHERITED credential is not, and the agent may prefer it over the
// one the seat names — presenting the wrong tenant's token to the vendor gateway.
var firstPartyCredentialEnv = []string{
	"ANTHROPIC_API_KEY",
	"ANTHROPIC_AUTH_TOKEN",
	"CLAUDE_CODE_OAUTH_TOKEN",
}

// thirdPartySeatModel resolves the model id a THIRD-PARTY seat's launch pins, and reports
// whether it changed the caller's posture and why.
//
// `fak accounts launch` deliberately pins a first-party model by default (defaultLaunchModel)
// so every seat starts on the same one regardless of its saved default, with a fallback CHAIN
// of further first-party ids behind it. For a vendor endpoint that default is not merely
// unhelpful, it is wrong twice over: the vendor serves its own model namespace, so `--model
// claude-opus-5` names a model that does not exist there, and each fallback names another one
// — turning one clean startup failure into a walk through a chain of ids the endpoint was
// never going to serve. The seat already carries the right answer in its own overlay
// ($ANTHROPIC_MODEL / $ANTHROPIC_DEFAULT_SONNET_MODEL), which is exactly what the empty
// --model defers to.
//
// So an UNSET --model resolves to "" (the seat's own default) rather than the first-party
// default. Because modelFallbackChain already declines on an empty primary, that one
// substitution also disables the first-party chain — no second knob, no way for the two to
// disagree.
//
// An EXPLICIT --model always wins, including for a third-party seat: an operator naming a
// vendor model id (or probing what the endpoint serves) is the case this must not fight. That
// is the same rule modelFallbackChain follows — respect an explicit choice, don't second-guess
// it — and it is why this keys off modelExplicit rather than comparing against the default id.
func thirdPartySeatModel(home accounts.Home, model string, modelExplicit bool) (resolved, why string, changed bool) {
	if !home.ThirdParty() || modelExplicit || strings.TrimSpace(model) == "" {
		return model, "", false
	}
	return "", fmt.Sprintf("seat %q is a third-party endpoint; not pinning the first-party default --model %q "+
		"(nor its fallback chain) since that namespace is the vendor's — deferring to the seat's own "+
		"$ANTHROPIC_MODEL. Pass --model explicitly to override.", home.Name, strings.TrimSpace(model)), true
}

// seatDeclaredCredentialEnv names the credential variables this seat's launch is configured
// to authenticate with, for the spawn broker's secret floor to pass through
// (newLaunchBrokerAttemptDeclaring). Everything else the floor holds out as before.
//
// Without this the floor and the launch contradict each other, and it is worth being precise
// about how: policy.StripInheritedSecrets drops any variable whose NAME contains TOKEN (and
// any whose VALUE looks like a secret), so a seat whose credential is $ANTHROPIC_AUTH_TOKEN —
// the bearer variable every third-party Anthropic-compatible gateway authenticates — has its
// `--api-key-env ANTHROPIC_AUTH_TOKEN` REFERENCE spliced into the child's argv while the
// value it points at is stripped from the child's environment. The launch then presents no
// credential at all and the upstream answers "Not logged in", which reads as a bad token or a
// bad endpoint rather than as a boundary doing its job. $ANTHROPIC_API_KEY escapes this only
// by accident of being name-shaped innocently and value-exempted by providerAPIKeyNames.
//
// The set is exactly two things, deduped, empties dropped:
//
//   - launchSeatAPIKeyEnv(home) — the variable this launch TELLS guard to bill via
//     --api-key-env, whether that came from the seat or the fleet-wide knob. Telling guard to
//     bill $X while stripping $X from guard's own environment cannot be right; declaring
//     precisely what the argv already references keeps the two in lockstep with no new policy.
//   - a third-party seat's own APIKeyEnv, since a vendor endpoint has no subscription path to
//     fall back on: that variable is the only credential the launch can possibly use.
//
// This is narrow on purpose. It declares only what the seat itself names, so an unrelated
// ambient secret is still held out, and the exemption does not follow the child to ITS spawns
// (the floor runs again per surface). A seat that names nothing declares nothing and its
// launch environment is bit-for-bit what it was before.
func seatDeclaredCredentialEnv(home accounts.Home) []string {
	var out []string
	seen := map[string]bool{}
	add := func(name string) {
		if name = strings.TrimSpace(name); name != "" && !seen[name] {
			seen[name] = true
			out = append(out, name)
		}
	}
	add(launchSeatAPIKeyEnv(home))
	if home.ThirdParty() {
		add(home.APIKeyEnv)
	}
	return out
}

// launchSeatEnv composes the environment an account-switched launch runs with, in strict
// precedence order (later wins, since the env is collapsed through a map keyed by name):
//
//	base            — the ambient environment the operator launched from, minus the
//	                  scrubbed credentials described below
//	seat overlay    — the seat's endpoint and extra env; naming a seat is how you override
//	                  the shell, so a seat's value beats an inherited one
//	CLAUDE_CONFIG_DIR — fak-owned and therefore LAST: the variable that makes this a seat
//	                  launch at all cannot be displaced, even if validation is bypassed
//
// For a THIRD-PARTY seat it also returns the inherited credential variables it dropped.
// This is not hygiene, it is correctness: a `fak guard` session exports its own
// ANTHROPIC_API_KEY (and ANTHROPIC_BASE_URL) into every child, so a seat launched from one
// inherits a first-party credential alongside the vendor endpoint. Overriding the endpoint
// alone is not enough — the agent can still pick the inherited key and send the wrong
// tenant's credential to the vendor, which reads as an auth failure against a correct
// config. The variable the seat itself names (APIKeyEnv) is of course kept: that IS the
// seat's credential. A first-party seat scrubs nothing, so its launch is unchanged.
//
// base is copied rather than appended to in place, so a caller's slice (os.Environ()'s
// backing array) is never aliased into the result.
func launchSeatEnv(base []string, home accounts.Home) (env, scrubbed []string) {
	drop := map[string]bool{}
	if home.ThirdParty() {
		keep := strings.TrimSpace(home.APIKeyEnv)
		for _, name := range firstPartyCredentialEnv {
			if name != keep {
				drop[name] = true
			}
		}
	}
	overlay := home.EnvOverlay()
	env = make([]string, 0, len(base)+len(overlay)+1)
	for _, kv := range base {
		name, _, _ := strings.Cut(kv, "=")
		if drop[name] {
			scrubbed = append(scrubbed, name)
			continue
		}
		env = append(env, kv)
	}
	env = append(env, overlay...)
	return append(env, "CLAUDE_CONFIG_DIR="+home.Dir), scrubbed
}
