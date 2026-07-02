package main

import "fmt"

// The managed-cache posture (--managed-cache, epic #1844 C6): should THIS guard session
// actively manage the provider prompt-cache on the outbound Anthropic wire, beyond
// forwarding the client's own cache_control bytes?
//
// Concretely, "active" turns on the stable-prefix 1h cache_control TTL upgrade
// (gateway.Config.CacheTTL1H → agent.UpgradeAnthropicStableCacheTTL1h): the stable
// system/tools head's existing breakpoint is extended from the default 5m tier to 1h, so a
// long session that goes idle past 5 minutes (a human stepping away, a slow tool, a rate-limit
// stall) re-enters on a 0.1x cache READ instead of re-writing the whole prefix at 1.25x.
// The upgrade is byte-safe by construction: it only touches an EXISTING stable-head
// breakpoint, refuses volatile heads, and returns identity on any ambiguity.
//
// Why the AUTO default keys on API-key billing and not on every session: the 1h tier doubles
// the write multiplier (2x vs 1.25x — break-even needs ~3 requests, which a wrapped agent
// session clears in its first minutes). On explicit API billing every one of those multipliers
// is real dollars the operator opted into managing, so fak managing the wire is the point.
// On a Pro/Max subscription the marginal token price is flat and the provider cache already
// rides the client's own breakpoints — fak stays passive there unless forced, so the default
// never speculates with a wire whose economics it cannot see.
const (
	guardManagedCacheAuto = "auto"
	guardManagedCacheOn   = "on"
	guardManagedCacheOff  = "off"
)

// guardManagedCacheInputs is the slice of the resolved upstream posture the managed-cache
// decision reads. Kept as a struct (not positional bools) so the auto rule is testable
// against exactly what cmdGuard resolved.
type guardManagedCacheInputs struct {
	// localModel: the --gguf/--local in-kernel branch — no provider prompt-cache wire exists.
	localModel bool
	// provider is the resolved upstream wire ("anthropic", "openai", ...). Only the
	// Anthropic wire carries cache_control, so only it can be actively managed.
	provider string
	// apiKey is the resolved upstream credential. On the OAuth path it holds the
	// subscription token (oauthSource names where it came from); on the explicit
	// --api-key-env path it holds the API key and oauthSource is empty; on plain
	// passthrough it is empty.
	apiKey string
	// oauthSource is non-empty exactly when the credential is a Pro/Max subscription
	// OAuth token — the flat-rate posture AUTO stays passive on.
	oauthSource string
}

// guardManagedCachePosture is the resolved decision plus the operator-facing reason,
// rendered once in the startup banner so the session's cache posture is explicit instead
// of inferred from flag defaults.
type guardManagedCachePosture struct {
	mode   string
	active bool
	reason string
}

// resolveGuardManagedCache maps --managed-cache MODE onto the session posture.
// AUTO activates only when fak KNOWS the session bills an API key on the Anthropic wire
// (the explicit --api-key-env opt-in resolved a key and no subscription token pinned):
// that is the one posture where every cache write/read multiplier is operator dollars and
// fak owns the outbound bytes. Everything it cannot prove stays passive — never speculate
// with someone else's billing.
func resolveGuardManagedCache(mode string, in guardManagedCacheInputs) (guardManagedCachePosture, error) {
	switch mode {
	case guardManagedCacheOff:
		return guardManagedCachePosture{mode: mode, active: false, reason: "disabled by --managed-cache off"}, nil
	case guardManagedCacheOn:
		return guardManagedCachePosture{mode: mode, active: true, reason: "forced by --managed-cache on"}, nil
	case guardManagedCacheAuto, "":
		p := guardManagedCachePosture{mode: guardManagedCacheAuto}
		switch {
		case in.localModel:
			p.reason = "local in-kernel model — no provider prompt-cache wire"
		case in.provider != "anthropic":
			p.reason = fmt.Sprintf("provider %q has no cache_control wire", in.provider)
		case in.oauthSource != "":
			p.reason = "subscription OAuth (flat-rate) — pass --managed-cache on to force"
		case in.apiKey == "":
			p.reason = "passthrough credential (billing unknown) — pass --managed-cache on to force"
		default:
			p.active = true
			p.reason = "API-key billing (--api-key-env) — cache economics are operator dollars"
		}
		return p, nil
	default:
		return guardManagedCachePosture{}, fmt.Errorf("--managed-cache %q: unknown mode (auto|on|off)", mode)
	}
}

// bannerLine renders the one-line posture note for the startup banner. ACTIVE names the
// lever it turns on (the 1h TTL upgrade) with its economics, so the operator reads what
// changed on the wire, not just a flag echo; passive names the reason and the override.
func (p guardManagedCachePosture) bannerLine() string {
	if p.active {
		return fmt.Sprintf("fak guard: managed cache — ACTIVE (%s): stable-prefix cache_control upgraded to the 1h TTL tier on the outbound wire, so an idle gap >5m re-enters on a 0.1x cache read instead of re-writing the prefix (1h writes cost 2x once; witness: fak_gateway_cache_ttl_upgrade_total)", p.reason)
	}
	return fmt.Sprintf("fak guard: managed cache — passive (%s); provider cache still applies via the client's own cache_control", p.reason)
}
