package vcachegov

// secret.go is the Law-D4 content classifier — the safety gate that runs BEFORE
// any warming economics. Issue #720 acceptance 4: secrets/PII/regulated content
// must NEVER be warmed into a provider prefix cache, because
//
//   - RETENTION: a warmed prefix sits in the provider's KV for at least the TTL,
//     replicated opaquely across shards, outside the warmer's deletion control.
//     Some models impose a minimum-retention floor that can exceed the TTL and
//     that zero-data-retention orgs cannot opt out of.
//   - SIDE CHANNEL: cache-hit latency is a membership oracle — "has anyone recently
//     processed string X?" — so warming a secret leaks its existence cross-tenant.
//
// The classifier is therefore fail-closed: the zero value (Cacheable) is the only
// state that permits warming, and every other state routes the prefix away from
// the implicit/auto prefix cache. The Governor's Classify consults Warmable before
// any λT math; the warm scheduler's Rank drops every non-cacheable candidate so a
// secret can never reach the Warmer even if a caller mis-files it in the warm set.

// SecretClassification is a prefix's Law-D4 content class. It is deliberately a
// closed three-valued set, not a boolean: "regulated" is distinct from "secret"
// because a regulated prefix MAY be cached, but only through a deletion-capable
// surface (Gemini CachedContent, which has a real delete primitive), never through
// the implicit/auto prefix cache the rest of vCache drives.
type SecretClassification string

const (
	// Cacheable (the zero value) is the only class permitted in the implicit/auto
	// prefix cache. A prefix that carries no secret, PII, or regulated content.
	Cacheable SecretClassification = ""
	// Secret is content that must never be warmed into ANY provider prefix cache:
	// credentials, API keys, tokens, PII, and anything else under a no-retention
	// policy. It takes the no-cache path — the full prefix is always re-sent and no
	// breakpoint ever precedes a secret byte.
	Secret SecretClassification = "secret"
	// SecretRegulated is content that is not a bare secret but is governed by a
	// retention/access policy (e.g. regulated customer data): it may be cached ONLY
	// through a surface with a real deletion primitive, never implicitly. The
	// Governor routes it to DecisionExplicitCache and refuses to warm it here.
	SecretRegulated SecretClassification = "regulated"
)

// Warmable reports whether a prefix of this class may enter the implicit/auto
// prefix cache the Governor budgets and pins. Only Cacheable is warmable; Secret
// and SecretRegulated are both refused, the former unconditionally and the latter
// pending an explicit deletion-capable surface the caller must drive itself.
func Warmable(c SecretClassification) bool { return c == Cacheable }

// ClassifyPrefix is the standalone Law-D4 classifier a canonicalizer or warmer
// calls before admitting a prefix to the warm set. It maps a coarse content label
// to the closed SecretClassification vocabulary so the rest of the Governor can
// stay branch-free on the secret axis. An empty/unknown label is fail-CLOSED to
// Secret: a prefix whose content class the canonicalizer could not establish is
// not trusted into a shared, retained provider cache.
func ClassifyPrefix(label string) SecretClassification {
	switch label {
	case "", "unknown":
		return Secret
	case "public", "cacheable", "system", "retrieved":
		return Cacheable
	case "regulated", "pii", "customer":
		return SecretRegulated
	case "secret", "credential", "api_key", "token", "private":
		return Secret
	default:
		return Secret
	}
}
