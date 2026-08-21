// cloudroute.go is the detector: given the environment a wrapped agent WOULD be
// launched with, decide whether that agent will talk to a request-signed cloud
// gateway (AWS Bedrock via SigV4, Google Vertex via ADC) instead of a
// bearer-token endpoint — because a base-URL repoint cannot redirect such a
// child, so fak's gateway would adjudicate nothing and must say so at launch
// rather than refuse for the wrong reason. See doc.go for the framing.

package cloudroute

import (
	"sort"
	"strings"
)

// Kind is the closed vocabulary of request-signed cloud routes this package
// recognizes. It is closed on purpose: each entry names a wire whose auth is a
// per-request SIGNATURE (or an ADC-minted short-lived token), which is precisely
// the property that makes ANTHROPIC_BASE_URL inert for the child.
type Kind string

const (
	// KindBedrock is Claude Code routed to AWS Bedrock: SigV4-signed requests to a
	// regional bedrock-runtime endpoint, credentials resolved by the AWS SDK chain
	// (SSO profile, role, instance metadata), no static bearer to hand fak.
	KindBedrock Kind = "bedrock"
	// KindVertex is Claude Code routed to Google Vertex AI: Application Default
	// Credentials mint a short-lived OAuth token per process, and the endpoint is
	// region-templated rather than free-form.
	KindVertex Kind = "vertex"
)

// SelectorFor names the single environment variable whose truthy value SELECTS
// each route. This is the detection signal, and it is deliberately the selector
// rather than the presence of credentials: an AWS profile in the environment of a
// subscription-routed session is incidental, while CLAUDE_CODE_USE_BEDROCK=1 is
// the operator declaring the wire.
var SelectorFor = map[Kind]string{
	KindBedrock: "CLAUDE_CODE_USE_BEDROCK",
	KindVertex:  "CLAUDE_CODE_USE_VERTEX",
}

// WaiverKey names the config/secretload key that DOWNGRADES the launch refusal to
// a loud warning: the operator has read the posture and wants guard's other
// properties (hook floor, tool brokering, transcript, sandbox) on a session whose
// model traffic fak will NOT see. It exists because "guard cannot adjudicate this
// wire" is a true statement about adjudication, not a reason to deny an operator
// the rest of the tool — but the default must be the refusal, because a session
// that silently adjudicates nothing is the worse failure.
const WaiverKey = "FAK_GUARD_ALLOW_UNSUPPORTED_UPSTREAM"

// bedrockEnvNames are the variables a Bedrock-routed child needs in order to keep
// the credential and region it already resolved in the operator's shell. They
// split into selectors/config (safe to inherit anywhere) and the SDK credential
// chain's own variables (inherited only when the route is actually detected, and
// only when already present — see Route.CredentialNames).
var bedrockEnvNames = []string{
	"CLAUDE_CODE_USE_BEDROCK",
	"ANTHROPIC_BEDROCK_BASE_URL",
	"CLAUDE_CODE_SKIP_BEDROCK_AUTH",
	"AWS_PROFILE",
	"AWS_DEFAULT_PROFILE",
	"AWS_REGION",
	"AWS_DEFAULT_REGION",
	"AWS_CONFIG_FILE",
	"AWS_SHARED_CREDENTIALS_FILE",
	"AWS_SDK_LOAD_CONFIG",
	"AWS_EC2_METADATA_DISABLED",
	"AWS_ROLE_ARN",
	"AWS_ROLE_SESSION_NAME",
	"AWS_WEB_IDENTITY_TOKEN_FILE",
	"AWS_CONTAINER_CREDENTIALS_FULL_URI",
	"AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
	"AWS_ACCESS_KEY_ID",
	"AWS_SECRET_ACCESS_KEY",
	"AWS_SESSION_TOKEN",
	"AWS_SESSION_EXPIRATION",
	"AWS_BEARER_TOKEN_BEDROCK",
}

// vertexEnvNames are the Vertex/ADC equivalents.
var vertexEnvNames = []string{
	"CLAUDE_CODE_USE_VERTEX",
	"ANTHROPIC_VERTEX_PROJECT_ID",
	"CLAUDE_CODE_SKIP_VERTEX_AUTH",
	"CLOUD_ML_REGION",
	"GOOGLE_APPLICATION_CREDENTIALS",
	"GOOGLE_CLOUD_PROJECT",
	"GOOGLE_CLOUD_QUOTA_PROJECT",
	"GCLOUD_PROJECT",
}

// credentialShaped marks the subset of the per-route names that the #2358
// inherited-secret floor (internal/policy.StripInheritedSecrets) strips by name
// or by value shape. They are listed separately because passing them to a child
// must be a DECLARED decision — the floor's StripInheritedSecretsExcept keep-set
// — taken only when the route was actually detected, never a blanket widening of
// the floor.
var credentialShaped = map[string]bool{
	"AWS_ACCESS_KEY_ID":                      true,
	"AWS_SECRET_ACCESS_KEY":                  true,
	"AWS_SESSION_TOKEN":                      true,
	"AWS_BEARER_TOKEN_BEDROCK":               true,
	"AWS_WEB_IDENTITY_TOKEN_FILE":            true,
	"AWS_CONTAINER_CREDENTIALS_FULL_URI":     true,
	"AWS_CONTAINER_CREDENTIALS_RELATIVE_URI": true,
	"GOOGLE_APPLICATION_CREDENTIALS":         true,
}

// Route is a detected request-signed cloud route: which wire, what selected it,
// and the facts an operator-facing refusal needs to name.
type Route struct {
	// Kind is the detected wire.
	Kind Kind
	// Selector is the environment variable whose truthy value selected it.
	Selector string
	// Observed names the route's variables that are actually SET in the inspected
	// environment, sorted. NAMES ONLY — never values, so this is safe to print in a
	// refusal, a bail file, or a doctor report.
	Observed []string
	// Waived reports that WaiverKey was truthy in the inspected environment, so the
	// caller should warn loudly instead of refusing.
	Waived bool
}

// InertRepointVars are the base-URL variables fak's gateway repoint writes into
// the child environment and that a request-signed child IGNORES. Naming them is
// the honest half of the refusal: it says exactly which mechanism went inert,
// rather than reporting a credential problem the operator does not have.
var InertRepointVars = []string{"ANTHROPIC_BASE_URL"}

// Detect reports whether environ selects a request-signed cloud route.
//
// environ is a "NAME=VALUE" slice — os.Environ() at the launch site, or the
// already-assembled child environment. Taking the snapshot as DATA (rather than
// reading the process env here) keeps this package pure, makes every case
// testable without t.Setenv, and lets the guard ask the question about the
// environment the child will really get rather than the one guard itself has.
//
// Bedrock is checked before Vertex only for determinism when a misconfigured host
// sets both; the returned Route names which one won, so the refusal is never
// ambiguous.
func Detect(environ []string) (Route, bool) {
	set := envMap(environ)
	waived := truthy(set[canon(WaiverKey)])
	for _, k := range []Kind{KindBedrock, KindVertex} {
		sel := SelectorFor[k]
		if !truthy(set[canon(sel)]) {
			continue
		}
		return Route{
			Kind:     k,
			Selector: sel,
			Observed: observed(set, EnvNames(k)),
			Waived:   waived,
		}, true
	}
	return Route{}, false
}

// EnvNames returns every environment variable name associated with a route, in a
// stable order. Callers use it two ways: to widen a deny-by-default sandbox
// allow-list (so a detected route's non-secret selectors survive the boundary),
// and to build the inherited-secret floor's declared keep-set.
func EnvNames(k Kind) []string {
	switch k {
	case KindBedrock:
		return append([]string(nil), bedrockEnvNames...)
	case KindVertex:
		return append([]string(nil), vertexEnvNames...)
	default:
		return nil
	}
}

// AllEnvNames returns the union of every route's names, sorted. This is the set a
// sandbox allow-list enumerates so that "the enterprise set is written down"
// rather than implicitly stripped — deny-by-default is preserved, the list just
// stops being a secret the operator has to rediscover.
func AllEnvNames() []string {
	seen := map[string]bool{}
	var out []string
	for _, k := range []Kind{KindBedrock, KindVertex} {
		for _, n := range EnvNames(k) {
			if !seen[n] {
				seen[n] = true
				out = append(out, n)
			}
		}
	}
	sort.Strings(out)
	return out
}

// NonSecretEnvNames is AllEnvNames minus the credential-shaped names: the subset
// that is pure route SELECTION and configuration (which cloud, which region,
// which profile, which config file). Only this subset belongs in a static
// allow-list; the credential-shaped remainder must stay a per-launch declared
// decision so a sandbox boundary never becomes a blanket credential conduit.
func NonSecretEnvNames() []string {
	var out []string
	for _, n := range AllEnvNames() {
		if !credentialShaped[n] {
			out = append(out, n)
		}
	}
	return out
}

// CredentialNames returns the route's credential-shaped names that are actually
// present in environ — the exact keep-set to declare to
// policy.StripInheritedSecretsExcept. Present-only is the narrowing rule: a
// declared name is permission for a variable to cross, never an instruction to
// invent one.
func (r Route) CredentialNames(environ []string) []string {
	set := envMap(environ)
	var out []string
	for _, n := range EnvNames(r.Kind) {
		if !credentialShaped[n] {
			continue
		}
		if _, ok := set[canon(n)]; ok {
			out = append(out, n)
		}
	}
	sort.Strings(out)
	return out
}

// Explain renders the operator-facing cause: the wire, what selected it, which
// repoint went inert, and the alternative that DOES work. One sentence per fact,
// no values — this string is what the refusal and the bail's config: block both
// carry, so it must be safe to log.
func (r Route) Explain() string {
	var b strings.Builder
	b.WriteString("the wrapped agent is routed to ")
	switch r.Kind {
	case KindBedrock:
		b.WriteString("AWS Bedrock (SigV4-signed requests)")
	case KindVertex:
		b.WriteString("Google Vertex AI (ADC-minted tokens)")
	default:
		b.WriteString("a request-signed cloud gateway")
	}
	b.WriteString(" by ")
	b.WriteString(r.Selector)
	b.WriteString(", so it authenticates each request itself and ignores ")
	b.WriteString(strings.Join(InertRepointVars, "/"))
	b.WriteString(" — fak's gateway repoint cannot take effect and the session would be adjudicated by nothing")
	if len(r.Observed) > 0 {
		b.WriteString(". Observed: ")
		b.WriteString(strings.Join(r.Observed, ", "))
	}
	return b.String()
}

// observed returns the names among want that are present in set, sorted.
func observed(set map[string]string, want []string) []string {
	var out []string
	for _, n := range want {
		if _, ok := set[canon(n)]; ok {
			out = append(out, n)
		}
	}
	sort.Strings(out)
	return out
}

// envMap indexes a "NAME=VALUE" slice by canonical name, first occurrence
// winning — the same precedence exec gives a duplicated name on the platforms
// where it is possible.
func envMap(environ []string) map[string]string {
	out := make(map[string]string, len(environ))
	for _, kv := range environ {
		i := strings.IndexByte(kv, '=')
		if i <= 0 {
			continue
		}
		k := canon(kv[:i])
		if _, dup := out[k]; dup {
			continue
		}
		out[k] = kv[i+1:]
	}
	return out
}

// canon folds an environment variable name for comparison. Every name this
// package knows is upper-case ASCII, so upper-casing can only match the variable
// intended — and it is required on Windows, where env names are case-insensitive
// and a shell may well have exported `Aws_Profile`.
func canon(k string) string { return strings.ToUpper(strings.TrimSpace(k)) }

// truthy is the shared "is this selector on" rule, matching the vocabulary
// internal/secretload already uses for FAK_SANDBOX_INHERIT_ENV so an operator
// meets one convention rather than two. An unset or empty value is false, which
// is why a merely-exported CLAUDE_CODE_USE_BEDROCK= never trips detection.
func truthy(v string) bool {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "1", "true", "yes", "y", "on":
		return true
	default:
		return false
	}
}
