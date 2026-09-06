package engine

// fleet_trust.go — the OPERATOR-DECLARED trust boundary for org-operated fleet
// hosts (#5421, track G of epic #5416). This is an ENFORCEMENT change, and it
// stays fail-closed in every direction.
//
// THE PROBLEM IT CLOSES. modelroute.ZoneFleet records that an operator wrote
// `kind: fleet` in a roster. That is an ASSERTION about who owns a machine, not a
// verified boundary: the kind requires only an explicit, parseable, non-loopback
// http(s) base_url, and deliberately NOT a cred_env (org-operated servers on
// private networks commonly have no API key). A plain `http://` endpoint named in
// a config file is not a trust boundary, so declaring the zone buys a
// tenant-scoped payload nothing: ZoneFleet.OnBox() is false, remoteRoute reads a
// "fleet:" route as REMOTE, and the residency gate (rank 12) denies. That default
// is preserved here byte for byte — an EMPTY boundary is exactly the pre-#5421
// floor, including for every fleet route.
//
// WHAT THIS ADDS is the one widening the issue asks for and only that: an operator
// may DECLARE specific fleet hosts as inside the org's trust boundary, over
// AUTHENTICATED transport, after which the residency gate admits a sensitive
// payload to exactly those and nothing else.
//
// IT DOES NOT MOVE THE RESIDENCY CLASSIFIER — the load-bearing distinction, and
// the reason this lands as a separate predicate rather than a new on-box family.
// remoteRoute / localRoute, PlacementZone.OnBox, and the tier-1 mirror
// modelroute.IsRemoteRoute are UNTOUCHED: a declared fleet host is still off-box
// and still reads REMOTE everywhere, so TestFleetZoneStaysRemoteAtTheFloor,
// TestFleetTargetIsRemoteToTheFloor, and TestTierOneRouteMirrorAgreesWithTheFloor
// all keep holding. The payload is admitted because the operator VOUCHED FOR THE
// DESTINATION, not because the bytes stayed home. Keeping the two apart is what
// satisfies the issue's requirement 3 (a route the floor cannot place must read
// remote, never on-box — the classifier is still the fail-closed one) and its
// requirement 4 (the tier-1 mirror needs no copy of this policy, so it cannot
// drift from it; residency and trust are answered by different predicates).
//
// DEFAULT-DENY IN EVERY DIRECTION:
//   - the default boundary is EMPTY; nothing is admitted until an operator declares.
//   - a declaration is verified WHOLE-OR-NOTHING: one bad host installs no policy.
//   - https ONLY, non-loopback, and a cred_env whose variable is actually SET, so
//     an unauthenticated or unverifiable host is refused at declaration time.
//   - the credential is RE-READ at adjudication time, so an operator who unsets the
//     secret closes the boundary behind them without editing the policy.
//   - only the "fleet:<account>/…" route shape names an account to match against;
//     a bare "fleet", a "fleet-a/kimi", or a "fleet/x" names none and stays denied.
//
// #5315 (Org Policy Plane) is the ADMINISTRATION half — where such an allowlist is
// centrally authored and distributed to a fleet. The issue asks for THIS half
// first on purpose: an allowlist nothing enforces is worse than none, because it
// reads as protection.

import (
	"fmt"
	"net/url"
	"os"
	"sort"
	"strings"
	"sync"
)

// FleetHost is ONE org-operated inference host an operator declares as inside the
// organization's trust boundary. It is a policy record, never a secret: CredEnv
// NAMES the environment variable holding the credential, mirroring
// modelroute.Account.CredEnv, so a declared boundary is safe to log and diff.
type FleetHost struct {
	// Account is the fleet account id exactly as it appears in the engine route
	// modelroute.Target.EngineRoute() stamps for a KindFleet target:
	// "fleet:<account>/<model>". It is matched case-insensitively, like the route.
	Account string
	// BaseURL is the host's endpoint. It MUST be https — the transport is the
	// authenticated half of the boundary, and a plain http:// endpoint is exactly
	// the "named in a config file" non-boundary this issue exists to reject. Note
	// this is strictly NARROWER than modelroute's fleet-account validation, which
	// still admits http for ATTRIBUTION; attribution and trust are different asks.
	BaseURL string
	// CredEnv NAMES the environment variable holding the bearer credential for the
	// host. It is required here even though `kind: fleet` does not require one:
	// an unauthenticated endpoint cannot be a trust boundary, whatever the roster
	// says about who owns it.
	CredEnv string
}

// fleetTrust is the process-wide declared boundary, keyed by lowercased account id.
// A registry mirrors how the rest of this floor is installed (abi.RegisterAdjudicator,
// abi.RegisterEngine): policy is declared once at boot, then read on the hot path.
var fleetTrust struct {
	mu    sync.RWMutex
	hosts map[string]FleetHost
}

// DeclareFleetTrustBoundary REPLACES the declared trust boundary with hosts — a
// policy load, not an append, so re-declaring is idempotent and a host an operator
// dropped from the policy is genuinely gone. Calling it with NO hosts clears the
// boundary back to the fail-closed default in which every fleet route is denied a
// sensitive payload.
//
// It is WHOLE-OR-NOTHING: every host is verified before any is installed, so a
// typo in the fifth entry can never leave the first four live as a half-applied
// policy that reads like protection. It returns the first verification failure and
// leaves the previously declared boundary untouched.
func DeclareFleetTrustBoundary(hosts ...FleetHost) error {
	next := make(map[string]FleetHost, len(hosts))
	for _, h := range hosts {
		acct, err := h.verify()
		if err != nil {
			return err
		}
		if _, dup := next[acct]; dup {
			return fmt.Errorf("engine: fleet trust boundary declares account %q twice — one host per account id", acct)
		}
		h.Account = acct
		next[acct] = h
	}
	fleetTrust.mu.Lock()
	fleetTrust.hosts = next
	fleetTrust.mu.Unlock()
	return nil
}

// FleetTrustBoundary returns the declared hosts in account order — the read-back an
// operator report or a boot log uses to show WHAT was widened. It returns a copy, so
// a caller cannot mutate the live policy, and it carries only env-var names.
func FleetTrustBoundary() []FleetHost {
	fleetTrust.mu.RLock()
	out := make([]FleetHost, 0, len(fleetTrust.hosts))
	for _, h := range fleetTrust.hosts {
		out = append(out, h)
	}
	fleetTrust.mu.RUnlock()
	sort.Slice(out, func(i, j int) bool { return out[i].Account < out[j].Account })
	return out
}

// verify checks the invariants that separate a DECLARED boundary from the bare
// ownership assertion `kind: fleet` already carries, and returns the canonical
// (lowercased) account id the route match keys on.
func (h FleetHost) verify() (string, error) {
	acct := strings.ToLower(strings.TrimSpace(h.Account))
	if acct == "" {
		return "", fmt.Errorf("engine: fleet trust boundary host needs an account id (the id in its \"fleet:<account>/<model>\" engine route)")
	}
	// ':' and '/' delimit the route's own segments and whitespace cannot survive a
	// route string, so an id containing them could never match the segment it claims
	// to name — refuse it here rather than declare a host nothing can reach.
	if strings.ContainsAny(acct, ":/") || strings.ContainsAny(acct, " \t\r\n") {
		return "", fmt.Errorf("engine: fleet trust boundary account %q must not contain ':', '/', or whitespace (it names one segment of a \"fleet:<account>/<model>\" route)", h.Account)
	}
	if h.BaseURL == "" {
		return "", fmt.Errorf("engine: fleet trust boundary host %q needs a base_url (the org endpoint being vouched for, e.g. https://gpu-07.corp.internal:8000/v1)", acct)
	}
	u, err := url.Parse(h.BaseURL)
	if err != nil || u.Hostname() == "" {
		return "", fmt.Errorf("engine: fleet trust boundary host %q base_url %q is not an absolute URL naming a host", acct, h.BaseURL)
	}
	if strings.ToLower(u.Scheme) != "https" {
		return "", fmt.Errorf("engine: fleet trust boundary host %q base_url %q must use https (got scheme %q) — "+
			"an unauthenticated transport is not a trust boundary, whoever owns the hardware", acct, h.BaseURL, u.Scheme)
	}
	if isLoopbackFleetHost(u.Hostname()) {
		return "", fmt.Errorf("engine: fleet trust boundary host %q base_url %q is a loopback host — an on-box server is already residency-exempt as a local/on-device route; the zones must stay disjoint", acct, h.BaseURL)
	}
	if h.CredEnv == "" {
		return "", fmt.Errorf("engine: fleet trust boundary host %q needs a cred_env (the env var NAME holding its credential) — "+
			"`kind: fleet` does not require one, but an unauthenticated host cannot be inside the boundary", acct)
	}
	if !isEnvVarName(h.CredEnv) {
		return "", fmt.Errorf("engine: fleet trust boundary host %q cred_env %q is not an env-var name "+
			"(it must NAME the variable holding the credential, e.g. CORP_FLEET_TOKEN — never the secret itself)", acct, h.CredEnv)
	}
	// The declaration is only as good as the credential behind it: a cred_env that
	// names an UNSET variable describes an unauthenticated host, so it is refused
	// here rather than admitted and silently unenforced.
	if os.Getenv(h.CredEnv) == "" {
		return "", fmt.Errorf("engine: fleet trust boundary host %q cred_env %q is unset — the authenticated transport it declares is not actually present", acct, h.CredEnv)
	}
	return acct, nil
}

// insideFleetTrustBoundary reports whether an engine route names a fleet host the
// operator declared inside the org trust boundary AND whose credential is still
// present. It is the ONLY widening of the residency gate; everything it cannot
// positively place stays denied.
//
// It does NOT report locality: a route this returns true for is still remote to
// remoteRoute and still off-box to modelroute.ZoneOfRoute. Trust and residency are
// deliberately separate answers about the same string.
func insideFleetTrustBoundary(route string) bool {
	acct, ok := fleetRouteAccount(route)
	if !ok {
		return false
	}
	fleetTrust.mu.RLock()
	h, declared := fleetTrust.hosts[acct]
	fleetTrust.mu.RUnlock()
	if !declared {
		return false
	}
	// Re-read at ADJUDICATION time, not just at declaration: an operator who
	// withdraws the secret has withdrawn the authenticated transport, and the
	// boundary must close behind them without needing the policy to be re-loaded.
	return os.Getenv(h.CredEnv) != ""
}

// fleetRouteAccount extracts the account id from the ONE fleet route shape that
// names one: "fleet:<account>" or "fleet:<account>/<model…>". The other three
// shapes localRoute-style matching accepts (a bare "fleet", "fleet-suffix",
// "fleet/path") carry no account segment, so there is nothing to match against the
// allowlist and they can never be admitted — fail-closed by construction.
//
// Only the FIRST '/' after the account ends it: upstream model ids may contain
// provider namespace slashes ("qwen/qwen3.6-27b"), exactly as EngineRoute documents.
func fleetRouteAccount(route string) (string, bool) {
	r := strings.ToLower(strings.TrimSpace(route))
	const fam = fleetRouteFamily + ":"
	if !strings.HasPrefix(r, fam) {
		return "", false
	}
	acct := r[len(fam):]
	if i := strings.Index(acct, "/"); i >= 0 {
		acct = acct[:i]
	}
	if acct == "" {
		return "", false
	}
	return acct, true
}

// fleetRouteFamily is the engine-route family prefix a KindFleet target stamps. It
// is deliberately NOT added to localRoute's on-box list — see this file's header.
const fleetRouteFamily = "fleet"

// isLoopbackFleetHost mirrors modelroute's loopback test on an already-parsed host.
func isLoopbackFleetHost(host string) bool {
	host = strings.ToLower(host)
	switch host {
	case "localhost", "127.0.0.1", "0.0.0.0", "::1":
		return true
	}
	return strings.HasPrefix(host, "127.")
}

// isEnvVarName reports whether s is a plausible environment-variable NAME rather
// than a pasted secret: leading letter or '_', then letters, digits, or '_'. It is
// the local counterpart of modelroute's envNameRE, kept dependency-free.
func isEnvVarName(s string) bool {
	if s == "" {
		return false
	}
	for i, r := range s {
		switch {
		case r == '_':
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z':
		case r >= '0' && r <= '9' && i > 0:
		default:
			return false
		}
	}
	return true
}
