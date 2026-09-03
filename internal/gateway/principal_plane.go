package gateway

// principal_plane.go — the CONTROL-PLANE half of the kernel-assigned authority
// principal (#2439, part of the harness-native program #2387 / #2397).
//
// principal.go (#2412) stamps an authority principal on the served MODEL turn and
// strips a relayed confirmation token before adjudication. It still trusts the caller
// for the label: classifyPrincipal reads a header, and the sibling isolation seam
// (principalFor, http_fak_endpoints.go) even falls back to a request BODY field. That
// is the hole #2439 closes on the control plane:
//
//   - The kernel ASSIGNS the label from the authenticated transport. kernelPrincipal
//     reads the route's transport floor and the trusted front door's relay header, and
//     never a body field. A relay label may DEMOTE a turn below its transport floor
//     (a scheduler honestly declaring itself a timer) but can never PROMOTE it: an
//     /a2a/* turn is peer-agent even when it writes "human" on the wire.
//   - Authority-CONSUMING control verbs (pause/resume and the policy-widening verbs on
//     POST /v1/fak/session/{id}/{verb}) are refused under a non-human principal with the
//     closed PRINCIPAL_NOT_HUMAN reason, so "a webhook paused the run" / "a timer widened
//     the budget" is unrepresentable rather than patched per channel.
//   - Cross-agent sends address a KERNEL-MINTED lease id. A lease-addressed target is
//     resolved strictly — unknown or expired REFUSES, and there is deliberately no
//     fallback to name routing, because that fallback IS the misroute: an expired
//     lease id degrading into the name its new holder now owns.
//   - Every control-plane event is journaled with the principal the kernel assigned,
//     refused or not, so a relayed authority attempt leaves a countable witness.

import (
	"crypto/rand"
	"encoding/hex"
	"net/http"
	"strings"
	"time"
)

// ---------------------------------------------------------------------------
// kernel-assigned principal
// ---------------------------------------------------------------------------

// principalRank orders the closed label set by how much authority it may consume:
// higher = more. Only the human rank consumes user authority; the ordering exists so a
// caller-supplied label can be intersected with the transport's floor and can therefore
// DEMOTE a turn but never PROMOTE it. PrincipalUnknown (and the zero value) rank lowest,
// keeping an unrecognized label fail-closed.
func principalRank(p Principal) int {
	switch p {
	case PrincipalHuman:
		return 5
	case PrincipalSelfModel:
		return 4
	case PrincipalPeerAgent:
		return 3
	case PrincipalTimer:
		return 2
	case PrincipalNetworkTool:
		return 1
	default:
		return 0
	}
}

// transportPrincipalFloor is the CEILING on the authority a route's transport may carry,
// derived from the authenticated transport alone — never from caller text. The /a2a/*
// plane is by construction a peer-agent relay, so a turn arriving there can never present
// as human however it labels itself. Every other route's floor is the direct interactive
// wire (human), which an honest relay label can still demote.
func transportPrincipalFloor(path string) Principal {
	if strings.HasPrefix(path, "/a2a/") {
		return PrincipalPeerAgent
	}
	return PrincipalHuman
}

// kernelPrincipal assigns one request's inbound AUTHORITY principal. It reads exactly two
// kernel-side facts: the route's transport floor and the trusted front door's relay-class
// header. It never reads a request body — that is the #2439 hardening over principalFor's
// body-principal fallback, since caller-supplied text may not name the authority a turn
// carries. The result is the WEAKER of the two, so a spoofed "human" header on a relay
// transport buys nothing.
func kernelPrincipal(r *http.Request) Principal {
	if r == nil {
		return PrincipalUnknown
	}
	floor := PrincipalHuman
	if r.URL != nil {
		floor = transportPrincipalFloor(r.URL.Path)
	}
	claimed := classifyPrincipal(r.Header.Get(inboundPrincipalHeader))
	if principalRank(claimed) < principalRank(floor) {
		return claimed
	}
	return floor
}

// ---------------------------------------------------------------------------
// authority-consuming control verbs
// ---------------------------------------------------------------------------

// controlVerbConsumesAuthority reports whether a POST /v1/fak/session/{id}/{verb} control
// verb spends USER authority — the closed set #2439 names: a drive-state change (the "run"
// verb, which carries pause/resume/stop) and a policy widening ("budget", "wall",
// "throughput"). "pace" and "priority" are deliberately OUT: they are scheduling hints that
// cannot widen what a session is permitted to do, so a machine source may still set them.
// "steer" is out for the same reason — a steer is INPUT, gated by the a2achan floor on
// caps/taint/scope, and is stamped with its principal rather than refused.
func controlVerbConsumesAuthority(verb string) bool {
	switch strings.ToLower(strings.TrimSpace(verb)) {
	case "run", "budget", "wall", "throughput", "pause", "resume", "stop":
		return true
	}
	return false
}

// stampedSteerPrincipal resolves the attribution string a steer reaches the bus with. Under
// the human principal the caller's own label passes through untouched (an operator steer
// stays empty; the "doomloop-guard" machine label of #3529 stays itself). Under any
// non-human principal the kernel's label REPLACES it, so a relayed steer cannot present on
// the bus as the operator by writing "operator" in its body.
func stampedSteerPrincipal(p Principal, claimed string) string {
	if p.IsHuman() {
		return claimed
	}
	return string(p)
}

// ---------------------------------------------------------------------------
// kernel-minted lease identities
// ---------------------------------------------------------------------------

// leaseIDPrefix marks a target as a KERNEL-MINTED lease id rather than a name. A target
// carrying it is resolved through the lease table and never degrades to name routing.
const leaseIDPrefix = "lease_"

// Closed refusal reasons for lease-addressed routing. A send naming a lease the kernel
// never minted, or one whose lease has expired, draws one of these instead of being
// delivered to whoever holds the underlying name now.
const (
	ReasonLeaseUnknown = "LEASE_UNKNOWN"
	ReasonLeaseExpired = "LEASE_EXPIRED"
)

// maxSessionLeases bounds the lease table, using the same generational reset as the other
// per-trace maps on the server so it cannot grow unbounded.
const maxSessionLeases = maxCtxRestoreSessions

// sessionLease binds a kernel-minted id to the name that held it and the instant the
// binding stops being deliverable. Name reuse after Expires is exactly the misroute this
// record exists to refuse.
type sessionLease struct {
	Name    string
	Expires time.Time
}

// mintSessionLease binds name to a fresh kernel-minted lease id valid for ttl and returns
// the id. The id is unguessable (crypto/rand) so a peer cannot address a lease it was never
// handed. A non-positive ttl mints an already-expired lease, which is legal and refuses on
// first use — never a lease that outlives its holder.
func (s *Server) mintSessionLease(name string, ttl time.Duration) (string, error) {
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	id := leaseIDPrefix + hex.EncodeToString(b)
	s.leaseMu.Lock()
	defer s.leaseMu.Unlock()
	if s.sessionLeases == nil {
		s.sessionLeases = make(map[string]sessionLease)
	}
	if len(s.sessionLeases) >= maxSessionLeases {
		s.sessionLeases = make(map[string]sessionLease) // generational reset, like traceOwner
	}
	s.sessionLeases[id] = sessionLease{Name: name, Expires: time.Now().Add(ttl)}
	return id, nil
}

// isLeaseAddressed reports whether a message target is a kernel-minted lease id rather than
// a plain name.
func isLeaseAddressed(target string) bool {
	return strings.HasPrefix(strings.TrimSpace(target), leaseIDPrefix)
}

// resolveLeaseTarget resolves an inbound message target to the name it may be delivered to.
// A target with no lease prefix is a legacy NAME address and is returned unchanged. A
// lease-addressed target is resolved strictly against the kernel's table: an id the kernel
// never minted refuses LEASE_UNKNOWN, and one past its expiry refuses LEASE_EXPIRED. Neither
// refusal falls back to name routing — that fallback is precisely the misroute where an
// expired lease id is delivered to the name's NEW holder.
func (s *Server) resolveLeaseTarget(target string, now time.Time) (name, reason string, ok bool) {
	target = strings.TrimSpace(target)
	if !isLeaseAddressed(target) {
		return target, "", true
	}
	s.leaseMu.RLock()
	lease, found := s.sessionLeases[target]
	s.leaseMu.RUnlock()
	if !found {
		return "", ReasonLeaseUnknown, false
	}
	if !now.Before(lease.Expires) {
		return "", ReasonLeaseExpired, false
	}
	return lease.Name, "", true
}

// ---------------------------------------------------------------------------
// control-plane principal journal
// ---------------------------------------------------------------------------

// maxControlPlaneEvents bounds the in-process control-plane journal. Oldest rows are
// dropped first, so a flood of refused relay attempts cannot exhaust memory while the most
// recent evidence stays readable.
const maxControlPlaneEvents = 1024

// ControlPlaneEvent is one journaled control-plane admission: which verb reached which
// session under which KERNEL-ASSIGNED principal, and whether the kernel refused it. Every
// control-plane event is recorded, admitted or refused, so "which principal drove this
// session" is answerable from the record rather than reconstructed from prose.
type ControlPlaneEvent struct {
	TraceID   string    `json:"trace_id"`
	Verb      string    `json:"verb"`
	Principal Principal `json:"principal"`
	Refused   bool      `json:"refused,omitempty"`
	Reason    string    `json:"reason,omitempty"`
}

// journalControlPrincipal records one control-plane event. A nil server is a safe no-op so
// the route stays callable from a bare handler test.
func (s *Server) journalControlPrincipal(ev ControlPlaneEvent) {
	if s == nil {
		return
	}
	if ev.Principal == "" {
		ev.Principal = PrincipalUnknown
	}
	s.controlPlaneMu.Lock()
	defer s.controlPlaneMu.Unlock()
	s.controlPlaneLog = append(s.controlPlaneLog, ev)
	if len(s.controlPlaneLog) > maxControlPlaneEvents {
		s.controlPlaneLog = append(s.controlPlaneLog[:0:0], s.controlPlaneLog[len(s.controlPlaneLog)-maxControlPlaneEvents:]...)
	}
}

// controlPlaneEvents returns a copy of the journaled control-plane events, oldest first.
func (s *Server) controlPlaneEvents() []ControlPlaneEvent {
	if s == nil {
		return nil
	}
	s.controlPlaneMu.Lock()
	defer s.controlPlaneMu.Unlock()
	return append([]ControlPlaneEvent(nil), s.controlPlaneLog...)
}

// refusedPrincipalAttempts counts journaled control-plane events the kernel refused because
// the inbound principal was not human — the countable witness behind a guard summary of
// relayed authority attempts.
func (s *Server) refusedPrincipalAttempts() int {
	n := 0
	for _, ev := range s.controlPlaneEvents() {
		if ev.Refused && ev.Reason == ReasonPrincipalNotHuman {
			n++
		}
	}
	return n
}
