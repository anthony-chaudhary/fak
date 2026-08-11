package fleetbus

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net"
	"strconv"
	"strings"
)

// IdentitySource explains which property supplies a serve instance's bus
// identity. Operators need this distinction because only the configured-address
// and explicit forms survive a restart; a process fallback is intentionally honest
// about being lifetime-scoped.
type IdentitySource string

const (
	// IdentityExplicit is an operator-provided --fleet-bus-id. The bus preserves a
	// valid configured token byte-for-byte.
	IdentityExplicit IdentitySource = "explicit"
	// IdentityConfiguredAddress is derived from a fixed listen address. Rebinding
	// the same address after a restart yields the same identity, while two serves
	// that can listen concurrently on distinct addresses yield distinct identities.
	IdentityConfiguredAddress IdentitySource = "configured-address"
	// IdentityProcessFallback is the named fallback for transports with no stable
	// configured address (stdio, an ephemeral :0 listener, or an absent/malformed
	// address). It stays unique among simultaneous processes but is not advertised
	// as restart-stable.
	IdentityProcessFallback IdentitySource = "process-fallback"
)

// ServeIdentityRequest is the transport information available when fak serve joins
// the fleet bus. PID is deliberately an input rather than read here so a restart is
// reproducible in tests and callers cannot accidentally hide process identity in a
// package global.
type ServeIdentityRequest struct {
	// ExplicitID is the optional operator override. A valid token is preserved
	// byte-for-byte; an unsafe value is normalized with the same narrow alphabet
	// bus path segments use.
	ExplicitID string
	// Machine is the host token carried by Instance.Machine.
	Machine string
	// Addr is the configured HTTP listen address.
	Addr string
	// PID identifies the current process and is used only by the named fallback.
	PID int
	// Stdio means Addr is not the active transport, even when the ordinary HTTP
	// flag still contains its default value.
	Stdio bool
}

// ServeIdentity is the resolved identity plus the address that should be published
// on Instance. RestartStable is a promise about the derivation, not a claim that a
// configured explicit name is collision-proof: two live processes explicitly given
// one name still deliberately share one bus identity.
type ServeIdentity struct {
	ID            string
	Addr          string
	Source        IdentitySource
	RestartStable bool
}

// ResolveServeIdentity resolves the default serve identity without consulting the
// process table or the mutable fleet roster.
//
// A fixed configured listen address supplies both properties the PID default could
// not hold at once:
//
//   - restart stability: the same machine + address hashes to the same token;
//   - simultaneous uniqueness: two serves that can run concurrently on one machine
//     must listen on distinct addresses, which hash to distinct tokens.
//
// stdio and ephemeral/malformed addresses have no stable endpoint to key on. They
// keep the old process-local shape under an explicit Source value so help and logs
// can state the narrower guarantee rather than overclaim restart safety.
func ResolveServeIdentity(req ServeIdentityRequest) ServeIdentity {
	if explicit := strings.TrimSpace(req.ExplicitID); explicit != "" {
		return ServeIdentity{
			ID:            normalizeIdentityToken(explicit, "serve"),
			Addr:          serveAdvertisedAddr(req),
			Source:        IdentityExplicit,
			RestartStable: true,
		}
	}

	machine := normalizeIdentityToken(strings.TrimSpace(req.Machine), "unknown-host")
	if !req.Stdio {
		if canonical, ok := canonicalFixedListenAddr(req.Addr); ok {
			sum := sha256.Sum256([]byte("fak-serve\x00" + machine + "\x00" + canonical))
			return ServeIdentity{
				ID:            fmt.Sprintf("serve-%s-addr-%s", truncateToken(machine, 48), hex.EncodeToString(sum[:])),
				Addr:          strings.TrimSpace(req.Addr),
				Source:        IdentityConfiguredAddress,
				RestartStable: true,
			}
		}
	}

	return ServeIdentity{
		ID:            normalizeIdentityToken(fmt.Sprintf("serve-%s-%d", machine, req.PID), "serve"),
		Addr:          serveAdvertisedAddr(req),
		Source:        IdentityProcessFallback,
		RestartStable: false,
	}
}

func serveAdvertisedAddr(req ServeIdentityRequest) string {
	if req.Stdio {
		return "stdio"
	}
	return strings.TrimSpace(req.Addr)
}

// canonicalFixedListenAddr returns a byte-stable comparison form for an address
// whose port is fixed. Port zero is deliberately rejected: the kernel has not bound
// the listener yet, so ":0" contains no stable identity information.
func canonicalFixedListenAddr(addr string) (string, bool) {
	addr = strings.TrimSpace(addr)
	if addr == "" {
		return "", false
	}
	host, portText, err := net.SplitHostPort(addr)
	if err != nil {
		return "", false
	}
	port, err := strconv.Atoi(portText)
	if err != nil || port <= 0 || port > 65535 {
		return "", false
	}
	host = strings.TrimSpace(host)
	if ip := net.ParseIP(host); ip != nil {
		host = ip.String()
	} else {
		host = strings.ToLower(host)
	}
	// Empty host is the wildcard address. Spell it rather than leave an empty
	// component so the hash input is explicit and easy to reproduce.
	if host == "" {
		host = "*"
	}
	return net.JoinHostPort(host, strconv.Itoa(port)), true
}

// normalizeIdentityToken maps arbitrary operator/host text into the ValidToken
// alphabet. It matches the CLI's long-standing sanitizeBusToken behavior, including
// the 128-byte ceiling, while allowing a caller-specific non-empty fallback.
func normalizeIdentityToken(s, fallback string) string {
	var b strings.Builder
	for i := 0; i < len(s) && b.Len() < 128; i++ {
		c := s[i]
		switch {
		case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c >= '0' && c <= '9':
			b.WriteByte(c)
		case c == '-', c == '_', c == '.':
			b.WriteByte(c)
		default:
			b.WriteByte('-')
		}
	}
	out := b.String()
	if out == "" || out == "." || out == ".." {
		out = fallback
	}
	if len(out) > 128 {
		out = out[:128]
	}
	return out
}

func truncateToken(s string, max int) string {
	if max <= 0 {
		return ""
	}
	if len(s) <= max {
		return s
	}
	return s[:max]
}
