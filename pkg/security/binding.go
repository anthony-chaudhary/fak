package security

import (
	"fmt"
	"log"
	"net"
	"strings"
)

var (
	// ErrWildcardBinding is returned when a daemon attempts to bind to
	// a wildcard address (0.0.0.0, [::], or empty host) without explicit opt-in.
	ErrWildcardBinding = ErrInsecureWildcardBinding
)

// ExtractHost extracts the hostname or IP component from a listen address string.
// It handles addresses with or without a port, bracketed IPv6 literals, and raw IPs.
func ExtractHost(addr string) string {
	addr = strings.TrimSpace(addr)
	if addr == "" {
		return ""
	}

	// Attempt standard host:port split
	if h, _, err := net.SplitHostPort(addr); err == nil {
		return strings.Trim(h, "[]")
	}

	// Handle bracketed IPv6 without port (e.g. "[::1]")
	if strings.HasPrefix(addr, "[") && strings.HasSuffix(addr, "]") {
		return addr[1 : len(addr)-1]
	}

	// Raw IP or hostname without port
	if ip := net.ParseIP(addr); ip != nil {
		return ip.String()
	}

	return addr
}

// IsWildcard reports whether the host portion of an address binds to all network interfaces.
func IsWildcard(host string) bool {
	host = strings.TrimSpace(host)
	if host == "*" {
		return true
	}
	return isWildcard(host)
}

// IsLoopback reports whether the host is a loopback address (e.g. 127.0.0.1, ::1, or localhost).
func IsLoopback(host string) bool {
	return isLoopback(strings.TrimSpace(host))
}

// IsTailscaleIP reports whether the IP falls within the Tailscale IPv4 CGNAT range
// (100.64.0.0/10) or Tailscale IPv6 ULA range (fd7a:115c:a1e0::/48).
func IsTailscaleIP(ip net.IP) bool {
	if ip == nil {
		return false
	}
	return defaultTailscaleNet.Contains(ip) || defaultTailscaleIPv6Net.Contains(ip)
}

// IsRFC1918IP reports whether the IP falls within an RFC 1918 private IPv4 subnet.
func IsRFC1918IP(ip net.IP) bool {
	return IsProhibitedPrivateEgress(ip)
}

// ValidateDaemonBinding checks that a daemon or server endpoint binding is restricted
// to loopback (127.0.0.1, localhost) or explicit Tailscale interface IPs unless
// allowInsecureLAN is true.
func ValidateDaemonBinding(addr string, allowInsecureLAN bool) error {
	if allowInsecureLAN {
		return nil
	}

	host := ExtractHost(addr)

	if IsWildcard(host) {
		return fmt.Errorf("%w: listen address %q attempted wildcard binding; must bind strictly to 127.0.0.1, localhost, or an explicit Tailscale IP (100.64.0.0/10) unless --allow-insecure-lan is set", ErrWildcardBinding, addr)
	}

	if IsLoopback(host) {
		return nil
	}

	ip := net.ParseIP(host)
	if ip != nil && IsTailscaleIP(ip) {
		return nil
	}

	if ip != nil && IsRFC1918IP(ip) {
		return fmt.Errorf("%w: listen address %q attempted binding to private LAN IP (%s); endpoints must strictly bind to loopback or Tailscale IP unless --allow-insecure-lan is set", ErrInsecureLANBinding, addr, ip.String())
	}

	if ip != nil {
		return fmt.Errorf("%w: listen address %q attempted binding to non-loopback, non-Tailnet IP (%s); endpoints must strictly bind to loopback or Tailscale IP unless --allow-insecure-lan is set", ErrInsecureLANBinding, addr, ip.String())
	}

	return fmt.Errorf("%w: listen address %q attempted binding to unverified hostname %q; endpoints must strictly bind to loopback or Tailscale IP unless --allow-insecure-lan is set", ErrInsecureLANBinding, addr, host)
}

// AssertDaemonBinding terminates the process immediately via log.Fatalf if
// ValidateDaemonBinding returns an error. It serves as a startup assertion
// for all unauthenticated servers and daemons.
func AssertDaemonBinding(addr string, allowInsecureLAN bool) {
	if err := ValidateDaemonBinding(addr, allowInsecureLAN); err != nil {
		log.Fatalf("FATAL: daemon socket binding violation: %v", err)
	}
}

// ListenGuarded validates the listen address against zero-trust ingress policies
// before delegating to net.Listen.
func ListenGuarded(network, addr string, allowInsecureLAN bool) (net.Listener, error) {
	if err := ValidateDaemonBinding(addr, allowInsecureLAN); err != nil {
		return nil, err
	}
	return net.Listen(network, addr)
}
