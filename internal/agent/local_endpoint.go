package agent

import (
	"net"
	"net/netip"
	"net/url"
	"strings"
)

// cgnatPrefix is the RFC 6598 shared address space, which Tailscale uses for
// its tailnet addresses; netip has no predicate for it.
var cgnatPrefix = netip.MustParsePrefix("100.64.0.0/10")

// IsLocalEndpoint reports whether baseURL points at a model server on this
// host or the local network rather than a hosted provider. Local servers
// re-prefill the whole context each turn, so callers relax their timeouts.
func IsLocalEndpoint(baseURL string) bool {
	baseURL = strings.TrimSpace(baseURL)
	if baseURL == "" {
		return false
	}
	u, err := url.Parse(baseURL)
	if err != nil {
		return false
	}
	host := strings.TrimSuffix(strings.ToLower(u.Hostname()), ".")
	if host == "" {
		return false
	}
	if host == "localhost" || strings.HasSuffix(host, ".localhost") {
		return true
	}
	if addr, err := netip.ParseAddr(host); err == nil {
		addr = addr.Unmap()
		return addr.IsLoopback() || addr.IsPrivate() || addr.IsLinkLocalUnicast() ||
			(addr.Is4() && cgnatPrefix.Contains(addr))
	}
	if net.ParseIP(host) != nil {
		return false
	}
	if !strings.Contains(host, ".") {
		return true
	}
	for _, suffix := range []string{".local", ".lan", ".internal", ".home.arpa"} {
		if strings.HasSuffix(host, suffix) {
			return true
		}
	}
	return false
}
