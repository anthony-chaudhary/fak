package fakclient

import (
	"net"
	"net/url"
)

// LoopbackFallbackURL returns an alternative URL to retry when a probe of a
// loopback LITERAL has failed at the transport level. On Windows the loopback
// may be reachable only on one address family (e.g. a WSL2 gateway is exposed
// as [::1] only), so a literal like 127.0.0.1 can fail while "localhost"
// (which resolves to both families, ::1 first) succeeds. It returns
// ("", false) when the URL is not a loopback literal, has no fallback, or is
// not a valid absolute URL. The caller must not use the fallback when a
// request SUCCEEDED.
func LoopbackFallbackURL(rawURL string) (string, bool) {
	u, err := url.Parse(rawURL)
	if err != nil {
		return "", false
	}
	host := u.Hostname()
	if host == "" {
		return "", false
	}
	ip := net.ParseIP(host)
	if ip == nil || !ip.IsLoopback() {
		return "", false
	}
	port := u.Port()
	u.Host = net.JoinHostPort("localhost", port)
	if port == "" {
		u.Host = "localhost"
	}
	return u.String(), true
}
