// Package security provides network ingress validation and host boundary isolation,
// ensuring daemons bind to loopback or Tailscale overlay interfaces by default and
// preventing lateral movement to private RFC 1918 subnets during tool execution.
package security

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	// DefaultBindAddress is the default loopback binding address.
	DefaultBindAddress = "127.0.0.1"

	// TailscaleCIDR defines the Carrier-Grade NAT (CGNAT) subnet allocated to Tailscale overlays.
	TailscaleCIDR = "100.64.0.0/10"

	// TailscaleIPv6CIDR defines the Tailscale IPv6 Unique Local Address (ULA) overlay prefix.
	TailscaleIPv6CIDR = "fd7a:115c:a1e0::/48"
)

// ProhibitedEgressCIDRs lists the RFC 1918 private subnets blocked by default to prevent lateral movement.
var ProhibitedEgressCIDRs = []string{
	"10.0.0.0/8",
	"172.16.0.0/12",
	"192.168.0.0/16",
}

// Pre-compiled CIDR network definitions.
var (
	defaultTailscaleNet     *net.IPNet
	defaultTailscaleIPv6Net *net.IPNet
	defaultRFC1918Nets      []*net.IPNet
)

func init() {
	var err error
	_, defaultTailscaleNet, err = net.ParseCIDR(TailscaleCIDR)
	if err != nil {
		panic(fmt.Sprintf("security: failed to parse default Tailscale CIDR %q: %v", TailscaleCIDR, err))
	}

	_, defaultTailscaleIPv6Net, err = net.ParseCIDR(TailscaleIPv6CIDR)
	if err != nil {
		panic(fmt.Sprintf("security: failed to parse default Tailscale IPv6 CIDR %q: %v", TailscaleIPv6CIDR, err))
	}

	for _, cidr := range ProhibitedEgressCIDRs {
		_, n, err := net.ParseCIDR(cidr)
		if err != nil {
			panic(fmt.Sprintf("security: failed to parse RFC 1918 CIDR %q: %v", cidr, err))
		}
		defaultRFC1918Nets = append(defaultRFC1918Nets, n)
	}
}

var (
	// ErrInsecureWildcardBinding is returned when a wildcard socket binding (0.0.0.0, ::, or empty host)
	// is attempted without explicit opt-in via allowInsecureLAN.
	ErrInsecureWildcardBinding = errors.New("security: insecure wildcard socket binding prohibited: requires allow-insecure-lan opt-in")

	// ErrInsecureLANBinding is returned when binding to a non-loopback, non-Tailscale interface
	// is attempted without explicit opt-in via allowInsecureLAN.
	ErrInsecureLANBinding = errors.New("security: insecure LAN socket binding prohibited: requires allow-insecure-lan opt-in")

	// ErrProhibitedPrivateEgress is returned when egress connection attempts to reach
	// a prohibited RFC 1918 private subnet without explicit opt-in via allowPrivateLAN.
	ErrProhibitedPrivateEgress = errors.New("security: egress to RFC 1918 private network prohibited: lateral movement blocked")

	// ErrInvalidAddress is returned when a socket address or target URL cannot be parsed.
	ErrInvalidAddress = errors.New("security: invalid socket address format")
)

// ParameterEnvelope encapsulates default network boundary configuration parameters.
type ParameterEnvelope struct {
	DefaultBindAddress    string   `json:"default_bind_address"`
	ProhibitedEgressCIDRs []string `json:"prohibited_egress_cidrs"`
	TailscaleCIDR         string   `json:"tailscale_cidr"`
}

// DefaultParameterEnvelope returns the baseline parameter envelope.
func DefaultParameterEnvelope() ParameterEnvelope {
	return ParameterEnvelope{
		DefaultBindAddress:    DefaultBindAddress,
		ProhibitedEgressCIDRs: append([]string(nil), ProhibitedEgressCIDRs...),
		TailscaleCIDR:         TailscaleCIDR,
	}
}

// SocketBindingGuard enforces zero-trust socket binding and egress host boundary isolation policies.
// It ensures daemons bind strictly to loopback or Tailscale overlay IPs by default, prohibits
// unauthenticated wildcard listeners, and blocks lateral egress to private RFC 1918 home subnets.
type SocketBindingGuard struct {
	AllowInsecureLAN      bool     `json:"allow_insecure_lan"`
	AllowPrivateLAN       bool     `json:"allow_private_lan"`
	DefaultBindAddress    string   `json:"default_bind_address"`
	TailscaleCIDR         string   `json:"tailscale_cidr"`
	ProhibitedEgressCIDRs []string `json:"prohibited_egress_cidrs"`

	parsedTailscaleNet    *net.IPNet
	parsedTailscaleIPv6   *net.IPNet
	parsedProhibitedCIDRs []*net.IPNet
	mu                    sync.RWMutex
}

// NewSocketBindingGuard constructs a SocketBindingGuard with the specified LAN ingress policy.
func NewSocketBindingGuard(allowInsecureLAN bool) *SocketBindingGuard {
	return NewSocketBindingGuardWithOptions(allowInsecureLAN, false)
}

// NewSocketBindingGuardWithOptions constructs a SocketBindingGuard with explicit ingress and egress policies.
func NewSocketBindingGuardWithOptions(allowInsecureLAN, allowPrivateLAN bool) *SocketBindingGuard {
	g := &SocketBindingGuard{
		AllowInsecureLAN:      allowInsecureLAN,
		AllowPrivateLAN:       allowPrivateLAN,
		DefaultBindAddress:    DefaultBindAddress,
		TailscaleCIDR:         TailscaleCIDR,
		ProhibitedEgressCIDRs: append([]string(nil), ProhibitedEgressCIDRs...),
	}
	g.initNets()
	return g
}

func (g *SocketBindingGuard) initNets() {
	g.mu.Lock()
	defer g.mu.Unlock()

	if g.TailscaleCIDR == "" {
		g.TailscaleCIDR = TailscaleCIDR
	}
	_, g.parsedTailscaleNet, _ = net.ParseCIDR(g.TailscaleCIDR)
	_, g.parsedTailscaleIPv6, _ = net.ParseCIDR(TailscaleIPv6CIDR)

	if len(g.ProhibitedEgressCIDRs) == 0 {
		g.ProhibitedEgressCIDRs = append([]string(nil), ProhibitedEgressCIDRs...)
	}
	g.parsedProhibitedCIDRs = nil
	for _, cidr := range g.ProhibitedEgressCIDRs {
		if _, n, err := net.ParseCIDR(cidr); err == nil {
			g.parsedProhibitedCIDRs = append(g.parsedProhibitedCIDRs, n)
		}
	}
}

// ParameterEnvelope returns the configured parameter envelope.
func (g *SocketBindingGuard) ParameterEnvelope() ParameterEnvelope {
	if g == nil {
		return DefaultParameterEnvelope()
	}
	g.mu.RLock()
	defer g.mu.RUnlock()

	defAddr := g.DefaultBindAddress
	if defAddr == "" {
		defAddr = DefaultBindAddress
	}
	tsCIDR := g.TailscaleCIDR
	if tsCIDR == "" {
		tsCIDR = TailscaleCIDR
	}
	proh := g.ProhibitedEgressCIDRs
	if len(proh) == 0 {
		proh = ProhibitedEgressCIDRs
	}

	return ParameterEnvelope{
		DefaultBindAddress:    defAddr,
		ProhibitedEgressCIDRs: append([]string(nil), proh...),
		TailscaleCIDR:         tsCIDR,
	}
}

func (g *SocketBindingGuard) isTailscale(host string) bool {
	ip := net.ParseIP(host)
	if ip == nil {
		return false
	}

	var ts4, ts6 *net.IPNet
	if g != nil {
		g.mu.RLock()
		ts4 = g.parsedTailscaleNet
		ts6 = g.parsedTailscaleIPv6
		g.mu.RUnlock()
	}
	if ts4 == nil {
		ts4 = defaultTailscaleNet
	}
	if ts6 == nil {
		ts6 = defaultTailscaleIPv6Net
	}

	if ip4 := ip.To4(); ip4 != nil {
		return ts4.Contains(ip4)
	}
	return ts6.Contains(ip)
}

// ValidateBinding validates whether the requested bind address conforms to loopback and Tailscale
// interface constraints, returning ErrInsecureWildcardBinding for wildcard violations or
// ErrInsecureLANBinding for other non-loopback/non-Tailscale addresses unless allowInsecureLAN is true.
func (g *SocketBindingGuard) ValidateBinding(addr string, allowInsecureLAN ...bool) error {
	allow := false
	if g != nil {
		allow = g.AllowInsecureLAN
	}
	if len(allowInsecureLAN) > 0 {
		allow = allowInsecureLAN[0]
	}

	host, err := parseBindingAddress(addr)
	if err != nil {
		return err
	}

	if isWildcard(host) {
		if !allow {
			return ErrInsecureWildcardBinding
		}
		return nil
	}

	if isLoopback(host) {
		return nil
	}

	if g.isTailscale(host) {
		return nil
	}

	if !allow {
		return ErrInsecureLANBinding
	}

	return nil
}

// GuardedListen validates the bind address against zero-trust ingress constraints before calling net.Listen.
func (g *SocketBindingGuard) GuardedListen(network, addr string, allowInsecureLAN ...bool) (net.Listener, error) {
	allow := false
	if g != nil {
		allow = g.AllowInsecureLAN
	}
	if len(allowInsecureLAN) > 0 {
		allow = allowInsecureLAN[0]
	}

	if network == "" {
		network = "tcp"
	}

	if err := g.ValidateBinding(addr, allow); err != nil {
		return nil, err
	}
	return net.Listen(network, addr)
}

// IsProhibitedPrivateEgress checks whether the provided IP belongs to an RFC 1918 private subnet
// (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16) to prevent lateral movement.
func (g *SocketBindingGuard) IsProhibitedPrivateEgress(ip net.IP) bool {
	if ip == nil {
		return false
	}
	ip4 := ip.To4()
	if ip4 == nil {
		return false
	}

	var nets []*net.IPNet
	if g != nil {
		g.mu.RLock()
		nets = g.parsedProhibitedCIDRs
		g.mu.RUnlock()
	}
	if len(nets) == 0 {
		nets = defaultRFC1918Nets
	}

	for _, cidr := range nets {
		if cidr.Contains(ip4) {
			return true
		}
	}
	return false
}

// ValidateEgressTarget inspects the target address or URL and blocks connections to private RFC 1918 subnets
// unless allowPrivateLAN is true.
func (g *SocketBindingGuard) ValidateEgressTarget(targetAddr string, allowPrivateLAN ...bool) error {
	allow := false
	if g != nil {
		allow = g.AllowPrivateLAN
	}
	if len(allowPrivateLAN) > 0 {
		allow = allowPrivateLAN[0]
	}
	if allow {
		return nil
	}

	host, err := parseTargetHost(targetAddr)
	if err != nil {
		return err
	}

	if ip := net.ParseIP(host); ip != nil {
		if g.IsProhibitedPrivateEgress(ip) {
			return ErrProhibitedPrivateEgress
		}
		return nil
	}

	if isLocalhost(host) {
		return nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	ips, lookupErr := net.DefaultResolver.LookupIP(ctx, "ip", host)
	if lookupErr == nil {
		for _, ip := range ips {
			if g.IsProhibitedPrivateEgress(ip) {
				return ErrProhibitedPrivateEgress
			}
		}
	}

	return nil
}

// GuardedDialContext validates the target address against egress rules before dialing.
func (g *SocketBindingGuard) GuardedDialContext(ctx context.Context, network, addr string, allowPrivateLAN ...bool) (net.Conn, error) {
	if err := g.ValidateEgressTarget(addr, allowPrivateLAN...); err != nil {
		return nil, err
	}
	var d net.Dialer
	return d.DialContext(ctx, network, addr)
}

// GuardedDial validates the target address against egress rules before dialing.
func (g *SocketBindingGuard) GuardedDial(network, addr string, allowPrivateLAN ...bool) (net.Conn, error) {
	return g.GuardedDialContext(context.Background(), network, addr, allowPrivateLAN...)
}

// SafeHTTPTransport returns an http.Transport configured with egress target validation
// on every dial, preventing SSRF and lateral movement across RFC 1918 subnets.
func (g *SocketBindingGuard) SafeHTTPTransport(allowPrivateLAN ...bool) *http.Transport {
	return &http.Transport{
		DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
			return g.GuardedDialContext(ctx, network, addr, allowPrivateLAN...)
		},
		ForceAttemptHTTP2:     true,
		MaxIdleConns:          100,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
	}
}

// ValidateBinding validates whether addr conforms to loopback and Tailscale interface constraints.
// Returns ErrInsecureWildcardBinding when wildcard binding is attempted without opt-in,
// or ErrInsecureLANBinding when non-loopback/non-Tailscale address is attempted without opt-in.
func ValidateBinding(addr string, allowInsecureLAN bool) error {
	var guard *SocketBindingGuard
	return guard.ValidateBinding(addr, allowInsecureLAN)
}

// GuardedListen validates the bind address against zero-trust ingress constraints before calling net.Listen.
func GuardedListen(network, addr string, allowInsecureLAN bool) (net.Listener, error) {
	var guard *SocketBindingGuard
	return guard.GuardedListen(network, addr, allowInsecureLAN)
}

// IsProhibitedPrivateEgress reports whether ip falls within any RFC 1918 private subnet
// (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16).
func IsProhibitedPrivateEgress(ip net.IP) bool {
	var guard *SocketBindingGuard
	return guard.IsProhibitedPrivateEgress(ip)
}

// ValidateEgressTarget checks whether targetAddr targets an RFC 1918 private address, returning
// ErrProhibitedPrivateEgress unless allowPrivateLAN is true.
func ValidateEgressTarget(targetAddr string, allowPrivateLAN bool) error {
	var guard *SocketBindingGuard
	return guard.ValidateEgressTarget(targetAddr, allowPrivateLAN)
}

// GuardedDialContext dials an outbound connection after verifying the target address does not violate egress policy.
func GuardedDialContext(ctx context.Context, network, addr string, allowPrivateLAN bool) (net.Conn, error) {
	var guard *SocketBindingGuard
	return guard.GuardedDialContext(ctx, network, addr, allowPrivateLAN)
}

// GuardedDial dials an outbound connection after verifying the target address does not violate egress policy.
func GuardedDial(network, addr string, allowPrivateLAN bool) (net.Conn, error) {
	var guard *SocketBindingGuard
	return guard.GuardedDial(network, addr, allowPrivateLAN)
}

// SafeHTTPTransport returns an http.Transport configured with egress target validation.
func SafeHTTPTransport(allowPrivateLAN bool) *http.Transport {
	var guard *SocketBindingGuard
	return guard.SafeHTTPTransport(allowPrivateLAN)
}

// IsLoopbackAddress reports whether the address resolves to a loopback interface.
func IsLoopbackAddress(addr string) bool {
	host, err := parseBindingAddress(addr)
	if err != nil {
		return false
	}
	return isLoopback(host)
}

// IsTailscaleAddress reports whether the address falls within the Tailscale CGNAT overlay subnet.
func IsTailscaleAddress(addr string) bool {
	host, err := parseBindingAddress(addr)
	if err != nil {
		return false
	}
	var guard *SocketBindingGuard
	return guard.isTailscale(host)
}

// IsWildcardAddress reports whether the address specifies a wildcard socket binding (0.0.0.0, ::, or empty).
func IsWildcardAddress(addr string) bool {
	host, err := parseBindingAddress(addr)
	if err != nil {
		return false
	}
	return isWildcard(host)
}

func parseBindingAddress(addr string) (string, error) {
	addr = strings.TrimSpace(addr)
	if addr == "" {
		return "", nil
	}

	host, port, err := net.SplitHostPort(addr)
	if err == nil {
		if port != "" {
			p, pErr := strconv.Atoi(port)
			if pErr != nil || p < 0 || p > 65535 {
				return "", fmt.Errorf("%w: invalid port %q in address %q", ErrInvalidAddress, port, addr)
			}
		}
		return host, nil
	}

	if ip := net.ParseIP(addr); ip != nil {
		return addr, nil
	}

	if strings.HasPrefix(addr, "[") && strings.HasSuffix(addr, "]") {
		inner := addr[1 : len(addr)-1]
		if ip := net.ParseIP(inner); ip != nil {
			return inner, nil
		}
	}

	if !strings.Contains(addr, ":") && !strings.Contains(addr, "/") {
		return addr, nil
	}

	return "", fmt.Errorf("%w: %s", ErrInvalidAddress, err.Error())
}

func parseTargetHost(targetAddr string) (string, error) {
	targetAddr = strings.TrimSpace(targetAddr)
	if targetAddr == "" {
		return "", ErrInvalidAddress
	}

	if strings.Contains(targetAddr, "://") {
		u, err := url.Parse(targetAddr)
		if err != nil {
			return "", fmt.Errorf("%w: %v", ErrInvalidAddress, err)
		}
		targetAddr = u.Host
		if targetAddr == "" {
			return "", ErrInvalidAddress
		}
	} else {
		if slashIdx := strings.Index(targetAddr, "/"); slashIdx != -1 {
			targetAddr = targetAddr[:slashIdx]
		}
	}

	host, _, err := net.SplitHostPort(targetAddr)
	if err == nil {
		return host, nil
	}

	if strings.HasPrefix(targetAddr, "[") && strings.HasSuffix(targetAddr, "]") {
		return targetAddr[1 : len(targetAddr)-1], nil
	}

	return targetAddr, nil
}

func isWildcard(host string) bool {
	if host == "" || host == "0.0.0.0" || host == "::" || host == "[::]" {
		return true
	}
	ip := net.ParseIP(host)
	if ip != nil {
		return ip.IsUnspecified()
	}
	return false
}

func isLoopback(host string) bool {
	if isLocalhost(host) {
		return true
	}
	ip := net.ParseIP(host)
	if ip != nil {
		return ip.IsLoopback()
	}
	return false
}

func isLocalhost(host string) bool {
	return strings.EqualFold(host, "localhost") || strings.HasSuffix(strings.ToLower(host), ".localhost")
}
