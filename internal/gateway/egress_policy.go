// Adapted from opensandbox-group/OpenSandbox specs/egress-api.yaml:1
// Clean-room Go policy evaluation and credential-vault placeholder substitution
// at the gateway egress boundary; no proxy server or hosted runtime coupling.

package gateway

import (
	"errors"
	"fmt"
	"net/netip"
	"sort"
	"strconv"
	"strings"
	"sync"
)

var (
	// ErrEgressInvalidTarget indicates a malformed destination host, IP, or port.
	ErrEgressInvalidTarget = errors.New("gateway: invalid egress destination")
	// ErrEgressDenied indicates outbound egress was blocked by default-deny or a deny rule.
	ErrEgressDenied = errors.New("gateway: egress destination denied")
	// ErrVaultPlaceholderMissing indicates an unresolved credential placeholder.
	ErrVaultPlaceholderMissing = errors.New("gateway: credential vault placeholder missing")
	// ErrVaultHostDisallowed indicates a credential is not authorized for the destination host.
	ErrVaultHostDisallowed = errors.New("gateway: credential not authorized for egress destination")
)

// EgressAction is an explicit allow or deny verdict.
type EgressAction string

const (
	EgressAllow EgressAction = "allow"
	EgressDeny  EgressAction = "deny"
)

// EgressRule is one ordered outbound policy rule matching either a CIDR/IP
// (normalized to netip.Prefix) or an exact/wildcard domain name.
type EgressRule struct {
	Action EgressAction
	Target string
	Ports  []uint16
}

// EgressPolicyConfig configures a default-deny EgressPolicy.
type EgressPolicyConfig struct {
	DefaultAction EgressAction
	Rules         []EgressRule
}

type EgressDestination struct {
	Host string
	IP   netip.Addr
	IsIP bool
	Port uint16
}

// EgressDecision is the deterministic policy verdict for a destination.
type EgressDecision struct {
	Action      EgressAction
	Allowed     bool
	Destination EgressDestination
	MatchedRule int // -1 when default-deny applied
	Reason      string
}

type egressTargetMatcher struct {
	raw        string
	isCIDR     bool
	prefix     netip.Prefix
	isWildcard bool
	domain     string // exact domain, or suffix without "*." when isWildcard
}

type compiledEgressRule struct {
	action  EgressAction
	matcher egressTargetMatcher
	ports   map[uint16]struct{}
}

// EgressPolicy evaluates outbound egress requests against ordered CIDR and
// domain/wildcard rules under a strict default-deny posture.
type EgressPolicy struct {
	defaultAction EgressAction
	rules         []compiledEgressRule
}

// NewEgressPolicy compiles and validates cfg under default-deny semantics.
func NewEgressPolicy(cfg EgressPolicyConfig) (*EgressPolicy, error) {
	def := cfg.DefaultAction
	if def == "" {
		def = EgressDeny
	}
	if def != EgressDeny {
		return nil, fmt.Errorf("gateway: egress policy default action must be %q, got %q", EgressDeny, cfg.DefaultAction)
	}

	compiled := make([]compiledEgressRule, 0, len(cfg.Rules))
	for i, r := range cfg.Rules {
		if r.Action != EgressAllow && r.Action != EgressDeny {
			return nil, fmt.Errorf("gateway: egress rule %d has invalid action %q", i, r.Action)
		}
		m, err := compileEgressTarget(r.Target)
		if err != nil {
			return nil, fmt.Errorf("gateway: egress rule %d target %q: %w", i, r.Target, err)
		}
		var ports map[uint16]struct{}
		if len(r.Ports) > 0 {
			ports = make(map[uint16]struct{}, len(r.Ports))
			for _, p := range r.Ports {
				if p == 0 {
					return nil, fmt.Errorf("gateway: egress rule %d has invalid port 0", i)
				}
				ports[p] = struct{}{}
			}
		}
		compiled = append(compiled, compiledEgressRule{
			action:  r.Action,
			matcher: m,
			ports:   ports,
		})
	}
	return &EgressPolicy{
		defaultAction: EgressDeny,
		rules:         compiled,
	}, nil
}

// ParseEgressDestination parses and normalizes host/IP and optional port
// ("example.com:443", "[2001:db8::1]:443", "::1", "10.0.0.1:8080"), rejecting
// invalid ports, URL userinfo/paths, IPv6 zone IDs, and unmapping IPv4-mapped
// IPv6 addresses so IPv4 CIDR rules cannot be bypassed via ::ffff:a.b.c.d.
func ParseEgressDestination(target string) (EgressDestination, error) {
	s := strings.TrimSpace(target)
	if s == "" {
		return EgressDestination{}, fmt.Errorf("%w: empty destination", ErrEgressInvalidTarget)
	}
	if strings.ContainsAny(s, " \t\r\n/@?#\\") {
		return EgressDestination{}, fmt.Errorf("%w: illegal character in %q", ErrEgressInvalidTarget, target)
	}

	var hostPart string
	var port uint16

	switch {
	case strings.HasPrefix(s, "["):
		end := strings.IndexByte(s, ']')
		if end < 0 {
			return EgressDestination{}, fmt.Errorf("%w: unclosed IPv6 bracket in %q", ErrEgressInvalidTarget, target)
		}
		inside := s[1:end]
		rest := s[end+1:]
		if !strings.Contains(inside, ":") {
			return EgressDestination{}, fmt.Errorf("%w: bracketed host must be IPv6 literal in %q", ErrEgressInvalidTarget, target)
		}
		if rest != "" {
			if !strings.HasPrefix(rest, ":") {
				return EgressDestination{}, fmt.Errorf("%w: unexpected trailing data after bracket in %q", ErrEgressInvalidTarget, target)
			}
			p, err := parseStrictPort(rest[1:])
			if err != nil {
				return EgressDestination{}, err
			}
			port = p
		}
		addr, err := parseNormalizedIP(inside)
		if err != nil {
			return EgressDestination{}, err
		}
		return EgressDestination{Host: addr.String(), IP: addr, IsIP: true, Port: port}, nil

	case strings.Count(s, ":") > 1:
		// Unbracketed IPv6 literal without port.
		addr, err := parseNormalizedIP(s)
		if err != nil {
			return EgressDestination{}, err
		}
		return EgressDestination{Host: addr.String(), IP: addr, IsIP: true, Port: 0}, nil

	default:
		if colon := strings.IndexByte(s, ':'); colon >= 0 {
			hostPart = s[:colon]
			p, err := parseStrictPort(s[colon+1:])
			if err != nil {
				return EgressDestination{}, err
			}
			port = p
		} else {
			hostPart = s
		}
	}

	if addr, err := netip.ParseAddr(hostPart); err == nil {
		if addr.Zone() != "" {
			return EgressDestination{}, fmt.Errorf("%w: IP zone is not permitted in %q", ErrEgressInvalidTarget, target)
		}
		addr = addr.Unmap()
		return EgressDestination{Host: addr.String(), IP: addr, IsIP: true, Port: port}, nil
	}

	dom, err := normalizeDomainHost(hostPart)
	if err != nil {
		return EgressDestination{}, fmt.Errorf("%w: %v", ErrEgressInvalidTarget, err)
	}
	return EgressDestination{Host: dom, Port: port}, nil
}

func parseStrictPort(raw string) (uint16, error) {
	if raw == "" {
		return 0, fmt.Errorf("%w: empty port", ErrEgressInvalidTarget)
	}
	for i := 0; i < len(raw); i++ {
		if raw[i] < '0' || raw[i] > '9' {
			return 0, fmt.Errorf("%w: non-numeric port %q", ErrEgressInvalidTarget, raw)
		}
	}
	v, err := strconv.Atoi(raw)
	if err != nil || v < 1 || v > 65535 {
		return 0, fmt.Errorf("%w: port %q out of range 1..65535", ErrEgressInvalidTarget, raw)
	}
	return uint16(v), nil
}

func parseNormalizedIP(raw string) (netip.Addr, error) {
	addr, err := netip.ParseAddr(raw)
	if err != nil {
		return netip.Addr{}, fmt.Errorf("%w: invalid IP %q", ErrEgressInvalidTarget, raw)
	}
	if addr.Zone() != "" {
		return netip.Addr{}, fmt.Errorf("%w: IPv6 zone is not permitted in %q", ErrEgressInvalidTarget, raw)
	}
	return addr.Unmap(), nil
}

func normalizeDomainHost(raw string) (string, error) {
	s := strings.ToLower(strings.TrimSpace(raw))
	s = strings.TrimSuffix(s, ".")
	if s == "" || len(s) > 253 {
		return "", fmt.Errorf("invalid domain %q", raw)
	}
	if strings.HasPrefix(s, ".") || strings.HasSuffix(s, ".") {
		return "", fmt.Errorf("invalid leading/trailing dot in domain %q", raw)
	}
	labels := strings.Split(s, ".")
	for _, lbl := range labels {
		if len(lbl) == 0 || len(lbl) > 63 {
			return "", fmt.Errorf("invalid domain label in %q", raw)
		}
		if lbl[0] == '-' || lbl[len(lbl)-1] == '-' {
			return "", fmt.Errorf("domain label cannot start/end with hyphen in %q", raw)
		}
		for i := 0; i < len(lbl); i++ {
			c := lbl[i]
			if !((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-') {
				return "", fmt.Errorf("illegal domain character %q in %q", c, raw)
			}
		}
	}
	return s, nil
}

func compileEgressTarget(raw string) (egressTargetMatcher, error) {
	s := strings.TrimSpace(raw)
	if s == "" {
		return egressTargetMatcher{}, errors.New("empty rule target")
	}
	if strings.ContainsAny(s, " \t\r\n@?#\\[]") {
		return egressTargetMatcher{}, fmt.Errorf("illegal character in rule target %q", raw)
	}

	if strings.Contains(s, "/") {
		pfx, err := netip.ParsePrefix(s)
		if err != nil {
			return egressTargetMatcher{}, fmt.Errorf("invalid CIDR %q: %w", raw, err)
		}
		if pfx.Addr().Zone() != "" {
			return egressTargetMatcher{}, fmt.Errorf("CIDR zone not permitted in %q", raw)
		}
		addr := pfx.Addr()
		bits := pfx.Bits()
		if addr.Is4In6() {
			if bits < 96 {
				return egressTargetMatcher{}, fmt.Errorf("ambiguous IPv4-mapped CIDR %q", raw)
			}
			addr = addr.Unmap()
			bits -= 96
		}
		norm := netip.PrefixFrom(addr, bits).Masked()
		return egressTargetMatcher{raw: raw, isCIDR: true, prefix: norm}, nil
	}

	if addr, err := netip.ParseAddr(s); err == nil {
		if addr.Zone() != "" {
			return egressTargetMatcher{}, fmt.Errorf("IP zone not permitted in %q", raw)
		}
		addr = addr.Unmap()
		return egressTargetMatcher{
			raw:    raw,
			isCIDR: true,
			prefix: netip.PrefixFrom(addr, addr.BitLen()),
		}, nil
	}

	if strings.Contains(s, ":") {
		return egressTargetMatcher{}, fmt.Errorf("rule target %q must not embed port or malformed IPv6", raw)
	}

	if strings.HasPrefix(s, "*.") {
		suffix, err := normalizeDomainHost(s[2:])
		if err != nil {
			return egressTargetMatcher{}, fmt.Errorf("invalid wildcard domain %q: %w", raw, err)
		}
		return egressTargetMatcher{
			raw:        raw,
			isWildcard: true,
			domain:     suffix,
		}, nil
	}

	if strings.Contains(s, "*") {
		return egressTargetMatcher{}, fmt.Errorf("wildcard only supported as leading '*.' label in %q", raw)
	}

	dom, err := normalizeDomainHost(s)
	if err != nil {
		return egressTargetMatcher{}, err
	}
	return egressTargetMatcher{raw: raw, domain: dom}, nil
}

func (m egressTargetMatcher) matches(dest EgressDestination) bool {
	if m.isCIDR {
		return dest.IsIP && m.prefix.Contains(dest.IP)
	}
	if dest.IsIP {
		return false
	}
	if m.isWildcard {
		return strings.HasSuffix(dest.Host, "."+m.domain)
	}
	return dest.Host == m.domain
}

// Evaluate checks target against the compiled policy in rule order. A nil policy
// or empty/invalid target fails closed.
func (p *EgressPolicy) Evaluate(target string) (EgressDecision, error) {
	dest, err := ParseEgressDestination(target)
	if err != nil {
		return EgressDecision{Action: EgressDeny, Allowed: false, MatchedRule: -1, Reason: err.Error()}, err
	}
	if p == nil {
		return EgressDecision{
			Action:      EgressDeny,
			Allowed:     false,
			Destination: dest,
			MatchedRule: -1,
			Reason:      "nil egress policy (default deny)",
		}, nil
	}
	for i, rule := range p.rules {
		if !rule.matcher.matches(dest) {
			continue
		}
		if len(rule.ports) > 0 {
			if dest.Port == 0 {
				continue
			}
			if _, ok := rule.ports[dest.Port]; !ok {
				continue
			}
		}
		allowed := rule.action == EgressAllow
		reason := fmt.Sprintf("matched rule %d (%s %s)", i, rule.action, rule.matcher.raw)
		return EgressDecision{
			Action:      rule.action,
			Allowed:     allowed,
			Destination: dest,
			MatchedRule: i,
			Reason:      reason,
		}, nil
	}
	return EgressDecision{
		Action:      EgressDeny,
		Allowed:     false,
		Destination: dest,
		MatchedRule: -1,
		Reason:      "default deny (no matching egress rule)",
	}, nil
}

// VaultCredential registers one secret placeholder and the destination hosts it
// is authorized to be injected toward at the gateway egress boundary.
type VaultCredential struct {
	Name         string
	Placeholder  string
	Secret       string `json:"-"`
	AllowedHosts []string
}

// String redacts Secret to prevent accidental credential leakage in logs or errors.
func (c VaultCredential) String() string {
	return fmt.Sprintf("VaultCredential{Name:%q Placeholder:%q Secret:<redacted> AllowedHosts:%v}",
		c.Name, c.Placeholder, c.AllowedHosts)
}

type compiledVaultCredential struct {
	name        string
	placeholder string
	secret      string
	matchers    []egressTargetMatcher
}

func (c *compiledVaultCredential) allowsDestination(dest EgressDestination) bool {
	for _, m := range c.matchers {
		if m.matches(dest) {
			return true
		}
	}
	return false
}

// EgressCredentialVault holds credential placeholders and injects real secret
// values only when resolving at an authorized egress destination. Safe for
// concurrent use.
type EgressCredentialVault struct {
	mu           sync.RWMutex
	byToken      map[string]*compiledVaultCredential
	placeholders []string // sorted longest-first for deterministic single-pass scanning
}

// NewEgressCredentialVault validates and stores creds. Empty secrets, missing
// AllowedHosts, or duplicate placeholders fail closed.
func NewEgressCredentialVault(creds ...VaultCredential) (*EgressCredentialVault, error) {
	v := &EgressCredentialVault{
		byToken: make(map[string]*compiledVaultCredential, len(creds)),
	}
	for _, c := range creds {
		if err := v.Add(c); err != nil {
			return nil, err
		}
	}
	return v, nil
}

// Add registers a credential in the vault.
func (v *EgressCredentialVault) Add(cred VaultCredential) error {
	if v == nil {
		return errors.New("gateway: cannot add credential to nil vault")
	}
	name := strings.TrimSpace(cred.Name)
	if name == "" {
		return errors.New("gateway: credential name must be non-empty")
	}
	if cred.Secret == "" {
		return fmt.Errorf("gateway: credential %q secret must be non-empty", name)
	}
	if len(cred.AllowedHosts) == 0 {
		return fmt.Errorf("gateway: credential %q must declare at least one AllowedHosts rule", name)
	}
	placeholder := strings.TrimSpace(cred.Placeholder)
	if placeholder == "" {
		placeholder = "{{" + name + "}}"
	}
	if strings.ContainsAny(placeholder, " \t\r\n") {
		return fmt.Errorf("gateway: credential %q placeholder contains whitespace", name)
	}

	matchers := make([]egressTargetMatcher, 0, len(cred.AllowedHosts))
	for _, h := range cred.AllowedHosts {
		m, err := compileEgressTarget(h)
		if err != nil {
			return fmt.Errorf("gateway: credential %q allowed host %q: %w", name, h, err)
		}
		matchers = append(matchers, m)
	}

	v.mu.Lock()
	defer v.mu.Unlock()
	if _, exists := v.byToken[placeholder]; exists {
		return fmt.Errorf("gateway: duplicate credential placeholder %q", placeholder)
	}
	v.byToken[placeholder] = &compiledVaultCredential{
		name:        name,
		placeholder: placeholder,
		secret:      cred.Secret,
		matchers:    matchers,
	}
	v.placeholders = append(v.placeholders, placeholder)
	sort.Slice(v.placeholders, func(i, j int) bool {
		if len(v.placeholders[i]) != len(v.placeholders[j]) {
			return len(v.placeholders[i]) > len(v.placeholders[j])
		}
		return v.placeholders[i] < v.placeholders[j]
	})
	return nil
}

// ResolvePlaceholder resolves a single placeholder token for target, failing
// closed on invalid target, unknown placeholder, or unauthorized destination host.
func (v *EgressCredentialVault) ResolvePlaceholder(target, placeholder string) (string, error) {
	dest, err := ParseEgressDestination(target)
	if err != nil {
		return "", err
	}
	if v == nil {
		return "", fmt.Errorf("%w: %q", ErrVaultPlaceholderMissing, placeholder)
	}
	v.mu.RLock()
	cred, ok := v.byToken[placeholder]
	v.mu.RUnlock()
	if !ok {
		return "", fmt.Errorf("%w: %q", ErrVaultPlaceholderMissing, placeholder)
	}
	if !cred.allowsDestination(dest) {
		return "", fmt.Errorf("%w: credential %q cannot be sent to %q", ErrVaultHostDisallowed, cred.name, dest.Host)
	}
	return cred.secret, nil
}

// Substitute performs single-pass fail-closed substitution of credential
// placeholders in input for target. Any unregistered {{...}} token or host
// mismatch returns an error without revealing secret values.
func (v *EgressCredentialVault) Substitute(target, input string) (string, error) {
	dest, err := ParseEgressDestination(target)
	if err != nil {
		return "", err
	}
	if input == "" {
		return "", nil
	}
	if v == nil {
		if tok, found := nextGenericPlaceholder(input, 0); found {
			return "", fmt.Errorf("%w: %q", ErrVaultPlaceholderMissing, tok)
		}
		return input, nil
	}

	v.mu.RLock()
	defer v.mu.RUnlock()

	var b strings.Builder
	cursor := 0
	for cursor < len(input) {
		matchPos := -1
		matchToken := ""
		var matchCred *compiledVaultCredential

		for _, tok := range v.placeholders {
			if idx := strings.Index(input[cursor:], tok); idx >= 0 {
				abs := cursor + idx
				if matchPos < 0 || abs < matchPos || (abs == matchPos && len(tok) > len(matchToken)) {
					matchPos = abs
					matchToken = tok
					matchCred = v.byToken[tok]
				}
			}
		}

		// Also scan for generic {{...}} tokens so unresolved placeholders fail closed
		// instead of leaking through to an outbound request.
		if genIdx, genTok, found := findGenericPlaceholder(input, cursor); found {
			if matchPos < 0 || genIdx < matchPos {
				matchPos = genIdx
				matchToken = genTok
				matchCred = v.byToken[genTok]
			}
		}

		if matchPos < 0 {
			if b.Len() == 0 {
				return input, nil
			}
			b.WriteString(input[cursor:])
			return b.String(), nil
		}

		if matchCred == nil {
			return "", fmt.Errorf("%w: %q", ErrVaultPlaceholderMissing, matchToken)
		}
		if !matchCred.allowsDestination(dest) {
			return "", fmt.Errorf("%w: credential %q cannot be sent to %q", ErrVaultHostDisallowed, matchCred.name, dest.Host)
		}
		if b.Len() == 0 {
			b.Grow(len(input) + len(matchCred.secret))
		}
		b.WriteString(input[cursor:matchPos])
		b.WriteString(matchCred.secret)
		cursor = matchPos + len(matchToken)
	}
	return b.String(), nil
}

// SubstituteHeaders returns a fresh header map with all placeholder tokens
// resolved for target, or an error if any header contains a missing or
// disallowed-host placeholder.
func (v *EgressCredentialVault) SubstituteHeaders(target string, headers map[string]string) (map[string]string, error) {
	if _, err := ParseEgressDestination(target); err != nil {
		return nil, err
	}
	if len(headers) == 0 {
		return nil, nil
	}
	out := make(map[string]string, len(headers))
	for k, val := range headers {
		resolved, err := v.Substitute(target, val)
		if err != nil {
			return nil, err
		}
		out[k] = resolved
	}
	return out, nil
}

func nextGenericPlaceholder(s string, start int) (string, bool) {
	_, tok, ok := findGenericPlaceholder(s, start)
	return tok, ok
}

func findGenericPlaceholder(s string, start int) (int, string, bool) {
	for i := start; i < len(s); {
		open := strings.Index(s[i:], "{{")
		if open < 0 {
			return 0, "", false
		}
		absOpen := i + open
		closeRel := strings.Index(s[absOpen+2:], "}}")
		if closeRel < 0 {
			return 0, "", false
		}
		inner := s[absOpen+2 : absOpen+2+closeRel]
		if isPlaceholderName(inner) {
			return absOpen, s[absOpen : absOpen+2+closeRel+2], true
		}
		i = absOpen + 2
	}
	return 0, "", false
}

func isPlaceholderName(inner string) bool {
	if inner == "" {
		return false
	}
	for i := 0; i < len(inner); i++ {
		c := inner[i]
		switch {
		case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c >= '0' && c <= '9':
		case c == '_', c == '.', c == ':', c == '-', c == '/':
		default:
			return false
		}
	}
	return true
}

// EgressBoundary pairs an EgressPolicy with an EgressCredentialVault so
// policy enforcement precedes credential injection on every outbound call.
type EgressBoundary struct {
	Policy *EgressPolicy
	Vault  *EgressCredentialVault
}

// AuthorizeAndSubstituteHeaders enforces policy first (failing closed on deny
// or nil policy) and only then resolves credential placeholders for target.
func (b *EgressBoundary) AuthorizeAndSubstituteHeaders(target string, headers map[string]string) (EgressDecision, map[string]string, error) {
	if b == nil || b.Policy == nil {
		dec, err := (*EgressPolicy)(nil).Evaluate(target)
		if err != nil {
			return dec, nil, err
		}
		return dec, nil, ErrEgressDenied
	}
	dec, err := b.Policy.Evaluate(target)
	if err != nil {
		return dec, nil, err
	}
	if !dec.Allowed {
		return dec, nil, fmt.Errorf("%w: %s", ErrEgressDenied, dec.Reason)
	}
	resolved, err := b.Vault.SubstituteHeaders(target, headers)
	if err != nil {
		return dec, nil, err
	}
	return dec, resolved, nil
}
