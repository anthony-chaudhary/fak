// Package egressfloor is the structural network-egress floor: a pure classifier
// that names a tool call reaching a blocked NETWORK DESTINATION — above all the
// cloud-instance METADATA endpoint (169.254.169.254 and its peers), whose only
// purpose from inside a VM is to hand out the box's cloud IAM credentials.
//
// WHY THIS LEAF EXISTS (the "fak guard on a random VM" floor). Move a coding agent
// onto an ephemeral cloud VM and the human steps away — there is nobody to click
// "approve" on a tool call. The single most-recurring security primitive every
// sandbox/cloud-agent vendor independently re-derived is "deny-by-default egress +
// block the cloud-metadata SSRF class," because a prompt-injected agent that can
// reach 169.254.169.254 reads the instance's role credentials and walks out of the
// VM with them. This leaf is fak's structural answer: it denies the metadata/
// link-local destination class BY SHAPE, with no model and no human in the loop, so
// the kernel that travels into the VM is useful there the moment it boots.
//
// It is a tier-1 FOUNDATION leaf: pure, allocation-light, imports only the frozen
// ABI + stdlib (net, net/url) — never os/exec, never cgo — so it is safe to fold on
// the live tool-call decision path that the adjudicator's egress rung runs it from.
// The dangerous-destination set is HARDWIRED (these addresses are never a legitimate
// agent destination), so the floor needs no policy opt-in; a policy may only ever
// extend it, never carve a hole in it.
//
// Invariant: egress floor evaluation is fail-closed and default-deny; unverified or link-local destinations are denied unconditionally.
//
// Guard: Classify and EnsureSendable guard against cloud-metadata SSRF exfiltration and unverified outbound platform delivery.
package egressfloor

import (
	"net"
	"net/url"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// ReasonEgressBlock is the refusal code an egress-blocked call cites. It is an
// OUT-OF-TREE reason (above abi.ReasonCoreMax = 1023): the closed core vocabulary in
// internal/abi is human-owned and additive-only, so this leaf reserves a code in the
// registered range and the adjudicator registers its name ("EGRESS_BLOCK") at init —
// the sanctioned RegisterReason extension path, not a core edit. A deny cites this so
// the audit summary and the in-band note distinguish a metadata-SSRF block from a
// generic POLICY_BLOCK, and the operator gets a destination-specific recovery hint.
const ReasonEgressBlock abi.ReasonCode = 1024

// ReasonEgressBlockName is the stable name registered for ReasonEgressBlock.
const ReasonEgressBlockName = "EGRESS_BLOCK"

// destinationKeys are the argument keys whose VALUE is a network destination in its
// entirety (a URL or a bare host) for the common coding-agent + MCP/http tools:
// WebFetch.url, an http tool's uri/endpoint/host/base_url/webhook/…. The whole value
// is parsed as a URL or host. Lower-cased at the match site so a `URL`/`Url` key still
// matches. This is a deliberate ALLOWLIST, not a scan-everything fallback: a content /
// text / body / code arg can legitimately MENTION a metadata address (a doc, a test, a
// security note) without REACHING it, and the floor must not refuse writing such a
// file. Only an arg that NAMES a destination — or a shell command that fetches one
// (commandKeys) — is inspected.
var destinationKeys = map[string]bool{
	"url": true, "uri": true, "endpoint": true, "host": true, "hostname": true,
	"base_url": true, "baseurl": true, "base-url": true, "server": true,
	"address": true, "addr": true, "remote": true, "upstream": true,
	"target_url": true, "callback_url": true, "webhook": true, "webhook_url": true,
	"proxy": true, "proxy_url": true,
}

// commandKeys are the argument keys that carry a SHELL COMMAND LINE (Bash.command,
// PowerShell.command, a generic exec arg). A command can embed a destination inside a
// `curl`/`wget`/`Invoke-WebRequest` invocation, so its value is SCANNED for URLs and
// bare metadata IPs rather than parsed whole.
var commandKeys = map[string]bool{
	"command": true, "cmd": true, "script": true,
}

// Classify inspects a decoded tool call and returns the first blocked network
// destination it reaches, with a short human label naming the class. host == ""
// means nothing was blocked (the overwhelmingly common case — a call with no
// destination, or a destination that is not metadata/link-local). It is pure and
// cheap: it touches only string args, fast-bails on the no-destination path, and
// never allocates on a clean call beyond the args iteration the caller already did.
//
// tool is accepted for future per-tool tightening but is not required for the
// metadata floor — a blocked host is blocked regardless of which tool reaches it, so
// every tool that can carry a URL (WebFetch, Bash, an MCP http tool) is covered by
// the same set. args is the already-decoded argument map (the adjudicator's
// decodeArgs output); a nil map yields ("", "").
//
// extraDenyHosts is an OPTIONAL operator-supplied block-list (from the policy
// manifest's egress.deny_hosts): exact, case-insensitive host names/IPs to refuse IN
// ADDITION to the hardwired metadata/link-local class. It only ever TIGHTENS the floor
// — there is no way to carve a hole in the hardwired set — so a deployment can block
// its own sensitive endpoints (an internal secrets service, a corp metadata mirror)
// without a code change. Empty/nil leaves the floor at the hardwired set.
func Classify(tool string, args map[string]any, extraDenyHosts ...string) (host, label string) {
	for k, v := range args {
		s, ok := v.(string)
		if !ok || s == "" {
			continue
		}
		key := strings.ToLower(k)
		switch {
		case destinationKeys[key]:
			if h, lbl := classifyDestinationWith(s, extraDenyHosts); h != "" {
				return h, lbl
			}
		case commandKeys[key]:
			if h, lbl := classifyCommandWith(s, extraDenyHosts); h != "" {
				return h, lbl
			}
		}
	}
	return "", ""
}

// Destinations returns every candidate destination host a tool call reaches, from both
// whole-value destination args (WebFetch.url, an http tool's uri/endpoint/…) and hosts
// embedded in a shell command line (curl/wget targets). It is the EXTRACTION half of
// Classify, exposed so a higher egress layer — the operator/community allow-block lists
// in internal/egresslist — can resolve the SAME hosts the hardwired floor inspects
// without re-deriving destination parsing, keeping the two layers in agreement about
// what "the destination" of a call is. Hosts come back normalized (no scheme, no port,
// no brackets) and de-duplicated in first-seen order; a call with no destination yields
// nil. Pure and DNS-free, like Classify.
func Destinations(tool string, args map[string]any) []string {
	var out []string
	seen := map[string]bool{}
	add := func(h string) {
		h = strings.ToLower(strings.TrimSpace(h))
		if h == "" || seen[h] {
			return
		}
		seen[h] = true
		out = append(out, h)
	}
	for k, v := range args {
		s, ok := v.(string)
		if !ok || s == "" {
			continue
		}
		key := strings.ToLower(k)
		switch {
		case destinationKeys[key]:
			add(destinationHost(s))
		case commandKeys[key]:
			for _, h := range commandHosts(s) {
				add(h)
			}
		}
	}
	return out
}

// destinationHost reduces a whole-value destination arg to its bare host — a URL host
// (scheme://host/…) via hostOf, or a bare "host"/"host:port" reduced by hostnameNoPort
// (pure parsing, never DNS). Returns "" when the value names no host.
func destinationHost(v string) string {
	h := hostOf(v)
	if h == "" {
		h = hostnameNoPort(strings.TrimSpace(v))
	}
	return h
}

// commandHosts extracts every candidate destination host embedded in a shell command
// line: each http(s):// URL host, plus any bare host-shaped token (a `curl
// 169.254.169.254` with no scheme still reaches the endpoint). Pure parsing, never DNS.
func commandHosts(cmd string) []string {
	var out []string
	for _, tok := range tokenizeCommand(cmd) {
		h := hostOf(tok)
		if h == "" {
			h = hostnameNoPort(tok)
		}
		if h != "" {
			out = append(out, h)
		}
	}
	return out
}

// classifyDestinationWith treats the whole value as a single destination and classifies
// its host against the hardwired metadata set plus any operator extra-deny hosts.
func classifyDestinationWith(v string, extra []string) (host, label string) {
	h := destinationHost(v)
	if h == "" {
		return "", ""
	}
	if blocked, lbl := classifyHostWith(h, extra); blocked {
		return h, lbl
	}
	return "", ""
}

// classifyCommandWith scans a shell command line for embedded destinations and classifies
// each host against the hardwired set plus any operator extra-deny hosts. Returns the
// first blocked host.
func classifyCommandWith(cmd string, extra []string) (host, label string) {
	for _, h := range commandHosts(cmd) {
		if blocked, lbl := classifyHostWith(h, extra); blocked {
			return h, lbl
		}
	}
	return "", ""
}

// classifyHostWith folds the hardwired ClassifyHost with an operator extra-deny list:
// the hardwired metadata/link-local class first (it can never be disabled), then an
// exact, case-insensitive match against the operator's extra hosts. A bracket-wrapped
// or ported host is already reduced to a bare host by the callers, so the extra-list
// match is a plain normalized-string compare.
func classifyHostWith(host string, extra []string) (blocked bool, label string) {
	if b, lbl := ClassifyHost(host); b {
		return true, lbl
	}
	if len(extra) == 0 {
		return false, ""
	}
	h := strings.ToLower(strings.Trim(strings.TrimSpace(host), "[]"))
	for _, e := range extra {
		if h == strings.ToLower(strings.Trim(strings.TrimSpace(e), "[]")) {
			return true, "operator-denied egress destination (policy egress.deny_hosts)"
		}
	}
	return false, ""
}

// tokenizeCommand splits a command line on shell whitespace and the common
// argument-glue characters (`=`, quotes) so an embedded URL or host surfaces as its
// own token, then keeps only tokens that could be a destination (contain a dot, a
// colon, or "://"). It is deliberately simple — a destination is recognized from its
// SHAPE, not from parsing shell grammar — because the floor only needs to find a
// metadata host, not to understand the whole command.
func tokenizeCommand(cmd string) []string {
	fields := strings.FieldsFunc(cmd, func(r rune) bool {
		switch r {
		case ' ', '\t', '\n', '\r', '"', '\'', '=', '`', '(', ')', '|', '&', ';', ',', '<', '>':
			return true
		}
		return false
	})
	out := fields[:0]
	for _, f := range fields {
		if strings.ContainsAny(f, ".:") {
			out = append(out, f)
		}
	}
	return out
}

// hostOf extracts the host of a value that parses as a URL with a scheme and an
// authority (scheme://host[:port]/…). It returns "" for a value that is not a
// scheme-bearing URL, so the caller can fall back to treating the value as a bare
// host. The port is stripped; a bracketed IPv6 host is unwrapped.
func hostOf(v string) string {
	if !strings.Contains(v, "://") {
		return ""
	}
	u, err := url.Parse(strings.TrimSpace(v))
	if err != nil || u.Host == "" {
		return ""
	}
	return hostnameNoPort(u.Host)
}

// hostnameNoPort strips a :port and unwraps [..] IPv6 brackets from an authority.
// It reuses net.SplitHostPort where the value has a port and trims brackets
// otherwise, so both "169.254.169.254:80" and "[fd00:ec2::254]" reduce to the host.
func hostnameNoPort(authority string) string {
	authority = strings.TrimSpace(authority)
	if h, _, err := net.SplitHostPort(authority); err == nil {
		return strings.Trim(h, "[]")
	}
	return strings.Trim(authority, "[]")
}

// blockedHostnames are the well-known cloud-metadata HOSTNAMES that resolve to the
// instance metadata service. The IP forms (169.254.169.254 and friends) are caught
// structurally by ClassifyHost's link-local/IP checks; these are the DNS names a
// client may use instead (so blocking the IP alone would be bypassable by name).
var blockedHostnames = map[string]string{
	"metadata.google.internal": "cloud-metadata endpoint (GCP)",
	"metadata.goog":            "cloud-metadata endpoint (GCP)",
	"metadata":                 "cloud-metadata endpoint (short name)",
}

// ClassifyHost reports whether a single host (a hostname or an IP literal, no port,
// no brackets) is a blocked egress destination, with a short label naming the class.
// The blocked set is the cloud-instance metadata / link-local family:
//
//   - any link-local address — IPv4 169.254.0.0/16 (the IMDS address 169.254.169.254
//     for AWS/GCP/Azure/DigitalOcean/Oracle/OpenStack, the ECS task-metadata address
//     169.254.170.2, and the rest of the /16) and IPv6 fe80::/10 — via
//     net.IP.IsLinkLocalUnicast;
//   - the AWS IMDSv6 ULA fd00:ec2::254;
//   - the Alibaba Cloud metadata address 100.100.100.100;
//   - the metadata DNS hostnames (metadata.google.internal & friends), so a request
//     by NAME is blocked exactly as the IP is.
//
// Everything else returns (false, ""). It is allocation-free on the common
// not-blocked path (a public hostname fails the cheap hostname-map lookup and parses
// to a nil IP).
func ClassifyHost(host string) (blocked bool, label string) {
	h := strings.ToLower(strings.TrimSpace(host))
	h = strings.Trim(h, "[]")
	if h == "" {
		return false, ""
	}
	if lbl, ok := blockedHostnames[h]; ok {
		return true, lbl
	}
	ip := net.ParseIP(h)
	if ip == nil {
		// net.ParseIP accepts only canonical dotted-decimal IPv4 / standard IPv6, but a
		// libc resolver (curl/wget/most HTTP clients, via inet_aton) also dials the
		// non-canonical forms — a bare 32-bit integer, 0x-hex, octal, per-octet radices.
		// `http://2852039166/` reaches 169.254.169.254, the canonical cloud-metadata SSRF
		// bypass, so normalize those to a canonical IP before the class checks below. This
		// can only ever ADD a block for a host that decodes INTO the metadata/link-local
		// class; a real hostname (has letters) or a public integer decodes elsewhere and
		// stays allowed, so it introduces no false positive on legitimate traffic.
		ip = looseIPv4(h)
	}
	if ip == nil {
		return false, ""
	}
	switch {
	case ip.Equal(awsIMDSv6):
		return true, "cloud-metadata endpoint (AWS IMDSv6 fd00:ec2::254)"
	case ip.Equal(alibabaMetadata):
		return true, "cloud-metadata endpoint (Alibaba 100.100.100.100)"
	case ip.IsLinkLocalUnicast():
		// 169.254.0.0/16 and fe80::/10 — the instance-metadata IMDS address lives here
		// (169.254.169.254), and a link-local destination is never a legitimate target
		// for agent-issued traffic regardless.
		return true, "link-local / cloud-metadata address (169.254.0.0/16, fe80::/10)"
	}
	return false, ""
}

// looseIPv4 parses the non-canonical IPv4 literals that net.ParseIP rejects but a libc
// resolver (inet_aton, used by curl/wget and most HTTP clients) still dials: a value of
// 1–4 dot-separated parts where each part may be decimal, 0x-hexadecimal, or leading-0
// octal. The parts are combined with inet_aton's part-count-dependent byte layout (a
// lone integer is the full 32 bits; a.b spills b into the low 24 bits; a.b.c spills c
// into the low 16). It returns nil for anything that is not an unambiguous alternate-
// radix IPv4 literal — a real DNS hostname has letters and never survives ParseUint —
// so ClassifyHost can only ever ADD a metadata/link-local block from it, never a new
// false positive on a legitimate hostname or an already-canonical dotted address (those
// are handled by net.ParseIP before this runs).
func looseIPv4(h string) net.IP {
	if h == "" {
		return nil
	}
	parts := strings.Split(h, ".")
	if len(parts) > 4 {
		return nil
	}
	vals := make([]uint64, len(parts))
	for i, p := range parts {
		v, ok := parseInetPart(p)
		if !ok {
			return nil
		}
		vals[i] = v
	}
	var ip uint32
	switch len(parts) {
	case 1: // a — the whole 32-bit address
		if vals[0] > 0xFFFFFFFF {
			return nil
		}
		ip = uint32(vals[0])
	case 2: // a.b — b is the low 24 bits
		if vals[0] > 0xFF || vals[1] > 0xFFFFFF {
			return nil
		}
		ip = uint32(vals[0])<<24 | uint32(vals[1])
	case 3: // a.b.c — c is the low 16 bits
		if vals[0] > 0xFF || vals[1] > 0xFF || vals[2] > 0xFFFF {
			return nil
		}
		ip = uint32(vals[0])<<24 | uint32(vals[1])<<16 | uint32(vals[2])
	default: // a.b.c.d — one byte each
		if vals[0] > 0xFF || vals[1] > 0xFF || vals[2] > 0xFF || vals[3] > 0xFF {
			return nil
		}
		ip = uint32(vals[0])<<24 | uint32(vals[1])<<16 | uint32(vals[2])<<8 | uint32(vals[3])
	}
	return net.IPv4(byte(ip>>24), byte(ip>>16), byte(ip>>8), byte(ip))
}

// parseInetPart parses one inet_aton octet-or-larger field: a 0x/0X prefix is
// hexadecimal, a lone leading 0 (with more digits) is octal, everything else is
// decimal. It returns ok == false for an empty field or any value ParseUint rejects, so
// a non-numeric token (a hostname label) can never masquerade as a numeric part.
func parseInetPart(p string) (uint64, bool) {
	if p == "" {
		return 0, false
	}
	base := 10
	switch {
	case len(p) >= 2 && (p[:2] == "0x" || p[:2] == "0X"):
		base, p = 16, p[2:]
		if p == "" {
			return 0, false
		}
	case len(p) >= 2 && p[0] == '0':
		base, p = 8, p[1:]
	}
	v, err := strconv.ParseUint(p, base, 64)
	if err != nil {
		return 0, false
	}
	return v, true
}

var (
	awsIMDSv6       = net.ParseIP("fd00:ec2::254")
	alibabaMetadata = net.ParseIP("100.100.100.100")
)
