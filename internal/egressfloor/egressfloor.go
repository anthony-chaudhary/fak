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
package egressfloor

import (
	"net"
	"net/url"
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
func Classify(tool string, args map[string]any) (host, label string) {
	for k, v := range args {
		s, ok := v.(string)
		if !ok || s == "" {
			continue
		}
		key := strings.ToLower(k)
		switch {
		case destinationKeys[key]:
			if h, lbl := classifyDestination(s); h != "" {
				return h, lbl
			}
		case commandKeys[key]:
			if h, lbl := classifyCommand(s); h != "" {
				return h, lbl
			}
		}
	}
	return "", ""
}

// classifyDestination treats the whole value as a single destination — a URL
// (scheme://host/…) or a bare host[:port] — and classifies its host. A scheme-bearing
// value yields its host via hostOf; a bare "host" or "host:port" (no scheme) is
// reduced by hostnameNoPort (pure parsing, never DNS).
func classifyDestination(v string) (host, label string) {
	h := hostOf(v)
	if h == "" {
		h = hostnameNoPort(strings.TrimSpace(v))
	}
	if h == "" {
		return "", ""
	}
	if blocked, lbl := ClassifyHost(h); blocked {
		return h, lbl
	}
	return "", ""
}

// classifyCommand scans a shell command line for embedded destinations: every
// http(s):// URL, plus any bare metadata/link-local IP token (a `curl 169.254.169.254`
// with no scheme still reaches the endpoint). Returns the first blocked host found.
func classifyCommand(cmd string) (host, label string) {
	for _, tok := range tokenizeCommand(cmd) {
		// A scheme-bearing URL (curl http://169.254.169.254/…) yields its host; a bare
		// token (curl 169.254.169.254, curl metadata.google.internal:80) is reduced to
		// host[:port] form by hostnameNoPort (pure parsing — net.SplitHostPort, never DNS).
		h := hostOf(tok)
		if h == "" {
			h = hostnameNoPort(tok)
		}
		if h == "" {
			continue
		}
		if blocked, lbl := ClassifyHost(h); blocked {
			return h, lbl
		}
	}
	return "", ""
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

var (
	awsIMDSv6       = net.ParseIP("fd00:ec2::254")
	alibabaMetadata = net.ParseIP("100.100.100.100")
)
