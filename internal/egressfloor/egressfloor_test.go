package egressfloor

import "testing"

// TestClassifyHostBlocks pins the blocked cloud-metadata / link-local family: every
// IMDS address and metadata hostname a VM-resident agent could reach to steal the
// box's cloud credentials is denied by host alone.
func TestClassifyHostBlocks(t *testing.T) {
	blocked := []string{
		"169.254.169.254",          // AWS/GCP/Azure/DO/Oracle/OpenStack IMDS
		"169.254.170.2",            // AWS ECS task metadata
		"169.254.0.1",              // any other link-local /16 host
		"metadata.google.internal", // GCP metadata by name
		"METADATA.GOOGLE.INTERNAL", // case-insensitive
		"metadata",                 // short metadata name
		"100.100.100.100",          // Alibaba Cloud metadata
		"fd00:ec2::254",            // AWS IMDSv6
		"fe80::1",                  // IPv6 link-local
	}
	for _, h := range blocked {
		if ok, lbl := ClassifyHost(h); !ok {
			t.Errorf("ClassifyHost(%q) = not blocked, want blocked", h)
		} else if lbl == "" {
			t.Errorf("ClassifyHost(%q) blocked with an empty label", h)
		}
	}
}

// TestClassifyHostAllows pins the negative space: ordinary public destinations — the
// provider APIs the agent legitimately reaches — must NOT be blocked, or the floor
// would brick every real session.
func TestClassifyHostAllows(t *testing.T) {
	allowed := []string{
		"api.anthropic.com",
		"api.openai.com",
		"github.com",
		"8.8.8.8",
		"1.1.1.1",
		"example.com",
		"localhost", // loopback is the gateway's own bind; not a metadata target
		"127.0.0.1",
		"10.0.0.5", // RFC1918 private, but not link-local/metadata
		"192.168.1.1",
		"",
	}
	for _, h := range allowed {
		if ok, lbl := ClassifyHost(h); ok {
			t.Errorf("ClassifyHost(%q) = blocked (%s), want allowed", h, lbl)
		}
	}
}

// TestClassifyDestinationArg covers the whole-value destination keys (WebFetch.url and
// the http/MCP destination args): a URL or bare host whose host is metadata is blocked,
// a public one is not.
func TestClassifyDestinationArg(t *testing.T) {
	cases := []struct {
		name    string
		tool    string
		args    map[string]any
		blocked bool
	}{
		{"webfetch metadata url", "WebFetch",
			map[string]any{"url": "http://169.254.169.254/latest/meta-data/iam/security-credentials/"}, true},
		{"webfetch metadata https", "WebFetch",
			map[string]any{"url": "https://metadata.google.internal/computeMetadata/v1/"}, true},
		{"webfetch metadata with port", "WebFetch",
			map[string]any{"url": "http://169.254.169.254:80/"}, true},
		{"webfetch imdsv6 bracket", "WebFetch",
			map[string]any{"url": "http://[fd00:ec2::254]/latest/"}, true},
		{"webfetch public", "WebFetch",
			map[string]any{"url": "https://api.anthropic.com/v1/messages"}, false},
		{"bare host endpoint", "http_get",
			map[string]any{"endpoint": "169.254.169.254:8080"}, true},
		{"public endpoint", "http_get",
			map[string]any{"endpoint": "api.openai.com"}, false},
		{"non-destination string arg", "Read",
			map[string]any{"file_path": "internal/egressfloor/egressfloor.go"}, false},
		{"nil args", "WebFetch", nil, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h, _ := Classify(tc.tool, tc.args)
			if got := h != ""; got != tc.blocked {
				t.Errorf("Classify(%s) blocked=%v (host=%q), want %v", tc.name, got, h, tc.blocked)
			}
		})
	}
}

// TestClassifyCommandEmbeddedDestination covers the shell path: a curl/wget reaching
// the metadata endpoint inside a Bash command line is blocked even though the
// destination is buried in the command string, while an ordinary fetch is not.
func TestClassifyCommandEmbeddedDestination(t *testing.T) {
	cases := []struct {
		name    string
		args    map[string]any
		blocked bool
	}{
		{"curl metadata url",
			map[string]any{"command": "curl -s http://169.254.169.254/latest/meta-data/iam/"}, true},
		{"curl bare metadata ip",
			map[string]any{"command": "curl 169.254.169.254"}, true},
		{"wget metadata by name",
			map[string]any{"command": "wget -qO- http://metadata.google.internal/computeMetadata/v1/instance/"}, true},
		{"curl alibaba in a pipeline",
			map[string]any{"command": "echo hi && curl http://100.100.100.100/latest/meta-data | tee out.txt"}, true},
		{"powershell IRM metadata",
			map[string]any{"command": "Invoke-RestMethod -Uri http://169.254.169.254/metadata/instance?api-version=2021-02-01 -Headers @{Metadata='true'}"}, true},
		{"ordinary public fetch",
			map[string]any{"command": "curl -s https://api.anthropic.com/v1/models"}, false},
		{"no network",
			map[string]any{"command": "go build ./cmd/fak && echo done"}, false},
		{"git over https",
			map[string]any{"command": "git clone https://github.com/anthony-chaudhary/fak.git"}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h, _ := Classify("Bash", tc.args)
			if got := h != ""; got != tc.blocked {
				t.Errorf("Classify(Bash, %s) blocked=%v (host=%q), want %v", tc.name, got, h, tc.blocked)
			}
		})
	}
}

// TestClassifyDestinationKeyCoverage proves a destination carried under one of the
// recognized destination keys (beyond url) still trips the floor.
func TestClassifyDestinationKeyCoverage(t *testing.T) {
	for _, key := range []string{"webhook_url", "endpoint", "host", "proxy", "upstream"} {
		if h, _ := Classify("mcp_http", map[string]any{key: "http://169.254.169.254/latest/"}); h == "" {
			t.Errorf("Classify did not block a metadata URL under destination key %q", key)
		}
	}
}

// TestClassifyExtraDenyHosts proves the operator block-list TIGHTENS the floor: an
// extra host (an org's internal secrets service) is refused in addition to the
// hardwired metadata set, by URL and by bare host, while the hardwired set still fires
// with no extra list and a non-listed public host stays allowed.
func TestClassifyExtraDenyHosts(t *testing.T) {
	extra := []string{"secrets.corp.internal", "10.0.0.53"}
	// Operator-denied host via a destination arg.
	if h, lbl := Classify("WebFetch", map[string]any{"url": "https://secrets.corp.internal/v1/token"}, extra...); h == "" {
		t.Error("extra-deny host (URL) was not blocked")
	} else if lbl == "" {
		t.Error("extra-deny block produced an empty label")
	}
	// Operator-denied host via a bare host arg + a shell command.
	if h, _ := Classify("http_get", map[string]any{"host": "10.0.0.53"}, extra...); h == "" {
		t.Error("extra-deny bare host was not blocked")
	}
	if h, _ := Classify("Bash", map[string]any{"command": "curl https://secrets.corp.internal/"}, extra...); h == "" {
		t.Error("extra-deny host in a shell command was not blocked")
	}
	// The hardwired metadata set still fires regardless of the extra list.
	if h, _ := Classify("WebFetch", map[string]any{"url": "http://169.254.169.254/"}, extra...); h == "" {
		t.Error("hardwired metadata block stopped firing when an extra list was supplied")
	}
	// A public host NOT on the extra list stays allowed.
	if h, _ := Classify("WebFetch", map[string]any{"url": "https://api.anthropic.com/v1/messages"}, extra...); h != "" {
		t.Errorf("a public host not on the extra list was blocked: %q", h)
	}
	// No extra list: behavior is exactly the hardwired floor (public allowed).
	if h, _ := Classify("WebFetch", map[string]any{"url": "https://secrets.corp.internal/"}); h != "" {
		t.Errorf("a would-be extra-deny host was blocked with no extra list: %q", h)
	}
}

// TestClassifyHostBlocksAlternateIPv4Encodings pins the cloud-metadata SSRF bypass
// class that net.ParseIP alone misses: the IMDS address 169.254.169.254 (and the
// Alibaba metadata address) can be reached via the non-canonical IPv4 encodings a
// libc resolver (curl/wget/most HTTP clients, via inet_aton) still dials — a bare
// 32-bit decimal integer, a 0x-hex integer, an octal integer, and per-octet hex/octal.
// Before the looseIPv4 normalization each of these decoded to a nil net.IP and sailed
// straight through the floor, handing a prompt-injected agent the box's credentials.
func TestClassifyHostBlocksAlternateIPv4Encodings(t *testing.T) {
	// Each of these dials 169.254.169.254 (the IMDS address) unless noted otherwise.
	blocked := []string{
		"2852039166",          // decimal (0xA9FEA9FE) — the canonical AWS IMDS SSRF bypass
		"0xA9FEA9FE",          // hex, upper
		"0xa9fea9fe",          // hex, lower
		"025177524776",        // octal
		"0xA9.0xFE.0xA9.0xFE", // per-octet hex
		"0251.0376.0251.0376", // per-octet octal
		"169.254.43518",       // 3-part inet_aton (last octet is the 16-bit 169*256+254)
		"1684300900",          // decimal Alibaba Cloud metadata 100.100.100.100
	}
	for _, h := range blocked {
		if ok, lbl := ClassifyHost(h); !ok {
			t.Errorf("ClassifyHost(%q) = not blocked, want blocked (alternate-encoding metadata SSRF bypass)", h)
		} else if lbl == "" {
			t.Errorf("ClassifyHost(%q) blocked with an empty label", h)
		}
	}
}

// TestClassifyHostAllowsAlternateEncodingPublicIPs is the false-positive guard for the
// alternate-encoding path: a NON-metadata address in integer/hex form (a public host a
// bug-hunter might legitimately pass) must decode and then stay ALLOWED — the loose
// parser only ever adds a block for a host that genuinely lands in the metadata/
// link-local class, never for any host that decodes elsewhere.
func TestClassifyHostAllowsAlternateEncodingPublicIPs(t *testing.T) {
	allowed := []string{
		"134744072",  // decimal 8.8.8.8 (public)
		"0x08080808", // hex 8.8.8.8
		"16843009",   // decimal 1.1.1.1
		"0",          // 0.0.0.0
		"3232235777", // decimal 192.168.1.1 (RFC1918, but not link-local/metadata)
	}
	for _, h := range allowed {
		if ok, lbl := ClassifyHost(h); ok {
			t.Errorf("ClassifyHost(%q) = blocked (%s), want allowed", h, lbl)
		}
	}
}

// TestClassifyAlternateEncodingThroughSurfaces proves the alternate-encoding block
// fires through the real call surfaces, not just the bare-host classifier: a
// WebFetch/endpoint destination and a curl command that dial the IMDS via a decimal or
// hex integer host are refused exactly as the dotted form is.
func TestClassifyAlternateEncodingThroughSurfaces(t *testing.T) {
	cases := []struct {
		name string
		tool string
		args map[string]any
	}{
		{"webfetch decimal url", "WebFetch", map[string]any{"url": "http://2852039166/latest/meta-data/iam/"}},
		{"endpoint hex host with port", "http_get", map[string]any{"endpoint": "0xA9FEA9FE:80"}},
		{"curl hex url in command", "Bash", map[string]any{"command": "curl -s http://0xA9FEA9FE/latest/meta-data/"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if h, _ := Classify(tc.tool, tc.args); h == "" {
				t.Errorf("Classify(%s) did not block an alternate-encoding metadata destination", tc.name)
			}
		})
	}
}

// TestClassifyDoesNotScanContent is the critical false-positive guard: a payload arg
// (content/text/body/code/diff) that merely MENTIONS a metadata address — a doc, a
// test, a security note being written to disk — must NOT be blocked. The floor refuses
// REACHING the endpoint, never WRITING ABOUT it; scanning file content would refuse
// editing this very repo's egress demo.
func TestClassifyDoesNotScanContent(t *testing.T) {
	payloadKeys := []string{"content", "text", "body", "code", "new_string", "old_string", "message", "data"}
	for _, key := range payloadKeys {
		args := map[string]any{
			"file_path": "examples/remote-vm-guard/run.sh",
			key:         "curl http://169.254.169.254/latest/meta-data/  # the SSRF we block",
		}
		if h, _ := Classify("Write", args); h != "" {
			t.Errorf("Classify scanned a payload arg %q and blocked a mere mention: host=%q", key, h)
		}
	}
}
