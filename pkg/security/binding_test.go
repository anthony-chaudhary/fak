package security

import (
	"errors"
	"net"
	"testing"
)

func TestDaemonSocketBindingGuard(t *testing.T) {
	tests := []struct {
		name             string
		addr             string
		allowInsecureLAN bool
		wantErr          error
	}{
		// 1. Wildcard addresses (Fail closed without flag)
		{
			name:             "wildcard IPv4 with port",
			addr:             "0.0.0.0:8080",
			allowInsecureLAN: false,
			wantErr:          ErrWildcardBinding,
		},
		{
			name:             "wildcard IPv4 bare",
			addr:             "0.0.0.0",
			allowInsecureLAN: false,
			wantErr:          ErrWildcardBinding,
		},
		{
			name:             "empty host with port (standard net.Listen wildcard)",
			addr:             ":8080",
			allowInsecureLAN: false,
			wantErr:          ErrWildcardBinding,
		},
		{
			name:             "empty string",
			addr:             "",
			allowInsecureLAN: false,
			wantErr:          ErrWildcardBinding,
		},
		{
			name:             "wildcard IPv6 with port",
			addr:             "[::]:8080",
			allowInsecureLAN: false,
			wantErr:          ErrWildcardBinding,
		},
		{
			name:             "wildcard IPv6 bare",
			addr:             "::",
			allowInsecureLAN: false,
			wantErr:          ErrWildcardBinding,
		},
		{
			name:             "asterisk wildcard",
			addr:             "*:8080",
			allowInsecureLAN: false,
			wantErr:          ErrWildcardBinding,
		},

		// 2. Loopback addresses (Allowed by default)
		{
			name:             "loopback IPv4 127.0.0.1 with port",
			addr:             "127.0.0.1:8080",
			allowInsecureLAN: false,
			wantErr:          nil,
		},
		{
			name:             "loopback IPv4 127.0.0.1 bare",
			addr:             "127.0.0.1",
			allowInsecureLAN: false,
			wantErr:          nil,
		},
		{
			name:             "loopback IPv4 other 127.x range",
			addr:             "127.0.1.5:9000",
			allowInsecureLAN: false,
			wantErr:          nil,
		},
		{
			name:             "localhost with port",
			addr:             "localhost:8080",
			allowInsecureLAN: false,
			wantErr:          nil,
		},
		{
			name:             "localhost bare",
			addr:             "localhost",
			allowInsecureLAN: false,
			wantErr:          nil,
		},
		{
			name:             "loopback IPv6 [::1] with port",
			addr:             "[::1]:8080",
			allowInsecureLAN: false,
			wantErr:          nil,
		},
		{
			name:             "loopback IPv6 bare",
			addr:             "::1",
			allowInsecureLAN: false,
			wantErr:          nil,
		},

		// 3. Tailscale CGNAT IP range (100.64.0.0/10 and fd7a:115c:a1e0::/48 - Allowed by default)
		{
			name:             "tailscale subnet start 100.64.0.1",
			addr:             "100.64.0.1:8080",
			allowInsecureLAN: false,
			wantErr:          nil,
		},
		{
			name:             "tailscale desktop node 100.64.0.11",
			addr:             "100.64.0.11:8000",
			allowInsecureLAN: false,
			wantErr:          nil,
		},
		{
			name:             "tailscale laptop node 100.64.0.12",
			addr:             "100.64.0.12:8131",
			allowInsecureLAN: false,
			wantErr:          nil,
		},
		{
			name:             "tailscale macbook node 100.64.0.13",
			addr:             "100.64.0.13:8131",
			allowInsecureLAN: false,
			wantErr:          nil,
		},
		{
			name:             "tailscale subnet end 100.127.255.254",
			addr:             "100.127.255.254:9000",
			allowInsecureLAN: false,
			wantErr:          nil,
		},
		{
			name:             "tailscale IPv6 ULA",
			addr:             "[fd7a:115c:a1e0::1]:8080",
			allowInsecureLAN: false,
			wantErr:          nil,
		},

		// 4. Non-Tailscale CGNAT boundary checks (Fail closed)
		{
			name:             "IP below Tailscale CGNAT 100.63.255.255",
			addr:             "100.63.255.255:8080",
			allowInsecureLAN: false,
			wantErr:          ErrInsecureLANBinding,
		},
		{
			name:             "IP above Tailscale CGNAT 100.128.0.1",
			addr:             "100.128.0.1:8080",
			allowInsecureLAN: false,
			wantErr:          ErrInsecureLANBinding,
		},

		// 5. RFC 1918 Private LAN IPs (Fail closed without explicit opt-in)
		{
			name:             "RFC 1918 Class A 10.0.0.1",
			addr:             "10.0.0.1:8080",
			allowInsecureLAN: false,
			wantErr:          ErrInsecureLANBinding,
		},
		{
			name:             "RFC 1918 Class A 10.254.1.1",
			addr:             "10.254.1.1:9090",
			allowInsecureLAN: false,
			wantErr:          ErrInsecureLANBinding,
		},
		{
			name:             "RFC 1918 Class B 172.16.0.1",
			addr:             "172.16.0.1:8080",
			allowInsecureLAN: false,
			wantErr:          ErrInsecureLANBinding,
		},
		{
			name:             "RFC 1918 Class B 172.31.255.254",
			addr:             "172.31.255.254:8080",
			allowInsecureLAN: false,
			wantErr:          ErrInsecureLANBinding,
		},
		{
			name:             "RFC 1918 Class C 192.168.1.1 router gateway",
			addr:             "192.168.1.1:80",
			allowInsecureLAN: false,
			wantErr:          ErrInsecureLANBinding,
		},
		{
			name:             "RFC 1918 Class C 192.168.1.50 LAN host",
			addr:             "192.168.1.50:8080",
			allowInsecureLAN: false,
			wantErr:          ErrInsecureLANBinding,
		},

		// 6. Public IPs and non-localhost hostnames (Fail closed without opt-in)
		{
			name:             "public DNS 8.8.8.8",
			addr:             "8.8.8.8:8080",
			allowInsecureLAN: false,
			wantErr:          ErrInsecureLANBinding,
		},
		{
			name:             "public IP 1.1.1.1",
			addr:             "1.1.1.1:8080",
			allowInsecureLAN: false,
			wantErr:          ErrInsecureLANBinding,
		},
		{
			name:             "arbitrary LAN hostname",
			addr:             "nas.local:8080",
			allowInsecureLAN: false,
			wantErr:          ErrInsecureLANBinding,
		},

		// 7. Overrides with allowInsecureLAN = true
		{
			name:             "override: wildcard IPv4 with allowInsecureLAN",
			addr:             "0.0.0.0:8080",
			allowInsecureLAN: true,
			wantErr:          nil,
		},
		{
			name:             "override: empty host with allowInsecureLAN",
			addr:             ":8080",
			allowInsecureLAN: true,
			wantErr:          nil,
		},
		{
			name:             "override: RFC 1918 192.168.1.50 with allowInsecureLAN",
			addr:             "192.168.1.50:8080",
			allowInsecureLAN: true,
			wantErr:          nil,
		},
		{
			name:             "override: RFC 1918 10.0.0.1 with allowInsecureLAN",
			addr:             "10.0.0.1:8080",
			allowInsecureLAN: true,
			wantErr:          nil,
		},
		{
			name:             "override: RFC 1918 172.16.0.1 with allowInsecureLAN",
			addr:             "172.16.0.1:8080",
			allowInsecureLAN: true,
			wantErr:          nil,
		},
		{
			name:             "override: public IP with allowInsecureLAN",
			addr:             "8.8.8.8:8080",
			allowInsecureLAN: true,
			wantErr:          nil,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidateDaemonBinding(tc.addr, tc.allowInsecureLAN)
			if tc.wantErr == nil {
				if err != nil {
					t.Fatalf("ValidateDaemonBinding(%q, %v) expected success, got error: %v", tc.addr, tc.allowInsecureLAN, err)
				}
			} else {
				if err == nil {
					t.Fatalf("ValidateDaemonBinding(%q, %v) expected error matching %v, got nil", tc.addr, tc.allowInsecureLAN, tc.wantErr)
				}
				if !errors.Is(err, tc.wantErr) {
					t.Fatalf("ValidateDaemonBinding(%q, %v) error = %v, want errors.Is(%v)", tc.addr, tc.allowInsecureLAN, err, tc.wantErr)
				}
			}
		})
	}
}

func TestExtractHost(t *testing.T) {
	cases := []struct {
		input string
		want  string
	}{
		{"127.0.0.1:8080", "127.0.0.1"},
		{"localhost:9000", "localhost"},
		{":8080", ""},
		{"[::1]:8080", "::1"},
		{"[::]:8080", "::"},
		{"127.0.0.1", "127.0.0.1"},
		{"localhost", "localhost"},
		{"::1", "::1"},
		{"[::1]", "::1"},
		{"100.64.0.1:5000", "100.64.0.1"},
		{"", ""},
	}

	for _, tc := range cases {
		got := ExtractHost(tc.input)
		if got != tc.want {
			t.Errorf("ExtractHost(%q) = %q, want %q", tc.input, got, tc.want)
		}
	}
}

func TestIPClassificationHelpers(t *testing.T) {
	// Tailscale IPs (100.64.0.0/10 CGNAT range; generic test address, not any
	// real node: the scrub placeholder Tailscale address is a public-leak-audit
	// needle, so fixtures must not carry it).
	tsIPv4 := net.ParseIP("100.64.0.20")
	if !IsTailscaleIP(tsIPv4) {
		t.Errorf("expected 100.64.0.20 to be identified as Tailscale IP")
	}

	tsIPv6 := net.ParseIP("fd7a:115c:a1e0::1")
	if !IsTailscaleIP(tsIPv6) {
		t.Errorf("expected fd7a:115c:a1e0::1 to be identified as Tailscale IP")
	}

	nonTsIPv4 := net.ParseIP("100.128.0.1")
	if IsTailscaleIP(nonTsIPv4) {
		t.Errorf("expected 100.128.0.1 NOT to be identified as Tailscale IP")
	}

	// RFC 1918 IPs
	rfc10 := net.ParseIP("10.0.1.2")
	rfc172 := net.ParseIP("172.20.5.1")
	rfc192 := net.ParseIP("192.168.1.100")
	publicIP := net.ParseIP("8.8.8.8")

	if !IsRFC1918IP(rfc10) {
		t.Errorf("expected 10.0.1.2 to be RFC 1918")
	}
	if !IsRFC1918IP(rfc172) {
		t.Errorf("expected 172.20.5.1 to be RFC 1918")
	}
	if !IsRFC1918IP(rfc192) {
		t.Errorf("expected 192.168.1.100 to be RFC 1918")
	}
	if IsRFC1918IP(publicIP) {
		t.Errorf("expected 8.8.8.8 NOT to be RFC 1918")
	}
	if IsRFC1918IP(nil) {
		t.Errorf("expected nil NOT to be RFC 1918")
	}
}

func TestListenGuarded(t *testing.T) {
	// Loopback binding should succeed
	ln, err := ListenGuarded("tcp", "127.0.0.1:0", false)
	if err != nil {
		t.Fatalf("ListenGuarded on loopback failed: %v", err)
	}
	defer ln.Close()

	// Wildcard binding without flag should fail before net.Listen
	_, err = ListenGuarded("tcp", "0.0.0.0:0", false)
	if err == nil {
		t.Fatalf("ListenGuarded on wildcard without flag expected error, got nil")
	}
	if !errors.Is(err, ErrWildcardBinding) {
		t.Fatalf("expected ErrWildcardBinding, got %v", err)
	}

	// RFC 1918 binding without flag should fail
	_, err = ListenGuarded("tcp", "192.168.1.1:0", false)
	if err == nil {
		t.Fatalf("ListenGuarded on RFC 1918 without flag expected error, got nil")
	}
	if !errors.Is(err, ErrInsecureLANBinding) {
		t.Fatalf("expected ErrInsecureLANBinding, got %v", err)
	}
}
