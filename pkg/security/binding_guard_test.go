package security

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"strings"
	"sync"
	"testing"
	"time"
)

// Test loopback addresses pass.
func TestValidateBinding_Loopback(t *testing.T) {
	cases := []string{
		"127.0.0.1:8080",
		"127.0.0.1:0",
		"127.0.0.1",
		"127.0.0.2:9000",
		"127.255.255.254:5000",
		"[::1]:8080",
		"[::1]:0",
		"::1",
		"[::1]",
		"localhost:8080",
		"localhost:0",
		"localhost",
		"sub.localhost:8080",
	}

	for _, addr := range cases {
		// Must pass when allowInsecureLAN is false
		if err := ValidateBinding(addr, false); err != nil {
			t.Errorf("ValidateBinding(%q, false) failed: %v", addr, err)
		}
		// Must pass when allowInsecureLAN is true
		if err := ValidateBinding(addr, true); err != nil {
			t.Errorf("ValidateBinding(%q, true) failed: %v", addr, err)
		}
		if !IsLoopbackAddress(addr) {
			t.Errorf("IsLoopbackAddress(%q) = false, want true", addr)
		}
		if IsWildcardAddress(addr) {
			t.Errorf("IsWildcardAddress(%q) = true, want false", addr)
		}
	}
}

// Test Tailscale CGNAT addresses pass (e.g. 100.100.50.1).
func TestValidateBinding_TailscaleCGNAT(t *testing.T) {
	validTailscale := []string{
		"100.64.0.1:8080",
		"100.100.50.1:8080",
		"100.100.50.1",
		"100.127.255.254:8080",
		"100.64.0.0:9090",
		"100.127.255.255:9090",
		"[fd7a:115c:a1e0::1]:8080",
		"[fd7a:115c:a1e0::ffff]:9090",
	}

	for _, addr := range validTailscale {
		// Must pass when allowInsecureLAN is false
		if err := ValidateBinding(addr, false); err != nil {
			t.Errorf("ValidateBinding(%q, false) failed: %v", addr, err)
		}
		// Must pass when allowInsecureLAN is true
		if err := ValidateBinding(addr, true); err != nil {
			t.Errorf("ValidateBinding(%q, true) failed: %v", addr, err)
		}
		if !IsTailscaleAddress(addr) {
			t.Errorf("IsTailscaleAddress(%q) = false, want true", addr)
		}
		if IsWildcardAddress(addr) {
			t.Errorf("IsWildcardAddress(%q) = true, want false", addr)
		}
	}

	// Non-Tailscale CGNAT addresses outside 100.64.0.0/10 must fail closed without opt-in
	invalidTailscale := []string{
		"100.63.255.255:8080",
		"100.128.0.1:8080",
		"101.64.0.1:8080",
	}

	for _, addr := range invalidTailscale {
		err := ValidateBinding(addr, false)
		if !errors.Is(err, ErrInsecureLANBinding) {
			t.Errorf("ValidateBinding(%q, false) = %v, want ErrInsecureLANBinding", addr, err)
		}
		if IsTailscaleAddress(addr) {
			t.Errorf("IsTailscaleAddress(%q) = true, want false", addr)
		}
	}
}

// Test wildcard 0.0.0.0 fails closed without opt-in, passes with opt-in.
func TestValidateBinding_Wildcard(t *testing.T) {
	wildcards := []string{
		"0.0.0.0:8080",
		"0.0.0.0:0",
		"0.0.0.0",
		"[::]:8080",
		"[::]:0",
		"::",
		"[::]",
		":8080",
		":0",
		"",
	}

	for _, addr := range wildcards {
		// Must fail closed without opt-in
		err := ValidateBinding(addr, false)
		if !errors.Is(err, ErrInsecureWildcardBinding) {
			t.Errorf("ValidateBinding(%q, false) = %v, want ErrInsecureWildcardBinding", addr, err)
		}

		// Must pass with opt-in
		if err := ValidateBinding(addr, true); err != nil {
			t.Errorf("ValidateBinding(%q, true) failed: %v", addr, err)
		}

		if !IsWildcardAddress(addr) {
			t.Errorf("IsWildcardAddress(%q) = false, want true", addr)
		}
	}
}

// Test LAN IP (e.g. 192.168.1.100) fails closed without opt-in.
func TestValidateBinding_LAN(t *testing.T) {
	lanAddresses := []string{
		"192.168.1.100:8080",
		"192.168.1.100",
		"192.168.50.25:9000",
		"10.0.0.5:8080",
		"10.200.1.1:80",
		"172.16.5.10:8080",
		"172.31.255.1:443",
		"8.8.8.8:8080",
		"1.1.1.1",
	}

	for _, addr := range lanAddresses {
		// Must fail closed without opt-in
		err := ValidateBinding(addr, false)
		if !errors.Is(err, ErrInsecureLANBinding) {
			t.Errorf("ValidateBinding(%q, false) = %v, want ErrInsecureLANBinding", addr, err)
		}

		// Must pass with opt-in
		if err := ValidateBinding(addr, true); err != nil {
			t.Errorf("ValidateBinding(%q, true) failed: %v", addr, err)
		}

		if IsLoopbackAddress(addr) {
			t.Errorf("IsLoopbackAddress(%q) = true, want false", addr)
		}
		if IsTailscaleAddress(addr) {
			t.Errorf("IsTailscaleAddress(%q) = true, want false", addr)
		}
		if IsWildcardAddress(addr) {
			t.Errorf("IsWildcardAddress(%q) = true, want false", addr)
		}
	}
}

// Test invalid address formats.
func TestValidateBinding_InvalidAddress(t *testing.T) {
	invalid := []string{
		"127.0.0.1:99999",
		"127.0.0.1:-1",
		"127.0.0.1:abc",
		"invalid:address:too:many:colons",
	}

	for _, addr := range invalid {
		err := ValidateBinding(addr, false)
		if !errors.Is(err, ErrInvalidAddress) {
			t.Errorf("ValidateBinding(%q, false) = %v, want ErrInvalidAddress", addr, err)
		}
	}
}

// Test RFC 1918 egress detection.
func TestIsProhibitedPrivateEgress(t *testing.T) {
	prohibited := []string{
		// 10.0.0.0/8
		"10.0.0.0",
		"10.0.0.1",
		"10.128.50.1",
		"10.255.255.255",
		// 172.16.0.0/12
		"172.16.0.0",
		"172.16.0.1",
		"172.24.100.5",
		"172.31.255.255",
		// 192.168.0.0/16
		"192.168.0.0",
		"192.168.1.1",
		"192.168.50.25",
		"192.168.255.255",
		// IPv4-mapped IPv6
		"::ffff:10.0.0.1",
		"::ffff:172.16.0.1",
		"::ffff:192.168.1.100",
	}

	for _, ipStr := range prohibited {
		ip := net.ParseIP(ipStr)
		if ip == nil {
			t.Fatalf("failed to parse IP %q", ipStr)
		}
		if !IsProhibitedPrivateEgress(ip) {
			t.Errorf("IsProhibitedPrivateEgress(%q) = false, want true", ipStr)
		}
	}

	permitted := []string{
		"127.0.0.1",
		"127.0.0.2",
		"100.64.0.1",
		"100.100.50.1",
		"100.127.255.254",
		"8.8.8.8",
		"1.1.1.1",
		"9.255.255.255",
		"11.0.0.0",
		"172.15.255.255",
		"172.32.0.0",
		"192.167.255.255",
		"192.169.0.0",
		"::1",
		"2001:db8::1",
		"::ffff:127.0.0.1",
		"::ffff:8.8.8.8",
	}

	for _, ipStr := range permitted {
		ip := net.ParseIP(ipStr)
		if ip == nil {
			t.Fatalf("failed to parse IP %q", ipStr)
		}
		if IsProhibitedPrivateEgress(ip) {
			t.Errorf("IsProhibitedPrivateEgress(%q) = true, want false", ipStr)
		}
	}

	if IsProhibitedPrivateEgress(nil) {
		t.Errorf("IsProhibitedPrivateEgress(nil) = true, want false")
	}
}

// Test ValidateEgressTarget blocking RFC 1918 egress.
func TestValidateEgressTarget(t *testing.T) {
	prohibitedTargets := []string{
		"192.168.1.1:80",
		"192.168.1.1",
		"10.0.0.1:22",
		"10.0.0.1",
		"172.16.0.1:443",
		"172.31.255.255:8080",
		"http://192.168.1.1:8080/setup",
		"http://192.168.1.1/admin",
		"https://10.200.1.5:8443/console",
		"192.168.1.1/admin",
		"[192.168.1.1]:8080",
		"[::ffff:192.168.1.1]:80",
	}

	for _, target := range prohibitedTargets {
		// Must fail closed without allowPrivateLAN
		err := ValidateEgressTarget(target, false)
		if !errors.Is(err, ErrProhibitedPrivateEgress) {
			t.Errorf("ValidateEgressTarget(%q, false) = %v, want ErrProhibitedPrivateEgress", target, err)
		}

		// Must pass when allowPrivateLAN is true
		if err := ValidateEgressTarget(target, true); err != nil {
			t.Errorf("ValidateEgressTarget(%q, true) failed: %v", target, err)
		}
	}

	permittedTargets := []string{
		"127.0.0.1:8080",
		"http://127.0.0.1:3000/api",
		"localhost:8080",
		"http://localhost:8080",
		"100.64.0.1:8080",
		"100.100.50.1:443",
		"8.8.8.8:53",
		"1.1.1.1:443",
		"172.32.0.1:80",
		"192.169.1.1:80",
	}

	for _, target := range permittedTargets {
		if err := ValidateEgressTarget(target, false); err != nil {
			t.Errorf("ValidateEgressTarget(%q, false) failed: %v", target, err)
		}
		if err := ValidateEgressTarget(target, true); err != nil {
			t.Errorf("ValidateEgressTarget(%q, true) failed: %v", target, err)
		}
	}

	// Empty target must fail with ErrInvalidAddress
	if err := ValidateEgressTarget("", false); !errors.Is(err, ErrInvalidAddress) {
		t.Errorf("ValidateEgressTarget(\"\", false) = %v, want ErrInvalidAddress", err)
	}
	if err := ValidateEgressTarget("   ", false); !errors.Is(err, ErrInvalidAddress) {
		t.Errorf("ValidateEgressTarget(\"   \", false) = %v, want ErrInvalidAddress", err)
	}
}

// Test GuardedListen lifecycle.
func TestGuardedListen_Lifecycle(t *testing.T) {
	// 1. Valid loopback binding succeeds and handles network traffic
	ln, err := GuardedListen("tcp", "127.0.0.1:0", false)
	if err != nil {
		t.Fatalf("GuardedListen(127.0.0.1:0, false) failed: %v", err)
	}
	defer ln.Close()

	addr := ln.Addr().String()
	if !strings.HasPrefix(addr, "127.0.0.1:") {
		t.Fatalf("expected listener on 127.0.0.1, got %s", addr)
	}

	serverDone := make(chan error, 1)
	go func() {
		conn, err := ln.Accept()
		if err != nil {
			serverDone <- err
			return
		}
		defer conn.Close()

		buf := make([]byte, 4)
		if _, err := io.ReadFull(conn, buf); err != nil {
			serverDone <- err
			return
		}
		if string(buf) != "PING" {
			serverDone <- errors.New("unexpected payload")
			return
		}
		_, err = conn.Write([]byte("PONG"))
		serverDone <- err
	}()

	clientConn, err := net.DialTimeout("tcp", addr, 2*time.Second)
	if err != nil {
		t.Fatalf("failed to dial listener at %s: %v", addr, err)
	}
	defer clientConn.Close()

	if _, err := clientConn.Write([]byte("PING")); err != nil {
		t.Fatalf("client write failed: %v", err)
	}

	reply := make([]byte, 4)
	if _, err := io.ReadFull(clientConn, reply); err != nil {
		t.Fatalf("client read failed: %v", err)
	}
	if string(reply) != "PONG" {
		t.Fatalf("expected PONG, got %s", string(reply))
	}

	if err := <-serverDone; err != nil {
		t.Fatalf("server handler error: %v", err)
	}

	// 2. Wildcard listener fails closed without opt-in
	_, err = GuardedListen("tcp", "0.0.0.0:0", false)
	if !errors.Is(err, ErrInsecureWildcardBinding) {
		t.Errorf("GuardedListen(0.0.0.0:0, false) = %v, want ErrInsecureWildcardBinding", err)
	}

	_, err = GuardedListen("tcp", ":0", false)
	if !errors.Is(err, ErrInsecureWildcardBinding) {
		t.Errorf("GuardedListen(:0, false) = %v, want ErrInsecureWildcardBinding", err)
	}

	// 3. LAN IP listener fails closed without opt-in
	_, err = GuardedListen("tcp", "192.168.1.100:0", false)
	if !errors.Is(err, ErrInsecureLANBinding) {
		t.Errorf("GuardedListen(192.168.1.100:0, false) = %v, want ErrInsecureLANBinding", err)
	}

	// 4. Wildcard listener succeeds with explicit opt-in
	wildcardLn, err := GuardedListen("tcp", "0.0.0.0:0", true)
	if err != nil {
		t.Fatalf("GuardedListen(0.0.0.0:0, true) failed: %v", err)
	}
	_ = wildcardLn.Close()
}

// Test ParameterEnvelope baseline defaults.
func TestParameterEnvelope(t *testing.T) {
	if DefaultBindAddress != "127.0.0.1" {
		t.Errorf("DefaultBindAddress = %q, want 127.0.0.1", DefaultBindAddress)
	}
	if TailscaleCIDR != "100.64.0.0/10" {
		t.Errorf("TailscaleCIDR = %q, want 100.64.0.0/10", TailscaleCIDR)
	}

	expectedCIDRs := map[string]bool{
		"10.0.0.0/8":     true,
		"172.16.0.0/12":  true,
		"192.168.0.0/16": true,
	}
	for _, cidr := range ProhibitedEgressCIDRs {
		if !expectedCIDRs[cidr] {
			t.Errorf("unexpected CIDR in ProhibitedEgressCIDRs: %s", cidr)
		}
		delete(expectedCIDRs, cidr)
	}
	if len(expectedCIDRs) > 0 {
		t.Errorf("missing expected RFC 1918 CIDRs: %v", expectedCIDRs)
	}

	envelope := DefaultParameterEnvelope()
	if envelope.DefaultBindAddress != "127.0.0.1" {
		t.Errorf("envelope.DefaultBindAddress = %q, want 127.0.0.1", envelope.DefaultBindAddress)
	}
	if envelope.TailscaleCIDR != "100.64.0.0/10" {
		t.Errorf("envelope.TailscaleCIDR = %q, want 100.64.0.0/10", envelope.TailscaleCIDR)
	}

	// JSON serialization verifies field name tags
	data, err := json.Marshal(envelope)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}
	jsonStr := string(data)
	if !strings.Contains(jsonStr, `"default_bind_address":"127.0.0.1"`) {
		t.Errorf("missing default_bind_address in JSON: %s", jsonStr)
	}
	if !strings.Contains(jsonStr, `"prohibited_egress_cidrs"`) {
		t.Errorf("missing prohibited_egress_cidrs in JSON: %s", jsonStr)
	}
	if !strings.Contains(jsonStr, `"tailscale_cidr":"100.64.0.0/10"`) {
		t.Errorf("missing tailscale_cidr in JSON: %s", jsonStr)
	}

	guard := NewSocketBindingGuard(false)
	guardEnv := guard.ParameterEnvelope()
	if guardEnv.DefaultBindAddress != "127.0.0.1" {
		t.Errorf("guardEnv.DefaultBindAddress = %q, want 127.0.0.1", guardEnv.DefaultBindAddress)
	}
}

// Test SocketBindingGuard struct instance methods.
func TestSocketBindingGuard_InstanceMethods(t *testing.T) {
	guard := NewSocketBindingGuard(false)

	if err := guard.ValidateBinding("127.0.0.1:8080"); err != nil {
		t.Errorf("guard.ValidateBinding(127.0.0.1:8080) failed: %v", err)
	}
	if err := guard.ValidateBinding("0.0.0.0:8080"); !errors.Is(err, ErrInsecureWildcardBinding) {
		t.Errorf("guard.ValidateBinding(0.0.0.0:8080) = %v, want ErrInsecureWildcardBinding", err)
	}
	// Per-call override
	if err := guard.ValidateBinding("0.0.0.0:8080", true); err != nil {
		t.Errorf("guard.ValidateBinding(0.0.0.0:8080, true) failed: %v", err)
	}

	ln, err := guard.GuardedListen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("guard.GuardedListen(127.0.0.1:0) failed: %v", err)
	}
	_ = ln.Close()

	_, err = guard.GuardedListen("tcp", "0.0.0.0:0")
	if !errors.Is(err, ErrInsecureWildcardBinding) {
		t.Errorf("guard.GuardedListen(0.0.0.0:0) = %v, want ErrInsecureWildcardBinding", err)
	}

	// Opt-in guard instance
	guardOptIn := NewSocketBindingGuardWithOptions(true, true)
	if err := guardOptIn.ValidateBinding("0.0.0.0:8080"); err != nil {
		t.Errorf("guardOptIn.ValidateBinding(0.0.0.0:8080) failed: %v", err)
	}
	if err := guardOptIn.ValidateEgressTarget("192.168.1.1:80"); err != nil {
		t.Errorf("guardOptIn.ValidateEgressTarget(192.168.1.1:80) failed: %v", err)
	}

	// Zero-value guard fails closed safely
	var zeroGuard SocketBindingGuard
	if err := zeroGuard.ValidateBinding("0.0.0.0:8080"); !errors.Is(err, ErrInsecureWildcardBinding) {
		t.Errorf("zeroGuard.ValidateBinding(0.0.0.0:8080) = %v, want ErrInsecureWildcardBinding", err)
	}
	if err := zeroGuard.ValidateBinding("127.0.0.1:8080"); err != nil {
		t.Errorf("zeroGuard.ValidateBinding(127.0.0.1:8080) failed: %v", err)
	}
}

// Test GuardedDial and SafeHTTPTransport egress protection.
func TestGuardedDial_And_Transport(t *testing.T) {
	_, err := GuardedDial("tcp", "192.168.1.1:80", false)
	if !errors.Is(err, ErrProhibitedPrivateEgress) {
		t.Errorf("GuardedDial(192.168.1.1:80, false) = %v, want ErrProhibitedPrivateEgress", err)
	}

	transport := SafeHTTPTransport(false)
	if transport == nil {
		t.Fatalf("SafeHTTPTransport returned nil")
	}

	// Attempt dialing private IP via transport DialContext
	_, err = transport.DialContext(context.Background(), "tcp", "10.0.0.1:80")
	if !errors.Is(err, ErrProhibitedPrivateEgress) {
		t.Errorf("transport.DialContext(10.0.0.1:80) = %v, want ErrProhibitedPrivateEgress", err)
	}
}

// Test concurrency and verify zero race conditions under go test -race.
func TestConcurrency_ZeroRace(t *testing.T) {
	const goroutines = 64
	var wg sync.WaitGroup
	wg.Add(goroutines)

	guard := NewSocketBindingGuard(false)

	for i := 0; i < goroutines; i++ {
		go func(id int) {
			defer wg.Done()

			// Concurrently validate loopback
			if err := ValidateBinding("127.0.0.1:8080", false); err != nil {
				t.Errorf("goroutine %d: ValidateBinding failed: %v", id, err)
			}

			// Concurrently validate Tailscale
			if err := ValidateBinding("100.100.50.1:8080", false); err != nil {
				t.Errorf("goroutine %d: ValidateBinding Tailscale failed: %v", id, err)
			}

			// Concurrently validate wildcard rejection
			if err := ValidateBinding("0.0.0.0:8080", false); !errors.Is(err, ErrInsecureWildcardBinding) {
				t.Errorf("goroutine %d: ValidateBinding wildcard = %v", id, err)
			}

			// Concurrently validate LAN rejection
			if err := ValidateBinding("192.168.1.100:8080", false); !errors.Is(err, ErrInsecureLANBinding) {
				t.Errorf("goroutine %d: ValidateBinding LAN = %v", id, err)
			}

			// Concurrently check egress
			ipPrivate := net.ParseIP("192.168.1.1")
			if !IsProhibitedPrivateEgress(ipPrivate) {
				t.Errorf("goroutine %d: IsProhibitedPrivateEgress = false", id)
			}

			ipPublic := net.ParseIP("8.8.8.8")
			if IsProhibitedPrivateEgress(ipPublic) {
				t.Errorf("goroutine %d: IsProhibitedPrivateEgress public = true", id)
			}

			if err := ValidateEgressTarget("192.168.1.1:80", false); !errors.Is(err, ErrProhibitedPrivateEgress) {
				t.Errorf("goroutine %d: ValidateEgressTarget = %v", id, err)
			}

			// Guard methods
			if err := guard.ValidateBinding("127.0.0.1:8080"); err != nil {
				t.Errorf("goroutine %d: guard.ValidateBinding failed: %v", id, err)
			}
			_ = guard.ParameterEnvelope()

			// Listen on ephemeral loopback port, accept 0 conns, close
			ln, err := GuardedListen("tcp", "127.0.0.1:0", false)
			if err != nil {
				t.Errorf("goroutine %d: GuardedListen failed: %v", id, err)
				return
			}
			_ = ln.Close()
		}(i)
	}

	wg.Wait()
}

func BenchmarkValidateBinding(b *testing.B) {
	benchmarks := []struct {
		name string
		addr string
	}{
		{"LoopbackIPv4", "127.0.0.1:8080"},
		{"LoopbackIPv6", "[::1]:8080"},
		{"Localhost", "localhost:8080"},
		{"TailscaleIPv4", "100.100.50.1:8080"},
		{"TailscaleIPv6", "[fd7a:115c:a1e0::1]:8080"},
		{"Wildcard", "0.0.0.0:8080"},
		{"LAN", "192.168.1.100:8080"},
	}

	for _, bm := range benchmarks {
		b.Run(bm.name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				_ = ValidateBinding(bm.addr, false)
			}
		})
	}
}

func BenchmarkDaemonSocketBindingGuard(b *testing.B) {
	benchmarks := []struct {
		name string
		addr string
	}{
		{"LoopbackIPv4", "127.0.0.1:8080"},
		{"Localhost", "localhost:8080"},
		{"TailscaleIPv4", "100.64.0.20:8000"},
		{"Wildcard", "0.0.0.0:8080"},
		{"LAN", "192.168.1.1:80"},
	}

	for _, bm := range benchmarks {
		b.Run(bm.name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				_ = ValidateDaemonBinding(bm.addr, false)
			}
		})
	}
}

func BenchmarkSocketBindingGuard_Instance(b *testing.B) {
	guard := NewSocketBindingGuard(false)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = guard.ValidateBinding("127.0.0.1:8080")
	}
}

func BenchmarkIsProhibitedPrivateEgress(b *testing.B) {
	ipPrivate := net.ParseIP("192.168.1.1")
	ipPublic := net.ParseIP("8.8.8.8")

	b.Run("RFC1918Private", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_ = IsProhibitedPrivateEgress(ipPrivate)
		}
	})
	b.Run("PublicIP", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_ = IsProhibitedPrivateEgress(ipPublic)
		}
	})
}

func BenchmarkValidateEgressTarget(b *testing.B) {
	b.Run("LiteralLoopback", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_ = ValidateEgressTarget("127.0.0.1:8080", false)
		}
	})
	b.Run("LiteralRFC1918", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_ = ValidateEgressTarget("192.168.1.1:80", false)
		}
	})
}
