package config

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/security"
)

// TestDefaultConfigExposesNoNonLoopbackDebugListener pins the secure debug
// defaults: pprof disabled entirely, metrics bound to loopback only. A
// default that listens on all interfaces exposes unauthenticated heap and
// CPU profiles plus process introspection to anyone who can reach the host.
func TestDefaultConfigExposesNoNonLoopbackDebugListener(t *testing.T) {
	cfg := DefaultConfig()

	if cfg.PprofAddr != "" {
		t.Fatalf("DefaultConfig().PprofAddr = %q, want \"\" (pprof must be disabled by default)", cfg.PprofAddr)
	}
	if cfg.MetricsAddr == "" {
		t.Fatalf("DefaultConfig().MetricsAddr = \"\", want a loopback bind (e.g. 127.0.0.1:9090)")
	}
	if !security.IsLoopbackAddress(cfg.MetricsAddr) {
		t.Fatalf("DefaultConfig().MetricsAddr = %q, want a loopback bind (e.g. 127.0.0.1:9090)", cfg.MetricsAddr)
	}
	if err := cfg.Validate(); err != nil {
		t.Fatalf("DefaultConfig().Validate() = %v, want nil", err)
	}
}

// TestValidatePprofLoopbackOnly enforces that an explicitly enabled pprof
// listener can never bind off loopback: pprof serves heap profiles, the
// process command line, and CPU/trace profiling without authentication.
func TestValidatePprofLoopbackOnly(t *testing.T) {
	for _, addr := range []string{"127.0.0.1:6060", "localhost:6060", "[::1]:6060", "127.8.8.8:6060"} {
		cfg := DefaultConfig()
		cfg.PprofAddr = addr
		if err := cfg.Validate(); err != nil {
			t.Errorf("Validate() with pprof_addr %q = %v, want nil (loopback bind allowed)", addr, err)
		}
	}
	for _, addr := range []string{":6060", "0.0.0.0:6060", "192.168.1.10:6060", "10.0.0.1:6060", "[::]:6060", "[fe80::1]:6060"} {
		cfg := DefaultConfig()
		cfg.PprofAddr = addr
		if err := cfg.Validate(); err == nil {
			t.Errorf("Validate() with pprof_addr %q = nil, want error (non-loopback pprof bind refused)", addr)
		}
	}
}

// TestValidateMetricsAddr: loopback is the safe default; a non-loopback
// metrics bind is an explicit opt-in (e.g. Prometheus scraping over the pod
// network) and stays allowed, but a malformed address is refused.
func TestValidateMetricsAddr(t *testing.T) {
	for _, addr := range []string{"127.0.0.1:9090", "localhost:9090", "[::1]:9090", "0.0.0.0:9090", ":9090", "192.168.1.10:9090"} {
		cfg := DefaultConfig()
		cfg.MetricsAddr = addr
		if err := cfg.Validate(); err != nil {
			t.Errorf("Validate() with metrics_addr %q = %v, want nil", addr, err)
		}
	}
	for _, addr := range []string{"9090", "not an addr"} {
		cfg := DefaultConfig()
		cfg.MetricsAddr = addr
		if err := cfg.Validate(); err == nil {
			t.Errorf("Validate() with metrics_addr %q = nil, want error (malformed address refused)", addr)
		}
	}
}

// TestLoadKeepsSecureDebugDefaults covers the JSON merge path: a config file
// that does not mention the debug surfaces must inherit the secure defaults,
// not the pre-hardening all-interfaces binds.
func TestLoadKeepsSecureDebugDefaults(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "cfg.json")
	if err := os.WriteFile(path, []byte("{}\n"), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.PprofAddr != "" {
		t.Errorf("loaded PprofAddr = %q, want \"\"", cfg.PprofAddr)
	}
	if !security.IsLoopbackAddress(cfg.MetricsAddr) {
		t.Errorf("loaded MetricsAddr = %q, want a loopback bind", cfg.MetricsAddr)
	}
}
