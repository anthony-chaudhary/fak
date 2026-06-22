// browser_test.go - unit tests for the pure, resource-free helpers in the
// webbench browser layer. DefaultConfig and New touch no browser, node
// process, network, or filesystem, so their contracts are deterministic and
// safe to assert directly.
package browser

import (
	"testing"
	"time"
)

// TestDefaultConfig pins the documented browser defaults: headless on and a
// 30-second timeout. A regression in either field (e.g. a changed timeout or
// a non-headless default) makes this test fail.
func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()

	if !cfg.Headless {
		t.Errorf("DefaultConfig().Headless = %v, want true", cfg.Headless)
	}

	wantTimeout := 30 * time.Second
	if cfg.Timeout != wantTimeout {
		t.Errorf("DefaultConfig().Timeout = %v, want %v", cfg.Timeout, wantTimeout)
	}

	// Cross-check the raw nanosecond magnitude so a unit mix-up (e.g.
	// 30*time.Millisecond) is caught even if someone rewrites the literal.
	if got := cfg.Timeout.Nanoseconds(); got != int64(30_000_000_000) {
		t.Errorf("DefaultConfig().Timeout = %d ns, want 30000000000 ns", got)
	}
}

// TestDefaultConfigIsStable verifies DefaultConfig is deterministic: repeated
// calls return equal values, confirming it is a pure constructor with no
// hidden state.
func TestDefaultConfigIsStable(t *testing.T) {
	a := DefaultConfig()
	b := DefaultConfig()

	if a != b {
		t.Errorf("DefaultConfig not deterministic: %+v != %+v", a, b)
	}
}

// TestNew checks that New succeeds without error and returns a usable, empty
// session regardless of the config it is handed. New currently constructs the
// session with zero-value script/context, so we assert that observable shape.
func TestNew(t *testing.T) {
	cfgs := []Config{
		DefaultConfig(),
		{Headless: false, Timeout: time.Second},
		{},
	}

	for _, cfg := range cfgs {
		b, err := New(cfg)
		if err != nil {
			t.Fatalf("New(%+v) returned error: %v", cfg, err)
		}
		if b == nil {
			t.Fatalf("New(%+v) returned nil Browser", cfg)
		}
		if b.script != "" {
			t.Errorf("New(%+v).script = %q, want empty", cfg, b.script)
		}
		if b.context != "" {
			t.Errorf("New(%+v).context = %q, want empty", cfg, b.context)
		}
	}
}
