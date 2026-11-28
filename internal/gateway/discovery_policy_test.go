package gateway

import (
	"errors"
	"testing"
)

func TestDiscoveryPolicy(t *testing.T) {
	t.Parallel()
	calls := []string{}
	rung := func(name, addr string, ok bool) DiscoveryRung {
		return DiscoveryRung{Name: name, Find: func() (string, bool) { calls = append(calls, name); return addr, ok }}
	}
	// The first rung that finds an address wins and later rungs are not consulted.
	addr, name, err := DiscoverMetricsEndpoint([]DiscoveryRung{
		rung("explicit", "", false),
		rung("annotation", "10.0.0.5:9090", true),
		rung("port-name", "10.0.0.6:8000", true),
	})
	if err != nil || addr != "10.0.0.5:9090" || name != "annotation" {
		t.Fatalf("got (%q, %q, %v)", addr, name, err)
	}
	if len(calls) != 2 {
		t.Fatalf("consulted %v, want only the first two rungs", calls)
	}
	// An explicit operator rung outranks inferred ones.
	addr, name, _ = DiscoverMetricsEndpoint([]DiscoveryRung{
		rung("explicit", "operator:1234", true),
		rung("annotation", "10.0.0.5:9090", true),
	})
	if name != "explicit" || addr != "operator:1234" {
		t.Fatalf("explicit must win, got (%q, %q)", addr, name)
	}
	// Nothing found is a typed error, not an empty address.
	if _, _, err := DiscoverMetricsEndpoint([]DiscoveryRung{rung("a", "", false), {Name: "nil"}}); !errors.Is(err, ErrNoDiscoveryEndpoint) {
		t.Fatalf("err = %v, want ErrNoDiscoveryEndpoint", err)
	}
}
