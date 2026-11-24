package routing

import (
	"reflect"
	"testing"
)

func TestBackgroundWorkClassOrdersHealthyHaloFirst(t *testing.T) {
	members := []WorkClassMember{
		{Name: "hive", Healthy: true},
		{Name: "strix-2", Local: true, Healthy: false},
		{Name: "nebius", Healthy: true},
		{Name: "strix-1", Local: true, Healthy: true},
		{Name: "strix-3", Local: true, Healthy: true},
	}
	got, applied := OrderForWorkClass(WorkClassBackground, members)
	want := []string{"strix-1", "strix-3", "hive", "nebius", "strix-2"}
	if !applied || !reflect.DeepEqual(got, want) {
		t.Fatalf("order = %v applied=%v, want %v applied=true", got, applied, want)
	}
}

func TestBackgroundWorkClassNoHealthyHaloKeepsDeclared(t *testing.T) {
	members := []WorkClassMember{
		{Name: "hive", Healthy: true},
		{Name: "strix-1", Local: true, Healthy: false},
	}
	got, applied := OrderForWorkClass(WorkClassBackground, members)
	if applied || !reflect.DeepEqual(got, []string{"hive", "strix-1"}) {
		t.Fatalf("order = %v applied=%v, want declared, applied=false", got, applied)
	}
}

func TestBackgroundWorkClassNoneKeepsDeclared(t *testing.T) {
	members := []WorkClassMember{{Name: "hive", Healthy: true}, {Name: "strix-1", Local: true, Healthy: true}}
	got, applied := OrderForWorkClass(WorkClassNone, members)
	if applied || !reflect.DeepEqual(got, []string{"hive", "strix-1"}) {
		t.Fatalf("order = %v applied=%v, want declared, applied=false", got, applied)
	}
}

func TestBackgroundWorkClassFor(t *testing.T) {
	cases := []struct {
		header, model string
		want          WorkClass
	}{
		{"background", "qwen", WorkClassBackground},
		{" Background ", "qwen", WorkClassBackground},
		{"", LocalFirstAlias, WorkClassBackground},
		{"", LocalFirstAlias + "/qwen", WorkClassBackground},
		{"", "local-first", WorkClassBackground},
		{"", "local-firstish", WorkClassNone},
		{"", LegacyHaloFirstAlias, WorkClassBackground},
		{"", "halo-first/qwen", WorkClassBackground},
		{"", "halo-firstish", WorkClassNone},
		{"interactive", "qwen", WorkClassNone},
		{"", "qwen", WorkClassNone},
	}
	for _, c := range cases {
		if got := WorkClassFor(c.header, c.model); got != c.want {
			t.Errorf("WorkClassFor(%q, %q) = %q, want %q", c.header, c.model, got, c.want)
		}
	}
}

func TestBackgroundWorkClassRule(t *testing.T) {
	r := WorkClassRule(WorkClassBackground, "alias:x")
	if r != RuleWorkClassBackground+"alias:x" {
		t.Fatalf("rule = %q", r)
	}
	if again := WorkClassRule(WorkClassBackground, r); again != r {
		t.Fatalf("not idempotent: %q", again)
	}
	if plain := WorkClassRule(WorkClassNone, "alias:x"); plain != "alias:x" {
		t.Fatalf("none rule = %q", plain)
	}
}

func TestWorkClassLocalFirstRuleAcceptsLegacyMarker(t *testing.T) {
	for rule, want := range map[string]bool{
		RuleLocalFirst + "m1->model:qwen":                         true,
		RuleWorkClassBackground + RuleLocalFirst + "m1->model:q":  true,
		"halo_first:m1->model:qwen":                               true,
		RuleWorkClassBackground + "halo_first:m1->model:qwen":     true,
		RuleWorkClassBackground + "alias:" + LegacyHaloFirstAlias: false,
		"model:m1": false,
	} {
		if got := IsLocalFirstRule(rule); got != want {
			t.Errorf("IsLocalFirstRule(%q) = %v, want %v", rule, got, want)
		}
	}
}

func TestBackgroundWorkClassLocalCapacityBound(t *testing.T) {
	if !WorkClassLocalHasCapacity(WorkClassLocalMaxInFlight - 1) {
		t.Fatal("a Halo below the bound must take background work")
	}
	if WorkClassLocalHasCapacity(WorkClassLocalMaxInFlight) {
		t.Fatal("a Halo at the bound must overflow background work")
	}
}

func TestBackgroundWorkClassLocalFirstRoute(t *testing.T) {
	hostedOnly := []WorkClassMember{{Name: "opencode-go", Healthy: true}, {Name: "hive-ai", Healthy: true}}
	haloRoute := WorkClassRoute{Model: "qwen", Members: []WorkClassMember{
		{Name: "strix-3", Local: true, Healthy: false},
		{Name: "strix-1", Local: true, Healthy: true},
		{Name: "opencode-go", Healthy: true},
	}}
	busyRoute := WorkClassRoute{Model: "busy", Members: []WorkClassMember{{Name: "strix-2", Local: true, Healthy: false}}}
	cases := []struct {
		name      string
		class     WorkClass
		primary   []WorkClassMember
		fallbacks []WorkClassRoute
		want      string
	}{
		{"background hops to the first fallback with a healthy Halo", WorkClassBackground, hostedOnly, []WorkClassRoute{busyRoute, haloRoute}, "qwen"},
		{"no class walks as declared", WorkClassNone, hostedOnly, []WorkClassRoute{haloRoute}, ""},
		{"own chain already has a healthy Halo", WorkClassBackground, haloRoute.Members, []WorkClassRoute{haloRoute}, ""},
		{"every fallback Halo is busy or ejected", WorkClassBackground, hostedOnly, []WorkClassRoute{busyRoute}, ""},
		{"no fallback declared", WorkClassBackground, hostedOnly, nil, ""},
	}
	for _, c := range cases {
		if got := LocalFirstRoute(c.class, c.primary, c.fallbacks); got != c.want {
			t.Errorf("%s: LocalFirstRoute = %q, want %q", c.name, got, c.want)
		}
	}
	if got := LocalPrefix(haloRoute.Members); !reflect.DeepEqual(got, []string{"strix-1"}) {
		t.Fatalf("LocalPrefix = %v, want [strix-1]", got)
	}
}

func TestBackgroundWorkClassEnv(t *testing.T) {
	got := WithBackgroundWorkClass([]string{"A=1"})
	if want := []string{"A=1", WorkClassEnv + "=" + string(WorkClassBackground)}; !reflect.DeepEqual(got, want) {
		t.Fatalf("env = %v, want %v", got, want)
	}
	explicit := []string{WorkClassEnv + "="}
	if got := WithBackgroundWorkClass(explicit); !reflect.DeepEqual(got, explicit) {
		t.Fatalf("explicit value overridden: %v", got)
	}
}
