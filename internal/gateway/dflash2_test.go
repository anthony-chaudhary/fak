package gateway

import (
	"context"
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// fak-test:runtime fast est=50ms lane=default
func TestDFlash2SelectorFailsClosed(t *testing.T) {
	for _, tc := range []struct {
		a, b, c       string
		selected, bad bool
	}{
		{"", "", "", false, false}, {"", "", "mtp", false, false}, {"", "", "dflash2", true, false},
		{"dflash2", "", "", true, false}, {"dflash2", "mtp", "", false, true},
		{"mtp", "", "dflash2", false, true}, {"", "", "", false, false},
	} {
		t.Run(tc.a+"/"+tc.b+"/"+tc.c, func(t *testing.T) {
			t.Setenv("FAK_SPECULATIVE", tc.a)
			t.Setenv("SPECULATIVE", tc.b)
			selected, err := dflash2Selected(Config{SpeculativeMode: tc.c})
			if selected != tc.selected || (err != nil) != tc.bad {
				t.Fatalf("selected=%v error=%v", selected, err)
			}
		})
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestDFlash2GatewayMissingIdentityRefuses(t *testing.T) {
	t.Setenv("FAK_SPECULATIVE", "")
	t.Setenv("SPECULATIVE", "")
	cfg := Config{SpeculativeMode: "dflash2"}
	if err := validateDFlash2Config(cfg); err == nil {
		t.Fatal("missing authenticated owner admitted")
	}
	if _, err := New(cfg); err == nil {
		t.Fatal("gateway ignored selected DFlash2")
	}
	cfg.DFlash2Target = &agent.DFlash2Target{}
	if err := validateDFlash2Config(cfg); err == nil {
		t.Fatal("empty opaque owner admitted")
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestDFlash2BindingFailureHasNoFallback(t *testing.T) {
	cause := errors.New("changed owner")
	p := &dflash2RefusedPlanner{model: "target", err: cause}
	if p.Model() != "target" {
		t.Fatal("model provenance")
	}
	if result, err := p.Complete(context.Background(), nil, nil); result != nil || !errors.Is(err, cause) {
		t.Fatalf("fallback result=%v err=%v", result, err)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestDFlash2ActualPlannerConstructionRefusesBinding(t *testing.T) {
	t.Setenv("FAK_SPECULATIVE", "")
	t.Setenv("SPECULATIVE", "")
	cfg := Config{SpeculativeMode: "dflash2", InKernelModel: &model.Model{}, DFlash2Target: &agent.DFlash2Target{}}
	p := newInKernelChatPlanner(cfg, "bound-target", func(string, ...any) {})
	if _, ok := p.(*dflash2RefusedPlanner); !ok {
		t.Fatalf("selected route fell through to %T", p)
	}
	if result, err := p.Complete(context.Background(), nil, nil); result != nil || err == nil {
		t.Fatalf("fallback result=%v err=%v", result, err)
	}
}
