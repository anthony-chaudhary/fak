package ggufload

import (
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// specCapableBackend has the op families of a verify chunk; nothing is called.
type specCapableBackend struct {
	compute.Backend
	compute.Inkling2Device
	compute.Inkling2PrefillDevice
	compute.Inkling2VerifyDevice
}

func (specCapableBackend) Name() string { return "spec-capable" }

type plainBackend struct{ compute.Backend }

func (plainBackend) Name() string { return "plain" }

func TestInkling2MetalSpecDefault(t *testing.T) {
	if d := Inkling2MetalSpecDefault(plainBackend{}); d != (Inkling2SpecDefault{}) {
		t.Fatalf("a backend without verify ops got %+v", d)
	}
	d := Inkling2MetalSpecDefault(specCapableBackend{})
	if d.Depth != inkling2SpecMaxDraft || d.Draft != inkling2DraftAuto {
		t.Fatalf("default %+v, want auto depth %d", d, inkling2SpecMaxDraft)
	}
	m := &Inkling2LoopModel{SpecDefault: d}
	cases := []struct {
		spec, draft string
		depth       int
		kind, label string
	}{
		{"", "", inkling2SpecMaxDraft, inkling2DraftAuto, "spec=lookup depth=8"},
		{"0", "", 0, inkling2DraftAuto, "spec=off"},
		{"1", "", 4, inkling2DraftAuto, "spec=lookup depth=4"},
		{"", "lookup", inkling2SpecMaxDraft, inkling2DraftLookup, "spec=lookup depth=8"},
		{"", "dflash2", inkling2SpecMaxDraft, inkling2DraftDFlash, "spec=dflash2 depth=8"},
		{"", "auto", inkling2SpecMaxDraft, inkling2DraftAuto, "spec=lookup depth=8"},
	}
	for _, c := range cases {
		t.Run(c.spec+"/"+c.draft, func(t *testing.T) {
			setOrUnset(t, "FAK_INKLING2_SPEC", c.spec)
			setOrUnset(t, "FAK_INKLING2_SPEC_DRAFT", c.draft)
			t.Setenv("FAK_INKLING2_MTP_DIR", "")
			if got := inkling2SpecDepthFor(m); got != c.depth {
				t.Errorf("depth %d, want %d", got, c.depth)
			}
			if got, err := inkling2SpecDraftKindFor(m); err != nil || got != c.kind {
				t.Errorf("draft %q, %v; want %q", got, err, c.kind)
			}
			if got := Inkling2SpecLabel(m, specCapableBackend{}); got != c.label {
				t.Errorf("label %q, want %q", got, c.label)
			}
		})
	}
	setOrUnset(t, "FAK_INKLING2_SPEC", "")
	if got := Inkling2SpecLabel(&Inkling2LoopModel{}, specCapableBackend{}); got != "spec=off" {
		t.Errorf("a model without a default is labelled %q", got)
	}
	if got := Inkling2SpecLabel(m, plainBackend{}); got != "spec=off" {
		t.Errorf("a backend without verify ops is labelled %q", got)
	}
}

// setOrUnset sets env to v, or unsets it for the test when v is empty.
func setOrUnset(t *testing.T, env, v string) {
	t.Helper()
	t.Setenv(env, v)
	if v == "" {
		os.Unsetenv(env)
	}
}
