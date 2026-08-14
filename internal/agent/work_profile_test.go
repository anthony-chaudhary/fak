package agent

import (
	"bytes"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/syspromptmmu"
)

func TestOwnedSystemBlockComposesWorkAndResponseProfiles(t *testing.T) {
	t.Setenv(syspromptmmu.WorkProfileEnvVar, "ponytail:medium")
	t.Setenv(syspromptmmu.StyleEnvVar, "caveman:high")
	block := BuildOwnedSystemBlock(nil, func(syspromptmmu.BaseEdit) bool { return true })
	if !block.CacheStable() {
		t.Fatalf("cache audit = %+v", block.Audit)
	}
	if block.WorkProfile != "ponytail:native:medium" || block.Style != "caveman:native:high" {
		t.Fatalf("profile readout = work %q style %q", block.WorkProfile, block.Style)
	}
	work := []byte("Work profile: Ponytail-inspired, native, medium intensity.")
	style := []byte("Signal-first level 3 (compressed)")
	wi, si := bytes.Index(block.Value, work), bytes.Index(block.Value, style)
	if wi < 0 || si < 0 || wi >= si {
		t.Fatalf("work/style overlays absent or wrong precedence: work=%d style=%d", wi, si)
	}
}

func TestOwnedSystemBlockDefaultsWorkProfileOff(t *testing.T) {
	_ = os.Unsetenv(syspromptmmu.WorkProfileEnvVar)
	block := BuildOwnedSystemBlock(nil, func(syspromptmmu.BaseEdit) bool { return true })
	if block.WorkProfile != "standard" || block.WorkProfileIntensity != "off" {
		t.Fatalf("default work profile = %+v", block)
	}
}
