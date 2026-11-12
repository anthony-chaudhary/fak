package main

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/engine"
	"github.com/anthony-chaudhary/fak/internal/researcharm"
)

// armDefaultFlags / armDefaultBuild are the OS-independent twins of the
// linux||darwin serveArmFlags / serveArmBuild helpers: the arm-admission DEFAULT
// is not a durable-store concern, so this witness must compile and run on every
// host. Distinct names keep both files loadable together.

func armDefaultFlags(t *testing.T, args ...string) *serveFlags {
	t.Helper()
	fs, sf := newServeFlagSet()
	if err := fs.Parse(append([]string{"--engine=mock", "--model=fak-mock", "--workspace-admission-permissive"}, args...)); err != nil {
		t.Fatal(err)
	}
	return sf
}

func armDefaultBuild(t *testing.T, sf *serveFlags) *serveRuntime {
	t.Helper()
	abi.RegisterEngine("mock", engine.MockEngine)
	rt := &serveRuntime{t0: time.Now(), explicitFlags: map[string]bool{"engine": true}}
	ingress, err := rt.buildGateway(sf)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		rt.srv.Close()
		if err := rt.closeServeArmLeases(); err != nil {
			t.Error(err)
		}
		if ingress != nil {
			_ = ingress.Close()
		}
	})
	return rt
}

// TestServeArmConcurrencyBurstExceedsLegacyCeiling is the objective's reproducer.
// The reported failure was
//
//	429 ... researcharm: arm concurrency limit reached: arm "gw-bench" at 16/16 active requests
//
// because every gateway-minted gw-* trace funnels into ONE shared arm ("gw-bench",
// internal/researcharm/origin.go inferArmFromTrace) and the ceiling was a hard
// 16. This test pins BOTH ends: at an explicit 16 the exact 16/16 refusal still
// reproduces (the mechanism is unchanged), and at the shipped default a 20-deep
// burst on that same arm is ADMITTED.
// fak-test:runtime medium est=1s lane=default
func TestServeArmConcurrencyBurstExceedsLegacyCeiling(t *testing.T) {
	t.Setenv("FAK_MEMORY_GOVERNOR", "0")

	// Reproduce the legacy ceiling: 16 admit, the 17th refuses with the exact
	// message from the reported 429, and the arm is attributed gw-bench.
	t.Run("legacy ceiling 16 refuses the 17th burst turn", func(t *testing.T) {
		rt := armDefaultBuild(t, armDefaultFlags(t, "--arm-max-concurrency=16"))
		coord := rt.armCoordinator
		for i := 0; i < 16; i++ {
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
			if _, err := coord.AdmitWithWait(context.Background(), req, "/v1/chat/completions", fmt.Sprintf("gw-%d", i+1), 0); err != nil {
				t.Fatalf("admission %d/16 refused below the ceiling: %v", i+1, err)
			}
		}
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
		_, err := coord.AdmitWithWait(context.Background(), req, "/v1/chat/completions", "gw-17", 0)
		if err == nil || !errors.Is(err, researcharm.ErrArmConcurrencyExceeded) {
			t.Fatalf("17th turn = %v, want ErrArmConcurrencyExceeded", err)
		}
		const want = `arm "gw-bench" at 16/16 active requests`
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("refusal %q does not carry the reported signature %q", err.Error(), want)
		}
	})

	// The shipped default removes that spurious ceiling for one arm's burst.
	rt := armDefaultBuild(t, armDefaultFlags(t))
	coord := rt.armCoordinator
	if coord == nil {
		t.Fatal("research-arm coordinator not configured")
	}
	if got := coord.DefaultMaxConcurrency(); got != defaultArmMaxConcurrency {
		t.Fatalf("wired arm default = %d, want %d", got, defaultArmMaxConcurrency)
	}
	if defaultArmMaxConcurrency <= 16 {
		t.Fatalf("shipped default %d does not exceed the legacy 16 the 429 was measured at", defaultArmMaxConcurrency)
	}

	const burst = 20 // > the legacy 16, <= the shipped default
	leases := make([]*researcharm.RequestLease, 0, burst)
	for i := 0; i < burst; i++ {
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
		lease, err := coord.AdmitWithWait(context.Background(), req, "/v1/chat/completions", fmt.Sprintf("gw-%d", i+1), 0)
		if err != nil {
			t.Fatalf("admission %d/%d refused under the raised default: %v", i+1, burst, err)
		}
		if lease.ArmID != "gw-bench" {
			t.Fatalf("admission %d attributed to arm %q, want gw-bench", i+1, lease.ArmID)
		}
		leases = append(leases, lease)
	}
	for _, lease := range leases {
		lease.Done(0, nil)
	}
	if got := coord.Snapshot().TotalInflight; got != 0 {
		t.Fatalf("in-flight after releases = %d, want 0", got)
	}
}

// TestServeArmMaxConcurrencyFlagOverridesDefault pins the tunable: the flag
// decides the ceiling (0 = unlimited), and the value is observable before any
// arm has appeared.
// fak-test:runtime medium est=1s lane=default
func TestServeArmMaxConcurrencyFlagOverridesDefault(t *testing.T) {
	t.Setenv("FAK_MEMORY_GOVERNOR", "0")
	for _, tc := range []struct {
		name  string
		value string
		want  int
	}{
		{"explicit ceiling", "32", 32},
		{"zero means unlimited", "0", 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rt := armDefaultBuild(t, armDefaultFlags(t, "--arm-max-concurrency="+tc.value))
			if got := rt.armCoordinator.DefaultMaxConcurrency(); got != tc.want {
				t.Fatalf("arm default = %d, want %d", got, tc.want)
			}
			if got := rt.armCoordinator.Snapshot().DefaultMaxConcurrency; got != tc.want {
				t.Fatalf("snapshot default = %d, want %d", got, tc.want)
			}
		})
	}
}

// TestServeArmMaxConcurrencyIsConfigSurfaceNotEnvironment witnesses the
// CONFIG_NOT_ENV relocation: a behavioral setting is not steered by an ambient
// env var, so the retired FAK_ARM_MAX_CONCURRENCY cannot silently reconfigure
// admission. The decided value comes from the flag (or its built-in default).
// fak-test:runtime medium est=1s lane=default
func TestServeArmMaxConcurrencyIsConfigSurfaceNotEnvironment(t *testing.T) {
	t.Setenv("FAK_MEMORY_GOVERNOR", "0")
	t.Setenv("FAK_ARM_MAX_CONCURRENCY", "3")
	rt := armDefaultBuild(t, armDefaultFlags(t))
	if got := rt.armCoordinator.DefaultMaxConcurrency(); got != defaultArmMaxConcurrency {
		t.Fatalf("env leaked into admission: default = %d, want %d (config surface only)", got, defaultArmMaxConcurrency)
	}
}

// TestServeArmMaxConcurrencyIsReportedInEffectiveConfig keeps the tunable on the
// reviewable config surface: `--print-effective-config` states its value and
// provenance so an operator can see the ceiling without guessing.
// fak-test:runtime fast est=1s lane=default
func TestServeArmMaxConcurrencyIsReportedInEffectiveConfig(t *testing.T) {
	rep := parseEffectiveServeConfig(t, "")
	if got := rep.Values["arm_max_concurrency"].Value; got != defaultArmMaxConcurrency {
		t.Fatalf("effective default = %v, want %d", got, defaultArmMaxConcurrency)
	}
	if got := rep.Values["arm_max_concurrency"].Source; got != "built-in" {
		t.Fatalf("effective default source = %q, want built-in", got)
	}
	rep = parseEffectiveServeConfig(t, "", "--arm-max-concurrency=64")
	if got := rep.Values["arm_max_concurrency"].Value; got != 64 {
		t.Fatalf("explicit effective value = %v, want 64", got)
	}
	if got := rep.Values["arm_max_concurrency"].Source; got != "flag" {
		t.Fatalf("explicit effective source = %q, want flag", got)
	}
}
