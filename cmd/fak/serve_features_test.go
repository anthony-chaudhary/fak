package main

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/deploymanifest"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/session"
)

func TestServeFeaturesFlag(t *testing.T) {
	switch os.Getenv("FAK_TEST_SERVE_FEATURES_HELPER") {
	case "print":
		cmdServe([]string{"--print-features", "--gguf", filepath.Join(os.TempDir(), "model-that-must-not-be-opened.gguf"), "--model", "invalid-model-that-must-not-resolve"})
		os.Exit(0)
	case "live":
		keepAwake := os.Getenv("FAK_TEST_SERVE_FEATURES_KEEP_AWAKE")
		if keepAwake == "" {
			keepAwake = "off"
		}
		args := []string{"--mock", "--addr", os.Getenv("FAK_TEST_SERVE_FEATURES_ADDR"), "--session-state", "off", "--keep-awake", keepAwake}
		if os.Getenv("FAK_TEST_SERVE_FEATURES_APPLIANCE") == "1" {
			args = append(args, "--appliance-observability")
		}
		cmdServe(args)
		os.Exit(0)
	}

	t.Run("model IO free early exit", func(t *testing.T) {
		cmd := exec.Command(os.Args[0], "-test.run=^TestServeFeaturesFlag$")
		cmd.Env = append(os.Environ(), "FAK_TEST_SERVE_FEATURES_HELPER=print")
		out, err := cmd.Output()
		if err != nil {
			t.Fatalf("fak serve --print-features failed before its early exit: %v", err)
		}
		var catalog gateway.FeatureCatalog
		if err := json.Unmarshal(out, &catalog); err != nil {
			t.Fatalf("stdout is not one feature catalog: %v\n%s", err, out)
		}
		if catalog.Schema != gateway.FeatureSchema || len(catalog.Features) != 42 {
			t.Fatalf("catalog schema/count = %q/%d, want %q/42", catalog.Schema, len(catalog.Features), gateway.FeatureSchema)
		}
		status := featureStatusByID(t, catalog)
		if got := status[gateway.FeatureNativeModel]; got.State != gateway.FeatureConfiguredStandby || got.Provenance != gateway.FeatureCLIFlag {
			t.Fatalf("native model = %#v, want CLI requested standby before model I/O", got)
		}
	})

	t.Run("mock boot publishes evaluated catalog", func(t *testing.T) {
		addr := reserveServeStartupAddr(t)
		stateDir := t.TempDir()
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestServeFeaturesFlag$")
		cmd.Env = append(os.Environ(),
			"FAK_TEST_SERVE_FEATURES_HELPER=live",
			"FAK_TEST_SERVE_FEATURES_ADDR="+addr,
			"FAK_SESSION_REGISTRY="+filepath.Join(stateDir, "sessions.json"),
			"HOME="+stateDir,
			"USERPROFILE="+stateDir,
			"XDG_CONFIG_HOME="+filepath.Join(stateDir, ".config"),
		)
		if err := cmd.Start(); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if cmd.ProcessState == nil || !cmd.ProcessState.Exited() {
				_ = cmd.Process.Kill()
				_ = cmd.Wait()
			}
		})

		catalog := readLiveFeatureCatalog(t, "http://"+addr+"/v1/fak/features", 12*time.Second)
		status := featureStatusByID(t, catalog)
		for _, id := range []gateway.ServeFeature{gateway.FeaturePolicyFloor, gateway.FeatureVDSO, gateway.FeatureDeferTools, gateway.FeatureMockPlanner} {
			if status[id].State != gateway.FeatureConfiguredActive {
				t.Errorf("live endpoint %s = %s, want configured_active", id, status[id].State)
			}
		}
		if status[gateway.FeatureNativeModel].State != gateway.FeatureDisabled {
			t.Errorf("live endpoint native_model = %s, want disabled", status[gateway.FeatureNativeModel].State)
		}
		cancel()
		_ = cmd.Wait()
	})
}

func TestServeFeatureCatalogReportsInstalledRuntimeTruth(t *testing.T) {
	tests := []struct {
		name  string
		env   string
		value string
		want  gateway.ServeFeature
	}{
		{name: "appliance dashboard installed", env: "FAK_TEST_SERVE_FEATURES_APPLIANCE", value: "1", want: gateway.FeatureApplianceObservability},
		{name: "while-active monitor running", env: "FAK_TEST_SERVE_FEATURES_KEEP_AWAKE", value: KeepAwakeWhileActive, want: gateway.FeatureKeepAwake},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			addr := reserveServeStartupAddr(t)
			stateDir := t.TempDir()
			ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestServeFeaturesFlag$")
			cmd.Env = append(os.Environ(),
				"FAK_TEST_SERVE_FEATURES_HELPER=live",
				"FAK_TEST_SERVE_FEATURES_ADDR="+addr,
				"FAK_SESSION_REGISTRY="+filepath.Join(stateDir, "sessions.json"),
				"HOME="+stateDir,
				"USERPROFILE="+stateDir,
				"XDG_CONFIG_HOME="+filepath.Join(stateDir, ".config"),
				tc.env+"="+tc.value,
			)
			if err := cmd.Start(); err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if cmd.ProcessState == nil || !cmd.ProcessState.Exited() {
					_ = cmd.Process.Kill()
					_ = cmd.Wait()
				}
			})

			catalog := readLiveFeatureCatalog(t, "http://"+addr+"/v1/fak/features", 12*time.Second)
			if got := featureStatusByID(t, catalog)[tc.want]; got.State != gateway.FeatureConfiguredActive {
				t.Fatalf("live %s = %#v, want configured_active after runtime installation", tc.want, got)
			}
			cancel()
			_ = cmd.Wait()
		})
	}
}

func TestServeFeatureCatalogUsesExactCUDAGraphRuntimeValue(t *testing.T) {
	for _, tc := range []struct {
		value string
		want  gateway.FeatureState
	}{
		{value: "1", want: gateway.FeatureConfiguredStandby},
		{value: "true", want: gateway.FeatureDisabled},
		{value: "yes", want: gateway.FeatureDisabled},
		{value: "on", want: gateway.FeatureDisabled},
	} {
		t.Run(tc.value, func(t *testing.T) {
			fs, sf := newServeFlagSet()
			if err := fs.Parse(nil); err != nil {
				t.Fatal(err)
			}
			catalog, err := evaluateServeFeatures(sf, deploymanifest.Defaults(), map[string]bool{}, func(key string) string {
				if key == "FAK_CUDA_GRAPH" {
					return tc.value
				}
				return ""
			}, nil)
			if err != nil {
				t.Fatal(err)
			}
			if got := featureStatusByID(t, catalog)[gateway.FeatureCUDAGraph]; got.State != tc.want {
				t.Fatalf("FAK_CUDA_GRAPH=%q catalog state = %s, want %s", tc.value, got.State, tc.want)
			}
		})
	}
}

func TestEvaluateServeFeaturesPrebootAndRuntime(t *testing.T) {
	fs, sf := newServeFlagSet()
	args := []string{
		"--vdso=false",
		"--remote-kv-mode=optional",
		"--remote-kv-backend=l3kv-blobhttp",
		"--remote-kv-url=http://cache.invalid",
		"--native",
		"--native-code-tools",
		"--native-code-workspace=.",
		"--debug-stats",
	}
	if err := fs.Parse(args); err != nil {
		t.Fatal(err)
	}
	explicit := explicitFlagNames(fs)
	getenv := func(key string) string {
		if key == "FAK_MEMORY_GOVERNOR" {
			return "1"
		}
		return ""
	}
	preboot, err := evaluateServeFeatures(sf, deploymanifest.Defaults(), explicit, getenv, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(preboot.Features) != 42 {
		t.Fatalf("preboot feature count = %d, want 42", len(preboot.Features))
	}
	pre := featureStatusByID(t, preboot)
	checks := map[gateway.ServeFeature]struct {
		state gateway.FeatureState
		src   gateway.FeatureProvenance
	}{
		gateway.FeatureVDSO:            {gateway.FeatureDisabled, gateway.FeatureCLIFlag},
		gateway.FeatureRemoteKV:        {gateway.FeatureConfiguredStandby, gateway.FeatureCLIFlag},
		gateway.FeatureNativeHarness:   {gateway.FeatureConfiguredStandby, gateway.FeatureCLIFlag},
		gateway.FeatureNativeCodeTools: {gateway.FeatureConfiguredStandby, gateway.FeatureCLIFlag},
		gateway.FeatureDebugStats:      {gateway.FeatureConfiguredStandby, gateway.FeatureCLIFlag},
		gateway.FeatureMemoryGovernor:  {gateway.FeatureConfiguredStandby, gateway.FeatureEnvVar},
	}
	for id, want := range checks {
		got := pre[id]
		if got.State != want.state || got.Provenance != want.src {
			t.Errorf("preboot %s = %s/%s, want %s/%s", id, got.State, got.Provenance, want.state, want.src)
		}
	}

	defaultsFS, defaults := newServeFlagSet()
	if err := defaultsFS.Parse(nil); err != nil {
		t.Fatal(err)
	}
	live, err := evaluateServeFeatures(defaults, deploymanifest.Defaults(), map[string]bool{}, func(string) string { return "" }, &serveRuntime{srv: &gateway.Server{}})
	if err != nil {
		t.Fatal(err)
	}
	running := featureStatusByID(t, live)
	for _, id := range []gateway.ServeFeature{gateway.FeaturePolicyFloor, gateway.FeatureVDSO, gateway.FeatureDeferTools, gateway.FeatureMockPlanner} {
		if running[id].State != gateway.FeatureConfiguredActive {
			t.Errorf("runtime %s = %s, want configured_active", id, running[id].State)
		}
	}
	if got := running[gateway.FeatureNativeModel].State; got != gateway.FeatureDisabled {
		t.Errorf("runtime native_model = %s, want disabled", got)
	}
	for _, id := range []gateway.ServeFeature{gateway.FeatureCompactHistory, gateway.FeatureDeferColdTools} {
		if running[id].State == gateway.FeatureConfiguredActive {
			t.Errorf("mock runtime reported Anthropic-only %s active", id)
		}
	}
}

func TestEvaluateServeFeaturesReportsBufferedMockElisionInstalled(t *testing.T) {
	fs, sf := newServeFlagSet()
	if err := fs.Parse(nil); err != nil {
		t.Fatal(err)
	}

	catalog, err := evaluateServeFeatures(sf, deploymanifest.Defaults(), map[string]bool{}, func(string) string { return "" }, &serveRuntime{srv: &gateway.Server{}})
	if err != nil {
		t.Fatal(err)
	}
	got := featureStatusByID(t, catalog)[gateway.FeatureElideResults]
	if got.State != gateway.FeatureConfiguredActive || got.Provenance != gateway.FeatureAutoDefault {
		t.Fatalf("default buffered mock elide_results = %s/%s, want %s/%s; the decoded planner path installs result elision",
			got.State, got.Provenance, gateway.FeatureConfiguredActive, gateway.FeatureAutoDefault)
	}
}

func TestEvaluateServeFeaturesReportsBufferedMockStaleReadElisionInstalled(t *testing.T) {
	fs, sf := newServeFlagSet()
	if err := fs.Parse(nil); err != nil {
		t.Fatal(err)
	}

	catalog, err := evaluateServeFeatures(sf, deploymanifest.Defaults(), map[string]bool{}, func(string) string { return "" }, &serveRuntime{srv: &gateway.Server{}})
	if err != nil {
		t.Fatal(err)
	}
	got := featureStatusByID(t, catalog)[gateway.FeatureElideStaleReads]
	if got.State != gateway.FeatureConfiguredActive || got.Provenance != gateway.FeatureAutoDefault {
		t.Fatalf("default buffered mock elide_stale_reads = %s/%s, want %s/%s; the decoded planner path installs stale-read elision",
			got.State, got.Provenance, gateway.FeatureConfiguredActive, gateway.FeatureAutoDefault)
	}
}

func TestServeFeatureFleetBusLifecycle(t *testing.T) {
	newServer := func(t *testing.T) *gateway.Server {
		t.Helper()
		s := &gateway.Server{}
		catalog, err := gateway.NewFeatureCatalog([]gateway.FeatureStatus{{
			Feature: gateway.FeatureFleetBus, State: gateway.FeatureConfiguredStandby,
			Provenance: gateway.FeatureCLIFlag, Description: "requested",
		}})
		if err != nil {
			t.Fatal(err)
		}
		if err := s.SetFeatureCatalog(catalog); err != nil {
			t.Fatal(err)
		}
		return s
	}
	state := func(t *testing.T, s *gateway.Server) gateway.FeatureState {
		t.Helper()
		return featureStatusByID(t, s.FeatureSnapshot())[gateway.FeatureFleetBus].State
	}

	t.Run("announce then stop", func(t *testing.T) {
		s := newServer(t)
		stop := startFleetBusLoop(context.Background(), t.TempDir(), "feature-lifecycle",
			time.Hour, session.NewTable(), false, serveGwBusApplier{srv: s, addr: "127.0.0.1:8080"})
		if got := state(t, s); got != gateway.FeatureConfiguredActive {
			t.Fatalf("after synchronous announce = %s, want configured_active", got)
		}
		stop()
		deadline := time.Now().Add(time.Second)
		for state(t, s) != gateway.FeatureConfiguredStandby && time.Now().Before(deadline) {
			time.Sleep(time.Millisecond)
		}
		if got := state(t, s); got != gateway.FeatureConfiguredStandby {
			t.Fatalf("after stop = %s, want configured_standby", got)
		}
	})

	t.Run("invalid bus root refuses", func(t *testing.T) {
		s := newServer(t)
		notDirectory := filepath.Join(t.TempDir(), "file")
		if err := os.WriteFile(notDirectory, []byte("not a directory"), 0o600); err != nil {
			t.Fatal(err)
		}
		stop := startFleetBusLoop(context.Background(), notDirectory, "feature-refusal",
			time.Hour, session.NewTable(), false, serveGwBusApplier{srv: s})
		stop()
		if got := state(t, s); got != gateway.FeatureRefusedUnavailable {
			t.Fatalf("invalid bus root = %s, want refused_unavailable", got)
		}
	})
}

func featureStatusByID(t *testing.T, catalog gateway.FeatureCatalog) map[gateway.ServeFeature]gateway.FeatureStatus {
	t.Helper()
	got := make(map[gateway.ServeFeature]gateway.FeatureStatus, len(catalog.Features))
	for _, status := range catalog.Features {
		if _, duplicate := got[status.Feature]; duplicate {
			t.Fatalf("duplicate feature %q", status.Feature)
		}
		got[status.Feature] = status
	}
	return got
}

func readLiveFeatureCatalog(t *testing.T, url string, timeout time.Duration) gateway.FeatureCatalog {
	t.Helper()
	deadline := time.Now().Add(timeout)
	client := &http.Client{Timeout: 250 * time.Millisecond}
	for time.Now().Before(deadline) {
		response, err := client.Get(url)
		if err == nil {
			body, readErr := io.ReadAll(response.Body)
			_ = response.Body.Close()
			if readErr == nil && response.StatusCode == http.StatusOK {
				var catalog gateway.FeatureCatalog
				if err := json.Unmarshal(body, &catalog); err != nil {
					t.Fatalf("decode live feature catalog: %v\n%s", err, body)
				}
				return catalog
			}
		}
		time.Sleep(25 * time.Millisecond)
	}
	t.Fatalf("serve did not publish feature catalog at %s", url)
	return gateway.FeatureCatalog{}
}
