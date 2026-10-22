package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/engine"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelengine"
)

// fak-test:runtime medium est=1s lane=default
func TestServePreemptionMetricsFollowSelectedLiveEngine(t *testing.T) {
	t.Setenv("FAK_NATIVE_KV_MAX_BLOCKS", "0")
	t.Setenv("FAK_MEMORY_GOVERNOR", "0")
	t.Setenv("FAK_ORG_ENROLLMENT_PATH", filepath.Join(t.TempDir(), "org-enrollment.json"))
	native := modelengine.New()
	native.Preload(model.NewSynthetic(modelengine.SyntheticConfig()))
	previousNative, previousMock := abi.Engine(modelengine.EngineID), abi.Engine("mock")
	abi.RegisterEngine(modelengine.EngineID, native)
	abi.RegisterEngine("mock", engine.MockEngine)
	t.Cleanup(func() {
		abi.RegisterEngine(modelengine.EngineID, previousNative)
		abi.RegisterEngine("mock", previousMock)
	})
	build := func(engineID string) *httptest.Server { return buildServeEngineGateway(t, engineID) }
	scrape := func(ts *httptest.Server) string {
		client := ts.Client()
		client.Timeout = 5 * time.Second
		response, err := client.Get(ts.URL + "/metrics")
		if err != nil {
			t.Fatal(err)
		}
		defer response.Body.Close()
		body, err := io.ReadAll(io.LimitReader(response.Body, 2<<20))
		if err != nil {
			t.Fatal(err)
		}
		if response.StatusCode != http.StatusOK {
			t.Fatalf("metrics status=%d", response.StatusCode)
		}
		return string(body)
	}
	served := build(modelengine.EngineID)
	if strings.Contains(scrape(served), "fak_sched_preempt_") {
		t.Fatal("fresh disabled engine exported preemption metrics")
	}
	native.SetKVPreemptionPolicy(modelengine.NativePreemptionPolicy{
		Mode: modelengine.NativePreemptRecompute, MaxBlocks: 1, BlockTokens: 128,
	})
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	requests := make([]abi.EngineRequest, 0, 2)
	for _, name := range []string{"first", "second"} {
		payload := []byte(`{"prompt":"` + name + `"}`)
		request, err := native.Admit(ctx, &abi.ToolCall{
			Tool: name, Args: abi.Ref{Kind: abi.RefInline, Inline: payload, Len: int64(len(payload))},
		})
		if err != nil {
			t.Fatal(err)
		}
		requests = append(requests, request)
	}
	var drained sync.WaitGroup
	for _, request := range requests {
		drained.Add(1)
		go func(request abi.EngineRequest) {
			defer drained.Done()
			for range request.Tokens() {
			}
		}(request)
	}
	drained.Wait()
	for _, request := range requests {
		result, err := request.Result()
		if err != nil || result == nil || result.Status != abi.StatusOK {
			t.Fatalf("native admission failed: result=%v error=%v", result, err)
		}
	}
	stats := native.KVPreemptionStats()
	if stats.Preemptions <= 0 || stats.RecomputeCount <= 0 || stats.Readmitted <= 0 {
		t.Fatalf("native pressure path did not execute: preemptions=%d recomputes=%d readmitted=%d", stats.Preemptions, stats.RecomputeCount, stats.Readmitted)
	}
	metrics := scrape(served)
	for name, value := range map[string]int64{
		"fak_sched_preempt_total":            stats.Preemptions,
		"fak_sched_preempt_recompute_total":  stats.RecomputeCount,
		"fak_sched_preempt_readmitted_total": stats.Readmitted,
		"fak_sched_preempt_max_blocks":       int64(stats.MaxBlocks),
	} {
		if !strings.Contains(metrics, fmt.Sprintf("%s %d\n", name, value)) {
			t.Fatalf("scraped %s did not match selected engine value %d", name, value)
		}
	}
	t.Logf("verified native preemption export: preemptions=%d recompute=%d readmitted=%d max_blocks=%d", stats.Preemptions, stats.RecomputeCount, stats.Readmitted, stats.MaxBlocks)
	if strings.Contains(scrape(build("mock")), "fak_sched_preempt_") {
		t.Fatal("mock-selected serve exported the unselected native engine's metrics")
	}
	native.SetKVPreemptionPolicy(modelengine.NativePreemptionPolicy{})
	if strings.Contains(scrape(served), "fak_sched_preempt_") {
		t.Fatal("disabled live engine retained preemption metrics")
	}
}

func buildServeEngineGateway(t *testing.T, engineID string) *httptest.Server {
	t.Helper()
	fs, sf := newServeFlagSet()
	if err := fs.Parse([]string{"--engine=" + engineID, "--workspace-admission-permissive"}); err != nil {
		t.Fatal(err)
	}
	rt := &serveRuntime{t0: time.Now(), explicitFlags: map[string]bool{"engine": true}}
	ingress, err := rt.buildGateway(sf)
	if err != nil {
		t.Fatal(err)
	}
	ts := httptest.NewServer(rt.srv.Handler())
	t.Cleanup(func() {
		ts.Close()
		rt.srv.Close()
		if err := rt.closeServeArmLeases(); err != nil {
			t.Error(err)
		}
		if ingress != nil {
			_ = ingress.Close()
		}
	})
	return ts
}

// fak-test:runtime medium est=1s lane=default
func TestServeStartupMessagesWarnOnMalformedNativeKVEnv(t *testing.T) {
	cases := []struct {
		name      string
		maxBlocks string
		wantWarn  bool
	}{
		{name: "malformed", maxBlocks: "abc", wantWarn: true},
		{name: "valid", maxBlocks: "0"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("FAK_NATIVE_KV_MAX_BLOCKS", tc.maxBlocks)
			t.Setenv("FAK_NATIVE_KV_BLOCK_TOKENS", "")
			t.Setenv("FAK_MEMORY_GOVERNOR", "0")
			t.Setenv("FAK_ORG_ENROLLMENT_PATH", filepath.Join(t.TempDir(), "org-enrollment.json"))
			native := modelengine.New()
			native.Preload(model.NewSynthetic(modelengine.SyntheticConfig()))
			previous := abi.Engine(modelengine.EngineID)
			abi.RegisterEngine(modelengine.EngineID, native)
			t.Cleanup(func() { abi.RegisterEngine(modelengine.EngineID, previous) })

			served := buildServeEngineGateway(t, modelengine.EngineID)
			vars := readServeStartupVars(t, served.URL+"/debug/vars", 5*time.Second)
			envErr := native.KVPreemptionEnvErr()
			if got := errors.Is(envErr, modelengine.ErrNativeKVEnvInvalid); got != tc.wantWarn {
				t.Fatalf("errors.Is(KVPreemptionEnvErr, ErrNativeKVEnvInvalid) = %v, want %v; err = %v", got, tc.wantWarn, envErr)
			}
			var found []gateway.StartupMessage
			for _, message := range vars.Startup.Messages {
				if message.Kind == "native-kv-env" {
					found = append(found, message)
				}
			}
			if !tc.wantWarn {
				if len(found) != 0 {
					t.Fatalf("valid native KV env produced startup messages %+v", found)
				}
				return
			}
			want := gateway.StartupMessage{Source: "serve", Kind: "native-kv-env", Level: "warning", Text: modelengine.NativeKVEnvNotice(envErr)}
			if len(found) != 1 || found[0] != want {
				t.Fatalf("native-kv-env startup messages = %+v, want exactly [%+v]", found, want)
			}
		})
	}
}
