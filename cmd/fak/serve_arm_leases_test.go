//go:build linux || darwin

package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/engine"
	"github.com/anthony-chaudhary/fak/internal/researcharm"
)

func serveArmFlags(t *testing.T, args ...string) *serveFlags {
	t.Helper()
	fs, sf := newServeFlagSet()
	if err := fs.Parse(append([]string{"--engine=mock", "--model=fak-mock", "--workspace-admission-permissive"}, args...)); err != nil {
		t.Fatal(err)
	}
	return sf
}
func serveArmBuild(t *testing.T, sf *serveFlags) *serveRuntime {
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
func serveArmHTTP(rt *serveRuntime, method, target string, body []byte, arm string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(method, target, bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	if arm != "" {
		req.Header.Set("X-Fak-Project-Arm", arm)
	}
	rec := httptest.NewRecorder()
	rt.srv.Handler().ServeHTTP(rec, req)
	return rec
}
func serveArmPath(t *testing.T) string {
	t.Helper()
	dir := filepath.Join(t.TempDir(), "authority")
	if err := os.Mkdir(dir, 0700); err != nil {
		t.Fatal(err)
	}
	return filepath.Join(dir, "leases.json")
}

// fak-test:runtime medium est=1s lane=default
func TestServeArmLeaseStoreHTTPRestart(t *testing.T) {
	// SW-VERIFIED: production flag/builder and control HTTP; no GPU or physical claim.
	t.Setenv("FAK_MEMORY_GOVERNOR", "0")
	path := serveArmPath(t)
	native := serveArmBuild(t, serveArmFlags(t, "--arm-lease-store="+path))
	payload, err := json.Marshal(researcharm.LeaseRequest{ArmID: "owner", Mode: researcharm.LeaseModeExclusive, TTL: time.Minute})
	if err != nil {
		t.Fatal(err)
	}
	acquired := serveArmHTTP(native, http.MethodPost, "/v1/fak/arms/lease", payload, "")
	if acquired.Code != http.StatusCreated {
		t.Fatalf("exclusive control acquisition: %d %s", acquired.Code, acquired.Body.String())
	}
	var original researcharm.LeaseInfo
	if err := json.Unmarshal(acquired.Body.Bytes(), &original); err != nil {
		t.Fatal(err)
	}
	if original.ID == "" || original.Token == "" {
		t.Fatal("control acknowledgement lacks identity/credential")
	}
	native.srv.Close()
	if err := native.closeServeArmLeases(); err != nil {
		t.Fatal(err)
	}
	var upstreamCalls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamCalls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`))
	}))
	defer upstream.Close()
	proxy := serveArmBuild(t, serveArmFlags(t, "--arm-lease-store="+path, "--base-url="+upstream.URL))
	listed := serveArmHTTP(proxy, http.MethodGet, "/v1/fak/arms/lease", nil, "")
	var leases []researcharm.LeaseInfo
	if listed.Code != http.StatusOK {
		t.Fatalf("reopened control list: %d %s", listed.Code, listed.Body.String())
	}
	if err := json.Unmarshal(listed.Body.Bytes(), &leases); err != nil {
		t.Fatal(err)
	}
	if len(leases) != 1 || leases[0].ID != original.ID || !leases[0].CreatedAt.Equal(original.CreatedAt) || !leases[0].ExpiresAt.Equal(original.ExpiresAt) {
		t.Fatalf("proxy restart changed acknowledged lease: %+v want %+v", leases, original)
	}
	chat := serveArmHTTP(proxy, http.MethodPost, "/v1/chat/completions", []byte(`{"model":"fak-mock","messages":[{"role":"user","content":"hi"}]}`), "peer")
	if chat.Code != http.StatusTooManyRequests {
		t.Fatalf("peer crossed proxy lease: %d %s", chat.Code, chat.Body.String())
	}
	if upstreamCalls.Load() != 0 {
		t.Fatal("denied peer reached upstream")
	}
	released := serveArmHTTP(proxy, http.MethodDelete, "/v1/fak/arms/lease?id="+url.QueryEscape(original.ID)+"&token="+url.QueryEscape(original.Token), nil, "")
	if released.Code != http.StatusOK {
		t.Fatalf("original credential lost: %d %s", released.Code, released.Body.String())
	}
	listed = serveArmHTTP(proxy, http.MethodGet, "/v1/fak/arms/lease", nil, "")
	if err := json.Unmarshal(listed.Body.Bytes(), &leases); err != nil {
		t.Fatal(err)
	}
	if listed.Code != http.StatusOK || len(leases) != 0 {
		t.Fatalf("released lease remains: %d %+v", listed.Code, leases)
	}
}

// fak-test:runtime medium est=1s lane=default
func TestServeArmLeaseStoreStartupContract(t *testing.T) {
	t.Setenv("FAK_MEMORY_GOVERNOR", "0")
	abi.RegisterEngine("mock", engine.MockEngine)
	t.Run("invalid configured authority refuses builder", func(t *testing.T) {
		path := serveArmPath(t)
		if err := os.WriteFile(path, []byte("corrupt"), 0600); err != nil {
			t.Fatal(err)
		}
		rt := &serveRuntime{t0: time.Now(), explicitFlags: map[string]bool{"engine": true}}
		if _, err := rt.buildGateway(serveArmFlags(t, "--arm-lease-store="+path)); err == nil {
			if rt.srv != nil {
				rt.srv.Close()
			}
			_ = rt.closeServeArmLeases()
			t.Fatal("invalid configured store silently used memory coordinator")
		}
		if rt.srv != nil {
			t.Fatal("failed authority published gateway before listener")
		}
	})
	t.Run("duplicate owner refuses builder", func(t *testing.T) {
		path := serveArmPath(t)
		first := serveArmBuild(t, serveArmFlags(t, "--arm-lease-store="+path))
		rt := &serveRuntime{t0: time.Now(), explicitFlags: map[string]bool{"engine": true}}
		if _, err := rt.buildGateway(serveArmFlags(t, "--arm-lease-store="+path)); err == nil {
			if rt.srv != nil {
				rt.srv.Close()
			}
			_ = rt.closeServeArmLeases()
			t.Fatal("duplicate store ownership admitted")
		}
		if rt.srv != nil {
			t.Fatal("duplicate owner published gateway")
		}
		if first.armCoordinator.Snapshot().DurableStatus != "bootstrap" {
			t.Fatal("failed competing boot changed first authority")
		}
	})
	t.Run("no flag preserves memory admission", func(t *testing.T) {
		rt := serveArmBuild(t, serveArmFlags(t))
		if rt.srv.ResearchArmCoordinator().Snapshot().DurableStatus != "" {
			t.Fatal("unconfigured serve adopted durable migration policy")
		}
		body, _ := json.Marshal(researcharm.LeaseRequest{ArmID: "legacy", Mode: researcharm.LeaseModeShared, TTL: time.Minute})
		got := serveArmHTTP(rt, http.MethodPost, "/v1/fak/arms/lease", body, "")
		if got.Code != http.StatusCreated {
			t.Fatalf("legacy initial shared lease refused: %d %s", got.Code, got.Body.String())
		}
	})
}
