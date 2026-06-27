package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func upHealthzServer(t *testing.T) *httptest.Server {
	t.Helper()
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/healthz" {
			w.WriteHeader(http.StatusOK)
			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	t.Cleanup(s.Close)
	return s
}

func downHealthzServer(t *testing.T) *httptest.Server {
	t.Helper()
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	t.Cleanup(s.Close)
	return s
}

func autoRowByName(rep autoDecisionReport, name string) (autoTargetRow, bool) {
	for _, r := range rep.Targets {
		if r.Name == name {
			return r, true
		}
	}
	return autoTargetRow{}, false
}

// TestCostClassOrderingBuiltins proves the documented cost ladder local<mac<gcp<anthropic
// and the exact ordinals the policy is built on.
func TestCostClassOrderingBuiltins(t *testing.T) {
	hermeticTargets(t)
	reg, err := loadComputeTargets("")
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]int{"local": 0, "mac": 110, "gcp": 120, "anthropic": 150}
	for name, w := range want {
		tg, ok := reg.resolve(name)
		if !ok {
			t.Fatalf("built-in %q missing", name)
		}
		if got := tg.costClass(); got != w {
			t.Errorf("costClass(%s) = %d, want %d", name, got, w)
		}
	}
	local, _ := reg.resolve("local")
	mac, _ := reg.resolve("mac")
	gcp, _ := reg.resolve("gcp")
	anth, _ := reg.resolve("anthropic")
	if !(local.costClass() < mac.costClass() && mac.costClass() < gcp.costClass() && gcp.costClass() < anth.costClass()) {
		t.Fatalf("cost ladder not strictly increasing: local=%d mac=%d gcp=%d anthropic=%d",
			local.costClass(), mac.costClass(), gcp.costClass(), anth.costClass())
	}
}

// TestAutoSelectPicksCheapestHealthy: two healthy targets, the cheaper (local) wins even
// when the registry order lists the pricier one first.
func TestAutoSelectPicksCheapestHealthy(t *testing.T) {
	up1, up2 := upHealthzServer(t), upHealthzServer(t)
	pricier := computeTarget{Name: "pricier", Kind: targetGatewayURL, GatewayURL: up2.URL, Locality: localityRemote, HealthzPath: "/healthz", CostNote: "paid GPU compute"}
	cheap := computeTarget{Name: "cheap", Kind: targetLocalSpawn, GatewayURL: up1.URL, SpawnSpec: "fak serve", Locality: localityLocal, HealthzPath: "/healthz", CostNote: "no per-token cost"}
	reg := &targetRegistry{targets: []computeTarget{pricier, cheap}} // pricier first on purpose

	hc := &http.Client{Timeout: 2 * time.Second}
	rep, winner, err := autoSelectComputeTarget(context.Background(), reg, hc, 2*time.Second)
	if err != nil {
		t.Fatalf("autoSelect: %v", err)
	}
	if winner.Name != "cheap" || rep.Winner != "cheap" {
		t.Fatalf("winner = %q / %q, want cheap (lowest cost_class)", winner.Name, rep.Winner)
	}
	cr, _ := autoRowByName(rep, "cheap")
	pr, _ := autoRowByName(rep, "pricier")
	if cr.Rank != 1 || pr.Rank != 2 {
		t.Fatalf("ranks = cheap %d / pricier %d, want 1 / 2", cr.Rank, pr.Rank)
	}
	if cr.Health.State != "up" || cr.Health.Provenance != "witnessed" {
		t.Fatalf("cheap health = %+v, want witnessed up", cr.Health)
	}
}

// TestAutoSelectFailsOverPastDead: the CHEAPEST target is down, so --auto fails over to
// the next-cheapest healthy one and never lands on the dead gateway.
func TestAutoSelectFailsOverPastDead(t *testing.T) {
	down, up := downHealthzServer(t), upHealthzServer(t)
	cheapDead := computeTarget{Name: "cheap", Kind: targetLocalSpawn, GatewayURL: down.URL, SpawnSpec: "fak serve", Locality: localityLocal, HealthzPath: "/healthz", CostNote: "no per-token cost"}
	pricierUp := computeTarget{Name: "pricier", Kind: targetGatewayURL, GatewayURL: up.URL, Locality: localityRemote, HealthzPath: "/healthz", CostNote: "paid GPU compute"}
	reg := &targetRegistry{targets: []computeTarget{cheapDead, pricierUp}}

	hc := &http.Client{Timeout: 2 * time.Second}
	rep, winner, err := autoSelectComputeTarget(context.Background(), reg, hc, 2*time.Second)
	if err != nil {
		t.Fatalf("autoSelect: %v", err)
	}
	if winner.Name != "pricier" {
		t.Fatalf("winner = %q, want failover to pricier (cheap is down)", winner.Name)
	}
	cr, _ := autoRowByName(rep, "cheap")
	if cr.Candidate || cr.Rank != 0 || cr.Health.State != "down" {
		t.Fatalf("dead cheap row = %+v, want excluded (candidate=false rank=0 health=down)", cr)
	}
}

// TestAutoSelectNoHealthyErrors: every reachable-probed target is down and there is no
// no-healthz fallback, so --auto returns an error and selects nothing (never a dead one).
func TestAutoSelectNoHealthyErrors(t *testing.T) {
	d1, d2 := downHealthzServer(t), downHealthzServer(t)
	reg := &targetRegistry{targets: []computeTarget{
		{Name: "x", Kind: targetGatewayURL, GatewayURL: d1.URL, Locality: localityRemote, HealthzPath: "/healthz"},
		{Name: "y", Kind: targetGatewayURL, GatewayURL: d2.URL, Locality: localityRemote, HealthzPath: "/healthz"},
	}}
	hc := &http.Client{Timeout: 2 * time.Second}
	rep, winner, err := autoSelectComputeTarget(context.Background(), reg, hc, 2*time.Second)
	if err == nil || winner != nil {
		t.Fatalf("all-down should error with no winner, got winner=%v err=%v", winner, err)
	}
	if rep.Winner != "" {
		t.Fatalf("report winner = %q, want empty", rep.Winner)
	}
}

// TestAutoQuotaHonestlyLabeled proves the quota signal is a [stub] for a metered provider
// seat and n/a otherwise — never a claimed live read.
func TestAutoQuotaHonestlyLabeled(t *testing.T) {
	if s := quotaForTarget(computeTarget{Kind: targetProviderProxy}); s.Provenance != "stub" {
		t.Fatalf("provider-proxy quota provenance = %q, want stub", s.Provenance)
	}
	if s := quotaForTarget(computeTarget{Kind: targetGatewayURL}); s.Provenance != "n/a" {
		t.Fatalf("gateway quota provenance = %q, want n/a", s.Provenance)
	}
}

// TestTUIAgentAutoJSONEmitsDecision: `fak c --auto --json` emits the ranked decision (not
// a launch plan). With every gateway built-in pointed at a closed port, only the
// no-healthz anthropic target survives, so it is the deterministic winner.
func TestTUIAgentAutoJSONEmitsDecision(t *testing.T) {
	hermeticTargets(t)
	// Closed loopback port => fast connection-refused => every gateway target is down.
	t.Setenv("FAK_MAC_GATEWAY", "http://127.0.0.1:1")
	t.Setenv("FAK_GLM_GCP_BASE_URL", "http://127.0.0.1:1/v1")
	t.Setenv("FAK_LOCAL_GATEWAY", "http://127.0.0.1:1")

	var stdout, stderr bytes.Buffer
	code := runTUI(&stdout, &stderr, []string{"agent", "--auto", "--json"})
	if code != 0 {
		t.Fatalf("--auto --json code=%d stderr=%s", code, stderr.String())
	}
	var rep autoDecisionReport
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("unmarshal auto decision: %v\n%s", err, stdout.String())
	}
	if rep.Schema != computeTargetAutoSchema {
		t.Fatalf("schema = %q, want %q", rep.Schema, computeTargetAutoSchema)
	}
	if rep.Winner != "anthropic" {
		t.Fatalf("winner = %q, want anthropic (the only reachable target)", rep.Winner)
	}
	// The anthropic row must carry an honestly-labeled stub quota signal.
	ar, ok := autoRowByName(rep, "anthropic")
	if !ok || ar.Quota.Provenance != "stub" {
		t.Fatalf("anthropic quota = %+v, want a [stub] signal", ar.Quota)
	}
}

// TestTUIAgentAutoConflicts proves --auto is mutually exclusive with a named target and
// with an explicit --gateway-url.
func TestTUIAgentAutoConflicts(t *testing.T) {
	hermeticTargets(t)
	t.Run("with positional target", func(t *testing.T) {
		var so, se bytes.Buffer
		if code := runTUI(&so, &se, []string{"agent", "mac", "--auto", "--json"}); code != 2 {
			t.Fatalf("code=%d, want 2; stderr=%s", code, se.String())
		} else if !strings.Contains(se.String(), "--auto selects a target automatically") {
			t.Fatalf("stderr=%q", se.String())
		}
	})
	t.Run("with explicit gateway-url", func(t *testing.T) {
		var so, se bytes.Buffer
		if code := runTUI(&so, &se, []string{"agent", "--auto", "--gateway-url", "http://node.example:8080", "--json"}); code != 2 {
			t.Fatalf("code=%d, want 2; stderr=%s", code, se.String())
		} else if !strings.Contains(se.String(), "cannot combine with an explicit --gateway-url") {
			t.Fatalf("stderr=%q", se.String())
		}
	})
}
