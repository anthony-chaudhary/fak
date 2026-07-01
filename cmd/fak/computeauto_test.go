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
	// AC3 fit signal: a remote target's /healthz proves liveness, not model fit — so its
	// fit must be the honest n/a, never a fabricated witnessed "ok".
	if pr.Fit.State != "n/a" || pr.Fit.Provenance != "n/a" {
		t.Fatalf("pricier (remote) fit = %+v, want n/a/n/a (/healthz proves liveness, not fit)", pr.Fit)
	}
	// The local in-kernel target's fit is a genuine read: witnessed host-memory advisory,
	// or n/a when the backend (cpu-ref floor) cannot probe — never a fabricated provenance.
	if cr.Fit.Provenance != "witnessed" && cr.Fit.Provenance != "n/a" {
		t.Fatalf("local fit provenance = %q, want witnessed or n/a", cr.Fit.Provenance)
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

// TestCredForTarget covers the launch-credential signal that completes the failover gate:
// a declared bearer env var must be set (present/absent are witnessed), while the anthropic
// OAuth proxy and a no-CredEnv local serve need none (n/a).
func TestCredForTarget(t *testing.T) {
	getenv := func(m map[string]string) func(string) string {
		return func(k string) string { return m[k] }
	}
	gw := computeTarget{Name: "gw", Kind: targetGatewayURL, GatewayURL: "http://mac-host:8080", Locality: localityRemote, CredEnv: "FAK_GATEWAY_KEY"}

	if s := credForTarget(gw, getenv(map[string]string{})); s.State != "absent" || s.Provenance != "witnessed" {
		t.Errorf("empty cred env: %+v, want witnessed absent", s)
	}
	if s := credForTarget(gw, getenv(map[string]string{"FAK_GATEWAY_KEY": "sekret"})); s.State != "present" || s.Provenance != "witnessed" {
		t.Errorf("set cred env: %+v, want witnessed present", s)
	}
	// A target that declares no CredEnv (the local in-kernel serve) needs none.
	local := computeTarget{Name: "local", Kind: targetLocalSpawn, GatewayURL: "http://127.0.0.1:8080", Locality: localityLocal}
	if s := credForTarget(local, getenv(map[string]string{})); s.State != "n/a" {
		t.Errorf("no-cred local target: %+v, want n/a", s)
	}
	// The anthropic provider-proxy uses OAuth via guard even with a CredEnv named.
	anth := computeTarget{Name: "anthropic", Kind: targetProviderProxy, GatewayURL: "https://api.anthropic.com", Locality: localityRemote, CredEnv: "ANTHROPIC_API_KEY"}
	if s := credForTarget(anth, getenv(map[string]string{})); s.State != "n/a" {
		t.Errorf("anthropic proxy: %+v, want n/a (OAuth via guard)", s)
	}
}

// TestAutoSelectExcludesCredAbsentTarget proves the failover gate is health AND credential:
// a CHEAPER target that is /healthz-up but whose declared bearer is empty is excluded (not
// crowned then dead at launch), so --auto falls over to the launchable one — and once the
// credential is present, the cheaper target wins.
func TestAutoSelectExcludesCredAbsentTarget(t *testing.T) {
	up1, up2 := upHealthzServer(t), upHealthzServer(t)
	// gwSecured is cheaper (local cost class) but DECLARES a credential; gwOpen is pricier
	// (remote) and declares none, so it is always launchable.
	gwSecured := computeTarget{Name: "gwSecured", Kind: targetGatewayURL, GatewayURL: up1.URL, Locality: localityLocal, HealthzPath: "/healthz", CredEnv: "FAK_TEST_ABSENT_KEY", CostNote: "no per-token cost"}
	gwOpen := computeTarget{Name: "gwOpen", Kind: targetGatewayURL, GatewayURL: up2.URL, Locality: localityRemote, HealthzPath: "/healthz", CostNote: "paid GPU compute"}
	reg := &targetRegistry{targets: []computeTarget{gwSecured, gwOpen}}
	hc := &http.Client{Timeout: 2 * time.Second}

	// Credential ABSENT: gwSecured is up but excluded; --auto fails over to gwOpen.
	t.Setenv("FAK_TEST_ABSENT_KEY", "")
	rep, winner, err := autoSelectComputeTarget(context.Background(), reg, hc, 2*time.Second)
	if err != nil {
		t.Fatalf("autoSelect (cred absent): %v", err)
	}
	if winner.Name != "gwOpen" {
		t.Fatalf("winner = %q, want gwOpen (gwSecured is up but un-credentialed)", winner.Name)
	}
	sr, _ := autoRowByName(rep, "gwSecured")
	if sr.Candidate || sr.Rank != 0 || sr.Health.State != "up" || sr.Cred.State != "absent" {
		t.Fatalf("gwSecured row = %+v, want excluded-with-witnessed-up-health-and-absent-cred", sr)
	}

	// Credential PRESENT: gwSecured is now launchable and, being cheaper, wins.
	t.Setenv("FAK_TEST_ABSENT_KEY", "sekret")
	_, winner2, err := autoSelectComputeTarget(context.Background(), reg, hc, 2*time.Second)
	if err != nil {
		t.Fatalf("autoSelect (cred present): %v", err)
	}
	if winner2.Name != "gwSecured" {
		t.Fatalf("winner = %q, want gwSecured (cheaper, now credentialed)", winner2.Name)
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

// TestAutoSelectFailoverLadderOrder: with THREE healthy targets at distinct cost classes,
// the rank ladder (winner=1, fallbacks=2,3) strictly follows ascending cost_class — proving
// the failover ORDER, not just the winner (the earlier tests have a degenerate 2-row ladder).
func TestAutoSelectFailoverLadderOrder(t *testing.T) {
	a, b, c := upHealthzServer(t), upHealthzServer(t), upHealthzServer(t)
	localCheap := computeTarget{Name: "localCheap", Kind: targetLocalSpawn, GatewayURL: a.URL, SpawnSpec: "fak serve", Locality: localityLocal, HealthzPath: "/healthz", CostNote: "no per-token cost"} // 0
	gwMid := computeTarget{Name: "gwMid", Kind: targetGatewayURL, GatewayURL: b.URL, Locality: localityRemote, HealthzPath: "/healthz", CostNote: "no per-token cost"}                                  // 110
	proxyHigh := computeTarget{Name: "proxyHigh", Kind: targetProviderProxy, GatewayURL: c.URL, Locality: localityRemote, HealthzPath: "/healthz", CostNote: "metered per token"}                       // 150
	reg := &targetRegistry{targets: []computeTarget{proxyHigh, localCheap, gwMid}}                                                                                                                      // shuffled

	hc := &http.Client{Timeout: 2 * time.Second}
	rep, winner, err := autoSelectComputeTarget(context.Background(), reg, hc, 2*time.Second)
	if err != nil {
		t.Fatalf("autoSelect: %v", err)
	}
	if winner.Name != "localCheap" {
		t.Fatalf("winner = %q, want localCheap (cheapest)", winner.Name)
	}
	for name, wantRank := range map[string]int{"localCheap": 1, "gwMid": 2, "proxyHigh": 3} {
		row, ok := autoRowByName(rep, name)
		if !ok {
			t.Fatalf("row %q missing from decision", name)
		}
		if row.Rank != wantRank {
			t.Errorf("rank(%s) = %d, want %d (ladder must follow ascending cost_class)", name, row.Rank, wantRank)
		}
	}
	lc, _ := autoRowByName(rep, "localCheap")
	gm, _ := autoRowByName(rep, "gwMid")
	ph, _ := autoRowByName(rep, "proxyHigh")
	if !(lc.CostClass < gm.CostClass && gm.CostClass < ph.CostClass) {
		t.Fatalf("cost ladder not strictly increasing: localCheap=%d gwMid=%d proxyHigh=%d", lc.CostClass, gm.CostClass, ph.CostClass)
	}
}

// TestRenderAutoDecision exercises the human decision table + the provenance cell tagging
// (the [stub]/[n/a] honesty surface) — which the --json path never reaches and no other
// test covered.
func TestRenderAutoDecision(t *testing.T) {
	up, down := upHealthzServer(t), downHealthzServer(t)
	gwUp := computeTarget{Name: "gwUp", Kind: targetGatewayURL, GatewayURL: up.URL, Locality: localityRemote, HealthzPath: "/healthz", CostNote: "no per-token cost"}       // 110, up -> winner
	gwDown := computeTarget{Name: "gwDown", Kind: targetGatewayURL, GatewayURL: down.URL, Locality: localityRemote, HealthzPath: "/healthz", CostNote: "no per-token cost"} // down -> excluded
	proxy := computeTarget{Name: "proxy", Kind: targetProviderProxy, GatewayURL: "https://api.example", Locality: localityRemote, CostNote: "metered per token"}            // no /healthz -> n/a, [stub] quota
	reg := &targetRegistry{targets: []computeTarget{gwUp, gwDown, proxy}}

	hc := &http.Client{Timeout: 2 * time.Second}
	rep, winner, err := autoSelectComputeTarget(context.Background(), reg, hc, 2*time.Second)
	if err != nil {
		t.Fatalf("autoSelect: %v", err)
	}
	if winner.Name != "gwUp" {
		t.Fatalf("winner = %q, want gwUp (cheapest healthy)", winner.Name)
	}
	var buf bytes.Buffer
	renderAutoDecision(&buf, rep)
	out := buf.String()
	for _, want := range []string{
		"gwUp *",                   // the winner row is marked
		"CRED",                     // the launch-credential column header
		"assumed-available [stub]", // a stub quota cell is tagged
		"[n/a]",                    // an n/a cell is tagged, never bare
		"down",                     // the excluded target shows its witnessed down state
		"policy: launchable first", // the honest policy line (health AND credential gate)
		"quota is a [stub] signal", // the honest quota footer
		"winner: gwUp",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("render missing %q:\n%s", want, out)
		}
	}
}

// TestTUIAgentAutoDryRunLaunchPlan exercises the non-JSON --auto launch glue (selectedTarget
// = winner.Name; the preflight skip): it logs the ranked decision to stderr and, with
// --dry-run, renders the WINNER's resolved launch plan to stdout.
func TestTUIAgentAutoDryRunLaunchPlan(t *testing.T) {
	hermeticTargets(t)
	up := upHealthzServer(t)
	t.Setenv("FAK_GATEWAY_KEY", "test-bearer")
	t.Setenv("FAK_MAC_GATEWAY", up.URL)                       // mac is up (cost 110) -> winner
	t.Setenv("FAK_GLM_GCP_BASE_URL", "http://127.0.0.1:1/v1") // gcp down
	t.Setenv("FAK_LOCAL_GATEWAY", "http://127.0.0.1:1")       // local down

	var stdout, stderr bytes.Buffer
	code := runTUI(&stdout, &stderr, []string{"agent", "--auto", "--dry-run", "--width", "1000"})
	if code != 0 {
		t.Fatalf("--auto --dry-run code=%d stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "RANK") || !strings.Contains(stderr.String(), "winner: mac") {
		t.Fatalf("ranked decision not logged to stderr:\n%s", stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "existing-fak-gateway") || !strings.Contains(out, up.URL) {
		t.Fatalf("dry-run plan did not resolve to the mac winner gateway %q:\n%s", up.URL, out)
	}
}
