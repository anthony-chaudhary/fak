package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/fleetaccounts"
)

func TestCodexCensusAgreesAcrossStatusWavePreflightAndResolve(t *testing.T) {
	root := t.TempDir()
	home := filepath.Join(root, "home")
	configHome := filepath.Join(root, "config")
	regDir := filepath.Join(root, "registry")
	for _, dir := range []string{home, configHome, regDir} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	writeCodexCensusHome(t, home, "alpha", `{"auth_mode":"chatgpt","tokens":{"access_token":"fixture-alpha","account_id":"uuid-alpha"}}`)
	writeCodexCensusHome(t, home, "beta", `{"auth_mode":"chatgpt","tokens":{"access_token":"fixture-beta","account_id":"uuid-beta"}}`)
	needsLogin := filepath.Join(home, ".codex-needs-login")
	if err := os.MkdirAll(needsLogin, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(needsLogin, "config.toml"), []byte("# no auth\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(home, ".codex-archive"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(regDir, "probe_ledger.jsonl"), nil, 0o600); err != nil {
		t.Fatal(err)
	}

	t.Setenv("FLEET_USER_HOME", home)
	t.Setenv("FLEET_CONFIG_HOME", configHome)
	t.Setenv("FLEET_REG_DIR", regDir)
	t.Setenv("FLEET_POLICY_PATH", filepath.Join(root, "missing-policy.json"))

	var stdout, stderr strings.Builder
	if code := runFleetAccounts(&stdout, &stderr, []string{"status", "--provider", "codex", "--json"}); code != 0 {
		t.Fatalf("status exit=%d stderr=%s", code, stderr.String())
	}
	var status fleetaccounts.StatusReport
	if err := json.Unmarshal([]byte(stdout.String()), &status); err != nil {
		t.Fatalf("decode status: %v\n%s", err, stdout.String())
	}
	wantReady := []string{"alpha", "beta"}
	if got := readyStatusTags(status.Accounts); !reflect.DeepEqual(got, wantReady) {
		t.Fatalf("status ready tags = %v, want %v; totals=%+v", got, wantReady, status.Totals)
	}
	if status.Totals.AuthBlockedAccounts != 1 || status.Totals.NonAccounts != 1 {
		t.Fatalf("status exclusions = auth:%d non-account:%d, want 1/1; accounts=%+v",
			status.Totals.AuthBlockedAccounts, status.Totals.NonAccounts, status.Accounts)
	}
	assertTypedCodexExclusion(t, status.Accounts, "needs-login", "auth", fleetaccounts.RootPresent)
	assertTypedCodexExclusion(t, status.Accounts, "archive", string(fleetaccounts.KindNonAccount), fleetaccounts.RootMalformed)

	paths := fleetaccounts.ResolvePaths(filepath.Join(root, "tools"))
	rows := fleetaccounts.AnnotatedRoster(paths.Home, paths.ConfigHome,
		fleetaccounts.LoadPolicy(paths), fleetaccounts.LoadRegistry(paths.RegistryPath))
	wave := fleetaccounts.AllocateWave(rows, fleetaccounts.WaveRequest{
		Count: 2, Product: "codex", WorkKind: "engineering",
	}, fleetaccounts.DefaultPolicy())
	if got := waveReadyTags(wave); !reflect.DeepEqual(got, wantReady) {
		t.Fatalf("wave eligible tags = %v, want %v; wave=%+v", got, wantReady, wave)
	}

	dispatchRows := dispatchAuthoritativeAccountRows(root)
	pool := dispatchtick.BuildSeatPool(dispatchRows, nil, "codex")
	if got := dispatchPoolReadyTags(pool); !reflect.DeepEqual(got, wantReady) {
		t.Fatalf("dispatch preflight eligible tags = %v, want %v; pool=%+v", got, wantReady, pool)
	}
	route := dispatchtick.RouteAccount(dispatchtick.AccountRouteInput{
		Rows: dispatchRows, Product: "codex", WorkKind: "engineering",
	})
	if !route.OK {
		t.Fatalf("resolve refused canonical Codex census: %+v", route)
	}
	if got := resolveReadyTags(route.SeatSelection); !reflect.DeepEqual(got, wantReady) {
		t.Fatalf("resolve eligible tags = %v, want %v; selection=%+v", got, wantReady, route.SeatSelection)
	}
}

func writeCodexCensusHome(t *testing.T, home, tag, auth string) {
	t.Helper()
	dir := filepath.Join(home, ".codex-"+tag)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "auth.json"), []byte(auth), 0o600); err != nil {
		t.Fatal(err)
	}
}

func readyStatusTags(accounts []fleetaccounts.StatusAccount) []string {
	var tags []string
	for _, account := range accounts {
		if account.Kind == string(fleetaccounts.KindWorker) && account.CapacityCounted &&
			(account.State == "ready" || account.State == "leased" || account.State == "full") {
			tags = append(tags, account.Tag)
		}
	}
	sort.Strings(tags)
	return tags
}

func assertTypedCodexExclusion(t *testing.T, accounts []fleetaccounts.StatusAccount, tag, state string, rootState fleetaccounts.RootState) {
	t.Helper()
	for _, account := range accounts {
		if account.Tag != tag {
			continue
		}
		if account.State != state || account.RootState != rootState || account.Reason == "" {
			t.Fatalf("exclusion %s = state:%q root:%q reason:%q, want %q/%q/typed reason",
				tag, account.State, account.RootState, account.Reason, state, rootState)
		}
		return
	}
	t.Fatalf("typed exclusion %q absent from status: %+v", tag, accounts)
}

func waveReadyTags(wave fleetaccounts.WaveResult) []string {
	tags := make([]string, 0, len(wave.Lanes))
	for _, lane := range wave.Lanes {
		tags = append(tags, lane.Tag)
	}
	sort.Strings(tags)
	return tags
}

func dispatchPoolReadyTags(pool dispatchtick.SeatPoolResult) []string {
	var tags []string
	for _, seat := range pool.Seats {
		if seat.Available {
			tags = append(tags, seat.Tag)
		}
	}
	sort.Strings(tags)
	return tags
}

func resolveReadyTags(selection dispatchtick.SeatSelection) []string {
	var tags []string
	for _, candidate := range selection.Candidates {
		if candidate.CanServe && candidate.SeatFree {
			tags = append(tags, candidate.Tag)
		}
	}
	sort.Strings(tags)
	return tags
}
