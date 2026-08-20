package main

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

func installReadyDispatchWorkerPreflightProbe(t *testing.T) {
	t.Helper()
	old := dispatchCodexWorkerPreflightProbe
	dispatchCodexWorkerPreflightProbe = func(_ context.Context, req dispatchWorkerPreflightRequest) (dispatchCodexPreflightObservation, error) {
		return readyDispatchCodexObservation(req), nil
	}
	t.Cleanup(func() { dispatchCodexWorkerPreflightProbe = old })
}

func readyDispatchCodexObservation(req dispatchWorkerPreflightRequest) dispatchCodexPreflightObservation {
	return dispatchCodexPreflightObservation{
		Authenticated: true,
		AccountType:   "chatgpt",
		Models:        []string{req.Model},
	}
}

func setDispatchWorkerPreflightProbe(t *testing.T, fn func(context.Context, dispatchWorkerPreflightRequest) (dispatchCodexPreflightObservation, error)) {
	t.Helper()
	old := dispatchCodexWorkerPreflightProbe
	dispatchCodexWorkerPreflightProbe = fn
	t.Cleanup(func() { dispatchCodexWorkerPreflightProbe = old })
}

func TestDispatchWorkerPreflightClassifiesFakeProviderOutcomes(t *testing.T) {
	now := time.Date(2026, 8, 19, 3, 30, 0, 0, time.UTC)
	req := dispatchWorkerPreflightRequest{
		Backend:         "codex",
		Account:         dispatchtick.Account{Tag: "seat-a", Dir: filepath.Join("C:", "codex-a")},
		Model:           "gpt-5.6-sol",
		Workspace:       filepath.Join("C:", "repo"),
		WorkKind:        "implementation",
		DeadlineSeconds: 3600,
		Guarded:         true,
		RouteDigest:     "sha256:route",
	}
	retryAt := now.Add(17 * time.Minute)
	tests := []struct {
		name string
		obs  dispatchCodexPreflightObservation
		err  error
		want string
	}{
		{
			name: "auth invalid",
			obs:  dispatchCodexPreflightObservation{AuthError: "refresh token revoked"},
			want: dispatchWorkerPreflightAuthInvalid,
		},
		{
			name: "model unsupported",
			obs: dispatchCodexPreflightObservation{
				Authenticated: true,
				AccountType:   "chatgpt",
				Models:        []string{"gpt-5.6-terra"},
			},
			want: dispatchWorkerPreflightModelUnsupported,
		},
		{
			name: "quota exhausted",
			obs: dispatchCodexPreflightObservation{
				Authenticated:  true,
				AccountType:    "chatgpt",
				Models:         []string{"gpt-5.6-sol"},
				QuotaExhausted: true,
				RetryAt:        retryAt,
			},
			want: dispatchWorkerPreflightQuotaExhausted,
		},
		{
			name: "transient upstream",
			err:  errors.New("upstream timeout"),
			want: dispatchWorkerPreflightTransientUpstream,
		},
		{
			name: "quota endpoint transient",
			obs: dispatchCodexPreflightObservation{
				Authenticated: true,
				AccountType:   "chatgpt",
				Models:        []string{"gpt-5.6-sol"},
				QuotaError:    "quota service temporarily unavailable",
			},
			want: dispatchWorkerPreflightTransientUpstream,
		},
		{
			name: "route misconfigured",
			obs: dispatchCodexPreflightObservation{
				RouteError: "model provider env_key is missing",
			},
			want: dispatchWorkerPreflightRouteMisconfigured,
		},
		{
			name: "ready",
			obs: dispatchCodexPreflightObservation{
				Authenticated: true,
				AccountType:   "chatgpt",
				Models:        []string{"gpt-5.6-sol"},
			},
			want: dispatchWorkerPreflightReady,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			setDispatchWorkerPreflightProbe(t, func(context.Context, dispatchWorkerPreflightRequest) (dispatchCodexPreflightObservation, error) {
				return tc.obs, tc.err
			})
			got := dispatchWorkerPreflight(context.Background(), req, now)
			if got.Verdict != tc.want {
				t.Fatalf("verdict = %q, want %q (%+v)", got.Verdict, tc.want, got)
			}
			if got.Ready != (tc.want == dispatchWorkerPreflightReady) {
				t.Fatalf("ready = %v for %s", got.Ready, tc.want)
			}
			if got.Evidence == "" || got.SeatToken == "" || got.CheckedAt.IsZero() || got.ExpiresAt.IsZero() {
				t.Fatalf("preflight omitted binding/freshness evidence: %+v", got)
			}
			if tc.want == dispatchWorkerPreflightQuotaExhausted && !got.CooldownUntil.Equal(retryAt) {
				t.Fatalf("cooldown = %s, want %s", got.CooldownUntil, retryAt)
			}
			if tc.want == dispatchWorkerPreflightTransientUpstream && !got.CooldownUntil.After(now) {
				t.Fatalf("transient cooldown = %s, want bounded backoff after %s", got.CooldownUntil, now)
			}
			if tc.want == dispatchWorkerPreflightReady && !got.Binds(req, now.Add(time.Second)) {
				t.Fatalf("READY evidence did not bind its exact request: %+v", got)
			}
			if tc.want == dispatchWorkerPreflightReady {
				otherModel := req
				otherModel.Model = "gpt-5.6-terra"
				if got.Binds(otherModel, now.Add(time.Second)) || got.Binds(req, got.ExpiresAt.Add(time.Nanosecond)) {
					t.Fatalf("READY evidence admitted a different model or expired launch: %+v", got)
				}
			}
		})
	}
}

func TestDispatchWorkerPreflightParsesCodexQuotaCooldown(t *testing.T) {
	reset := time.Date(2026, 8, 19, 4, 0, 0, 0, time.UTC).Unix()
	exhausted, retryAt, errText := dispatchCodexQuotaFromResult(json.RawMessage(`{
		"rateLimits": {
			"rateLimitReachedType": "workspace_member_usage_limit_reached",
			"primary": {"usedPercent": 100, "resetsAt": ` + strconv.FormatInt(reset, 10) + `}
		},
		"rateLimitsByLimitId": null
	}`))
	if errText != "" || !exhausted || retryAt.Unix() != reset {
		t.Fatalf("quota exhausted=%v retry=%s err=%q", exhausted, retryAt, errText)
	}
}

func TestDispatchWorkerPreflightHardRefusalsSkipLeaseAndWorkerSpawn(t *testing.T) {
	tests := []struct {
		name    string
		guarded bool
		obs     dispatchCodexPreflightObservation
		want    string
	}{
		{
			name:    "invalid auth",
			guarded: true,
			obs:     dispatchCodexPreflightObservation{AuthError: "refresh token revoked"},
			want:    dispatchWorkerPreflightAuthInvalid,
		},
		{
			name:    "unsupported model",
			guarded: true,
			obs: dispatchCodexPreflightObservation{
				Authenticated: true,
				AccountType:   "chatgpt",
				Models:        []string{"gpt-5.6-terra"},
			},
			want: dispatchWorkerPreflightModelUnsupported,
		},
		{
			name: "unguarded launch",
			obs: dispatchCodexPreflightObservation{
				Authenticated: true,
				AccountType:   "chatgpt",
				Models:        []string{"gpt-5.6-sol"},
			},
			want: dispatchWorkerPreflightRouteMisconfigured,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			root, _ := dispatchCodexGateFixture(t, false)
			if tc.guarded {
				t.Setenv("FLEET_DOGFOOD_GUARD_BASEURL", healthyDispatchProvider(t)+"/v1")
			} else {
				t.Setenv("FLEET_DOGFOOD_GUARD_BASEURL", "")
			}
			setDispatchWorkerPreflightProbe(t, func(context.Context, dispatchWorkerPreflightRequest) (dispatchCodexPreflightObservation, error) {
				return tc.obs, nil
			})
			oldBroker := launchSpawnBroker
			oldSpawner := dispatchIssueWorkerSpawner
			spawned := false
			launchSpawnBroker = func(a launchBrokerAttempt) launchBrokerGrant {
				return allowLaunchBrokerGrant(a, "unit-test-allow")
			}
			dispatchIssueWorkerSpawner = func([]string, map[string]string, string, string, int, string, string, string, []string, dispatchtick.Account, *dispatchtick.Membership, string, string, float64) (dispatchSpawnResult, error) {
				spawned = true
				return dispatchSpawnResult{}, nil
			}
			t.Cleanup(func() {
				launchSpawnBroker = oldBroker
				dispatchIssueWorkerSpawner = oldSpawner
			})

			out, _, code := runDispatchAt("tick", "--workspace", root, "--backend", "codex", "--lane", "docs", "--no-refresh", "--no-loop-ledger", "--live", "--json")
			if code != 1 {
				t.Fatalf("exit = %d, want hard refusal (output %s)", code, out)
			}
			var got map[string]any
			if err := json.Unmarshal([]byte(out), &got); err != nil {
				t.Fatalf("decode: %v\n%s", err, out)
			}
			if spawned || got["action"] != "worker_preflight_refused" || got["verdict"] != tc.want || got["ok"] != false {
				t.Fatalf("spawned=%v receipt=%#v", spawned, got)
			}
			if _, acquired := got["lease"]; acquired {
				t.Fatalf("hard refusal created a lane lease: %#v", got["lease"])
			}
			if got["admitted_workers"] != float64(0) {
				t.Fatalf("admitted_workers = %#v, want 0", got["admitted_workers"])
			}
			if !tc.guarded {
				preflight := mapAt(got, "worker_preflight")
				reason := dispatchMapString(preflight, "reason")
				if preflight["guarded"] != false || !strings.Contains(reason, "FLEET_DOGFOOD_GUARD_BASEURL") || !strings.Contains(reason, "fak guard --base-url") {
					t.Fatalf("unguarded refusal omitted exact remedy: %#v", preflight)
				}
			}
		})
	}
}

func TestDispatchWorkerPreflightReadyLaunchesExactAccountModelAndEvidence(t *testing.T) {
	root, _ := dispatchCodexGateFixture(t, false)
	t.Setenv("FLEET_DOGFOOD_GUARD_BASEURL", healthyDispatchProvider(t)+"/v1")

	var probed dispatchWorkerPreflightRequest
	setDispatchWorkerPreflightProbe(t, func(_ context.Context, req dispatchWorkerPreflightRequest) (dispatchCodexPreflightObservation, error) {
		probed = req
		return readyDispatchCodexObservation(req), nil
	})
	oldBroker := launchSpawnBroker
	oldSpawner := dispatchIssueWorkerSpawner
	var capturedCommand []string
	var capturedEnv map[string]string
	var capturedAccount dispatchtick.Account
	launchSpawnBroker = func(a launchBrokerAttempt) launchBrokerGrant {
		return allowLaunchBrokerGrant(a, "unit-test-allow")
	}
	dispatchIssueWorkerSpawner = func(argv []string, env map[string]string, cwd, runsDir string, issue int, lane, backend, leaseID string, tree []string, account dispatchtick.Account, membership *dispatchtick.Membership, baseSHA, stdinPayload string, probeS float64) (dispatchSpawnResult, error) {
		capturedCommand = append([]string(nil), argv...)
		capturedEnv = copyStringMap(env)
		capturedAccount = account
		logPath := filepath.Join(runsDir, "resolve-12-20260819-033000.log")
		if err := os.MkdirAll(filepath.Dir(logPath), 0o755); err != nil {
			return dispatchSpawnResult{}, err
		}
		if err := os.WriteFile(logPath, []byte("# fak-spawn\nworking\n"), 0o644); err != nil {
			return dispatchSpawnResult{}, err
		}
		return dispatchSpawnResult{PID: 6849, Log: logPath, Issue: issue, Lane: lane, Backend: backend, LeaseID: leaseID, Tree: tree}, nil
	}
	t.Cleanup(func() {
		launchSpawnBroker = oldBroker
		dispatchIssueWorkerSpawner = oldSpawner
	})

	out, errb, code := runDispatchAt("tick", "--workspace", root, "--backend", "codex", "--lane", "docs", "--no-refresh", "--no-loop-ledger", "--live", "--json")
	if code != 0 {
		t.Fatalf("exit = %d, want launched fixture (stderr %s)\n%s", code, errb, out)
	}
	var got map[string]any
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("decode: %v\n%s", err, out)
	}
	t.Cleanup(func() { releaseInProcessLaneLease(root, mapAt(got, "lease")) })
	preflight := mapAt(got, "worker_preflight")
	if got["action"] != "spawned" || got["verdict"] != "SPAWNED" || dispatchMapString(preflight, "verdict") != dispatchWorkerPreflightReady {
		t.Fatalf("launch receipt = %#v", got)
	}
	if probed.Account.Tag == "" || capturedAccount.Tag != probed.Account.Tag || capturedAccount.Dir != probed.Account.Dir {
		t.Fatalf("preflight account=%+v spawn account=%+v", probed.Account, capturedAccount)
	}
	if probed.Model != guardCodexDefaultModelID || !slices.Contains(capturedCommand, probed.Model) {
		t.Fatalf("preflight model=%q command=%#v", probed.Model, capturedCommand)
	}
	if capturedEnv["FAK_WORKER_PREFLIGHT_EVIDENCE"] == "" ||
		capturedEnv["FAK_WORKER_PREFLIGHT_EVIDENCE"] != dispatchMapString(preflight, "evidence") ||
		capturedEnv["FAK_WORKER_PREFLIGHT_MODEL"] != probed.Model ||
		capturedEnv["FAK_WORKER_PREFLIGHT_SEAT"] != dispatchMapString(preflight, "seat_token") {
		t.Fatalf("spawn env did not carry exact preflight evidence: env=%#v preflight=%#v", capturedEnv, preflight)
	}
}
