package main

import (
	"fmt"
	"strings"
	"testing"
	"time"
)

func TestReleaseShipCIState(t *testing.T) {
	tests := []struct{ status, conclusion, want string }{
		{"queued", "", "pending"},
		{"in_progress", "", "pending"},
		{"completed", "success", "green"},
		{"completed", "failure", "failed"},
		{"completed", "timed_out", "failed"},
		{"completed", "cancelled", "cancelled_or_starved"},
		{"completed", "stale", "cancelled_or_starved"},
		{"completed", "", "missing"},
	}
	for _, tt := range tests {
		t.Run(tt.status+"/"+tt.conclusion, func(t *testing.T) {
			if got := releaseShipCIState(tt.status, tt.conclusion); got != tt.want {
				t.Fatalf("got %q, want %q", got, tt.want)
			}
		})
	}
}

// fak-test:runtime fast est=20ms
func TestReleaseShipAncestorCIDoesNotOverrideExactFailure(t *testing.T) {
	for _, tc := range []struct {
		name, status, conclusion string
		adopt, noRuns            bool
	}{
		{name: "failure", status: "completed", conclusion: "failure"},
		{name: "timeout", status: "completed", conclusion: "timed_out"},
		{name: "action required", status: "completed", conclusion: "action_required"},
		{name: "startup failure", status: "completed", conclusion: "startup_failure"},
		{name: "unrecognized conclusion", status: "completed", conclusion: "unknown"},
		{name: "pending", status: "in_progress", adopt: true},
		{name: "missing conclusion", status: "completed", adopt: true},
		{name: "cancelled", status: "completed", conclusion: "cancelled", adopt: true},
		{name: "no signal", adopt: true, noRuns: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sourceQueries, ancestorQueries, probes := 0, 0, 0
			restore := stubReleaseShipRunner(t, func(cwd, name string, args []string, env []string, timeout time.Duration) (int, string) {
				switch {
				case name == "gh" && sameArgs(args, "run", "list", "--workflow", "ci.yml", "--commit", "source-sha", "--event", "push", "--limit", "1", "--json", "databaseId,status,conclusion,url,headSha,event"):
					sourceQueries++
					if tc.noRuns {
						return 0, `[]`
					}
					return 0, fmt.Sprintf(`[{"event":"push","headSha":"source-sha","status":%q,"conclusion":%q}]`, tc.status, tc.conclusion)
				case name == "gh" && sameArgs(args, "run", "list", "--workflow", "ci.yml", "--branch", "main", "--status", "completed", "--event", "push", "--limit", "20", "--json", "databaseId,status,conclusion,url,headSha,updatedAt,event"):
					ancestorQueries++
					return 0, `[{"event":"push","status":"completed","conclusion":"success","headSha":"ancestor-sha"}]`
				case name == "git" && sameArgs(args, "merge-base", "--is-ancestor", "ancestor-sha", "source-sha"):
					probes++
					return 0, ""
				default:
					t.Fatalf("unexpected command: %s %v", name, args)
					return 127, "unexpected"
				}
			})
			defer restore()
			var result releaseShipResult
			ci := releaseShipSourceCI(&result, t.TempDir(), releaseShipOptions{workflow: "ci.yml", trunk: "main", adoptAncestorCI: true}, "source-sha")
			wantAncestorQueries := 0
			if tc.adopt {
				wantAncestorQueries = 1
			}
			if sourceQueries != 1 || ancestorQueries != wantAncestorQueries || probes != wantAncestorQueries || releaseShipPayloadOK(ci) != tc.adopt {
				t.Fatalf("CI=%#v queries=%d/%d probes=%d; want adopted=%v", ci, sourceQueries, ancestorQueries, probes, tc.adopt)
			}
			if tc.adopt {
				if ci["status"] != "green" || ci["adopted_ancestor_sha"] != "ancestor-sha" {
					t.Fatalf("inconclusive source lost ancestor adoption: %#v", ci)
				}
			} else {
				run, _ := ci["run"].(map[string]any)
				if ci["status"] != "failed" || ci["conclusion"] != tc.conclusion || ci["source_sha"] != "source-sha" || run["headSha"] != "source-sha" || ci["adopted_ancestor_sha"] != nil {
					t.Fatalf("exact failure evidence was replaced: %#v", ci)
				}
			}
		})
	}
}

// All commands are stubbed and the appearance deadline is already expired.
// fak-test:runtime fast est=20ms
func TestReleaseShipCIRequiresPushEvent(t *testing.T) {
	for _, tc := range []struct {
		name, payload, status string
		code                  int
		appeared              bool
	}{
		{name: "missing event", payload: `[{"status":"completed","conclusion":"success"}]`, status: "missing"},
		{name: "null event", payload: `[{"event":null,"status":"completed","conclusion":"success"}]`, status: "missing"},
		{name: "empty event", payload: `[{"event":"","status":"completed","conclusion":"success"}]`, status: "missing"},
		{name: "manual success", payload: `[{"event":"workflow_dispatch","status":"completed","conclusion":"success"}]`, status: "missing"},
		{name: "pull request", payload: `[{"event":"pull_request","status":"completed","conclusion":"success"}]`, status: "missing"},
		{name: "scheduled", payload: `[{"event":"schedule","status":"completed","conclusion":"success"}]`, status: "missing"},
		{name: "unknown event", payload: `[{"event":"unknown","status":"completed","conclusion":"success"}]`, status: "missing"},
		{name: "wrong case", payload: `[{"event":"PUSH","status":"completed","conclusion":"success"}]`, status: "missing"},
		{name: "padded event", payload: `[{"event":" push ","status":"completed","conclusion":"success"}]`, status: "missing"},
		{name: "number event", payload: `[{"event":1,"status":"completed","conclusion":"success"}]`, status: "missing"},
		{name: "boolean event", payload: `[{"event":true,"status":"completed","conclusion":"success"}]`, status: "missing"},
		{name: "array event", payload: `[{"event":["push"],"status":"completed","conclusion":"success"}]`, status: "missing"},
		{name: "object event", payload: `[{"event":{"name":"push"},"status":"completed","conclusion":"success"}]`, status: "missing"},
		{name: "null row", payload: `[null]`, status: "missing"},
		{name: "push success", payload: `[{"event":"push","status":"completed","conclusion":"success"}]`, status: "green", appeared: true},
		{name: "push pending", payload: `[{"event":"push","status":"in_progress","conclusion":""}]`, status: "pending", appeared: true},
		{name: "push failure", payload: `[{"event":"push","status":"completed","conclusion":"failure"}]`, status: "failed", appeared: true},
		{name: "push cancelled", payload: `[{"event":"push","status":"completed","conclusion":"cancelled"}]`, status: "cancelled_or_starved", appeared: true},
		{name: "manual cannot hide push failure", payload: `[{"event":"workflow_dispatch","status":"completed","conclusion":"success"},{"event":"push","status":"completed","conclusion":"failure"}]`, status: "failed", appeared: true},
		{name: "no runs", payload: `[]`, status: "missing"},
		{name: "gh error", payload: `unavailable`, code: 1, status: "unavailable"},
		{name: "non-json", payload: `not-json`, status: "non_json"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			restore := stubReleaseShipRunner(t, func(cwd, name string, args []string, env []string, timeout time.Duration) (int, string) {
				fields := "databaseId,status,conclusion,url,headSha,event"
				if calls == 1 {
					fields = "databaseId,status,conclusion,url,event"
				}
				if name != "gh" || !sameArgs(args, "run", "list", "--workflow", "ci.yml", "--commit", "source-sha", "--event", "push", "--limit", "1", "--json", fields) {
					t.Fatalf("unexpected CI command: %s %v", name, args)
				}
				calls++
				return tc.code, tc.payload
			})
			defer restore()
			opts := releaseShipOptions{workflow: "ci.yml", ciAppearTimeout: -time.Nanosecond}
			var source, appearance releaseShipResult
			ci := releaseShipSourceCI(&source, t.TempDir(), opts, "source-sha")
			if ci["status"] != tc.status || releaseShipPayloadOK(ci) != (tc.status == "green") {
				t.Fatalf("source CI = %#v, want status %s", ci, tc.status)
			}
			if tc.status == "missing" && (ci["run"] != nil || !strings.Contains(stringFromAny(ci["reason"]), "push")) {
				t.Fatalf("unqualified evidence must be omitted with a reason: %#v", ci)
			}
			waitReleaseShipCIAppears(&appearance, t.TempDir(), opts, "source-sha")
			if calls != 2 || (len(appearance.Warnings) == 0) != tc.appeared {
				t.Fatalf("calls=%d warnings=%v, want appeared=%v", calls, appearance.Warnings, tc.appeared)
			}
			if !tc.appeared && tc.code == 0 && tc.status != "non_json" && !strings.Contains(appearance.Warnings[0], "release_tag will make the final CI decision") {
				t.Fatalf("appearance timeout lost final-tag handoff: %v", appearance.Warnings)
			}
		})
	}
}

// fak-test:runtime fast est=20ms
func TestReleaseShipAncestorCIRequiresPushEvent(t *testing.T) {
	const untrusted = `[{"status":"completed","conclusion":"success","headSha":"untrusted"},{"event":null,"conclusion":"success","headSha":"untrusted"},{"event":"workflow_dispatch","conclusion":"success","headSha":"untrusted"},{"event":"pull_request","conclusion":"success","headSha":"untrusted"},{"event":"schedule","conclusion":"success","headSha":"untrusted"},{"event":"unknown","conclusion":"success","headSha":"untrusted"},{"event":[],"conclusion":"success","headSha":"untrusted"},{"event":{},"conclusion":"success","headSha":"untrusted"}]`
	for _, tc := range []struct {
		name, pushWorkflow string
		adopt, reachable   bool
	}{
		{name: "opt-in required", pushWorkflow: "ci.yml", reachable: true},
		{name: "reject both workflows", adopt: true},
		{name: "primary workflow", pushWorkflow: "ci.yml", adopt: true, reachable: true},
		{name: "fallback workflow", pushWorkflow: "ci-fast.yml", adopt: true, reachable: true},
		{name: "unreachable push", pushWorkflow: "ci.yml", adopt: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			queries, probes := 0, 0
			restore := stubReleaseShipRunner(t, func(cwd, name string, args []string, env []string, timeout time.Duration) (int, string) {
				if name == "gh" && sameArgs(args, "run", "list", "--workflow", "ci.yml", "--commit", "source-sha", "--event", "push", "--limit", "1", "--json", "databaseId,status,conclusion,url,headSha,event") {
					return 0, `[{"event":"workflow_dispatch","status":"completed","conclusion":"success","headSha":"source-sha"}]`
				}
				if name == "gh" && len(args) > 3 {
					workflow := "ci.yml"
					if queries == 1 {
						workflow = "ci-fast.yml"
					}
					if !tc.adopt || !sameArgs(args, "run", "list", "--workflow", workflow, "--branch", "main", "--status", "completed", "--event", "push", "--limit", "20", "--json", "databaseId,status,conclusion,url,headSha,updatedAt,event") {
						t.Fatalf("unexpected ancestor query: %s %v", name, args)
					}
					queries++
					if workflow == tc.pushWorkflow {
						return 0, strings.TrimSuffix(untrusted, "]") + `,{"event":"push","status":"completed","conclusion":"success","headSha":"ancestor-sha"}]`
					}
					return 0, untrusted
				}
				if name != "git" || !sameArgs(args, "merge-base", "--is-ancestor", "ancestor-sha", "source-sha") {
					t.Fatalf("untrusted run reached ancestry probe: %s %v", name, args)
				}
				probes++
				if tc.reachable {
					return 0, ""
				}
				return 1, "not an ancestor"
			})
			defer restore()
			var result releaseShipResult
			ci := releaseShipSourceCI(&result, t.TempDir(), releaseShipOptions{workflow: "ci.yml", trunk: "main", adoptAncestorCI: tc.adopt}, "source-sha")
			wantGreen := tc.adopt && tc.reachable && tc.pushWorkflow != ""
			wantQueries, wantProbes := 0, 0
			if tc.adopt {
				wantQueries = 2
				if tc.pushWorkflow != "" {
					wantProbes = 1
				}
				if wantGreen && tc.pushWorkflow == "ci.yml" {
					wantQueries = 1
				}
			}
			if releaseShipPayloadOK(ci) != wantGreen || queries != wantQueries || probes != wantProbes {
				t.Fatalf("CI=%#v queries=%d probes=%d; want green=%v queries=%d probes=%d", ci, queries, probes, wantGreen, wantQueries, wantProbes)
			}
			if wantGreen && (ci["adopted_ancestor_sha"] != "ancestor-sha" || ci["workflow"] != tc.pushWorkflow) {
				t.Fatalf("wrong ancestor evidence: %#v", ci)
			}
			if !wantGreen && (ci["status"] != "missing" || ci["run"] != nil) {
				t.Fatalf("untrusted source must remain missing: %#v", ci)
			}
		})
	}
}
