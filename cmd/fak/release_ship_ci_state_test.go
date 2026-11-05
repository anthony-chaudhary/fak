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
				case name == "git" && sameArgs(args, "rev-parse", "--is-shallow-repository"):
					return 0, "false\n"
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
				if name == "git" && sameArgs(args, "rev-parse", "--is-shallow-repository") {
					if !tc.adopt {
						t.Fatal("history probe without ancestor opt-in")
					}
					return 0, "false\n"
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

// An unreadable or inconsistent primary history cannot justify secondary adoption.
// fak-test:runtime fast est=40ms
func TestReleaseShipAncestorCIRequiresReadableWorkflowHistory(t *testing.T) {
	const green = `[{"event":"push","status":"completed","conclusion":"success","headSha":"ancestor-sha"}]`
	for _, tc := range []struct {
		name, output, reason string
		code                 int
		adopt                bool
	}{
		{name: "query failed despite green output", code: 1, output: green, reason: "could not be queried"},
		{name: "invalid JSON", output: "not-json", reason: "not valid JSON"},
		{name: "null is not empty history", output: "null", reason: "not a JSON array"},
		{name: "success without status", output: `[{"event":"push","conclusion":"success","headSha":"ancestor-sha"}]`, reason: "incomplete or inconsistent"},
		{name: "success with unknown status", output: `[{"event":"push","status":"unknown","conclusion":"success","headSha":"ancestor-sha"}]`, reason: "incomplete or inconsistent"},
		{name: "success with nonstring status", output: `[{"event":"push","status":1,"conclusion":"success","headSha":"ancestor-sha"}]`, reason: "incomplete or inconsistent"},
		{name: "pending success", output: `[{"event":"push","status":"in_progress","conclusion":"success","headSha":"ancestor-sha"}]`, reason: "incomplete or inconsistent"},
		{name: "pending failure before green", output: `[{"event":"push","status":"queued","conclusion":"failure","headSha":"red-sha"},{"event":"push","status":"completed","conclusion":"success","headSha":"ancestor-sha"}]`, reason: "incomplete or inconsistent"},
		{name: "pending failure after green", output: `[{"event":"push","status":"completed","conclusion":"success","headSha":"ancestor-sha"},{"event":"push","status":"queued","conclusion":"failure","headSha":"red-sha"}]`, reason: "incomplete or inconsistent"},
		{name: "completed without conclusion", output: `[{"event":"push","status":"completed","headSha":"red-sha"}]`, reason: "incomplete or inconsistent"},
		{name: "completed with nonstring conclusion", output: `[{"event":"push","status":"completed","conclusion":false,"headSha":"red-sha"}]`, reason: "incomplete or inconsistent"},
		{name: "valid empty primary permits secondary", output: "[]", adopt: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			historyProbes, primaryQueries, secondaryQueries, ancestryProbes := 0, 0, 0, 0
			restore := stubReleaseShipRunner(t, func(cwd, name string, args []string, env []string, timeout time.Duration) (int, string) {
				switch {
				case name == "git" && sameArgs(args, "rev-parse", "--is-shallow-repository"):
					historyProbes++
					return 0, "false\n"
				case name == "gh" && sameArgs(args, "run", "list", "--workflow", "ci.yml", "--branch", "main", "--status", "completed", "--event", "push", "--limit", "20", "--json", "databaseId,status,conclusion,url,headSha,updatedAt,event"):
					primaryQueries++
					return tc.code, tc.output
				case name == "gh" && sameArgs(args, "run", "list", "--workflow", "ci-fast.yml", "--branch", "main", "--status", "completed", "--event", "push", "--limit", "20", "--json", "databaseId,status,conclusion,url,headSha,updatedAt,event"):
					secondaryQueries++
					return 0, green
				case name == "git" && sameArgs(args, "merge-base", "--is-ancestor", "ancestor-sha", "source-sha"):
					ancestryProbes++
					return 0, ""
				default:
					t.Fatalf("unexpected command: %s %v", name, args)
					return 127, "unexpected"
				}
			})
			defer restore()
			fallback := map[string]any{
				"ok": false, "status": "pending", "source_sha": "source-sha",
				"run": map[string]any{"headSha": "source-sha", "databaseId": 7},
			}
			var result releaseShipResult
			ci := releaseShipAdoptAncestorCI(&result, t.TempDir(), releaseShipOptions{workflow: "ci.yml", trunk: "main"}, "source-sha", fallback)
			wantSecondary := 0
			if tc.adopt {
				wantSecondary = 1
			}
			if historyProbes != 1 || primaryQueries != 1 || secondaryQueries != wantSecondary || ancestryProbes != wantSecondary {
				t.Fatalf("queries/probes=%d/%d/%d/%d; want complete primary evidence before secondary=%d", historyProbes, primaryQueries, secondaryQueries, ancestryProbes, wantSecondary)
			}
			if fallback["ancestor_ci_blocker"] != nil || fallback["status"] != "pending" || releaseShipPayloadOK(fallback) {
				t.Fatalf("original source evidence mutated: %#v", fallback)
			}
			if tc.adopt {
				if !releaseShipPayloadOK(ci) || ci["status"] != "green" || ci["adopted_ancestor_sha"] != "ancestor-sha" || ci["workflow"] != "ci-fast.yml" || ci["ancestor_ci_blocker"] != nil {
					t.Fatalf("valid empty primary lost secondary adoption: %#v", ci)
				}
				return
			}
			run, _ := ci["run"].(map[string]any)
			blocker, _ := ci["ancestor_ci_blocker"].(map[string]any)
			if releaseShipPayloadOK(ci) || ci["status"] != "pending" || ci["source_sha"] != "source-sha" || run["headSha"] != "source-sha" || run["databaseId"] != 7 || ci["adopted_ancestor_sha"] != nil || blocker["workflow"] != "ci.yml" || !strings.Contains(stringFromAny(blocker["reason"]), tc.reason) {
				t.Fatalf("unreadable primary history lost its barrier or source evidence: %#v", ci)
			}
		})
	}
}

// All command results describe a small source-history graph; no real git or gh runs.
// fak-test:runtime fast est=20ms
func TestReleaseShipAncestorCIDoesNotCrossFailedHistory(t *testing.T) {
	const red = `{"event":"push","status":"completed","conclusion":"failure","headSha":"red-sha"}`
	const oldGreen = `{"event":"push","status":"completed","conclusion":"success","headSha":"old-green"}`
	const fixedGreen = `{"event":"push","status":"completed","conclusion":"success","headSha":"fixed-green"}`
	for _, tc := range []struct {
		name, rows, adopted string
		ancestry            map[string]int
	}{
		{name: "intervening red", rows: "[" + red + "," + oldGreen + "]", ancestry: map[string]int{
			"red-sha->source-sha": 0, "old-green->source-sha": 0, "red-sha->old-green": 1,
		}},
		{name: "green listed before intervening red", rows: "[" + oldGreen + "," + red + "]", ancestry: map[string]int{
			"red-sha->source-sha": 0, "old-green->source-sha": 0, "red-sha->old-green": 1,
		}},
		{name: "same SHA red and green", rows: "[" + red + `,{"event":"push","status":"completed","conclusion":"success","headSha":"red-sha"}]`, ancestry: map[string]int{
			"red-sha->source-sha": 0,
		}},
		{name: "green supersedes older red", rows: "[" + fixedGreen + "," + red + "]", adopted: "fixed-green", ancestry: map[string]int{
			"red-sha->source-sha": 0, "fixed-green->source-sha": 0, "red-sha->fixed-green": 0,
		}},
		{name: "unrelated red", rows: "[" + red + "," + oldGreen + "]", adopted: "old-green", ancestry: map[string]int{
			"red-sha->source-sha": 1, "old-green->source-sha": 0,
		}},
		{name: "later candidate supersedes red", rows: "[" + oldGreen + "," + red + "," + fixedGreen + "]", adopted: "fixed-green", ancestry: map[string]int{
			"red-sha->source-sha": 0, "old-green->source-sha": 0, "red-sha->old-green": 1,
			"fixed-green->source-sha": 0, "red-sha->fixed-green": 0,
		}},
		{name: "one superseded red does not hide merged red", rows: "[" + red + `,{"event":"push","status":"completed","conclusion":"failure","headSha":"merged-red"},` + fixedGreen + "]", ancestry: map[string]int{
			"red-sha->source-sha": 0, "merged-red->source-sha": 0,
			"fixed-green->source-sha": 0, "red-sha->fixed-green": 0, "merged-red->fixed-green": 1,
		}},
		{name: "red blocks secondary workflow", rows: "[" + red + "]", ancestry: map[string]int{
			"red-sha->source-sha": 0,
		}},
		{name: "red source ancestry unreadable", rows: "[" + red + "," + oldGreen + "]", ancestry: map[string]int{
			"red-sha->source-sha": 128,
		}},
		{name: "red supersession unreadable", rows: "[" + red + "," + oldGreen + "]", ancestry: map[string]int{
			"red-sha->source-sha": 0, "old-green->source-sha": 0, "red-sha->old-green": 128,
		}},
		{name: "red SHA missing", rows: `[{"event":"push","status":"completed","conclusion":"failure"},` + oldGreen + "]"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sourceQueries, ancestorQueries := 0, 0
			probed := map[string]bool{}
			restore := stubReleaseShipRunner(t, func(cwd, name string, args []string, env []string, timeout time.Duration) (int, string) {
				switch {
				case name == "git" && sameArgs(args, "rev-parse", "--is-shallow-repository"):
					return 0, "false\n"
				case name == "gh" && sameArgs(args, "run", "list", "--workflow", "ci.yml", "--commit", "source-sha", "--event", "push", "--limit", "1", "--json", "databaseId,status,conclusion,url,headSha,event"):
					sourceQueries++
					return 0, `[{"event":"push","status":"in_progress","conclusion":"","headSha":"source-sha","databaseId":7}]`
				case name == "gh" && sameArgs(args, "run", "list", "--workflow", "ci.yml", "--branch", "main", "--status", "completed", "--event", "push", "--limit", "20", "--json", "databaseId,status,conclusion,url,headSha,updatedAt,event"):
					ancestorQueries++
					return 0, tc.rows
				case name == "git" && len(args) == 4 && args[0] == "merge-base" && args[1] == "--is-ancestor":
					key := args[2] + "->" + args[3]
					if code, ok := tc.ancestry[key]; ok {
						probed[key] = true
						return code, ""
					}
				}
				t.Fatalf("unexpected command: %s %v", name, args)
				return 127, "unexpected"
			})
			defer restore()
			var result releaseShipResult
			ci := releaseShipSourceCI(&result, t.TempDir(), releaseShipOptions{workflow: "ci.yml", trunk: "main", adoptAncestorCI: true}, "source-sha")
			if sourceQueries != 1 || ancestorQueries != 1 || len(probed) != len(tc.ancestry) {
				t.Fatalf("queries=%d/%d ancestry=%v; want one query each and %v", sourceQueries, ancestorQueries, probed, tc.ancestry)
			}
			if tc.adopted != "" {
				if !releaseShipPayloadOK(ci) || ci["adopted_ancestor_sha"] != tc.adopted || ci["ancestor_ci_blocker"] != nil {
					t.Fatalf("safe ancestor not adopted: %#v", ci)
				}
				return
			}
			run, _ := ci["run"].(map[string]any)
			blocker, _ := ci["ancestor_ci_blocker"].(map[string]any)
			failed, _ := blocker["run"].(map[string]any)
			if releaseShipPayloadOK(ci) || ci["status"] != "pending" || ci["source_sha"] != "source-sha" || run["headSha"] != "source-sha" || ci["adopted_ancestor_sha"] != nil || blocker["workflow"] != "ci.yml" || failed["conclusion"] != "failure" || stringFromAny(blocker["reason"]) == "" {
				t.Fatalf("failed ancestor lost its barrier or source evidence: %#v", ci)
			}
		})
	}
}

// A non-ancestor answer from a shallow graph is not complete history evidence.
// fak-test:runtime fast est=20ms
func TestReleaseShipAncestorCIRequiresCompleteHistory(t *testing.T) {
	for _, tc := range []struct {
		name, output string
		code         int
	}{
		{name: "shallow", output: "true\n"},
		{name: "empty"},
		{name: "malformed", output: "unknown\n"},
		{name: "failed probe", code: 128, output: "false\n"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			restore := stubReleaseShipRunner(t, func(cwd, name string, args []string, env []string, timeout time.Duration) (int, string) {
				calls++
				if name != "git" || !sameArgs(args, "rev-parse", "--is-shallow-repository") {
					t.Fatalf("unverified history reached ancestor query/probe: %s %v", name, args)
				}
				return tc.code, tc.output
			})
			defer restore()
			fallback := map[string]any{"ok": false, "status": "missing", "source_sha": "source-sha"}
			var result releaseShipResult
			ci := releaseShipAdoptAncestorCI(&result, t.TempDir(), releaseShipOptions{workflow: "ci.yml", trunk: "main"}, "source-sha", fallback)
			blocker, _ := ci["ancestor_ci_blocker"].(map[string]any)
			if calls != 1 || releaseShipPayloadOK(ci) || ci["status"] != "missing" || ci["source_sha"] != "source-sha" || blocker["workflow"] != "ci.yml" || !strings.Contains(stringFromAny(blocker["reason"]), "history") || fallback["ancestor_ci_blocker"] != nil {
				t.Fatalf("incomplete history escaped refusal or mutated original evidence: calls=%d CI=%#v fallback=%#v", calls, ci, fallback)
			}
		})
	}
}
