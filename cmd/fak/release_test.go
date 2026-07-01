package main

import (
	"bytes"
	"encoding/json"
	"io"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestReleaseDefaultsToStatusAndPassesFlags(t *testing.T) {
	old := releaseRunStatus
	defer func() { releaseRunStatus = old }()

	var gotArgs []string
	releaseRunStatus = func(stdout, stderr io.Writer, args []string) int {
		gotArgs = append([]string(nil), args...)
		return 7
	}

	var out, errb bytes.Buffer
	rc := runRelease(&out, &errb, []string{"--json", "--skip-gh"})
	if rc != 7 {
		t.Fatalf("exit = %d, want 7", rc)
	}
	if !reflect.DeepEqual(gotArgs, []string{"--json", "--skip-gh"}) {
		t.Fatalf("args = %#v", gotArgs)
	}
}

func TestReleaseStatusSubcommandUsesNativeRunner(t *testing.T) {
	old := releaseRunStatus
	defer func() { releaseRunStatus = old }()

	var gotArgs []string
	releaseRunStatus = func(stdout, stderr io.Writer, args []string) int {
		gotArgs = append([]string(nil), args...)
		return 3
	}

	rc := runRelease(io.Discard, io.Discard, []string{"status", "--json", "--skip-cut-plan"})
	if rc != 3 {
		t.Fatalf("exit = %d, want 3", rc)
	}
	if !reflect.DeepEqual(gotArgs, []string{"--json", "--skip-cut-plan"}) {
		t.Fatalf("args = %#v", gotArgs)
	}
}

func TestReleaseDispatchesKnownHelper(t *testing.T) {
	old := releaseRunScript
	defer func() { releaseRunScript = old }()

	var gotScript string
	var gotArgs []string
	releaseRunScript = func(root, script string, args []string, stdout, stderr io.Writer) int {
		gotScript = script
		gotArgs = append([]string(nil), args...)
		return 0
	}

	rc := runRelease(io.Discard, io.Discard, []string{"publish", "--version", "1.2.3", "--json"})
	if rc != 0 {
		t.Fatalf("exit = %d, want 0", rc)
	}
	if gotScript != "release_publish.py" {
		t.Fatalf("script = %q, want release_publish.py", gotScript)
	}
	if !reflect.DeepEqual(gotArgs, []string{"--version", "1.2.3", "--json"}) {
		t.Fatalf("args = %#v", gotArgs)
	}
}

func TestReleaseDispatchesShipHelper(t *testing.T) {
	old := releaseRunShip
	defer func() { releaseRunShip = old }()

	var gotArgs []string
	releaseRunShip = func(stdout, stderr io.Writer, args []string) int {
		gotArgs = append([]string(nil), args...)
		return 9
	}

	rc := runRelease(io.Discard, io.Discard, []string{"ship", "--execute", "--json"})
	if rc != 9 {
		t.Fatalf("exit = %d, want 9", rc)
	}
	if !reflect.DeepEqual(gotArgs, []string{"--execute", "--json"}) {
		t.Fatalf("args = %#v", gotArgs)
	}
}

func TestReleaseDispatchesStableHelper(t *testing.T) {
	old := releaseRunScript
	defer func() { releaseRunScript = old }()

	var gotScript string
	var gotArgs []string
	releaseRunScript = func(root, script string, args []string, stdout, stderr io.Writer) int {
		gotScript = script
		gotArgs = append([]string(nil), args...)
		return 0
	}

	rc := runRelease(io.Discard, io.Discard, []string{"stable-context", "--codename", "2026-06-bedrock", "--json"})
	if rc != 0 {
		t.Fatalf("exit = %d, want 0", rc)
	}
	if gotScript != "stable_release_context.py" {
		t.Fatalf("script = %q, want stable_release_context.py", gotScript)
	}
	if !reflect.DeepEqual(gotArgs, []string{"--codename", "2026-06-bedrock", "--json"}) {
		t.Fatalf("args = %#v", gotArgs)
	}
}

func TestReleaseExecuteCutAddsSkipDryRun(t *testing.T) {
	args := releaseArgs("cut", []string{"--execute", "--json"})
	if !reflect.DeepEqual(args, []string{"--execute", "--json", "--skip-dry-run"}) {
		t.Fatalf("args = %#v", args)
	}
	already := releaseArgs("tag", []string{"--execute", "--skip-dry-run", "--json"})
	if !reflect.DeepEqual(already, []string{"--execute", "--skip-dry-run", "--json"}) {
		t.Fatalf("already = %#v", already)
	}
	dry := releaseArgs("cut", []string{"--json"})
	if !reflect.DeepEqual(dry, []string{"--json"}) {
		t.Fatalf("dry = %#v", dry)
	}
}

func TestReleaseUnknownSubcommand(t *testing.T) {
	var errb bytes.Buffer
	rc := runRelease(io.Discard, &errb, []string{"nope"})
	if rc != 2 {
		t.Fatalf("exit = %d, want 2", rc)
	}
	if !strings.Contains(errb.String(), "unknown subcommand") {
		t.Fatalf("missing unknown-subcommand error:\n%s", errb.String())
	}
	if !strings.Contains(errb.String(), "stable|stable-context") {
		t.Fatalf("help does not surface stable release helpers:\n%s", errb.String())
	}
}

func TestReleaseUsageSurfacesCanonicalPath(t *testing.T) {
	var out bytes.Buffer
	releaseUsage(&out)
	text := out.String()
	for _, want := range []string{
		"release_decide -> release_lock -> release_cut",
		"release_tag",
		"release_publish",
		"release-artifacts verification",
		"ship --execute",
		"staleness",
		"stable|stable-context",
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("usage missing %q:\n%s", want, text)
		}
	}
}

func TestReleaseStatusNativeJSONEnvelope(t *testing.T) {
	oldRun := releaseStatusRunJSON
	oldNow := releaseStatusNow
	defer func() {
		releaseStatusRunJSON = oldRun
		releaseStatusNow = oldNow
	}()

	var scripts []string
	releaseStatusRunJSON = func(root string, timeout time.Duration, script string, args ...string) (map[string]any, int, error) {
		scripts = append(scripts, script)
		switch script {
		case "release_context.py":
			return map[string]any{
				"head_sha":                "abc123",
				"current_branch":          "main",
				"last_tag":                "v0.1.0",
				"latest_any_tag":          "v0.1.0",
				"commits_since_tag":       []any{},
				"files_touched_since_tag": []any{},
				"tag_drift":               map[string]any{},
				"ci_on_head":              map[string]any{"status": "green"},
				"workflows_parse_ok":      map[string]any{"ok": true},
			}, 0, nil
		case "release_decide.py":
			return map[string]any{
				"decision":       "hold",
				"reason":         "nothing release-worthy pending",
				"blockers":       []any{},
				"warnings":       []any{},
				"last_tag":       "v0.1.0",
				"latest_any_tag": "v0.1.0",
			}, 2, nil
		case "release_lock.py":
			return map[string]any{"held": true, "lock": map[string]any{"owner": "peer"}}, 0, nil
		default:
			t.Fatalf("unexpected helper script %s", script)
		}
		return nil, 1, nil
	}
	releaseStatusNow = func() time.Time { return time.Date(2026, 6, 29, 12, 0, 0, 0, time.UTC) }

	var out, errb bytes.Buffer
	rc := runReleaseStatus(&out, &errb, []string{"--json", "--skip-gh", "--skip-cut-plan", "--limit-commits", "5"})
	if rc != 0 {
		t.Fatalf("exit = %d, stderr: %s", rc, errb.String())
	}
	var got map[string]any
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out.String())
	}
	if got["schema"] != releaseStatusSchema {
		t.Fatalf("schema = %v, want %s", got["schema"], releaseStatusSchema)
	}
	rolling := got["rolling"].(map[string]any)
	if rolling["last_tag"] != "v0.1.0" {
		t.Fatalf("rolling.last_tag = %v", rolling["last_tag"])
	}
	if cut := rolling["cut_plan"].(map[string]any); cut["skipped"] != true {
		t.Fatalf("cut_plan = %#v, want skipped", cut)
	}
	if gh := got["github_release"].(map[string]any); gh["status"] != "skipped" {
		t.Fatalf("github_release = %#v, want skipped", gh)
	}
	if _, ok := got["development_head"]; !ok {
		t.Fatalf("top-level development_head missing: %#v", got)
	}
	if _, ok := got["release_head"]; !ok {
		t.Fatalf("top-level release_head missing: %#v", got)
	}
	if got["development_branch"] == "" || got["release_branch"] == "" || got["latest_tag"] != "v0.1.0" {
		t.Fatalf("top-level branch fields missing: %#v", got)
	}
	regime := got["branch_regime"].(map[string]any)
	if regime["latest_tag"] != "v0.1.0" || regime["release_lock_held"] != true {
		t.Fatalf("branch_regime = %#v, want latest tag and held lock", regime)
	}
	blockers, _ := regime["promotion_blockers"].([]any)
	if len(blockers) == 0 {
		t.Fatalf("branch_regime promotion blockers empty: %#v", regime)
	}
	shadow := got["shadow_cutover"].(map[string]any)
	if shadow["checklist"] != "docs/branch-regime-shadow-cutover.md" || shadow["decision"] == "" {
		t.Fatalf("shadow_cutover missing checklist/decision: %#v", shadow)
	}
	if len(scripts) != 3 || scripts[0] != "release_context.py" || scripts[1] != "release_decide.py" || scripts[2] != "release_lock.py" {
		t.Fatalf("helpers = %#v, want context+decide+release_lock", scripts)
	}
}

func TestReleaseStatusShadowCutoverDecision(t *testing.T) {
	mainOnly := releaseStatusShadowCutover(map[string]any{
		"development_branch": "main",
		"release_branch":     "main",
		"release_source":     "main",
		"public_front_door":  "main",
		"promotion_blockers": []string{},
	}, map[string]any{"ok": true})
	if mainOnly["decision"] != "hold" || mainOnly["ready"] != false {
		t.Fatalf("main-only shadow decision = %#v, want hold", mainOnly)
	}
	if !containsString(releaseStatusStringSlice(mainOnly["blockers"]), "BRANCH_ROLES_NOT_SPLIT") {
		t.Fatalf("main-only blockers = %#v, want BRANCH_ROLES_NOT_SPLIT", mainOnly["blockers"])
	}

	splitRolesWithOpenProofGaps := releaseStatusShadowCutover(map[string]any{
		"development_branch": "dev",
		"release_branch":     "main",
		"release_source":     "dev",
		"public_front_door":  "main",
		"promotion_blockers": []string{},
	}, map[string]any{"ok": true})
	if splitRolesWithOpenProofGaps["decision"] != "hold" || splitRolesWithOpenProofGaps["ready"] != false {
		t.Fatalf("split-role shadow decision = %#v, want hold until proof gaps clear", splitRolesWithOpenProofGaps)
	}
	if blockers := releaseStatusStringSlice(splitRolesWithOpenProofGaps["blockers"]); len(blockers) != 0 {
		t.Fatalf("split-role blockers = %#v, want none", blockers)
	}
	if gaps := releaseStatusStringSlice(splitRolesWithOpenProofGaps["proof_gaps"]); !containsString(gaps, "PILOT_COHORT_WITNESS") {
		t.Fatalf("proof gaps = %#v, want pilot cohort witness gap", gaps)
	}
	if gaps := releaseStatusStringSlice(splitRolesWithOpenProofGaps["proof_gaps"]); len(gaps) > 0 && releaseStatusBool(splitRolesWithOpenProofGaps["ready"]) {
		t.Fatalf("shadow cutover reported ready=true with proof gaps: %#v", splitRolesWithOpenProofGaps)
	}
}

func TestReleaseStatusRenderIncludesBranchRegime(t *testing.T) {
	status := map[string]any{
		"rolling": map[string]any{
			"last_tag":          "v1.2.3",
			"commits_since_tag": 9,
			"decision":          map[string]any{"decision": "hold", "reason": "waiting"},
		},
		"stable":      map[string]any{"latest_stable": nil},
		"next_action": map[string]any{"kind": "wait", "detail": "nothing release-worthy pending"},
		"branch_regime": map[string]any{
			"development_branch": "dev",
			"release_branch":     "main",
			"development_ahead":  float64(3),
			"release_ahead":      float64(0),
			"drift":              "development_ahead",
			"promotion_blocked":  true,
			"promotion_blockers": []string{"DEVELOPMENT_CI_RED"},
		},
		"shadow_cutover": map[string]any{
			"decision":   "hold",
			"blockers":   []string{"PROMOTION_BLOCKED_DEVELOPMENT_CI_RED"},
			"proof_gaps": []string{"PILOT_COHORT_WITNESS"},
		},
	}
	out := renderReleaseStatus(status)
	for _, want := range []string{
		"last tag: v1.2.3",
		"branch regime: dev is 3 commit(s) ahead of main; promotion blocked: DEVELOPMENT_CI_RED",
		"shadow cutover: hold; blocker: PROMOTION_BLOCKED_DEVELOPMENT_CI_RED",
		"commits since tag: 9",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("render missing %q:\n%s", want, out)
		}
	}
}

func TestReleaseStatusRenderShowsProofGapWhenNoBlocker(t *testing.T) {
	out := releaseStatusRenderShadowCutover(map[string]any{
		"decision":   "hold",
		"blockers":   []string{},
		"proof_gaps": []string{"PILOT_COHORT_WITNESS", "FINAL_DECISION_RECORD"},
	})
	if !strings.Contains(out, "shadow cutover: hold; proof gap: PILOT_COHORT_WITNESS (+1 more)") {
		t.Fatalf("render missing proof-gap reason:\n%s", out)
	}
}

func TestReleaseStatusRelevantDirtyPathClassifier(t *testing.T) {
	for _, path := range []string{
		"VERSION",
		".github/workflows/ci.yml",
		"docs/releases/v1.2.3.md",
		"tools/release_status.py",
		"tools/stable_release_promote.py",
	} {
		if !releaseStatusIsRelevantDirtyPath(path) {
			t.Fatalf("%s should be release-relevant", path)
		}
	}
	for _, path := range []string{
		"tools/control_pane.loops.json",
		"docs/dispatch-loop.md",
		"cmd/fak/release.go",
	} {
		if releaseStatusIsRelevantDirtyPath(path) {
			t.Fatalf("%s should not be release-relevant", path)
		}
	}
}
