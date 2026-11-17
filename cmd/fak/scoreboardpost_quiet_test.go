package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/scoreboard"
	"github.com/anthony-chaudhary/fak/internal/slackquiet"
)

// TestScoreboardPostFlowQuietSkipsBeforeResolving is the contract that makes the brake
// usable from CI: with FAK_SLACK_QUIET engaged the post is a clean SKIP that exits 0
// WITHOUT a channel or a token. That ordering is the point — a muted fleet must not need
// the secret it is muting, and the skip must land before the enqueue-on-error path so a
// braked post is never spooled and delivered stale on the first drain after unmuting.
func TestScoreboardPostFlowQuietSkipsBeforeResolving(t *testing.T) {
	clearSlackEnv(t)
	t.Setenv(slackquiet.EnvVar, "1")

	var stdout, stderr bytes.Buffer
	// No channel flag, no token flag, and a resolver that would fail: if the brake were
	// checked after resolution this would exit 2 with the no-channel hint.
	code := scoreboardPostFlow(&stdout, &stderr, scoreboard.Update{Title: "fleet"}, scoreboardPostOpts{
		prefix:        "scoreboard post",
		noChannelHint: "no channel: pass --channel",
		resolveChan:   func() string { return "" },
		resolveToken:  func() string { return "" },
	})
	if code != 0 {
		t.Fatalf("code = %d, want 0 (a braked post is a skip, not a failure); stderr=%s", code, stderr.String())
	}
	if got := stdout.String(); !strings.Contains(got, slackquiet.EnvVar) || !strings.Contains(got, "nothing posted") {
		t.Fatalf("skip should name the switch and say nothing was posted:\n%s", got)
	}
	if stderr.Len() != 0 {
		t.Fatalf("a braked post must not write to stderr: %s", stderr.String())
	}
}

// TestScoreboardPostFlowDisengagedStillResolves pins the fail-safe direction: with the
// switch unset the flow behaves exactly as before, so a fresh clone and a fresh CI runner
// are unaffected by the brake's existence.
func TestScoreboardPostFlowDisengagedStillResolves(t *testing.T) {
	clearSlackEnv(t)
	t.Setenv(slackquiet.EnvVar, "off") // not in the opt-in vocabulary => not quiet

	var stdout, stderr bytes.Buffer
	code := scoreboardPostFlow(&stdout, &stderr, scoreboard.Update{Title: "fleet"}, scoreboardPostOpts{
		prefix:        "scoreboard post",
		noChannelHint: "no channel: pass --channel",
		resolveChan:   func() string { return "" },
		resolveToken:  func() string { return "" },
	})
	if code != 2 {
		t.Fatalf("code = %d, want 2 (the unbraked flow still requires a channel)", code)
	}
	if !strings.Contains(stderr.String(), "no channel") {
		t.Fatalf("stderr should carry the no-channel hint: %s", stderr.String())
	}
}

// TestScoreboardPostFlowQuietBeatsDryRunOrdering: dry_run already prints the card and
// exits 0, so the brake must not change that path — a dry run is a render, not a write.
func TestScoreboardPostFlowQuietBeatsDryRunOrdering(t *testing.T) {
	clearSlackEnv(t)
	t.Setenv(slackquiet.EnvVar, "1")

	var stdout, stderr bytes.Buffer
	code := scoreboardPostFlow(&stdout, &stderr, scoreboard.Update{Title: "fleet", Detail: "CARD"}, scoreboardPostOpts{dryRun: true})
	if code != 0 {
		t.Fatalf("code = %d, want 0", code)
	}
	if !strings.Contains(stdout.String(), "CARD") {
		t.Fatalf("a dry run under the brake must still render the card:\n%s", stdout.String())
	}
}
