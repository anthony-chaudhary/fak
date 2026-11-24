package workflows

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/testenv/repotest"
)

// workflowDir returns .github/workflows, the package directory go test runs in.
func workflowDir(t *testing.T) string {
	t.Helper()
	dir := repotest.PackageDir(t)
	if fi, err := os.Stat(dir); err == nil && fi.IsDir() {
		return dir
	}
	t.Fatalf("could not locate the workflows directory")
	return ""
}

// tokenEnvLine matches the job-level env line that arms a Slack feeder: the secret
// binding is the marker that a workflow can post, so it is the marker that must also
// carry the mute switch.
var tokenEnvLine = regexp.MustCompile(`(?m)^([ \t]*)FAK_SCOREBOARD_TOKEN:.*secrets\.FAK_SCOREBOARD_TOKEN.*$`)

// TestEverySlackFeederCarriesTheQuietSwitch is the CI half of the global brake's
// contract. The Go brake (internal/slackquiet) mutes every surface that posts through
// slackwire, but a workflow only reaches that code if the switch is in the job's
// environment — and a workflow that binds FAK_SCOREBOARD_TOKEN without binding
// FAK_SLACK_QUIET is a feeder the operator cannot mute. That is exactly the "N switches
// to flip and N to forget" failure the brake exists to end, so it is asserted rather
// than trusted: every workflow that arms a Slack post must also carry the switch.
//
// The switch is bound from the FAK_SLACK_QUIET repo VARIABLE (not a secret): it is not
// sensitive, and a variable is what an operator can flip from the UI without a
// re-encrypt. An unset variable expands to the empty string, which slackquiet reads as
// "not quiet" — so a repo that never sets it behaves exactly as before.
func TestEverySlackFeederCarriesTheQuietSwitch(t *testing.T) {
	dir := workflowDir(t)
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("read workflows dir: %v", err)
	}

	armed, guarded := 0, 0
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".yml") {
			continue
		}
		raw, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			t.Fatalf("read %s: %v", e.Name(), err)
		}
		content := strings.ReplaceAll(string(raw), "\r\n", "\n")
		locs := tokenEnvLine.FindAllStringSubmatchIndex(content, -1)
		if len(locs) == 0 {
			continue // not a Slack feeder
		}
		armed += len(locs)
		for _, loc := range locs {
			indent := content[loc[2]:loc[3]]
			// The switch must sit in the SAME env block as the token: same indentation,
			// immediately adjacent. A switch bound at a different scope (a step env, or
			// the workflow level) would not reach the post step's process.
			want := indent + "FAK_SLACK_QUIET: ${{ vars.FAK_SLACK_QUIET }}"
			if !strings.Contains(content, want) {
				t.Errorf("%s: binds FAK_SCOREBOARD_TOKEN at indent %d but has no %q in the same env block — this feeder cannot be muted by FAK_SLACK_QUIET",
					e.Name(), len(indent), strings.TrimSpace(want))
				continue
			}
			guarded++
		}
	}

	if armed == 0 {
		t.Fatal("found no workflow binding FAK_SCOREBOARD_TOKEN; the guard's marker has drifted and it is now vacuous")
	}
	if guarded != armed {
		t.Fatalf("guarded %d of %d Slack-feeder env blocks; every one must carry the switch", guarded, armed)
	}
	t.Logf("%d Slack-feeder env block(s) across %s carry FAK_SLACK_QUIET", guarded, dir)
}

// TestQuietSwitchIsAVariableNotASecret pins the binding form. A secret would work, but
// it would make the mute switch unreadable and unflippable from the repo UI, and it
// would imply the value is sensitive when it is a single word. More importantly, a
// secret that is unset expands to empty — same as a variable — so the only thing the
// distinction buys is operator friction.
func TestQuietSwitchIsAVariableNotASecret(t *testing.T) {
	dir := workflowDir(t)
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("read workflows dir: %v", err)
	}
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".yml") {
			continue
		}
		raw, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			t.Fatalf("read %s: %v", e.Name(), err)
		}
		if strings.Contains(string(raw), "secrets.FAK_SLACK_QUIET") {
			t.Errorf("%s binds FAK_SLACK_QUIET from secrets; it must be a repo variable (vars.FAK_SLACK_QUIET)", e.Name())
		}
	}
}
