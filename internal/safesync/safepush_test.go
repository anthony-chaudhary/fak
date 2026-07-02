package safesync

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"
)

func TestDecidePush(t *testing.T) {
	if DecidePush(PushAhead) != PushRetry {
		t.Error("ahead (remote already in HEAD) should RETRY — the rejection was a race")
	}
	for _, d := range []PushDivergence{PushBehind, PushDiverged, "weird"} {
		if DecidePush(d) != PushStop {
			t.Errorf("divergence %q should STOP (integrate first), not retry", d)
		}
	}
}

func TestIsNonFastForward(t *testing.T) {
	rejections := []string{
		" ! [rejected]        main -> main (non-fast-forward)",
		"Updates were rejected because the tip of your current branch is behind",
		"! [rejected] main -> main (fetch first)",
	}
	for _, r := range rejections {
		if !isNonFastForward(r) {
			t.Errorf("should detect non-ff: %q", r)
		}
	}
	notNFF := []string{
		"remote: Permission to repo denied to user",
		"fatal: could not read from remote repository",
		"remote rejected: PUBLIC_LEAK secret detected",
		"",
	}
	for _, r := range notNFF {
		if isNonFastForward(r) {
			t.Errorf("should NOT classify as non-ff (must surface as-is): %q", r)
		}
	}
}

// scriptedRunner replays canned RunResults for a sequence of git calls, matched by the
// subcommand (args[0]); push results are consumed in order so a retry sees the next one.
type scriptedRunner struct {
	push      []RunResult // consumed one per `git push`
	pushIdx   int
	fetch     RunResult
	ancestors map[string]int // key "a..b" -> merge-base --is-ancestor exit code
	branch    RunResult      // rev-parse --abbrev-ref HEAD
	calls     []string
}

func (s *scriptedRunner) run(_ context.Context, _ string, args ...string) RunResult {
	s.calls = append(s.calls, strings.Join(args, " "))
	switch {
	case args[0] == "push":
		r := s.push[s.pushIdx]
		if s.pushIdx < len(s.push)-1 {
			s.pushIdx++
		}
		return r
	case args[0] == "fetch":
		return s.fetch
	case args[0] == "rev-parse":
		if s.branch.Stdout == nil && s.branch.Code == 0 {
			return RunResult{Stdout: []byte("main\n")}
		}
		return s.branch
	case args[0] == "merge-base" && len(args) >= 4 && args[1] == "--is-ancestor":
		return RunResult{Code: s.ancestors[args[2]+".."+args[3]]}
	default:
		return RunResult{Code: 0}
	}
}

func nonFF() RunResult {
	return RunResult{Code: 1, Stderr: []byte(" ! [rejected] main -> main (non-fast-forward)\nUpdates were rejected because the tip of your current branch is behind")}
}

func TestSafePush_CleanFirstPush(t *testing.T) {
	sr := &scriptedRunner{push: []RunResult{{Code: 0}}}
	res, err := SafePush(context.Background(), PushOptions{Repo: ".", Branch: "main", Runner: sr.run})
	if err != nil {
		t.Fatalf("SafePush: %v", err)
	}
	if !res.Pushed || res.Attempts != 1 || res.Reason != "" {
		t.Fatalf("clean push = %+v, want pushed in 1 attempt", res)
	}
}

func TestSafePush_TransientRaceRetries(t *testing.T) {
	// First push rejected non-ff; after fetch the remote ref is an ANCESTOR of HEAD
	// (we already contain it — a race); second push succeeds.
	sr := &scriptedRunner{
		push:      []RunResult{nonFF(), {Code: 0}},
		fetch:     RunResult{Code: 0},
		ancestors: map[string]int{"origin/main..HEAD": 0}, // remote IS ancestor of HEAD -> ahead
	}
	res, err := SafePush(context.Background(), PushOptions{Repo: ".", Branch: "main", Runner: sr.run})
	if err != nil {
		t.Fatalf("SafePush: %v", err)
	}
	if !res.Pushed || res.Attempts != 2 || res.Divergence != string(PushAhead) {
		t.Fatalf("transient-race push = %+v, want pushed on attempt 2 after an 'ahead' reclassify", res)
	}
	// It must have fetched between the two pushes (re-classify), never merged/forced.
	joined := strings.Join(sr.calls, "|")
	if !strings.Contains(joined, "fetch origin main") {
		t.Errorf("expected a fetch between pushes; calls=%v", sr.calls)
	}
	for _, c := range sr.calls {
		if strings.HasPrefix(c, "merge ") || strings.Contains(c, "--force") || strings.Contains(c, "stash") || strings.Contains(c, "reset") {
			t.Errorf("SafePush must be non-destructive; saw %q", c)
		}
	}
}

func TestSafePush_BehindStops(t *testing.T) {
	// Rejected non-ff; after fetch HEAD is an ancestor of the remote (genuinely behind):
	// STOP with a clear integrate-then-push next step, never auto-merge.
	sr := &scriptedRunner{
		push:      []RunResult{nonFF()},
		fetch:     RunResult{Code: 0},
		ancestors: map[string]int{"origin/main..HEAD": 1, "HEAD..origin/main": 0}, // remote not in HEAD; HEAD in remote -> behind
	}
	res, err := SafePush(context.Background(), PushOptions{Repo: ".", Branch: "main", Runner: sr.run})
	if err != nil {
		t.Fatalf("SafePush: %v", err)
	}
	if res.Pushed || res.Reason != PushReasonBehind || res.Divergence != string(PushBehind) {
		t.Fatalf("behind push = %+v, want STOP with reason BEHIND", res)
	}
	if !strings.Contains(res.Detail, "integrate in place") || strings.Contains(res.Detail, "force") == false {
		// detail must name the safe next step and warn against force
		if !strings.Contains(res.Detail, "never force-push") {
			t.Errorf("BEHIND detail should guide integrate-then-push and warn against force: %q", res.Detail)
		}
	}
}

func TestSafePush_NonNFFErrorSurfaces(t *testing.T) {
	// A hook/secret rejection is NOT non-ff and must surface immediately, not retry.
	sr := &scriptedRunner{push: []RunResult{{Code: 1, Stderr: []byte("remote rejected: PUBLIC_LEAK secret detected")}}}
	res, err := SafePush(context.Background(), PushOptions{Repo: ".", Branch: "main", Runner: sr.run})
	if err != nil {
		t.Fatalf("SafePush: %v", err)
	}
	if res.Pushed || res.Reason != PushReasonError || res.Attempts != 1 {
		t.Fatalf("non-ff error push = %+v, want PUSH_ERROR on attempt 1 (no retry)", res)
	}
}

func TestSafePush_GitUnavailable(t *testing.T) {
	sr := &scriptedRunner{push: []RunResult{{Err: errors.New("git not found")}}}
	res, err := SafePush(context.Background(), PushOptions{Repo: ".", Branch: "main", Runner: sr.run})
	if err != nil {
		t.Fatalf("SafePush: %v", err)
	}
	if res.Reason != PushReasonGitMissing {
		t.Fatalf("git-missing push = %+v, want GIT_UNAVAILABLE", res)
	}
}

// swallowPushSleep replaces the retry backoff for the duration of a test,
// recording each wait instead of actually sleeping.
func swallowPushSleep(t *testing.T) *[]time.Duration {
	t.Helper()
	var waits []time.Duration
	prev := pushSleep
	pushSleep = func(d time.Duration) { waits = append(waits, d) }
	t.Cleanup(func() { pushSleep = prev })
	return &waits
}

func TestIsTransientPushNetwork(t *testing.T) {
	transient := []string{
		"fatal: unable to access 'https://github.com/x/y/': Could not resolve host: github.com",
		"fatal: the remote end hung up unexpectedly",
		"error: RPC failed; HTTP 502 curl 22 The requested URL returned error: 502",
		"fatal: unable to access 'https://github.com/x/y/': Failed to connect to github.com port 443: Connection timed out",
		"fatal: early EOF",
	}
	for _, m := range transient {
		if !isTransientPushNetwork(m) {
			t.Errorf("should classify transient network: %q", m)
		}
	}
	permanent := []string{
		"remote: Permission to repo denied to user",
		"fatal: Authentication failed for 'https://github.com/x/y/'",
		"remote rejected: PUBLIC_LEAK secret detected",
		" ! [rejected] main -> main (non-fast-forward)",
		"remote: error: GH013: Repository rule violations found",
		"",
	}
	for _, m := range permanent {
		if isTransientPushNetwork(m) {
			t.Errorf("must NOT retry a permanent failure as network-transient: %q", m)
		}
	}
}

func TestSafePush_TransientNetworkRetriesThenPushes(t *testing.T) {
	// The first push loses a network blip (remote hung up); the retry lands. No
	// fetch/divergence dance should run for a network failure — the remote did
	// not reject anything, the transport just dropped.
	waits := swallowPushSleep(t)
	sr := &scriptedRunner{
		push: []RunResult{
			{Code: 128, Stderr: []byte("fatal: the remote end hung up unexpectedly")},
			{Code: 0},
		},
	}
	res, err := SafePush(context.Background(), PushOptions{Repo: ".", Branch: "main", Runner: sr.run})
	if err != nil {
		t.Fatalf("SafePush: %v", err)
	}
	if !res.Pushed || res.Attempts != 2 || res.Reason != "" {
		t.Fatalf("network-blip push = %+v, want pushed on attempt 2", res)
	}
	if len(*waits) != 1 || (*waits)[0] <= 0 {
		t.Fatalf("waits = %v, want exactly one positive backoff before the re-push", *waits)
	}
	for _, c := range sr.calls {
		if strings.HasPrefix(c, "fetch") {
			t.Errorf("a network blip must not trigger the fetch/divergence dance; calls=%v", sr.calls)
		}
	}
}

func TestSafePush_PersistentNetworkFailureIsUnreachable(t *testing.T) {
	// A network failure that outlives every retry surfaces as REMOTE_UNREACHABLE
	// (the retry-eligible class), never PUSH_ERROR (the halt class).
	swallowPushSleep(t)
	sr := &scriptedRunner{
		push: []RunResult{{Code: 128, Stderr: []byte("fatal: unable to access 'https://github.com/x/y/': Could not resolve host: github.com")}},
	}
	res, err := SafePush(context.Background(), PushOptions{Repo: ".", Branch: "main", MaxRetries: 3, Runner: sr.run})
	if err != nil {
		t.Fatalf("SafePush: %v", err)
	}
	if res.Pushed || res.Reason != PushReasonUnreachable || res.Attempts != 3 {
		t.Fatalf("persistent network push = %+v, want REMOTE_UNREACHABLE after 3 attempts", res)
	}
	if !strings.Contains(res.Detail, "Could not resolve host") {
		t.Fatalf("detail should carry the raw failure headline, got %q", res.Detail)
	}
}

func TestSafePush_RaceRetryBacksOffJittered(t *testing.T) {
	// The non-ff race retry must not re-push in lockstep: a backoff (with jitter
	// in (0, base]) runs between attempts.
	waits := swallowPushSleep(t)
	sr := &scriptedRunner{
		push:      []RunResult{nonFF(), {Code: 0}},
		fetch:     RunResult{Code: 0},
		ancestors: map[string]int{"origin/main..HEAD": 0},
	}
	res, err := SafePush(context.Background(), PushOptions{Repo: ".", Branch: "main", Runner: sr.run})
	if err != nil {
		t.Fatalf("SafePush: %v", err)
	}
	if !res.Pushed || res.Attempts != 2 {
		t.Fatalf("race push = %+v, want pushed on attempt 2", res)
	}
	if len(*waits) != 1 {
		t.Fatalf("waits = %v, want exactly one backoff between the race re-push", *waits)
	}
	base := 250 * time.Millisecond // attempt 1: 1²×250ms
	if (*waits)[0] < base/2 || (*waits)[0] > base {
		t.Fatalf("backoff %v outside jitter band [%v, %v]", (*waits)[0], base/2, base)
	}
}

func TestSafePush_RetriesExhausted(t *testing.T) {
	// Always rejected non-ff and always reclassifies as a race (ahead) -> exhausts retries.
	sr := &scriptedRunner{
		push:      []RunResult{nonFF()}, // the single canned result repeats (idx clamps)
		fetch:     RunResult{Code: 0},
		ancestors: map[string]int{"origin/main..HEAD": 0},
	}
	res, err := SafePush(context.Background(), PushOptions{Repo: ".", Branch: "main", MaxRetries: 2, Runner: sr.run})
	if err != nil {
		t.Fatalf("SafePush: %v", err)
	}
	if res.Pushed || res.Reason != PushReasonExhausted || res.Attempts != 2 {
		t.Fatalf("exhausted push = %+v, want RETRIES_EXHAUSTED after 2 attempts", res)
	}
}
