package safesync

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"
)

func twoPointClock(start, end time.Time) func() time.Time {
	calls := 0
	return func() time.Time {
		calls++
		if calls == 1 {
			return start
		}
		return end
	}
}

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

func TestSafePush_ReconcilesConcurrentPublicationAfterHookFailure(t *testing.T) {
	sr := &scriptedRunner{
		push:      []RunResult{{Code: 1, Stderr: []byte("DUPLICATION (advisory): stale hook output")}},
		fetch:     RunResult{Code: 0},
		ancestors: map[string]int{"HEAD..origin/main": 0},
	}
	res, err := SafePush(context.Background(), PushOptions{Repo: ".", Branch: "main", Runner: sr.run})
	if err != nil {
		t.Fatalf("SafePush: %v", err)
	}
	if !res.Pushed || res.Reason != "" || res.Attempts != 1 {
		t.Fatalf("concurrently published push = %+v, want PUSHED on attempt 1", res)
	}
	if !strings.Contains(res.Detail, "concurrent publication") {
		t.Fatalf("detail = %q, want concurrent-publication explanation", res.Detail)
	}
	wantCalls := []string{"push origin main", "fetch origin main", "merge-base --is-ancestor HEAD origin/main"}
	if strings.Join(sr.calls, "|") != strings.Join(wantCalls, "|") {
		t.Fatalf("calls = %v, want %v", sr.calls, wantCalls)
	}
}

func TestSafePush_HeadlineNamesTheBlockingGateNotAnEarlierAdvisory(t *testing.T) {
	// A pre-push run prints its gates in order and only the last one can be the blocker, so a
	// gate that merely WARNED reaches stderr first. Reporting the first line sends the operator
	// off to dedupe code that never blocked the push, while the gate they actually have to clear
	// scrolls past unread.
	sr := &scriptedRunner{push: []RunResult{{Code: 1, Stderr: []byte(
		"DUPLICATION (advisory): added Go code clones a tracked site in origin/main..HEAD.\n" +
			"  internal/x/y.go:12-34 (7 windows)\n" +
			"CLAIM_UNWITNESSED (blocked): a commit in origin/main..HEAD claims something its diff does not witness.\n" +
			"  fix (not yet pushed): amend the subject to match what the diff does")}}}
	res, err := SafePush(context.Background(), PushOptions{Repo: ".", Branch: "main", Runner: sr.run})
	if err != nil {
		t.Fatalf("SafePush: %v", err)
	}
	if res.Pushed || res.Reason != PushReasonError {
		t.Fatalf("hook rejection = %+v, want PUSH_ERROR", res)
	}
	if !strings.Contains(res.Detail, "CLAIM_UNWITNESSED (blocked)") {
		t.Errorf("detail must name the gate that refused the push, got %q", res.Detail)
	}
	if strings.Contains(res.Detail, "DUPLICATION") {
		t.Errorf("an advisory must never be reported as the refusal, got %q", res.Detail)
	}
}

func TestSafePush_HeadlineFallsBackToFirstLineWithoutABlockedGate(t *testing.T) {
	// Not every rejection comes from a fak gate: a remote-side hook or an auth failure carries no
	// `(blocked)` marker, and there the first line is still the most actionable headline.
	sr := &scriptedRunner{push: []RunResult{{Code: 1, Stderr: []byte(
		"remote: error: hook declined to update refs/heads/main\n" +
			" ! [remote rejected] main -> main (hook declined)")}}}
	res, err := SafePush(context.Background(), PushOptions{Repo: ".", Branch: "main", Runner: sr.run})
	if err != nil {
		t.Fatalf("SafePush: %v", err)
	}
	if res.Reason != PushReasonError {
		t.Fatalf("hook rejection = %+v, want PUSH_ERROR", res)
	}
	if res.Detail != "remote: error: hook declined to update refs/heads/main" {
		t.Errorf("without a blocked gate the first line is the headline, got %q", res.Detail)
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
		code, ok := s.ancestors[args[2]+".."+args[3]]
		if !ok {
			code = 1
		}
		return RunResult{Code: code}
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

func TestSafePushVelocityQualifiesPublishedResult(t *testing.T) {
	start := time.Unix(1_700_000_000, 0)
	sr := &scriptedRunner{push: []RunResult{{Code: 0}}}
	res, err := SafePush(context.Background(), PushOptions{
		Repo: ".", Branch: "main", Runner: sr.run,
		VelocityBudget: time.Second,
		Now:            twoPointClock(start, start.Add(250*time.Millisecond)),
	})
	if err != nil {
		t.Fatal(err)
	}
	v := res.Velocity
	if !v.Qualified || v.ElapsedMS != 250 || v.BudgetMS != 1000 || v.BudgetRatio != 0.25 || v.Score == nil || *v.Score != 100 || v.Grade != "A" {
		t.Fatalf("velocity = %+v, want qualified 250ms/1s ratio=.25 score=100/A", v)
	}
	if len(v.Notes) != 1 || !strings.Contains(v.Notes[0], "published") {
		t.Fatalf("velocity notes = %v, want publication qualification", v.Notes)
	}
}

func TestSafePushVelocityDegradesBeyondBudget(t *testing.T) {
	start := time.Unix(1_700_000_000, 0)
	sr := &scriptedRunner{push: []RunResult{{Code: 0}}}
	res, err := SafePush(context.Background(), PushOptions{
		Repo: ".", Branch: "main", Runner: sr.run,
		VelocityBudget: time.Second,
		Now:            twoPointClock(start, start.Add(2*time.Second)),
	})
	if err != nil {
		t.Fatal(err)
	}
	v := res.Velocity
	if !v.Qualified || v.BudgetRatio != 2 || v.Score == nil || *v.Score != 50 || v.Grade != "F" {
		t.Fatalf("velocity = %+v, want qualified ratio=2 score=50/F", v)
	}
}

func TestSafePushVelocityNeverRewardsFastRefusal(t *testing.T) {
	start := time.Unix(1_700_000_000, 0)
	sr := &scriptedRunner{push: []RunResult{{Code: 1, Stderr: []byte("remote rejected: policy")}}}
	res, err := SafePush(context.Background(), PushOptions{
		Repo: ".", Branch: "main", Runner: sr.run,
		VelocityBudget: time.Second,
		Now:            twoPointClock(start, start.Add(time.Millisecond)),
	})
	if err != nil {
		t.Fatal(err)
	}
	v := res.Velocity
	if v.Qualified || v.Score != nil || v.Grade != "UNSCORED" || v.ElapsedMS != 1 || v.BudgetRatio != 0.001 {
		t.Fatalf("velocity = %+v, want 1ms refusal retained but unscored", v)
	}
	b, err := json.Marshal(res)
	if err != nil {
		t.Fatal(err)
	}
	jsonText := string(b)
	for _, want := range []string{`"velocity":`, `"score":null`, `"grade":"UNSCORED"`, `"budget_ratio":0.001`} {
		if !strings.Contains(jsonText, want) {
			t.Fatalf("JSON missing %s: %s", want, jsonText)
		}
	}
}

func TestSafePushVelocityIncludesRetryBackoff(t *testing.T) {
	current := time.Unix(1_700_000_000, 0)
	start := current
	var waited time.Duration
	previousWait := pushWait
	pushWait = func(ctx context.Context, d time.Duration) error {
		waited += d
		current = current.Add(d)
		return ctx.Err()
	}
	t.Cleanup(func() { pushWait = previousWait })
	sr := &scriptedRunner{
		push:      []RunResult{nonFF(), {Code: 0}},
		fetch:     RunResult{Code: 0},
		ancestors: map[string]int{"origin/main..HEAD": 0},
	}
	runner := func(ctx context.Context, repo string, args ...string) RunResult {
		res := sr.run(ctx, repo, args...)
		current = current.Add(10 * time.Millisecond)
		return res
	}
	res, err := SafePush(context.Background(), PushOptions{
		Repo: ".", Branch: "main", Runner: runner,
		VelocityBudget: time.Second,
		Now:            func() time.Time { return current },
	})
	if err != nil {
		t.Fatal(err)
	}
	wantElapsed := current.Sub(start).Milliseconds()
	if waited <= 0 || res.Attempts != 2 || res.Velocity.ElapsedMS != wantElapsed {
		t.Fatalf("attempts=%d waited=%v velocity=%+v, want elapsed=%dms including backoff", res.Attempts, waited, res.Velocity, wantElapsed)
	}
}

func TestSafePushVelocityRetainsTimingOnGoError(t *testing.T) {
	start := time.Unix(1_700_000_000, 0)
	sr := &scriptedRunner{branch: RunResult{Code: -1, Err: errors.New("git unavailable")}}
	res, err := SafePush(context.Background(), PushOptions{
		Repo: ".", Runner: sr.run, VelocityBudget: time.Second,
		Now: twoPointClock(start, start.Add(12*time.Millisecond)),
	})
	if err == nil {
		t.Fatal("SafePush error = nil, want git unavailable")
	}
	if res.Velocity.Qualified || res.Velocity.Score != nil || res.Velocity.ElapsedMS != 12 || !strings.Contains(strings.Join(res.Velocity.Notes, " "), "INTERNAL_ERROR") {
		t.Fatalf("velocity = %+v, want 12ms INTERNAL_ERROR unscored", res.Velocity)
	}
}

func TestSafePushRejectsSubMillisecondVelocityBudget(t *testing.T) {
	start := time.Unix(1_700_000_000, 0)
	res, err := SafePush(context.Background(), PushOptions{
		Repo: ".", Branch: "main", VelocityBudget: 500 * time.Microsecond,
		Now: twoPointClock(start, start),
	})
	if err == nil || !strings.Contains(err.Error(), "at least 1ms") {
		t.Fatalf("error = %v, want budget validation", err)
	}
	if res.Velocity.BudgetMS != 1 || res.Velocity.Score != nil || res.Velocity.Grade != "UNSCORED" {
		t.Fatalf("velocity = %+v, want present unscored minimum-resolution evidence", res.Velocity)
	}
}

func TestScorePushVelocityClampsNegativeClockSkew(t *testing.T) {
	v := ScorePushVelocity(PushResult{Pushed: true}, -time.Second, time.Second, nil)
	if v.ElapsedMS != 0 || v.BudgetRatio != 0 || !v.Qualified || v.Score == nil || *v.Score != 100 {
		t.Fatalf("velocity = %+v, want zero-elapsed qualified score after skew clamp", v)
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

func TestSafePush_SourceRefspecClassifiesAgainstSource(t *testing.T) {
	sr := &scriptedRunner{
		push:      []RunResult{nonFF(), {Code: 0}},
		fetch:     RunResult{Code: 0},
		ancestors: map[string]int{"origin/main..abc123": 0}, // remote IS ancestor of pushed source -> retry
	}
	res, err := SafePush(context.Background(), PushOptions{
		Repo:      ".",
		Remote:    "origin",
		Branch:    "main",
		SourceRef: "abc123",
		TargetRef: "refs/heads/main",
		Runner:    sr.run,
	})
	if err != nil {
		t.Fatalf("SafePush: %v", err)
	}
	if !res.Pushed || res.Attempts != 2 || res.Divergence != string(PushAhead) {
		t.Fatalf("source-refspec push = %+v, want pushed on attempt 2 after an 'ahead' reclassify", res)
	}
	joined := strings.Join(sr.calls, "|")
	if !strings.Contains(joined, "push origin abc123:refs/heads/main") {
		t.Fatalf("push must publish the exact source refspec, calls=%v", sr.calls)
	}
	if strings.Contains(joined, "push origin main") {
		t.Fatalf("source-refspec push must not fall back to mutable branch-tip push, calls=%v", sr.calls)
	}
	if !strings.Contains(joined, "merge-base --is-ancestor origin/main abc123") {
		t.Fatalf("source-refspec push must classify the pushed source, calls=%v", sr.calls)
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
	if !strings.Contains(res.Detail, "fak sync apply --fetch --remote origin --branch main") {
		t.Errorf("BEHIND detail should name the safe sync recovery command: %q", res.Detail)
	}
	if !strings.Contains(res.Detail, "write set is clean") {
		t.Errorf("BEHIND detail should explain the clean-write-set guard: %q", res.Detail)
	}
	if !strings.Contains(res.Detail, "never force-push") {
		t.Errorf("BEHIND detail should warn against force-push: %q", res.Detail)
	}
	if strings.Contains(res.Detail, "git merge") {
		t.Errorf("BEHIND detail should not steer agents to raw git merge: %q", res.Detail)
	}
}

func TestSafePush_DivergedDisjointStops(t *testing.T) {
	customRunner := func(ctx context.Context, repo string, args ...string) RunResult {
		switch {
		case args[0] == "push":
			return nonFF()
		case args[0] == "fetch":
			return RunResult{Code: 0}
		case args[0] == "rev-parse":
			return RunResult{Stdout: []byte("main\n")}
		case args[0] == "merge-base" && len(args) >= 4 && args[1] == "--is-ancestor":
			return RunResult{Code: 1} // neither is ancestor -> diverged
		case args[0] == "merge-base":
			return RunResult{Stdout: []byte("base123\n")}
		case args[0] == "diff" && args[len(args)-1] == "HEAD":
			return RunResult{Stdout: []byte("pkg/a.go\n")}
		case args[0] == "diff" && args[len(args)-1] == "origin/main":
			return RunResult{Stdout: []byte("pkg/b.go\n")}
		default:
			return RunResult{Code: 0}
		}
	}
	res, err := SafePush(context.Background(), PushOptions{Repo: ".", Branch: "main", Runner: customRunner})
	if err != nil {
		t.Fatalf("SafePush: %v", err)
	}
	if res.Pushed || res.Reason != ReasonDivergedDisjoint || res.Divergence != string(PushDiverged) {
		t.Fatalf("diverged push = %+v, want Reason=%s Divergence=%s", res, ReasonDivergedDisjoint, PushDiverged)
	}
	if !strings.Contains(res.Detail, "disjoint paths") {
		t.Errorf("detail should mention disjoint paths: %q", res.Detail)
	}
}

func TestSafePush_DivergedOverlapStops(t *testing.T) {
	customRunner := func(ctx context.Context, repo string, args ...string) RunResult {
		switch {
		case args[0] == "push":
			return nonFF()
		case args[0] == "fetch":
			return RunResult{Code: 0}
		case args[0] == "rev-parse":
			return RunResult{Stdout: []byte("main\n")}
		case args[0] == "merge-base" && len(args) >= 4 && args[1] == "--is-ancestor":
			return RunResult{Code: 1} // neither is ancestor -> diverged
		case args[0] == "merge-base":
			return RunResult{Stdout: []byte("base123\n")}
		case args[0] == "diff" && args[len(args)-1] == "HEAD":
			return RunResult{Stdout: []byte("pkg/a.go\n")}
		case args[0] == "diff" && args[len(args)-1] == "origin/main":
			return RunResult{Stdout: []byte("pkg/a.go\npkg/b.go\n")}
		default:
			return RunResult{Code: 0}
		}
	}
	res, err := SafePush(context.Background(), PushOptions{Repo: ".", Branch: "main", Runner: customRunner})
	if err != nil {
		t.Fatalf("SafePush: %v", err)
	}
	if res.Pushed || res.Reason != ReasonDivergedOverlap || res.Divergence != string(PushDiverged) {
		t.Fatalf("diverged push = %+v, want Reason=%s Divergence=%s", res, ReasonDivergedOverlap, PushDiverged)
	}
	if !strings.Contains(res.Detail, "overlapping paths") {
		t.Errorf("detail should mention overlapping paths: %q", res.Detail)
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
// recording each wait instead of actually sleeping (still honoring an
// already-cancelled ctx, like the real pushWait).
func swallowPushSleep(t *testing.T) *[]time.Duration {
	t.Helper()
	var waits []time.Duration
	prev := pushWait
	pushWait = func(ctx context.Context, d time.Duration) error {
		waits = append(waits, d)
		return ctx.Err()
	}
	t.Cleanup(func() { pushWait = prev })
	return &waits
}

func TestIsTransientPushNetwork(t *testing.T) {
	transient := []string{
		"fatal: unable to access 'https://github.com/x/y/': Could not resolve host: github.com",
		"fatal: the remote end hung up unexpectedly",
		"error: RPC failed; HTTP 502 curl 22 The requested URL returned error: 502",
		"fatal: unable to access 'https://github.com/x/y/': Failed to connect to github.com port 443: Connection timed out",
		"fatal: unable to access 'https://github.com/x/y.git/': Recv failure: Connection was reset",
		"fatal: early EOF",
		// GitHub's most common transient SSH throttle, both OpenSSH phrasings.
		"kex_exchange_identification: Connection closed by remote host\r\nfatal: Could not read from remote repository.",
		"ssh_exchange_identification: Connection closed by remote host",
		// curl 52: the server dropped before answering.
		"fatal: unable to access 'https://github.com/x/y.git/': Empty reply from server",
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
		// A permanent 403 DRAGS transport trailers that match transient needles
		// ("unexpected disconnect", "the remote end hung up") — the permanent
		// marker must win or SafePush spins retries against an auth wall.
		"error: RPC failed; HTTP 403 curl 22 The requested URL returned error: 403\n" +
			"send-pack: unexpected disconnect while reading sideband packet\n" +
			"fatal: the remote end hung up unexpectedly",
		// SSH auth rejection: "Permission denied (publickey)" is permanent even
		// though SSH transport failures are otherwise retryable.
		"git@github.com: Permission denied (publickey).\r\nfatal: Could not read from remote repository.",
	}
	for _, m := range permanent {
		if isTransientPushNetwork(m) {
			t.Errorf("must NOT retry a permanent failure as network-transient: %q", m)
		}
	}
}

func TestSafePush_403WithTransportTrailersSurfacesImmediately(t *testing.T) {
	// The full multi-line 403 blob must exit PUSH_ERROR on attempt 1 — no
	// retries, no REMOTE_UNREACHABLE — despite its transient-looking trailers.
	waits := swallowPushSleep(t)
	sr := &scriptedRunner{push: []RunResult{{Code: 128, Stderr: []byte(
		"error: RPC failed; HTTP 403 curl 22 The requested URL returned error: 403\n" +
			"send-pack: unexpected disconnect while reading sideband packet\n" +
			"fatal: the remote end hung up unexpectedly")}}}
	res, err := SafePush(context.Background(), PushOptions{Repo: ".", Branch: "main", Runner: sr.run})
	if err != nil {
		t.Fatalf("SafePush: %v", err)
	}
	if res.Pushed || res.Reason != PushReasonError || res.Attempts != 1 {
		t.Fatalf("403 push = %+v, want PUSH_ERROR on attempt 1 (permanent wins over trailers)", res)
	}
	if len(*waits) != 0 {
		t.Fatalf("no backoff should fire for a permanent rejection, got %v", *waits)
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

func TestSafePush_CancelledMidBackoffIsHonest(t *testing.T) {
	// A ctx cancelled during the backoff surfaces as CANCELLED — never as a
	// misattributed PUSH_ERROR/GIT_UNAVAILABLE from the next (dead) git spawn.
	swallowPushSleep(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // dead before the first backoff; the swallowed pushWait returns ctx.Err()
	sr := &scriptedRunner{
		push: []RunResult{{Code: 128, Stderr: []byte("fatal: the remote end hung up unexpectedly")}},
	}
	res, err := SafePush(ctx, PushOptions{Repo: ".", Branch: "main", Runner: sr.run})
	if err != nil {
		t.Fatalf("SafePush: %v", err)
	}
	if res.Pushed || res.Reason != PushReasonCancelled || res.Attempts != 1 {
		t.Fatalf("cancelled push = %+v, want CANCELLED after attempt 1", res)
	}
	if !strings.Contains(res.Detail, "cancelled during retry backoff") {
		t.Fatalf("detail should name the cancellation, got %q", res.Detail)
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
