package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gitdaily"
	"github.com/anthony-chaudhary/fak/internal/metrics"
	"github.com/anthony-chaudhary/fak/internal/treedoctor"
)

// TestGitDailyEmitUnitTaskScheduler is the "cron thing" acceptance witness on Windows:
// the emitted unit fires THIS verb, on a daily cadence, under the reboot-surviving S4U
// principal, and catches up a firing the box slept through. A daily job registered
// without those last two properties silently stops being daily.
func TestGitDailyEmitUnitTaskScheduler(t *testing.T) {
	root := t.TempDir()
	var out, errOut bytes.Buffer
	if code := runGitDaily(&out, &errOut, []string{"--emit-unit", "taskscheduler", "--root", root}); code != 0 {
		t.Fatalf("exit = %d, stderr = %s", code, errOut.String())
	}
	got := out.String()
	for _, want := range []string{
		"Register-ScheduledTask",
		"-TaskName 'fak-git-daily'",
		"-Execute 'fak' -Argument 'git-daily --root " + root,
		"New-TimeSpan -Seconds 86400", // the --interval default: once a day
		"-LogonType S4U",
		"-RunLevel Limited",
		"-Principal $principal",
		"-StartWhenAvailable",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("emitted unit missing %q\n---\n%s", want, got)
		}
	}
	// #3322 forbids the legacy path outright: it cannot express a principal.
	if strings.Contains(got, "schtasks /Create") {
		t.Errorf("emitted the legacy schtasks form:\n%s", got)
	}
}

// TestGitDailyEmitUnitPosixTargets keeps the other two schedulers honest: each fires the
// same verb with the same cadence, so an operator gets the identical contract whichever
// box they install it on.
func TestGitDailyEmitUnitPosixTargets(t *testing.T) {
	root := t.TempDir()
	for _, tc := range []struct{ target, want string }{
		{"systemd", "ExecStart=/usr/local/bin/fak git-daily --root "},
		{"launchd", "<string>--root</string>"},
	} {
		var out, errOut bytes.Buffer
		code := runGitDaily(&out, &errOut, []string{
			"--emit-unit", tc.target, "--fak-bin", "/usr/local/bin/fak", "--root", root,
		})
		if code != 0 {
			t.Fatalf("%s: exit = %d, stderr = %s", tc.target, code, errOut.String())
		}
		if !strings.Contains(out.String(), tc.want) {
			t.Errorf("%s unit missing %q\n---\n%s", tc.target, tc.want, out.String())
		}
		if !strings.Contains(out.String(), "git-daily") {
			t.Errorf("%s unit does not fire the verb\n---\n%s", tc.target, out.String())
		}
	}
}

// TestGitDailyEmitUnitRefusesWithoutARoot: a unit whose command relies on cwd discovery
// exits 2 every day from the scheduler's own working directory while looking installed.
// The refusal moves that failure to install time, where someone is watching.
func TestGitDailyEmitUnitRefusesWithoutARoot(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := emitGitDailyUnit(&out, &errOut, "taskscheduler", "", "fak", "", 24*time.Hour); code != 2 {
		t.Fatalf("exit = %d, want 2 (usage)", code)
	}
	if out.Len() != 0 {
		t.Fatalf("emitted a rootless unit: %s", out.String())
	}
	if !strings.Contains(errOut.String(), "--root") {
		t.Errorf("stderr = %q, want it to name the missing flag", errOut.String())
	}
}

// TestGitDailyEmitUnitRejectsUnknownTarget: a typo must be a usage error, never a
// silently-empty unit an operator installs and believes is scheduled.
func TestGitDailyEmitUnitRejectsUnknownTarget(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := runGitDaily(&out, &errOut, []string{"--emit-unit", "cron"}); code != 2 {
		t.Fatalf("exit = %d, want 2 (usage)", code)
	}
	if out.Len() != 0 {
		t.Fatalf("emitted a unit for an unknown target: %s", out.String())
	}
	if !strings.Contains(errOut.String(), "unknown --emit-unit") {
		t.Errorf("stderr = %q, want it to name the bad target", errOut.String())
	}
}

// TestRenderGitDailyStatusSurfacesTheStuckStreak: the readback exists to answer "has
// this been folding, or deferring for a week?" — so a streak has to be visible without
// the operator diffing the rows themselves.
func TestRenderGitDailyStatusSurfacesTheStuckStreak(t *testing.T) {
	var out bytes.Buffer
	writeGitDailyStatus(&out, "C:/repo/.git/"+gitdaily.LedgerName, []gitdaily.Row{
		{Schema: gitdaily.Schema, Day: "2026-08-02", LooseBefore: 4200, LooseAfter: 12, PacksBefore: 3, PacksAfter: 3},
		{Schema: gitdaily.Schema, Day: "2026-08-03", LooseBefore: 5000, LooseAfter: 5000, GraceRefused: "LOCKED"},
		{Schema: gitdaily.Schema, Day: "2026-08-04", LooseBefore: 9000, LooseAfter: 9000, GraceRefused: "LOCKED"},
	})
	got := out.String()
	for _, want := range []string{"folded 4188 loose", "fold tier LOCKED", "held back LOCKED for 2 consecutive runs"} {
		if !strings.Contains(got, want) {
			t.Errorf("status render missing %q\n---\n%s", want, got)
		}
	}
}

// TestRenderGitDailyStatusEmptyLedger: a clone that has never ticked says so plainly,
// rather than rendering a blank block an operator would read as "ran, found nothing".
func TestRenderGitDailyStatusEmptyLedger(t *testing.T) {
	var out bytes.Buffer
	writeGitDailyStatus(&out, "/tmp/x.jsonl", nil)
	if !strings.Contains(out.String(), "no runs recorded yet") {
		t.Errorf("empty-ledger render = %q", out.String())
	}
}

// TestRenderGitDailyTextSkips keeps the two skips legible: both are normal outcomes of a
// coarse OS trigger, so neither may read as a failure, and each must say what to do next.
func TestRenderGitDailyTextSkips(t *testing.T) {
	var out bytes.Buffer
	writeGitDailyText(&out, gitdaily.Result{
		Schema: gitdaily.Schema, Day: "2026-08-04", Apply: true,
		Skipped: gitdaily.SkipAlreadyRanToday, LastRunDay: "2026-08-04",
	})
	if got := out.String(); !strings.Contains(got, "ALREADY_RAN_TODAY") || !strings.Contains(got, "--force") {
		t.Errorf("already-ran render = %q, want the reason and the way back in", got)
	}

	out.Reset()
	writeGitDailyText(&out, gitdaily.Result{
		Schema: gitdaily.Schema, Day: "2026-08-04", Apply: true, Skipped: gitdaily.SkipTickBusy,
	})
	if got := out.String(); !strings.Contains(got, "TICK_BUSY") {
		t.Errorf("busy render = %q", got)
	}
}

// TestGitDailyStatusReportIsValidJSON pins the machine-readable envelope: the ledger path
// travels WITH the rows, so a consumer never has to re-derive which file it read.
func TestGitDailyStatusReportIsValidJSON(t *testing.T) {
	b, err := json.Marshal(gitDailyStatusReport{
		Schema: "fak-git-daily-status/1",
		Ledger: "/repo/.git/" + gitdaily.LedgerName,
		Rows:   []gitdaily.Row{{Schema: gitdaily.Schema, Day: "2026-08-04"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	var back map[string]any
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatal(err)
	}
	if back["ledger"] == "" || back["schema"] != "fak-git-daily-status/1" {
		t.Fatalf("envelope = %s", b)
	}
}

// TestGitDailyStatusRendersOutcomeCounters is the #5586 acceptance: the success /
// refusal / error tally has to be readable off the EXISTING --status surface, so a
// regression shows up in the readout an operator already runs instead of needing a bug
// report to go looking for it.
func TestGitDailyStatusRendersOutcomeCounters(t *testing.T) {
	var out bytes.Buffer
	writeGitDailyStatus(&out, "C:/repo/.git/"+gitdaily.LedgerName, []gitdaily.Row{
		{Schema: gitdaily.Schema, Day: "2026-08-01", LooseBefore: 4200, LooseAfter: 12},
		{Schema: gitdaily.Schema, Day: "2026-08-02", GraceRefused: "LOCKED"},
		{Schema: gitdaily.Schema, Day: "2026-08-03", GraceRefused: "LOCKED"},
		{Schema: gitdaily.Schema, Day: "2026-08-04", GraceRefused: "POSTURE_DRIFT", Incident: true},
	})
	got := out.String()
	for _, want := range []string{
		"4 recorded runs (2026-08-01..2026-08-04)",
		"1 ok, 2 refused, 1 error",
		"folded 4188 loose objects",
		"refused LOCKED x2",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("status render missing %q\n---\n%s", want, got)
		}
	}
}

// TestGitDailyStatusCountersAreHonestAboutSkips: PRUNE_OFF is the configured posture of
// a healthy default run (the grace-prune tier is opt-in), so a clean history must read
// as all-ok. If it read as "refused", the counter would cry wolf on every single default
// clone and nobody would look at it when a real LOCKED streak arrived.
func TestGitDailyStatusCountersAreHonestAboutSkips(t *testing.T) {
	var out bytes.Buffer
	writeGitDailyStatus(&out, "/tmp/x.jsonl", []gitdaily.Row{
		{Schema: gitdaily.Schema, Day: "2026-08-04", LooseBefore: 4129, LooseAfter: 0, GracePruneRefused: "PRUNE_OFF"},
	})
	got := out.String()
	if !strings.Contains(got, "1 ok, 0 refused, 0 error") {
		t.Errorf("default-posture history did not read as healthy\n---\n%s", got)
	}
	// The wording must not let an operator read the tally as trigger fires: skipped
	// ticks (ALREADY_RAN_TODAY / TICK_BUSY) deliberately write no row.
	if !strings.Contains(got, "recorded runs") {
		t.Errorf("counter line does not qualify what it counts\n---\n%s", got)
	}
}

// TestGitDailyStatusReportCarriesOutcomes pins the machine-readable half of the same
// ask: the counters ride in the --status --json envelope, folded from exactly the rows
// beside them, so the two can never disagree.
func TestGitDailyStatusReportCarriesOutcomes(t *testing.T) {
	rows := []gitdaily.Row{
		{Schema: gitdaily.Schema, Day: "2026-08-03", LooseBefore: 100, LooseAfter: 0},
		{Schema: gitdaily.Schema, Day: "2026-08-04", GraceRefused: "LOCKED"},
	}
	b, err := json.Marshal(gitDailyStatusReport{
		Schema:   "fak-git-daily-status/1",
		Ledger:   "/repo/.git/" + gitdaily.LedgerName,
		Outcomes: gitdaily.FoldOutcomes(rows),
		Rows:     rows,
	})
	if err != nil {
		t.Fatal(err)
	}
	var back struct {
		Outcomes gitdaily.Outcomes `json:"outcomes"`
		Rows     []gitdaily.Row    `json:"rows"`
	}
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatal(err)
	}
	if back.Outcomes.Runs != len(back.Rows) {
		t.Fatalf("counters (%d runs) disagree with the rows they shipped with (%d): %s",
			back.Outcomes.Runs, len(back.Rows), b)
	}
	if back.Outcomes.OK != 1 || back.Outcomes.Refused != 1 || back.Outcomes.Reasons["LOCKED"] != 1 {
		t.Fatalf("outcome envelope = %s", b)
	}
}

// TestGitDailyStatusRealLedgerCapture is the #5586 WITNESS, pinned: the five
// `fak-git-daily/1` rows below are the verbatim contents of this clone's own ledger
// (C:\work\fak\.git\fak-git-daily.jsonl) on 2026-08-05, and the asserted line is the
// readout `fak git-daily --status 50` actually printed from them:
//
//	git-daily status — C:\work\fak\.git\fak-git-daily.jsonl
//	5 recorded runs (2026-08-04..2026-08-05): 5 ok, 0 refused, 0 error; folded 10038 loose objects
//
// Replaying the captured BYTES (not hand-built structs) through the same Status ->
// writeGitDailyStatus path the operator surface uses is what keeps the capture honest:
// if the fold, the parse, or the rendering drifts, this test fails instead of the
// readout quietly starting to lie. Note the first row folded 0 loose (4129 -> 4129) yet
// still counts ok — a tick that reaped locks and found the object DB already packed did
// its job; only a refusal or an incident is a non-ok outcome.
func TestGitDailyStatusRealLedgerCapture(t *testing.T) {
	captured := []string{
		`{"schema":"fak-git-daily/1","day":"2026-08-04","at":"2026-08-04T11:14:37-07:00","lease_locks_reaped":0,"index_locks_reaped":7,"lock_actions":1,"loose_before":4129,"loose_after":4129,"packs_before":4,"packs_after":4,"grace_prune_refused":"PRUNE_OFF"}`,
		`{"schema":"fak-git-daily/1","day":"2026-08-04","at":"2026-08-04T11:18:58-07:00","lease_locks_reaped":0,"index_locks_reaped":0,"lock_actions":0,"loose_before":4129,"loose_after":0,"packs_before":4,"packs_after":2,"grace_prune_refused":"PRUNE_OFF"}`,
		`{"schema":"fak-git-daily/1","day":"2026-08-04","at":"2026-08-04T11:59:46-07:00","lease_locks_reaped":0,"index_locks_reaped":0,"lock_actions":1,"loose_before":48,"loose_after":0,"packs_before":2,"packs_after":4,"grace_prune_refused":"PRUNE_OFF"}`,
		`{"schema":"fak-git-daily/1","day":"2026-08-04","at":"2026-08-04T12:15:16-07:00","lease_locks_reaped":0,"index_locks_reaped":0,"lock_actions":1,"loose_before":34,"loose_after":0,"packs_before":4,"packs_after":4,"grace_prune_refused":"PRUNE_OFF"}`,
		`{"schema":"fak-git-daily/1","day":"2026-08-05","at":"2026-08-05T04:39:17-07:00","lease_locks_reaped":0,"index_locks_reaped":1,"lock_actions":2,"loose_before":5827,"loose_after":0,"packs_before":4,"packs_after":4,"grace_prune_refused":"PRUNE_OFF"}`,
	}
	path := filepath.Join(t.TempDir(), gitdaily.LedgerName)
	if err := os.WriteFile(path, []byte(strings.Join(captured, "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	var out bytes.Buffer
	writeGitDailyStatus(&out, path, gitdaily.Status(path, 50))
	const want = "5 recorded runs (2026-08-04..2026-08-05): 5 ok, 0 refused, 0 error; folded 10038 loose objects"
	if got := out.String(); !strings.Contains(got, want) {
		t.Fatalf("captured readout drifted\nwant line: %s\n---got---\n%s", want, got)
	}
}

// gitDailyCapturedLedger writes the verbatim `fak-git-daily/1` rows of this clone's own
// ledger (C:\work\fak\.git\fak-git-daily.jsonl, captured 2026-08-05) to a temp file and
// returns its path. Shared by the #5587 score tests so they replay the SAME real bytes
// the #5586 status test does — if the two readouts ever disagree about one ledger, that
// is the bug the card exists to make impossible.
func gitDailyCapturedLedger(t *testing.T) string {
	t.Helper()
	captured := []string{
		`{"schema":"fak-git-daily/1","day":"2026-08-04","at":"2026-08-04T11:14:37-07:00","lease_locks_reaped":0,"index_locks_reaped":7,"lock_actions":1,"loose_before":4129,"loose_after":4129,"packs_before":4,"packs_after":4,"grace_prune_refused":"PRUNE_OFF"}`,
		`{"schema":"fak-git-daily/1","day":"2026-08-04","at":"2026-08-04T11:18:58-07:00","lease_locks_reaped":0,"index_locks_reaped":0,"lock_actions":0,"loose_before":4129,"loose_after":0,"packs_before":4,"packs_after":2,"grace_prune_refused":"PRUNE_OFF"}`,
		`{"schema":"fak-git-daily/1","day":"2026-08-04","at":"2026-08-04T11:59:46-07:00","lease_locks_reaped":0,"index_locks_reaped":0,"lock_actions":1,"loose_before":48,"loose_after":0,"packs_before":2,"packs_after":4,"grace_prune_refused":"PRUNE_OFF"}`,
		`{"schema":"fak-git-daily/1","day":"2026-08-04","at":"2026-08-04T12:15:16-07:00","lease_locks_reaped":0,"index_locks_reaped":0,"lock_actions":1,"loose_before":34,"loose_after":0,"packs_before":4,"packs_after":4,"grace_prune_refused":"PRUNE_OFF"}`,
		`{"schema":"fak-git-daily/1","day":"2026-08-05","at":"2026-08-05T04:39:17-07:00","lease_locks_reaped":0,"index_locks_reaped":1,"lock_actions":2,"loose_before":5827,"loose_after":0,"packs_before":4,"packs_after":4,"grace_prune_refused":"PRUNE_OFF"}`,
	}
	path := filepath.Join(t.TempDir(), gitdaily.LedgerName)
	if err := os.WriteFile(path, []byte(strings.Join(captured, "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// TestGitDailyScoreProjectionMatchesStatus is the #5587 witness at the CLI seam: the
// graded readout and the `--status` readout are folded from ONE ledger read, so the two
// operator surfaces cannot drift apart and quietly disagree about the same history.
//
// Everything asserted here is CLOCK-INDEPENDENT on purpose. The adoption axis measures
// the newest row against time.Now(), so a captured ledger goes stale as the calendar
// advances; asserting the letter grade or the exit code against a frozen capture would
// pass today and fail tomorrow for no code change. The counters, the folded volume and
// the streak are properties of the ROWS alone, so they are the honest thing to pin here —
// the grading rules themselves are pinned deterministically in internal/metrics.
func TestGitDailyScoreProjectionMatchesStatus(t *testing.T) {
	path := gitDailyCapturedLedger(t)
	rows := gitdaily.Status(path, 0)

	in := gitDailyHealthInput(rows, path)
	tally := gitdaily.FoldOutcomes(rows)

	if in.Runs != tally.Runs || in.OK != tally.OK || in.Refused != tally.Refused || in.Errors != tally.Errors {
		t.Errorf("graded input disagrees with the --status fold:\n  score  %d runs / %d ok / %d refused / %d error\n  status %d runs / %d ok / %d refused / %d error",
			in.Runs, in.OK, in.Refused, in.Errors, tally.Runs, tally.OK, tally.Refused, tally.Errors)
	}
	if in.LooseFolded != tally.LooseFolded {
		t.Errorf("graded loose volume = %d, --status says %d", in.LooseFolded, tally.LooseFolded)
	}
	if in.FirstDay != tally.FirstDay || in.LastDay != tally.LastDay {
		t.Errorf("graded window = %s..%s, --status says %s..%s", in.FirstDay, in.LastDay, tally.FirstDay, tally.LastDay)
	}
	if in.LedgerPath != path {
		t.Errorf("graded input names ledger %q, want the one it read: %q", in.LedgerPath, path)
	}
	// Today must be the LOCAL day key the ledger's own day keys are written in, or the
	// recency axis compares two different calendars and reads a day stale every evening.
	if _, err := time.Parse(gitdaily.DayLayout, in.Today); err != nil {
		t.Errorf("graded today %q does not parse as a %s day key: %v", in.Today, gitdaily.DayLayout, err)
	}

	// The clock-independent half of the captured fragment: five ok ticks that folded
	// 10038 loose objects, with no trailing non-ok streak.
	const want = "runs=5 ok=5 refused=0 error=0 folded=10038 streak=0"
	got := metrics.GitDailyHealthFragment(in, metrics.GradeGitDailyHealth(in))
	if !strings.Contains(got, want) {
		t.Fatalf("captured score fragment drifted\nwant substring: %s\n---got---\n%s", want, got)
	}
}

// TestGitDailyScoreOnAnEmptyLedgerIsNotHealthy pins the exit-code contract the cron/CI
// caller gates on, using the one case that is genuinely clock-independent: a ledger with
// no rows. "Never ran" must not read as "healthy" — an unwitnessed job is exactly the
// silent failure (#4602) this verb exists to surface, and a card that graded absence as
// OK would exit 0 forever on a job that stopped years ago.
func TestGitDailyScoreOnAnEmptyLedgerIsNotHealthy(t *testing.T) {
	path := filepath.Join(t.TempDir(), gitdaily.LedgerName)
	in := gitDailyHealthInput(gitdaily.Status(path, 0), path)
	payload := metrics.GradeGitDailyHealth(in)

	if payload.OK {
		t.Errorf("an empty ledger graded OK; --score would exit 0 on a job that never ran")
	}
	// The defect has to NAME the witness it looked at, so an operator knows which file to
	// go check rather than being told a bare grade.
	var named bool
	for _, k := range payload.KPIs {
		for _, d := range k.Defects {
			if strings.Contains(d, path) {
				named = true
			}
		}
	}
	if !named {
		t.Errorf("no defect names the ledger %q it was scored from: %+v", path, payload.KPIs)
	}
}

func TestGitDailyStatusSurfacesWeeklyAdoptionFold(t *testing.T) {
	rows := []gitdaily.Row{
		{At: "2026-08-03T12:00:00Z"},
		{At: "2026-08-04T12:00:00Z", GraceRefused: "LOCKED"},
		{At: "2026-08-10T12:00:00Z", Incident: true},
	}
	var out bytes.Buffer
	writeGitDailyStatus(&out, "/tmp/gitdaily.jsonl", rows)
	for _, want := range []string{
		"week 2026-08-03: 2 recorded runs, 1 ok, 1 refused, 0 error",
		"week 2026-08-10: 1 recorded runs, 0 ok, 0 refused, 1 error",
	} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("status missing %q\n---\n%s", want, out.String())
		}
	}

	report := gitDailyStatusReport{
		Schema:   "fak-git-daily-status/1",
		Rows:     rows,
		Outcomes: gitdaily.FoldOutcomes(rows),
		Weekly:   gitdaily.FoldOutcomesByWeek(rows),
	}
	b, err := json.Marshal(report)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), `"weekly":[{"week":"2026-08-03","total":2,"ok":1,"refused":1,"errors":0}`) {
		t.Fatalf("JSON weekly fold missing: %s", b)
	}
}

func TestRunGitDailyRejectsInvalidGoCacheFlags(t *testing.T) {
	tests := []struct {
		name string
		args []string
		want string
	}{
		{name: "negative high bytes", args: []string{"--go-cache-high-bytes=-1"}, want: "--go-cache-high-bytes must be >= 0"},
		{name: "negative low bytes", args: []string{"--go-cache-low-bytes=-1"}, want: "--go-cache-low-bytes must be >= 0"},
		{name: "negative min age", args: []string{"--go-cache-min-age=-1s"}, want: "--go-cache-min-age must be >= 0"},
		{name: "negative min free bytes", args: []string{"--go-cache-min-free-bytes=-1"}, want: "--go-cache-min-free-bytes must be >= 0"},
		{name: "negative max entries", args: []string{"--go-cache-max-entries=-1"}, want: "--go-cache-max-entries must be >= 0"},
		{name: "negative deadline", args: []string{"--go-cache-deadline=-1s"}, want: "--go-cache-deadline must be >= 0"},
		{name: "low above high", args: []string{"--go-cache-high-bytes=10", "--go-cache-low-bytes=11"}, want: "--go-cache-low-bytes must be <= --go-cache-high-bytes"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			code := runGitDaily(&stdout, &stderr, tt.args)
			if code != 2 {
				t.Fatalf("exit = %d, want 2; stderr = %q", code, stderr.String())
			}
			if stdout.Len() != 0 {
				t.Fatalf("stdout = %q, want empty", stdout.String())
			}
			if !strings.Contains(stderr.String(), tt.want) {
				t.Fatalf("stderr = %q, want substring %q", stderr.String(), tt.want)
			}
		})
	}
}

func TestRunGitDailyGoCacheFlagsPropagateAndDisable(t *testing.T) {
	repoRoot := t.TempDir()
	initGitDailyTestRepo(t, repoRoot)
	t.Setenv(treedoctor.GoTmpDirEnv, filepath.Join(t.TempDir(), "gotmp"))

	oldRun := gitDailyRun
	defer func() { gitDailyRun = oldRun }()

	t.Run("propagates values", func(t *testing.T) {
		gitDailyRun = func(_ context.Context, _ gitdaily.Runner, opts gitdaily.Options) gitdaily.Result {
			if opts.GoCacheDir != treedoctor.GoCacheRootFromEnv(os.Getenv, os.UserCacheDir) {
				t.Fatalf("GoCacheDir = %q", opts.GoCacheDir)
			}
			if opts.GoCacheOptions.ActiveBuild == nil {
				t.Fatal("ActiveBuild is nil")
			}
			if opts.GoCacheOptions.HighBytes != 100 {
				t.Fatalf("HighBytes = %d, want 100", opts.GoCacheOptions.HighBytes)
			}
			if opts.GoCacheOptions.LowBytes != 50 {
				t.Fatalf("LowBytes = %d, want 50", opts.GoCacheOptions.LowBytes)
			}
			if opts.GoCacheOptions.MinAge != 2*time.Minute {
				t.Fatalf("MinAge = %v, want 2m", opts.GoCacheOptions.MinAge)
			}
			if opts.GoCacheOptions.MinFreeBytes != 200 {
				t.Fatalf("MinFreeBytes = %d, want 200", opts.GoCacheOptions.MinFreeBytes)
			}
			if opts.GoCacheOptions.MaxWalkEntries != 7 {
				t.Fatalf("MaxWalkEntries = %d, want 7", opts.GoCacheOptions.MaxWalkEntries)
			}
			if opts.GoCacheOptions.Deadline != 3*time.Second {
				t.Fatalf("Deadline = %v, want 3s", opts.GoCacheOptions.Deadline)
			}
			return gitdaily.Result{Apply: true, Day: "2026-09-01"}
		}
		var stdout, stderr bytes.Buffer
		code := runGitDaily(&stdout, &stderr, []string{"--root", repoRoot, "--go-cache-high-bytes=100", "--go-cache-low-bytes=50", "--go-cache-min-age=2m", "--go-cache-min-free-bytes=200", "--go-cache-max-entries=7", "--go-cache-deadline=3s"})
		if code != 0 {
			t.Fatalf("exit = %d, stderr = %s", code, stderr.String())
		}
	})

	t.Run("disable skips only go cache", func(t *testing.T) {
		gitDailyRun = func(_ context.Context, _ gitdaily.Runner, opts gitdaily.Options) gitdaily.Result {
			if opts.GoCacheDir != "" {
				t.Fatalf("GoCacheDir = %q, want empty", opts.GoCacheDir)
			}
			if opts.GoCacheOptions.ActiveBuild == nil {
				t.Fatal("ActiveBuild is nil")
			}
			if opts.GoTmpDir != os.Getenv(treedoctor.GoTmpDirEnv) {
				t.Fatalf("GoTmpDir = %q, want %q", opts.GoTmpDir, os.Getenv(treedoctor.GoTmpDirEnv))
			}
			if !opts.PruneWorktrees {
				t.Fatal("PruneWorktrees = false, want true")
			}
			return gitdaily.Result{Apply: true, Day: "2026-09-01"}
		}
		var stdout, stderr bytes.Buffer
		code := runGitDaily(&stdout, &stderr, []string{"--root", repoRoot, "--prune-worktrees", "--go-cache=false"})
		if code != 0 {
			t.Fatalf("exit = %d, stderr = %s", code, stderr.String())
		}
	})
}

func initGitDailyTestRepo(t *testing.T, root string) {
	t.Helper()
	cmd := exec.Command("git", "init", root)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git init %s: %v\n%s", root, err, out)
	}
}

func TestWriteGitDailyTextIncludesGoCacheReceiptAndCleanupHints(t *testing.T) {
	var out bytes.Buffer
	writeGitDailyText(&out, gitdaily.Result{GoCache: treedoctor.GoCacheReport{Root: filepath.Join("tmp", "go-build"), BytesBefore: 10, BytesAfter: 4, BytesAfterSemantics: "projected", ScanComplete: true, CleanupHints: []string{"use tree-doctor GOTMP reaper", "use git-daily --prune-worktrees"}}})
	text := out.String()
	for _, want := range []string{"Go build cache: 10 -> 4 bytes (projected)", "tree-doctor GOTMP", "--prune-worktrees"} {
		if !strings.Contains(text, want) {
			t.Fatalf("missing %q in %s", want, text)
		}
	}
}
