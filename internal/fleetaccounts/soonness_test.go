package fleetaccounts

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

// snNow anchors the undated time-only reset forms deterministically.
var snNow = time.Date(2026, time.June, 30, 10, 0, 0, 0, time.UTC)

// TestResetSoonnessOrdersSoonerHigher pins the core property the walled-tier tie-break relies
// on: a nearer future reset scores higher than a farther one, both strictly inside [0,1).
func TestResetSoonnessOrdersSoonerHigher(t *testing.T) {
	soon, ok1 := ResetSoonness("11am", snNow) // +1h
	late, ok2 := ResetSoonness("3pm", snNow)  // +5h
	if !ok1 || !ok2 {
		t.Fatalf("both future resets must parse: 11am ok=%v, 3pm ok=%v", ok1, ok2)
	}
	if !(soon > late) {
		t.Fatalf("nearer reset must score higher: 11am=%v, 3pm=%v", soon, late)
	}
	for _, tc := range []struct {
		name string
		v    float64
	}{{"11am", soon}, {"3pm", late}} {
		if tc.v < 0 || tc.v >= 1 {
			t.Fatalf("%s soonness = %v, want in [0,1)", tc.name, tc.v)
		}
	}
}

// TestResetSoonnessAtNowIsNearOne checks a reset essentially at now scores near the top of the
// band (about to free up).
func TestResetSoonnessAtNowIsNearOne(t *testing.T) {
	// 10am == now; the daily-reset slack rolls a just-passed time-only reset to tomorrow, so
	// use a couple minutes ahead to stay "today" and near now.
	v, ok := ResetSoonness("10:01am", snNow)
	if !ok {
		t.Fatal("a near-future reset must parse")
	}
	if v < 0.9 {
		t.Fatalf("a reset ~1min out should score near 1, got %v", v)
	}
}

// TestResetSoonnessUnparseableOrExpired checks the ok=false paths: empty, garbage, and an
// already-expired dated reset all report no soonness (the account is not waiting on it).
func TestResetSoonnessUnparseableOrExpired(t *testing.T) {
	for _, s := range []string{"", "whenever", "next tuesday"} {
		if v, ok := ResetSoonness(s, snNow); ok {
			t.Fatalf("ResetSoonness(%q) = (%v,true), want ok=false", s, v)
		}
	}
	// A dated reset comfortably in the past (last month) is expired -> no soonness.
	if v, ok := ResetSoonness("May 1, 3pm", snNow); ok {
		t.Fatalf("expired dated reset -> (%v,true), want ok=false", v)
	}
}

// TestResetSoonnessFarFutureIsZeroButOk checks a dated reset beyond the soonness horizon is
// still a valid future reset (ok=true) but scores 0 — future, yet no sooner than the horizon.
func TestResetSoonnessFarFutureIsZeroButOk(t *testing.T) {
	v, ok := ResetSoonness("Aug 1, 3pm", snNow) // ~32 days out, past the 24h horizon
	if !ok {
		t.Fatal("a far-future dated reset must still parse as future")
	}
	if v != 0 {
		t.Fatalf("far-future reset soonness = %v, want 0 (beyond horizon)", v)
	}
}

// TestResetParsesWeeklyAtAndWeekdayForms pins parity with fleet_accounts._reset_is_future on
// the DATED weekly reset shapes Claude actually emits: an "at" separator and an optional
// leading weekday token ("Mon Jun 25 at 1pm"). Before the weekday-strip + "at"-layout fix these
// parsed as nil (unknown), which throttleIsActive treats fail-closed as a still-active weekly
// cap — walling a healthy seat indefinitely and suppressing its fresh-OK probes. now is well
// before the reset so every form must read as a live future weekly.
func TestResetParsesWeeklyAtAndWeekdayForms(t *testing.T) {
	now := time.Date(2026, time.June, 20, 10, 0, 0, 0, time.UTC)
	future := []string{
		"Jun 25 at 1pm",
		"Jun 25 at 1:30pm",
		"Mon Jun 25 at 1pm",
		"Mon Jun 25 at 1:30pm",
		"Jun 25, 1pm",
		"Jun 25 at 1pm (America/Los_Angeles)",
	}
	for _, f := range future {
		if _, ok := resetTime(f, now); !ok {
			t.Errorf("resetTime(%q) not parsed; a real weekly form must parse", f)
		}
		if r := resetIsFuture(f, now); r == nil || !*r {
			t.Errorf("resetIsFuture(%q) = %v, want future — an unparsed weekly walls the seat fail-closed", f, r)
		}
	}
	// A narrative form we do not model stays unknown (nil), matching Python's None.
	if r := resetIsFuture("resets Monday", now); r != nil {
		t.Errorf("resetIsFuture(%q) = %v, want nil (unknown)", "resets Monday", r)
	}
}

// TestResetIsFutureUnchanged guards the refactor: resetIsFuture now reads the shared resetTime
// core, and must still report future/expired/unknown exactly as before.
func TestResetIsFutureUnchanged(t *testing.T) {
	if r := resetIsFuture("3pm", snNow); r == nil || !*r {
		t.Fatalf("3pm (5h ahead) should be future, got %v", r)
	}
	if r := resetIsFuture("", snNow); r != nil {
		t.Fatalf("empty reset should be unknown (nil), got %v", r)
	}
	if r := resetIsFuture("May 1, 3pm", snNow); r == nil || *r {
		t.Fatalf("May 1 (expired dated) should be past (false), got %v", r)
	}
}

func TestResetTimeResolvesWeekdayNamedWeeklyResets(t *testing.T) {
	la, err := time.LoadLocation("America/Los_Angeles")
	if err != nil {
		t.Fatal(err)
	}
	// Thursday after 09:00: both full and abbreviated Monday forms must name
	// the following Monday, not today's time-only occurrence.
	now := time.Date(2026, 6, 4, 12, 0, 0, 0, la)
	wantMonday := time.Date(2026, 6, 8, 9, 0, 0, 0, la)
	for _, raw := range []string{"Monday at 9am", "mon 9am"} {
		t.Run(raw, func(t *testing.T) {
			got, ok := resetTime(raw, now)
			if !ok || !got.Equal(wantMonday) {
				t.Fatalf("resetTime(%q) = %v, %v; want %v, true", raw, got, ok, wantMonday)
			}
		})
	}

	// Same weekday at/before now rolls a full week rather than reopening early.
	mondayAtNine := time.Date(2026, 6, 8, 9, 0, 0, 0, la)
	got, ok := resetTime("Monday at 9am", mondayAtNine)
	wantNextWeek := time.Date(2026, 6, 15, 9, 0, 0, 0, la)
	if !ok || !got.Equal(wantNextWeek) {
		t.Fatalf("same-instant Monday = %v, %v; want %v, true", got, ok, wantNextWeek)
	}

	// An explicit date remains authoritative even if its weekday token is wrong.
	nowBeforeDate := time.Date(2026, 5, 28, 12, 0, 0, 0, la)
	got, ok = resetTime("Mon Jun 3 at 9am", nowBeforeDate)
	wantDate := time.Date(2026, 6, 3, 9, 0, 0, 0, la)
	if !ok || !got.Equal(wantDate) {
		t.Fatalf("explicit date = %v, %v; want %v, true", got, ok, wantDate)
	}
}

func TestWeekdayNamedWeeklyThrottleTracksItsResetBoundary(t *testing.T) {
	la, err := time.LoadLocation("America/Los_Angeles")
	if err != nil {
		t.Fatal(err)
	}
	before := time.Date(2026, 6, 4, 12, 0, 0, 0, la)
	reset, ok := resetTime("Monday at 9am", before)
	if !ok {
		t.Fatal("weekday reset did not parse")
	}
	thr := map[string]any{"reset": "Monday at 9am", "weekly": "Monday at 9am"}
	if got := DisambiguateCap(thr, CapObservation{}, before, DefaultCapPolicy()); !got.Active || !got.WeeklyActive {
		t.Fatalf("before reset: active=%v weekly_active=%v; want true/true", got.Active, got.WeeklyActive)
	}
	// Persist the concrete reset instant, as the runtime does when it carries a
	// parsed cap forward. It must release immediately after that boundary rather
	// than reinterpret the recurring weekday as another future week.
	thr["reset"] = reset.Format("Jan 2, 3:04pm")
	thr["weekly"] = reset.Format("Jan 2, 3:04pm")
	after := reset.Add(time.Minute)
	if got := DisambiguateCap(thr, CapObservation{}, after, DefaultCapPolicy()); got.Active || got.WeeklyActive {
		t.Fatalf("after reset: active=%v weekly_active=%v; want false/false", got.Active, got.WeeklyActive)
	}
}

func TestWeekdayResetFixtureMatchesPythonParser(t *testing.T) {
	type fixture struct {
		Reset   string `json:"reset"`
		NowUTC  string `json:"now_utc"`
		WantUTC string `json:"want_utc"`
	}
	fixturePath := filepath.Join("testdata", "weekday_reset_parity.json")
	data, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatal(err)
	}
	var fixtures []fixture
	if err := json.Unmarshal(data, &fixtures); err != nil {
		t.Fatal(err)
	}
	if len(fixtures) < 6 {
		t.Fatalf("weekday fixture count = %d; want at least 6", len(fixtures))
	}
	for _, fx := range fixtures {
		now, err := time.Parse(time.RFC3339, fx.NowUTC)
		if err != nil {
			t.Fatalf("now %q: %v", fx.NowUTC, err)
		}
		want, err := time.Parse(time.RFC3339, fx.WantUTC)
		if err != nil {
			t.Fatalf("want %q: %v", fx.WantUTC, err)
		}
		got, ok := resetTime(fx.Reset, now)
		if !ok || !got.Equal(want) {
			t.Errorf("Go resetTime(%q) = %v, %v; want %v, true", fx.Reset, got, ok, want)
		}
	}

	python, prefix := fleetaccountsPython(t)
	// runtime.Caller paths are module-relative under -trimpath (the isolated
	// buildcheck/validate compile), so the caller-derived root is only a first
	// candidate; walking up from the test's working directory (go test runs in
	// the package dir) finds the real checkout in both compile modes.
	repoRoot := ""
	if _, sourceFile, _, ok := runtime.Caller(0); ok {
		candidate := filepath.Clean(filepath.Join(filepath.Dir(sourceFile), "..", ".."))
		if _, err := os.Stat(filepath.Join(candidate, "tools", "issue_resolve_dispatch.py")); err == nil {
			repoRoot = candidate
		}
	}
	if repoRoot == "" {
		dir, err := os.Getwd()
		if err != nil {
			t.Fatalf("resolve working directory: %v", err)
		}
		for i := 0; i < 8 && repoRoot == ""; i++ {
			if _, err := os.Stat(filepath.Join(dir, "tools", "issue_resolve_dispatch.py")); err == nil {
				repoRoot = dir
				break
			}
			parent := filepath.Dir(dir)
			if parent == dir {
				break
			}
			dir = parent
		}
	}
	if repoRoot == "" {
		t.Fatal("locate tools/issue_resolve_dispatch.py above the package directory")
	}
	script := `
import datetime as dt, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(sys.argv[1]) / "tools"))
import issue_resolve_dispatch as parser
fixtures = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8-sig"))
for fixture in fixtures:
    now = dt.datetime.fromisoformat(fixture["now_utc"].replace("Z", "+00:00")).replace(tzinfo=None)
    got = parser._parse_reset_to_utc(fixture["reset"], now)
    want = dt.datetime.fromisoformat(fixture["want_utc"].replace("Z", "+00:00")).replace(tzinfo=None)
    if got != want:
        raise SystemExit(f'{fixture["reset"]!r}: Python got {got!r}, want {want!r}')
`
	args := append(append([]string{}, prefix...), "-c", script, repoRoot, filepath.Join(repoRoot, "internal", "fleetaccounts", fixturePath))
	cmd := exec.Command(python, args...)
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("Python reset parser disagrees with shared fixture: %v\n%s", err, output)
	}
}

func fleetaccountsPython(t *testing.T) (string, []string) {
	t.Helper()
	for _, candidate := range []struct {
		name   string
		prefix []string
	}{{"python3", nil}, {"python", nil}, {"py", []string{"-3"}}} {
		if path, err := exec.LookPath(candidate.name); err == nil {
			return path, candidate.prefix
		}
	}
	t.Skip("no Python interpreter on PATH; cannot witness Go/Python reset parity")
	return "", nil
}
