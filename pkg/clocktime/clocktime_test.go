package clocktime

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"
)

// TestMain installs a synthetic IANA-style zone ("Syn/Dst": EST -5h until
// 2026-03-08 07:00 UTC, then EDT -4h) via $ZONEINFO so DST tests never depend
// on host tzdata availability.
func TestMain(m *testing.M) {
	dir, err := os.MkdirTemp("", "clocktime-zoneinfo-*")
	if err != nil {
		os.Exit(1)
	}
	defer os.RemoveAll(dir)
	syn := filepath.Join(dir, "Syn")
	if err := os.MkdirAll(syn, 0o755); err != nil {
		panic(err)
	}
	if err := os.WriteFile(filepath.Join(syn, "Dst"), dstTzfile(), 0o644); err != nil {
		panic(err)
	}
	// time.LoadLocation consults $ZONEINFO before platform sources; set it
	// before any test so "Syn/Dst" resolves to the synthetic zone on any host.
	os.Setenv("ZONEINFO", dir)
	code := m.Run()
	os.Unsetenv("ZONEINFO")
	os.Exit(code)
}

// dstTzfile builds a minimal TZif v0 byte stream: EST (-5h) until
// 2026-03-08 07:00 UTC, then EDT (-4h) onward. Two transitions are needed
// (epoch -> EST, T -> EDT) because Go applies tx[i].index over
// [tx[i].when, tx[i+1].when) and holds the last index past the final
// transition.
func dstTzfile() []byte {
	T := time.Date(2026, 3, 8, 7, 0, 0, 0, time.UTC).Unix()
	var b []byte
	u32 := func(v uint32) { b = append(b, byte(v>>24), byte(v>>16), byte(v>>8), byte(v)) }
	i32 := func(v int32) {
		u := uint32(v)
		b = append(b, byte(u>>24), byte(u>>16), byte(u>>8), byte(u))
	}
	b = append(b, 'T', 'Z', 'i', 'f', 0) // version byte must be NUL, '2', or '3'
	b = append(b, make([]byte, 15)...)   // reserved padding
	u32(0)                               // ttisgmtcnt
	u32(0)                               // ttisstdcnt
	u32(0)                               // leapcnt
	u32(2)                               // timecnt
	u32(2)                               // typecnt
	u32(8)                               // charcnt
	i32(0)                               // transition 0: epoch -> EST
	i32(int32(T))                        // transition 1: T -> EDT
	b = append(b, 0, 1)                  // type indices
	i32(-5 * 3600)                       // type 0: EST, std
	b = append(b, 0, 3)
	i32(-4 * 3600) // type 1: EDT, dst
	b = append(b, 1, 3)
	b = append(b, 'E', 'S', 'T', 0, 'E', 'D', 'T', 0) // abbreviations, NUL-terminated
	return b
}

func loadSynLoc(t *testing.T) *time.Location {
	t.Helper()
	loc, err := time.LoadLocation("Syn/Dst")
	if err != nil {
		t.Fatalf("synthetic zone Syn/Dst unavailable: %v", err)
	}
	return loc
}

// utcSchedule returns the default windows evaluated in UTC so tests are
// independent of the host's local zone.
func utcSchedule() Schedule {
	s := DefaultSchedule()
	s.Location = "UTC"
	return s
}

// fak-test:runtime fast est=50ms lane=default
func TestClassifyDefaultWindows(t *testing.T) {
	s := utcSchedule()
	// Sanity-anchor the calendar: these dates carry the weekdays the table
	// assumes.
	for _, d := range []struct {
		date time.Time
		want time.Weekday
	}{
		{time.Date(2026, 11, 9, 0, 0, 0, 0, time.UTC), time.Monday},
		{time.Date(2026, 11, 14, 0, 0, 0, 0, time.UTC), time.Saturday},
		{time.Date(2026, 11, 15, 0, 0, 0, 0, time.UTC), time.Sunday},
	} {
		if d.date.Weekday() != d.want {
			t.Fatalf("test calendar drift: %v is %v, want %v", d.date, d.date.Weekday(), d.want)
		}
	}
	cases := []struct {
		at   time.Time
		kind WindowKind
		win  string
	}{
		{time.Date(2026, 11, 9, 3, 0, 0, 0, time.UTC), WindowQuiet, "quiet-weeknight"},               // wrap tail
		{time.Date(2026, 11, 9, 5, 59, 0, 0, time.UTC), WindowQuiet, "quiet-weeknight"},              // wrap tail edge
		{time.Date(2026, 11, 9, 6, 0, 0, 0, time.UTC), WindowBusinessOffPeak, "business-offpeak-am"}, // boundary in
		{time.Date(2026, 11, 9, 7, 30, 0, 0, time.UTC), WindowBusinessOffPeak, "business-offpeak-am"},
		{time.Date(2026, 11, 9, 8, 59, 0, 0, time.UTC), WindowBusinessOffPeak, "business-offpeak-am"},
		{time.Date(2026, 11, 9, 9, 0, 0, 0, time.UTC), WindowBusinessPeak, "business-peak"}, // boundary in
		{time.Date(2026, 11, 9, 12, 0, 0, 0, time.UTC), WindowBusinessPeak, "business-peak"},
		{time.Date(2026, 11, 9, 16, 59, 0, 0, time.UTC), WindowBusinessPeak, "business-peak"},
		{time.Date(2026, 11, 9, 17, 0, 0, 0, time.UTC), WindowBusinessOffPeak, "business-offpeak-pm"}, // boundary out/in
		{time.Date(2026, 11, 9, 21, 59, 0, 0, time.UTC), WindowBusinessOffPeak, "business-offpeak-pm"},
		{time.Date(2026, 11, 9, 22, 0, 0, 0, time.UTC), WindowQuiet, "quiet-weeknight"}, // boundary in
		{time.Date(2026, 11, 9, 23, 59, 0, 0, time.UTC), WindowQuiet, "quiet-weeknight"},
		{time.Date(2026, 11, 14, 0, 0, 0, 0, time.UTC), WindowQuiet, "quiet-weekend"},
		{time.Date(2026, 11, 14, 12, 0, 0, 0, time.UTC), WindowQuiet, "quiet-weekend"},
		{time.Date(2026, 11, 15, 23, 59, 0, 0, time.UTC), WindowQuiet, "quiet-weekend"},
	}
	for _, tc := range cases {
		c := Classify(s, tc.at)
		if c.Kind != tc.kind || c.Window != tc.win {
			t.Errorf("Classify(%v) = %q/%q, want %q/%q", tc.at, c.Kind, c.Window, tc.kind, tc.win)
		}
		if c.Location != "UTC" || c.Basis != "windows:5@UTC" {
			t.Errorf("Classify(%v) location/basis = %q/%q, want %q/%q", tc.at, c.Location, c.Basis, "UTC", "windows:5@UTC")
		}
		if !c.At.Equal(tc.at) {
			t.Errorf("Classify(%v).At = %v, want same instant", tc.at, c.At)
		}
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestDefaultScheduleCoversEveryHour(t *testing.T) {
	s := utcSchedule()
	start := time.Date(2026, 11, 9, 0, 0, 0, 0, time.UTC) // Monday
	for i := 0; i < 7*24; i++ {
		at := start.Add(time.Duration(i) * time.Hour)
		c := Classify(s, at)
		if c.Kind == WindowUnknown {
			t.Fatalf("default schedule leaves %v (%v) unclassified", at, at.Weekday())
		}
		if at.Weekday() == time.Saturday || at.Weekday() == time.Sunday {
			if c.Kind != WindowQuiet || c.Window != "quiet-weekend" {
				t.Errorf("weekend %v = %q/%q, want QUIET/quiet-weekend", at, c.Kind, c.Window)
			}
		} else if c.Kind == WindowQuiet && c.Window != "quiet-weeknight" {
			t.Errorf("weekday quiet %v named %q, want quiet-weeknight", at, c.Window)
		}
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestWrapPastMidnight(t *testing.T) {
	s := Schedule{
		Location: "UTC",
		Windows:  []Window{{Name: "night", Kind: WindowMaintenance, Start: "22:00", End: "06:00"}},
	}
	cases := []struct {
		at   time.Time
		kind WindowKind
	}{
		{time.Date(2026, 11, 9, 23, 0, 0, 0, time.UTC), WindowMaintenance},
		{time.Date(2026, 11, 9, 5, 59, 0, 0, time.UTC), WindowMaintenance},
		{time.Date(2026, 11, 9, 6, 0, 0, 0, time.UTC), WindowUnknown}, // half-open end
		{time.Date(2026, 11, 9, 21, 59, 0, 0, time.UTC), WindowUnknown},
		{time.Date(2026, 11, 9, 22, 0, 0, 0, time.UTC), WindowMaintenance},
	}
	for _, tc := range cases {
		if c := Classify(s, tc.at); c.Kind != tc.kind {
			t.Errorf("wrap window at %v = %q, want %q", tc.at, c.Kind, tc.kind)
		}
	}
	allDay := Schedule{
		Location: "UTC",
		Windows:  []Window{{Name: "allday", Kind: WindowQuiet, Start: "00:00", End: "00:00"}},
	}
	for _, h := range []int{0, 12, 23} {
		at := time.Date(2026, 11, 9, h, 59, 0, 0, time.UTC)
		if c := Classify(allDay, at); c.Kind != WindowQuiet || c.Window != "allday" {
			t.Errorf("all-day window at %v = %q/%q, want QUIET/allday", at, c.Kind, c.Window)
		}
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestDayFilteringAndEmptyDays(t *testing.T) {
	monOnly := Schedule{
		Location: "UTC",
		Windows:  []Window{{Name: "mon", Kind: WindowBusinessPeak, Days: []time.Weekday{time.Monday}, Start: "09:00", End: "17:00"}},
	}
	noon := func(day int) time.Time { return time.Date(2026, 11, day, 12, 0, 0, 0, time.UTC) }
	if c := Classify(monOnly, noon(9)); c.Kind != WindowBusinessPeak { // Monday
		t.Errorf("Monday noon = %q, want BUSINESS_PEAK", c.Kind)
	}
	if c := Classify(monOnly, noon(10)); c.Kind != WindowUnknown { // Tuesday
		t.Errorf("Tuesday noon = %q, want UNKNOWN", c.Kind)
	}
	if c := Classify(monOnly, noon(15)); c.Kind != WindowUnknown { // Sunday
		t.Errorf("Sunday noon = %q, want UNKNOWN", c.Kind)
	}
	daily := Schedule{
		Location: "UTC",
		Windows:  []Window{{Name: "daily", Kind: WindowQuiet, Start: "00:00", End: "00:00"}},
	}
	if c := Classify(daily, noon(15)); c.Kind != WindowQuiet { // empty Days = every day
		t.Errorf("empty-Days Sunday noon = %q, want QUIET", c.Kind)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestPrecedenceFirstMatchWins(t *testing.T) {
	s := Schedule{
		Location: "UTC",
		Windows: []Window{
			{Name: "wide", Kind: WindowQuiet, Start: "00:00", End: "23:59"},
			{Name: "narrow", Kind: WindowMaintenance, Start: "12:00", End: "13:00"},
		},
	}
	// Both windows contain 12:30; the earlier slice entry must win.
	if c := Classify(s, time.Date(2026, 11, 9, 12, 30, 0, 0, time.UTC)); c.Kind != WindowQuiet || c.Window != "wide" {
		t.Errorf("12:30 = %q/%q, want QUIET/wide (first match)", c.Kind, c.Window)
	}
	if c := Classify(s, time.Date(2026, 11, 9, 13, 30, 0, 0, time.UTC)); c.Kind != WindowQuiet {
		t.Errorf("13:30 = %q, want QUIET", c.Kind)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestZeroValueFailClosed(t *testing.T) {
	at := time.Date(2026, 11, 9, 12, 0, 0, 0, time.UTC)
	for _, s := range []Schedule{
		{},
		{Location: "UTC"},
		{Location: "Local"},
	} {
		c := Classify(s, at)
		if c.Kind != WindowUnknown || c.Window != "" {
			t.Errorf("windowless schedule %v = %q/%q, want UNKNOWN/\"\"", s, c.Kind, c.Window)
		}
		if c.UntilNext != -1 || !c.NextAt.IsZero() || c.NextKind != WindowUnknown {
			t.Errorf("windowless schedule next = %v/%v/%v, want -1/zero/UNKNOWN", c.NextKind, c.NextAt, c.UntilNext)
		}
		if c.Basis != "none" {
			t.Errorf("windowless schedule basis = %q, want \"none\"", c.Basis)
		}
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestUnloadableLocationFailClosed(t *testing.T) {
	s := Schedule{
		Location: "Not/ARealZone",
		Windows:  []Window{{Name: "peak", Kind: WindowBusinessPeak, Start: "09:00", End: "17:00"}},
	}
	c := Classify(s, time.Date(2026, 11, 9, 12, 0, 0, 0, time.UTC))
	if c.Kind != WindowUnknown {
		t.Errorf("unloadable location = %q, want UNKNOWN", c.Kind)
	}
	if c.Location != "Not/ARealZone" || c.Basis != "location:Not/ARealZone" {
		t.Errorf("location/basis = %q/%q, want requested name echoed", c.Location, c.Basis)
	}
	if c.UntilNext != -1 || !c.NextAt.IsZero() {
		t.Errorf("next = %v/%v, want -1/zero", c.NextAt, c.UntilNext)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestLoadScheduleRoundTrip(t *testing.T) {
	want := Schedule{
		Location: "UTC",
		Windows: []Window{
			{Name: "maint", Kind: WindowMaintenance, Days: []time.Weekday{time.Saturday}, Start: "02:00", End: "04:00"},
			{Name: "night", Kind: WindowQuiet, Start: "22:00", End: "06:00"},
		},
	}
	data, err := json.Marshal(want)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "sched.json")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := LoadSchedule(path)
	if err != nil {
		t.Fatalf("LoadSchedule(%q): %v", path, err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("round trip = %+v, want %+v", got, want)
	}
	// Behavior of the loaded schedule: Saturday 03:00 hits both windows;
	// slice order puts "maint" first.
	sat := time.Date(2026, 11, 14, 3, 0, 0, 0, time.UTC)
	if c := Classify(got, sat); c.Kind != WindowMaintenance || c.Window != "maint" {
		t.Errorf("loaded schedule Sat 03:00 = %q/%q, want MAINTENANCE/maint", c.Kind, c.Window)
	}
	if c := Classify(got, time.Date(2026, 11, 14, 5, 0, 0, 0, time.UTC)); c.Kind != WindowQuiet {
		t.Errorf("loaded schedule Sat 05:00 = %q, want QUIET", c.Kind)
	}
	if c := Classify(got, time.Date(2026, 11, 9, 3, 0, 0, 0, time.UTC)); c.Kind != WindowQuiet {
		t.Errorf("loaded schedule Mon 03:00 = %q, want QUIET", c.Kind)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestLoadScheduleErrors(t *testing.T) {
	def := DefaultSchedule()
	got, err := LoadSchedule("")
	if err != nil {
		t.Fatalf("empty path: %v", err)
	}
	if !reflect.DeepEqual(got, def) {
		t.Errorf("empty path = %+v, want default schedule", got)
	}
	write := func(name, content string) string {
		p := filepath.Join(t.TempDir(), name)
		if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	if _, err := LoadSchedule(write("bad.json", "{ not json")); err == nil {
		t.Error("malformed JSON: want error, got nil")
	}
	if _, err := LoadSchedule(write("loc.json", `{"location":"Not/ARealZone","windows":[{"name":"x","kind":"QUIET","start":"00:00","end":"01:00"}]}`)); !errors.Is(err, ErrInvalidLocation) {
		t.Errorf("unloadable location: err = %v, want ErrInvalidLocation", err)
	}
	if _, err := LoadSchedule(write("start.json", `{"location":"UTC","windows":[{"name":"x","kind":"QUIET","start":"25:00","end":"01:00"}]}`)); !errors.Is(err, ErrInvalidWindow) {
		t.Errorf("bad start: err = %v, want ErrInvalidWindow", err)
	}
	if _, err := LoadSchedule(write("end.json", `{"location":"UTC","windows":[{"name":"x","kind":"QUIET","start":"00:00","end":"9:00"}]}`)); !errors.Is(err, ErrInvalidWindow) {
		t.Errorf("bad end: err = %v, want ErrInvalidWindow", err)
	}
	if _, err := LoadSchedule(write("day.json", `{"location":"UTC","windows":[{"name":"x","kind":"QUIET","days":[7],"start":"00:00","end":"01:00"}]}`)); !errors.Is(err, ErrInvalidWindow) {
		t.Errorf("bad weekday: err = %v, want ErrInvalidWindow", err)
	}
	if _, err := LoadSchedule(filepath.Join(t.TempDir(), "missing.json")); err == nil {
		t.Error("missing file: want error, got nil")
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestFromEnv(t *testing.T) {
	t.Setenv(EnvConfigPath, "")
	s, basis := FromEnv()
	if basis != "default" || !reflect.DeepEqual(s, DefaultSchedule()) {
		t.Errorf("unset env = basis %q, %+v; want default", basis, s)
	}

	want := Schedule{
		Location: "UTC",
		Windows:  []Window{{Name: "quiet", Kind: WindowQuiet, Start: "00:00", End: "00:00"}},
	}
	data, _ := json.Marshal(want)
	path := filepath.Join(t.TempDir(), "env-sched.json")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv(EnvConfigPath, path)
	s, basis = FromEnv()
	if basis != "config:"+path || !reflect.DeepEqual(s, want) {
		t.Errorf("configured env = basis %q, %+v; want config:%q", basis, s, path)
	}

	broken := filepath.Join(t.TempDir(), "missing.json")
	t.Setenv(EnvConfigPath, broken)
	s, basis = FromEnv()
	if basis != "config-error:"+broken {
		t.Errorf("broken env basis = %q, want config-error:%q", basis, broken)
	}
	if !reflect.DeepEqual(s, DefaultSchedule()) {
		t.Errorf("broken env schedule = %+v, want default fallback", s)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestUntilNextMonotonicAndMinusOne(t *testing.T) {
	s := utcSchedule()
	day := func(h, m int) time.Time { return time.Date(2026, 11, 9, h, m, 0, 0, time.UTC) }
	c1 := Classify(s, day(10, 0))  // PEAK until 17:00
	c2 := Classify(s, day(14, 0))  // same upcoming transition, closer
	c3 := Classify(s, day(17, 30)) // crossed into off-peak PM
	if !c1.NextAt.Equal(c2.NextAt) {
		t.Errorf("NextAt drifted inside one window: %v vs %v", c1.NextAt, c2.NextAt)
	}
	if !(c1.UntilNext > c2.UntilNext && c2.UntilNext > 0) {
		t.Errorf("UntilNext not monotonic: %v then %v", c1.UntilNext, c2.UntilNext)
	}
	if c1.UntilNext != 7*time.Hour || c2.UntilNext != 3*time.Hour {
		t.Errorf("UntilNext = %v / %v, want 7h / 3h", c1.UntilNext, c2.UntilNext)
	}
	if c1.NextKind != WindowBusinessOffPeak || !c1.NextAt.Equal(day(17, 0)) {
		t.Errorf("next from 10:00 = %v @%v, want BUSINESS_OFFPEAK @17:00", c1.NextKind, c1.NextAt)
	}
	if c3.Kind != WindowBusinessOffPeak || c3.UntilNext != 4*time.Hour+30*time.Minute || c3.NextKind != WindowQuiet {
		t.Errorf("from 17:30 = %v until %v next %v, want OFFPEAK/4h30m/QUIET", c3.Kind, c3.UntilNext, c3.NextKind)
	}
	if !c3.NextAt.After(c1.NextAt) {
		t.Errorf("post-crossing NextAt %v not after %v", c3.NextAt, c1.NextAt)
	}

	// -1 case: a single all-day window never changes kind.
	never := Schedule{
		Location: "UTC",
		Windows:  []Window{{Name: "always", Kind: WindowMaintenance, Start: "00:00", End: "00:00"}},
	}
	c := Classify(never, day(12, 0))
	if c.Kind != WindowMaintenance || c.UntilNext != -1 || !c.NextAt.IsZero() || c.NextKind != WindowMaintenance {
		t.Errorf("constant-kind schedule = %v until %v next %v@%v, want MAINTENANCE/-1/zero/MAINTENANCE", c.Kind, c.UntilNext, c.NextKind, c.NextAt)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestDSTSafetySynthetic(t *testing.T) {
	loc := loadSynLoc(t)
	s := defaultIn(loc) // default windows, evaluated in the synthetic zone

	// Fixture sanity: the zone really does shift by one hour across the
	// 2026-03-08 07:00 UTC transition.
	pre := time.Date(2026, 3, 2, 12, 0, 0, 0, loc)  // Monday, EST
	post := time.Date(2026, 3, 9, 12, 0, 0, 0, loc) // Monday, EDT
	if pre.UTC().Format("15:04") != "17:00" || post.UTC().Format("15:04") != "16:00" {
		t.Fatalf("fixture offsets wrong: pre=%s post=%s UTC", pre.UTC().Format("15:04"), post.UTC().Format("15:04"))
	}

	// Same wall-clock span, different absolute anchors: 17:00 local is
	// 22:00 UTC pre-transition but 21:00 UTC post-transition.
	cp := Classify(s, pre)
	cq := Classify(s, post)
	if cp.Kind != WindowBusinessPeak || cq.Kind != WindowBusinessPeak {
		t.Fatalf("peak hours misclassified: pre=%v post=%v", cp.Kind, cq.Kind)
	}
	if want := time.Date(2026, 3, 2, 22, 0, 0, 0, time.UTC); !cp.NextAt.Equal(want) {
		t.Errorf("pre-transition NextAt = %v, want %v (17:00 EST)", cp.NextAt, want)
	}
	if want := time.Date(2026, 3, 9, 21, 0, 0, 0, time.UTC); !cq.NextAt.Equal(want) {
		t.Errorf("post-transition NextAt = %v, want %v (17:00 EDT)", cq.NextAt, want)
	}
	if cp.UntilNext != 5*time.Hour || cq.UntilNext != 5*time.Hour {
		t.Errorf("wall-clock span changed: pre=%v post=%v, want 5h both", cp.UntilNext, cq.UntilNext)
	}

	// Representation invariance: the same instant carried in a foreign zone
	// classifies identically.
	foreign := post.In(time.FixedZone("Foreign", 3*3600))
	cf := Classify(s, foreign)
	if cf.Kind != cq.Kind || cf.Window != cq.Window || !cf.At.Equal(cq.At) {
		t.Errorf("foreign-zone input diverged: %v/%v vs %v/%v", cf.Kind, cf.At, cq.Kind, cq.At)
	}

	// Boundary straddle across the shifted offset: 09:00 EDT is 13:00 UTC.
	edge := time.Date(2026, 3, 9, 12, 0, 0, 0, time.UTC) // 08:00 EDT
	cEdge := Classify(s, edge)
	if cEdge.Kind != WindowBusinessOffPeak || cEdge.UntilNext != time.Hour {
		t.Errorf("08:00 EDT = %v until %v, want OFFPEAK/1h", cEdge.Kind, cEdge.UntilNext)
	}
	if want := time.Date(2026, 3, 9, 13, 0, 0, 0, time.UTC); !cEdge.NextAt.Equal(want) {
		t.Errorf("straddle NextAt = %v, want %v (09:00 EDT)", cEdge.NextAt, want)
	}
	if cIn := Classify(s, cEdge.NextAt); cIn.Kind != WindowBusinessPeak {
		t.Errorf("09:00 EDT = %v, want BUSINESS_PEAK", cIn.Kind)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestDayRolloverInScheduleLocation(t *testing.T) {
	loc := loadSynLoc(t) // EDT (-4h) in November
	s := defaultIn(loc)
	// Monday 01:30 UTC is Sunday 21:30 in the schedule's location: the
	// weekend window must apply, not the Monday night window.
	at := time.Date(2026, 11, 9, 1, 30, 0, 0, time.UTC)
	c := Classify(s, at)
	if c.Kind != WindowQuiet || c.Window != "quiet-weekend" {
		t.Errorf("Mon 01:30 UTC = %q/%q, want QUIET/quiet-weekend (Sun 21:30 local)", c.Kind, c.Window)
	}
}

// defaultIn returns the default windows with their Location pointed at the
// given concrete zone, for tests that need a named DST-capable location.
func defaultIn(loc *time.Location) Schedule {
	return Schedule{
		Location: loc.String(),
		Windows:  DefaultSchedule().Windows,
	}
}
