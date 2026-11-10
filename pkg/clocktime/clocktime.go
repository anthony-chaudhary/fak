package clocktime

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sort"
	"time"
)

// WindowKind names a calendar operating regime.
type WindowKind string

const (
	WindowBusinessPeak    WindowKind = "BUSINESS_PEAK"
	WindowBusinessOffPeak WindowKind = "BUSINESS_OFFPEAK"
	WindowQuiet           WindowKind = "QUIET"
	WindowMaintenance     WindowKind = "MAINTENANCE"
	WindowUnknown         WindowKind = "UNKNOWN"
)

// Window is one named span of wall-clock time in the schedule's location.
// Start and End are "HH:MM"; End <= Start wraps past midnight, so 00:00-
// 00:00 is the all-day window. Days empty means every day.
type Window struct {
	Name  string
	Kind  WindowKind
	Days  []time.Weekday
	Start string
	End   string
}

// Schedule is a set of windows evaluated in one location. Location "" or
// "Local" means system local; anything else must load via time.LoadLocation.
type Schedule struct {
	Location string
	Windows  []Window
}

// Classification is the verdict for one instant plus the next transition out
// of the current kind. UntilNext is -1 when no next transition is computable
// (then NextAt is the zero Time and NextKind equals Kind). Basis names the
// schedule source; see the package doc for the closed grammar.
type Classification struct {
	Kind      WindowKind
	Window    string
	At        time.Time
	Location  string
	NextKind  WindowKind
	NextAt    time.Time
	UntilNext time.Duration
	Basis     string
}

// Sentinel errors for config validation (LoadSchedule / Validate).
var (
	ErrInvalidLocation = errors.New("clocktime: invalid location")
	ErrInvalidWindow   = errors.New("clocktime: invalid window")
)

// EnvConfigPath is the environment variable holding a schedule JSON path.
const EnvConfigPath = "FAK_CLOCKTIME_CONFIG"

// DefaultSchedule returns the built-in schedule: QUIET weekdays 22:00-06:00
// plus all-day Sat/Sun; BUSINESS_PEAK weekdays 09:00-17:00; BUSINESS_OFFPEAK
// weekdays 06:00-09:00 and 17:00-22:00; no MAINTENANCE windows. Slice order
// is the precedence order; together these windows cover every instant of the
// week, so the default schedule never classifies UNKNOWN.
func DefaultSchedule() Schedule {
	return Schedule{
		Location: "",
		Windows: []Window{
			{Name: "quiet-weeknight", Kind: WindowQuiet, Days: weekdays(), Start: "22:00", End: "06:00"},
			{Name: "quiet-weekend", Kind: WindowQuiet, Days: weekend(), Start: "00:00", End: "00:00"},
			{Name: "business-peak", Kind: WindowBusinessPeak, Days: weekdays(), Start: "09:00", End: "17:00"},
			{Name: "business-offpeak-am", Kind: WindowBusinessOffPeak, Days: weekdays(), Start: "06:00", End: "09:00"},
			{Name: "business-offpeak-pm", Kind: WindowBusinessOffPeak, Days: weekdays(), Start: "17:00", End: "22:00"},
		},
	}
}

func weekdays() []time.Weekday {
	return []time.Weekday{time.Monday, time.Tuesday, time.Wednesday, time.Thursday, time.Friday}
}

func weekend() []time.Weekday {
	return []time.Weekday{time.Saturday, time.Sunday}
}

// Validate fail-closed checks a schedule: the location must load and every
// window must carry parseable "HH:MM" bounds and in-range weekdays. It is the
// strict gate LoadSchedule applies; Classify instead degrades per-window so a
// hand-built malformed schedule can never invent coverage.
func (s Schedule) Validate() error {
	if _, _, ok := resolveLocation(s.Location); !ok {
		return fmt.Errorf("%w: %q", ErrInvalidLocation, s.Location)
	}
	for i, w := range s.Windows {
		if _, ok := parseHM(w.Start); !ok {
			return fmt.Errorf("%w: index %d start %q", ErrInvalidWindow, i, w.Start)
		}
		if _, ok := parseHM(w.End); !ok {
			return fmt.Errorf("%w: index %d end %q", ErrInvalidWindow, i, w.End)
		}
		for _, d := range w.Days {
			if d < time.Sunday || d > time.Saturday {
				return fmt.Errorf("%w: index %d day %d", ErrInvalidWindow, i, int(d))
			}
		}
	}
	return nil
}

// Classify evaluates t against s in s's location and reports the matching
// window (first match in slice order wins), plus the next instant at which
// the kind changes. A windowless schedule or an unloadable location yields
// UNKNOWN with UntilNext -1 (fail-closed); see the package doc for the Basis
// grammar and honesty rules.
func Classify(s Schedule, t time.Time) Classification {
	display := s.Location
	if display == "" {
		display = "Local"
	}
	if len(s.Windows) == 0 {
		return Classification{
			Kind:      WindowUnknown,
			At:        t,
			Location:  display,
			NextKind:  WindowUnknown,
			UntilNext: -1,
			Basis:     "none",
		}
	}
	loc, display, ok := resolveLocation(s.Location)
	if !ok {
		return Classification{
			Kind:      WindowUnknown,
			At:        t,
			Location:  display,
			NextKind:  WindowUnknown,
			UntilNext: -1,
			Basis:     "location:" + display,
		}
	}
	tl := t.In(loc)
	pws := parseWindows(s.Windows)
	kind, name := matchAt(pws, tl)
	c := Classification{
		Kind:      kind,
		Window:    name,
		At:        tl,
		Location:  display,
		NextKind:  kind,
		UntilNext: -1,
		Basis:     fmt.Sprintf("windows:%d@%s", len(s.Windows), display),
	}
	if next, nk, found := nextTransition(pws, loc, tl, kind); found {
		c.NextAt = next
		c.NextKind = nk
		c.UntilNext = next.Sub(t)
	}
	return c
}

// LoadSchedule reads a schedule from a JSON file. An empty path yields the
// default schedule; malformed JSON, an unreadable file, or a failing
// Validate (unloadable location, bad window bounds or weekdays) is an error.
func LoadSchedule(path string) (Schedule, error) {
	if path == "" {
		return DefaultSchedule(), nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return Schedule{}, err
	}
	var s Schedule
	if err := json.Unmarshal(data, &s); err != nil {
		return Schedule{}, fmt.Errorf("clocktime: decode %s: %w", path, err)
	}
	if err := s.Validate(); err != nil {
		return Schedule{}, err
	}
	return s, nil
}

// FromEnv resolves the schedule from the FAK_CLOCKTIME_CONFIG environment
// variable. Unset or empty yields the default schedule with basis "default";
// otherwise it loads the configured path with basis "config:<path>", falling
// back to the default schedule with basis "config-error:<path>" when loading
// fails (fail-closed: the default never assumes business hours beyond its own
// explicit windows).
func FromEnv() (Schedule, string) {
	path := os.Getenv(EnvConfigPath)
	if path == "" {
		return DefaultSchedule(), "default"
	}
	s, err := LoadSchedule(path)
	if err != nil {
		return DefaultSchedule(), "config-error:" + path
	}
	return s, "config:" + path
}

// parsedWindow is a window with pre-parsed minute bounds.
type parsedWindow struct {
	name     string
	kind     WindowKind
	days     []time.Weekday
	startMin int
	endMin   int
	wrap     bool
}

// parseWindows drops windows whose bounds do not parse (fail-closed: they
// never match). wrap follows the documented rule End <= Start.
func parseWindows(ws []Window) []parsedWindow {
	out := make([]parsedWindow, 0, len(ws))
	for _, w := range ws {
		sm, okS := parseHM(w.Start)
		em, okE := parseHM(w.End)
		if !okS || !okE {
			continue
		}
		out = append(out, parsedWindow{
			name:     w.Name,
			kind:     w.Kind,
			days:     w.Days,
			startMin: sm,
			endMin:   em,
			wrap:     em <= sm,
		})
	}
	return out
}

// matchAt returns the first window (slice order) containing the wall-clock
// instant tl, or UNKNOWN. Intervals are half-open [Start, End).
func matchAt(pws []parsedWindow, tl time.Time) (WindowKind, string) {
	minutes := tl.Hour()*60 + tl.Minute()
	wd := tl.Weekday()
	for _, pw := range pws {
		if !dayMatches(pw.days, wd) {
			continue
		}
		if pw.wrap {
			if minutes >= pw.startMin || minutes < pw.endMin {
				return pw.kind, pw.name
			}
		} else if minutes >= pw.startMin && minutes < pw.endMin {
			return pw.kind, pw.name
		}
	}
	return WindowUnknown, ""
}

func dayMatches(days []time.Weekday, wd time.Weekday) bool {
	if len(days) == 0 {
		return true
	}
	for _, d := range days {
		if d == wd {
			return true
		}
	}
	return false
}

// nextTransition scans forward up to eight local days (every weekday appears,
// and window boundaries recur weekly in wall-clock time) for the earliest
// boundary strictly after tl at which the kind differs from cur. It returns
// found=false when no such boundary exists, which for a periodic schedule
// means the kind provably never changes.
func nextTransition(pws []parsedWindow, loc *time.Location, tl time.Time, cur WindowKind) (time.Time, WindowKind, bool) {
	base := time.Date(tl.Year(), tl.Month(), tl.Day(), 0, 0, 0, 0, loc)
	candidates := make([]time.Time, 0, len(pws)*16)
	for _, pw := range pws {
		for d := 0; d <= 7; d++ {
			day := base.AddDate(0, 0, d)
			start := day.Add(time.Duration(pw.startMin) * time.Minute)
			end := day.Add(time.Duration(pw.endMin) * time.Minute)
			if pw.wrap {
				end = end.AddDate(0, 0, 1)
			}
			if start.After(tl) {
				candidates = append(candidates, start)
			}
			if end.After(tl) {
				candidates = append(candidates, end)
			}
		}
	}
	sort.Slice(candidates, func(i, j int) bool { return candidates[i].Before(candidates[j]) })
	for _, c := range candidates {
		if k, _ := matchAt(pws, c.In(loc)); k != cur {
			return c, k, true
		}
	}
	return time.Time{}, cur, false
}

// resolveLocation maps the schedule's Location field onto a concrete
// *time.Location. "" and "Local" mean system local; anything else must load
// as an IANA name. The second return is the display name ("Local" or the
// requested IANA name); the third is false when the location cannot load.
func resolveLocation(name string) (*time.Location, string, bool) {
	switch name {
	case "", "Local":
		return time.Local, "Local", true
	default:
		l, err := time.LoadLocation(name)
		if err != nil {
			return nil, name, false
		}
		return l, name, true
	}
}

// parseHM parses a strict two-digit "HH:MM" into minutes since midnight.
func parseHM(v string) (int, bool) {
	if len(v) != 5 || v[2] != ':' {
		return 0, false
	}
	digits := [4]byte{v[0], v[1], v[3], v[4]}
	h, m := 0, 0
	for i, d := range digits {
		if d < '0' || d > '9' {
			return 0, false
		}
		if i < 2 {
			h = h*10 + int(d-'0')
		} else {
			m = m*10 + int(d-'0')
		}
	}
	if h > 23 || m > 59 {
		return 0, false
	}
	return h*60 + m, true
}
