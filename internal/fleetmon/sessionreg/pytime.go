package sessionreg

import (
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// isoUTC is Python's aware-UTC datetime.isoformat(): microseconds only when non-zero.
func isoUTC(t time.Time) string {
	t = t.UTC()
	s := t.Format("2006-01-02T15:04:05")
	if us := t.Nanosecond() / 1000; us != 0 {
		s += fmt.Sprintf(".%06d", us)
	}
	return s + "+00:00"
}

// mtimeUTC reproduces datetime.fromtimestamp(os.stat(p).st_mtime, utc): the stat float is
// sec + nsec*1e-9, and fromtimestamp rounds its fraction to microseconds half-to-even.
// Deriving it the same way keeps seen_utc and age_min bit-identical to the Python writer.
func mtimeUTC(mod time.Time) time.Time {
	ns := mod.UnixNano()
	sec, nsec := ns/1e9, ns%1e9
	if nsec < 0 {
		sec--
		nsec += 1e9
	}
	f := float64(sec) + float64(nsec)*1e-9
	whole, frac := math.Modf(f)
	us := math.RoundToEven(frac * 1e6)
	if us >= 1e6 {
		us -= 1e6
		whole++
	} else if us < 0 {
		us += 1e6
		whole--
	}
	return time.Unix(int64(whole), int64(us)*1000).UTC()
}

// minutesSince is Python's (now - then).total_seconds() / 60.0 on microsecond datetimes.
func minutesSince(now, then time.Time) float64 {
	us := now.Truncate(time.Microsecond).Sub(then.Truncate(time.Microsecond)).Microseconds()
	return float64(us) / 1e6 / 60.0
}

var isoRE = regexp.MustCompile(`^(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2})(?::(\d{2})(?::(\d{2})(?:[.,](\d{1,6}))?)?)?)?(?:([+-])(\d{2}):?(\d{2})(?::?(\d{2})(?:\.(\d{1,6}))?)?)?$`)

// parseUTC is the fleet tools' _parse_utc: datetime.fromisoformat(str(raw).replace("Z",
// "+00:00")), naive read as UTC, converted to UTC; nil when unparseable or empty.
func parseUTC(raw any) *time.Time {
	if !truthy(raw) {
		return nil
	}
	s := strings.ReplaceAll(pyStr(raw), "Z", "+00:00")
	m := isoRE.FindStringSubmatch(s)
	if m == nil {
		return nil
	}
	atoi := func(x string) int {
		if x == "" {
			return 0
		}
		n, _ := strconv.Atoi(x)
		return n
	}
	frac := 0
	if m[7] != "" {
		frac = atoi((m[7] + "000000")[:6])
	}
	t := time.Date(atoi(m[1]), time.Month(atoi(m[2])), atoi(m[3]), atoi(m[4]), atoi(m[5]), atoi(m[6]), frac*1000, time.UTC)
	if t.Year() != atoi(m[1]) || int(t.Month()) != atoi(m[2]) || t.Day() != atoi(m[3]) ||
		atoi(m[4]) > 23 || atoi(m[5]) > 59 || atoi(m[6]) > 59 {
		return nil
	}
	if m[8] != "" {
		off := time.Duration(atoi(m[9]))*time.Hour + time.Duration(atoi(m[10]))*time.Minute + time.Duration(atoi(m[11]))*time.Second
		if atoi(m[9]) > 23 || atoi(m[10]) > 59 {
			return nil
		}
		if m[8] == "+" {
			t = t.Add(-off)
		} else {
			t = t.Add(off)
		}
	}
	return &t
}
