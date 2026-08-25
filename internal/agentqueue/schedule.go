package agentqueue

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"
)

type MisfirePolicy string

const (
	MisfireSkip    MisfirePolicy = "skip"
	MisfireCatchUp MisfirePolicy = "catch_up"
)

type DailyWindow struct {
	ID       string        `json:"id"`
	Timezone string        `json:"timezone"`
	Start    string        `json:"start"`
	Stop     string        `json:"stop"`
	Misfire  MisfirePolicy `json:"misfire"`
}

type WindowOccurrence struct {
	ID       string    `json:"id"`
	StartsAt time.Time `json:"starts_at"`
	StopsAt  time.Time `json:"stops_at"`
}

// OccurrenceAt returns the daily local-time window containing now. Ambiguous
// fold times choose Go's deterministic first occurrence. Nonexistent DST-gap
// boundaries are rejected instead of silently shifting operator intent.
func (w DailyWindow) OccurrenceAt(now time.Time) (WindowOccurrence, bool, error) {
	loc, startMinute, stopMinute, err := w.validate()
	if err != nil {
		return WindowOccurrence{}, false, err
	}
	local := now.In(loc)
	for _, day := range []time.Time{local, local.AddDate(0, 0, -1)} {
		start, err := localBoundary(day, startMinute, loc)
		if err != nil {
			return WindowOccurrence{}, false, err
		}
		stopDay := day
		if stopMinute <= startMinute {
			stopDay = stopDay.AddDate(0, 0, 1)
		}
		stop, err := localBoundary(stopDay, stopMinute, loc)
		if err != nil {
			return WindowOccurrence{}, false, err
		}
		if !now.Before(start) && now.Before(stop) {
			return makeOccurrence(w.ID, start, stop), true, nil
		}
	}
	return WindowOccurrence{}, false, nil
}

// Due reports occurrences whose start was crossed since the previous tick.
// Skip drops missed starts; catch_up emits the newest still-open occurrence.
func (w DailyWindow) Due(previous, now time.Time) ([]WindowOccurrence, error) {
	if now.Before(previous) {
		return nil, errors.New("agentqueue: schedule clock moved backwards")
	}
	loc, startMinute, stopMinute, err := w.validate()
	if err != nil {
		return nil, err
	}
	var due []WindowOccurrence
	first := previous.In(loc).AddDate(0, 0, -1)
	last := now.In(loc)
	for day := first; !day.After(last); day = day.AddDate(0, 0, 1) {
		start, err := localBoundary(day, startMinute, loc)
		if err != nil {
			return nil, err
		}
		stopDay := day
		if stopMinute <= startMinute {
			stopDay = stopDay.AddDate(0, 0, 1)
		}
		stop, err := localBoundary(stopDay, stopMinute, loc)
		if err != nil {
			return nil, err
		}
		if start.After(previous) && !start.After(now) {
			occ := makeOccurrence(w.ID, start, stop)
			switch w.Misfire {
			case MisfireSkip:
				if now.Equal(start) {
					due = append(due, occ)
				}
			case MisfireCatchUp:
				if now.Before(stop) {
					due = []WindowOccurrence{occ}
				}
			}
		}
	}
	return due, nil
}

func (w DailyWindow) validate() (*time.Location, int, int, error) {
	if strings.TrimSpace(w.ID) == "" {
		return nil, 0, 0, errors.New("agentqueue: window id is required")
	}
	loc, err := time.LoadLocation(w.Timezone)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("agentqueue: timezone %q: %w", w.Timezone, err)
	}
	start, err := clockMinute(w.Start)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("agentqueue: start: %w", err)
	}
	stop, err := clockMinute(w.Stop)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("agentqueue: stop: %w", err)
	}
	if start == stop {
		return nil, 0, 0, errors.New("agentqueue: start and stop must differ")
	}
	if w.Misfire != MisfireSkip && w.Misfire != MisfireCatchUp {
		return nil, 0, 0, fmt.Errorf("agentqueue: unsupported misfire policy %q", w.Misfire)
	}
	return loc, start, stop, nil
}

func clockMinute(value string) (int, error) {
	parts := strings.Split(value, ":")
	if len(parts) != 2 {
		return 0, errors.New("time must be HH:MM")
	}
	hour, e1 := strconv.Atoi(parts[0])
	minute, e2 := strconv.Atoi(parts[1])
	if e1 != nil || e2 != nil || hour < 0 || hour > 23 || minute < 0 || minute > 59 || len(parts[0]) != 2 || len(parts[1]) != 2 {
		return 0, errors.New("time must be zero-padded HH:MM")
	}
	return hour*60 + minute, nil
}

func localBoundary(day time.Time, minute int, loc *time.Location) (time.Time, error) {
	y, m, d := day.In(loc).Date()
	h, min := minute/60, minute%60
	boundary := time.Date(y, m, d, h, min, 0, 0, loc)
	by, bm, bd := boundary.In(loc).Date()
	if by != y || bm != m || bd != d || boundary.In(loc).Hour() != h || boundary.In(loc).Minute() != min {
		return time.Time{}, fmt.Errorf("agentqueue: local boundary %04d-%02d-%02dT%02d:%02d does not exist in %s", y, m, d, h, min, loc)
	}
	return boundary, nil
}

func makeOccurrence(id string, start, stop time.Time) WindowOccurrence {
	sum := sha256.Sum256([]byte(id + "\x00" + start.UTC().Format(time.RFC3339Nano) + "\x00" + stop.UTC().Format(time.RFC3339Nano)))
	return WindowOccurrence{ID: "window:" + hex.EncodeToString(sum[:16]), StartsAt: start, StopsAt: stop}
}
