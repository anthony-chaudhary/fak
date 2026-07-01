package fleetaccounts

import (
	"bytes"
	"encoding/json"
	"math"
	"sort"
	"strings"
)

// seatscore.go — the deterministic per-seat health fold for routing decisions.
//
// A "seat" is a rate-limit pool (the PoolKey unit BuildSeatPool hands out): the thing a
// dispatch actually draws quota from. The passive availability fold in status.go answers
// "is this seat blocked right now?" from the live registry; this file answers the
// complementary, HISTORY-based question the router also wants: "how well has this seat
// been doing lately?" — folding a rolling ledger of dispatch outcomes into one health
// score per seat so the router can prefer seats that recently succeeded over seats that
// recently failed, timed out, or hit the wall.
//
// The fold is a pure function of the event slice — no clock, no disk, no global state —
// so a fixture ledger folds into the SAME scores every run (the #1804 witness). It is
// also SECRET-FREE by construction: a SeatEvent carries only a seat identity and one
// closed-vocabulary outcome token, and a SeatScore carries only counts + a derived score.
// There is no field for a token, a prompt, a reason string, or any credential, so a seat
// health report can never leak one.

// SeatOutcome is the closed vocabulary of terminal + in-flight dispatch outcomes a seat
// ledger records. Started is in-flight (a dispatch began); the other four are terminal.
type SeatOutcome string

const (
	// SeatStarted marks a dispatch that began on the seat (in-flight, not yet resolved).
	SeatStarted SeatOutcome = "started"
	// SeatWitnessed marks a dispatch that produced a witnessed, verified result — the
	// only outcome that counts as a success (a self-reported "done" is not witnessed).
	SeatWitnessed SeatOutcome = "witnessed"
	// SeatFailed marks a dispatch that concluded without a witnessed result.
	SeatFailed SeatOutcome = "failed"
	// SeatThrottled marks a dispatch refused/cut by a usage wall on the seat's pool.
	SeatThrottled SeatOutcome = "throttled"
	// SeatTimedOut marks a dispatch that exceeded its deadline without concluding.
	SeatTimedOut SeatOutcome = "timed_out"
)

// seatOutcomeAliases maps the loose tokens a real ledger tends to carry onto the closed
// vocabulary. Keys are already separator-normalized (lowercase, '-'/space -> '_').
var seatOutcomeAliases = map[string]SeatOutcome{
	"started":      SeatStarted,
	"start":        SeatStarted,
	"dispatched":   SeatStarted,
	"launched":     SeatStarted,
	"witnessed":    SeatWitnessed,
	"success":      SeatWitnessed,
	"succeeded":    SeatWitnessed,
	"ok":           SeatWitnessed,
	"done":         SeatWitnessed,
	"shipped":      SeatWitnessed,
	"failed":       SeatFailed,
	"fail":         SeatFailed,
	"failure":      SeatFailed,
	"error":        SeatFailed,
	"errored":      SeatFailed,
	"throttled":    SeatThrottled,
	"throttle":     SeatThrottled,
	"rate_limited": SeatThrottled,
	"ratelimited":  SeatThrottled,
	"walled":       SeatThrottled,
	"timed_out":    SeatTimedOut,
	"timeout":      SeatTimedOut,
	"timedout":     SeatTimedOut,
	"deadline":     SeatTimedOut,
}

// ParseSeatOutcome normalizes a raw ledger token to a SeatOutcome. It is tolerant of
// case, of '-'/space vs '_' separators, and of the common aliases (success/ok -> witnessed,
// timeout -> timed_out, rate_limited -> throttled). ok is false for an empty or unknown
// token, which the fold ignores rather than mis-bucketing.
func ParseSeatOutcome(raw string) (SeatOutcome, bool) {
	key := strings.ToLower(strings.TrimSpace(raw))
	key = strings.ReplaceAll(key, "-", "_")
	key = strings.ReplaceAll(key, " ", "_")
	if key == "" {
		return "", false
	}
	if out, ok := seatOutcomeAliases[key]; ok {
		return out, true
	}
	return "", false
}

// SeatEvent is one row of a seat outcome ledger. It is deliberately minimal — a seat
// identity (the PoolKey the dispatch drew on) and one outcome token — so the ledger, and
// anything folded from it, holds no secret. Unknown/empty Outcome tokens are ignored.
type SeatEvent struct {
	Seat    string      `json:"seat"`
	Outcome SeatOutcome `json:"outcome"`
}

// SeatScore is the folded rolling health of one seat. Started counts every dispatch that
// began; Witnessed/Failed/Throttled/TimedOut are the terminal tallies. Resolved is the
// number of terminal outcomes (the denominator of the rates); InFlight is the count of
// starts not yet accounted for by a terminal outcome. Health is the witnessed fraction of
// resolved dispatches in [0,1], rounded for stable comparison; Scored is false when no
// terminal outcome has landed yet (Health is then 0 and carries no evidence).
type SeatScore struct {
	Seat      string  `json:"seat"`
	Started   int     `json:"started"`
	Witnessed int     `json:"witnessed"`
	Failed    int     `json:"failed"`
	Throttled int     `json:"throttled"`
	TimedOut  int     `json:"timed_out"`
	Resolved  int     `json:"resolved"`
	InFlight  int     `json:"in_flight"`
	Health    float64 `json:"health"`
	Scored    bool    `json:"scored"`
}

// seatHealthPrecision rounds Health to a fixed number of decimals so two seats with the
// same underlying ratio compare equal regardless of float representation noise.
const seatHealthPrecision = 1e4

// finalizeSeatScore derives Resolved/InFlight/Health/Scored from the raw tallies. Health
// is witnessed / (witnessed + failed + throttled + timed_out): the fraction of concluded
// dispatches that produced a witnessed result. Throttled and timed-out count against the
// seat because the score exists to steer routing AWAY from seats that are currently
// costly to dispatch on, not only ones that produced wrong work.
func finalizeSeatScore(s *SeatScore) {
	s.Resolved = s.Witnessed + s.Failed + s.Throttled + s.TimedOut
	if inflight := s.Started - s.Resolved; inflight > 0 {
		s.InFlight = inflight
	}
	if s.Resolved == 0 {
		s.Health, s.Scored = 0, false
		return
	}
	h := float64(s.Witnessed) / float64(s.Resolved)
	s.Health = math.Round(h*seatHealthPrecision) / seatHealthPrecision
	s.Scored = true
}

// FoldSeatScores folds a seat outcome ledger into one deterministic SeatScore per seat.
// The result is ordered best-for-routing first: scored seats before unscored, then higher
// Health, then more Resolved evidence, then seat identity — a total order, so the fold is
// reproducible byte-for-byte across runs. Events with an unknown/empty outcome are counted
// nowhere (they never mis-bucket), but their seat still appears if it has other events.
func FoldSeatScores(events []SeatEvent) []SeatScore {
	idx := map[string]int{}
	scores := make([]SeatScore, 0)
	seatAt := func(seat string) int {
		if i, ok := idx[seat]; ok {
			return i
		}
		i := len(scores)
		idx[seat] = i
		scores = append(scores, SeatScore{Seat: seat})
		return i
	}
	for _, e := range events {
		i := seatAt(e.Seat)
		out, ok := ParseSeatOutcome(string(e.Outcome))
		if !ok {
			continue
		}
		switch out {
		case SeatStarted:
			scores[i].Started++
		case SeatWitnessed:
			scores[i].Witnessed++
		case SeatFailed:
			scores[i].Failed++
		case SeatThrottled:
			scores[i].Throttled++
		case SeatTimedOut:
			scores[i].TimedOut++
		}
	}
	for i := range scores {
		finalizeSeatScore(&scores[i])
	}
	sort.SliceStable(scores, func(i, j int) bool {
		a, b := scores[i], scores[j]
		if a.Scored != b.Scored {
			return a.Scored && !b.Scored
		}
		if a.Health != b.Health {
			return a.Health > b.Health
		}
		if a.Resolved != b.Resolved {
			return a.Resolved > b.Resolved
		}
		return a.Seat < b.Seat
	})
	return scores
}

// DecodeSeatLedger parses a seat outcome ledger from bytes. It accepts either a JSON array
// of SeatEvent objects or the JSONL form (one JSON object per line — the shape the fleet's
// probe/dispatch ledgers already use). Blank lines are skipped; a malformed row is a hard
// error so a corrupt ledger is not silently under-counted.
func DecodeSeatLedger(data []byte) ([]SeatEvent, error) {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 {
		return nil, nil
	}
	if trimmed[0] == '[' {
		var evs []SeatEvent
		if err := json.Unmarshal(trimmed, &evs); err != nil {
			return nil, err
		}
		return evs, nil
	}
	var evs []SeatEvent
	for _, line := range bytes.Split(trimmed, []byte("\n")) {
		line = bytes.TrimSpace(line)
		if len(line) == 0 {
			continue
		}
		var e SeatEvent
		if err := json.Unmarshal(line, &e); err != nil {
			return nil, err
		}
		evs = append(evs, e)
	}
	return evs, nil
}

// FoldSeatLedger is the one-call convenience the CLI/report path uses: decode a raw ledger
// and fold it into deterministic seat scores.
func FoldSeatLedger(data []byte) ([]SeatScore, error) {
	evs, err := DecodeSeatLedger(data)
	if err != nil {
		return nil, err
	}
	return FoldSeatScores(evs), nil
}
