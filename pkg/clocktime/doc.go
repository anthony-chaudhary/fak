// Package clocktime is fak-private's CALENDAR TIME-AWARENESS layer: it
// classifies an instant into named calendar windows (business peak, off-peak,
// quiet, maintenance) in a schedule's own location, and reports when the
// classification next changes.
//
// # Why this package exists
//
// The system already measures DURATION and STUCKNESS (elapsed seconds, stuck
// thresholds, cadence intervals) but has zero CALENDAR awareness: nothing
// knows that 03:00 on a Saturday is a different operating regime from 14:00
// on a Tuesday. docs/OPS.md reserves UTC elapsed durations for cadence
// `every`; this package owns the complementary axis - calendar-local
// time-of-day and day-of-week windows. It deliberately does NOT use
// UTC-day logic: windows are evaluated in the schedule's location, so a
// 22:00-06:00 quiet window means 22:00-06:00 THERE, whatever the UTC day.
//
// # Model
//
// A Schedule is a named IANA location ("" or "Local" = system local) plus an
// ordered list of Windows. A Window applies on its listed Days (empty Days =
// every day) between Start and End, both "HH:MM" in the schedule's location.
// Intervals are half-open [Start, End): the start instant is inside the
// window, the end instant is not. End <= Start wraps past midnight (22:00-
// 06:00 spans two local days); under that rule 00:00-00:00 is the all-day
// window. Deterministic precedence: the FIRST matching window in slice order
// wins; no match => UNKNOWN.
//
// # Evaluation
//
// Classify converts the input instant into the schedule's location and reads
// wall-clock weekday + minute there. All forward arithmetic (next
// transition) is done with time.Date in that location, so window boundaries
// stay anchored to local wall-clock across DST transitions by construction -
// no UTC-day assumptions anywhere.
//
// # Fail-closed and honesty rules
//
//   - A zero-value (windowless) Schedule classifies as UNKNOWN, never as
//     business hours.
//   - An unloadable location classifies as UNKNOWN; the requested name is
//     echoed in Classification.Location and Basis, never guessed.
//   - A window with an unparseable Start/End never matches (skipped), so a
//     hand-built malformed schedule degrades toward UNKNOWN instead of
//     inventing coverage. LoadSchedule rejects such schedules up front.
//   - UntilNext is -1 whenever no next transition is computable (no windows,
//     unloadable location, or the kind provably never changes); NextAt is the
//     zero Time and NextKind equals the current Kind in that case.
//   - Nothing is fabricated: every reported field is either measured from the
//     input instant or derived by the rules above, and Basis names what was
//     used.
//
// # Basis grammar (closed set)
//
// Classification.Basis names the schedule source structurally:
//
//	"none"                  windowless schedule (zero-value or empty Windows)
//	"location:<name>"       requested location failed to load (UNKNOWN)
//	"windows:<N>@<loc>"     N windows classified in location <loc>
//
// where <loc> is "Local" or the IANA name. The semantic source label
// ("default" / "config:<path>" / "config-error:<path>") is carried by
// FromEnv's second return value, because a Schedule value alone cannot say
// which file it came from.
//
// # Placement
//
// Autonomous-factory scheduling support (when may work proceed / when is it
// quiet), which BOUNDARY.md keeps private. Standard library only.
package clocktime
