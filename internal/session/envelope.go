package session

// envelope.go — issue #1573 (managed-context epic #1570, product track): the
// USER-FACING budget-envelope syntax. Everything else in this package (Budget,
// Pace, Throughput, TimeBudget) is the RUNTIME's internal drive-state vocabulary —
// correct, but scattered across five axes an operator has to know the shape of
// before they can start a managed run with a stated ceiling. This file is the one
// surface a user (or a CLI flag set / API request body) states a goal against:
// "run this, but no more than N tokens / M minutes / T turns / $D / a throughput
// floor" — and get back the SAME deterministic Budget/Pace/TimeBudget the runtime
// already enforces, so there is exactly one parse between what the user typed and
// what the table debits.
//
// THE GAP IT CLOSES. #1584 (TimeBudget) and #1585 (Throughput/ComposePace) gave the
// runtime new axes to enforce, but nothing let a user STATE a goal across all of
// them in one place — a caller wanting "cap this run at 50k tokens and 10 minutes"
// had to know to construct a session.Budget AND a session.TimeBudget by hand, two
// separate types with two separate zero-value conventions (Unbounded=-1 vs
// LimitNanos<=0). Envelope is the single parse surface: one struct, one string
// syntax, one Parse function, that PRODUCES those internal types rather than
// replacing them — see ToBudget/ToTimeBudget/ToPace below, each a pure projection
// onto the existing runtime primitives. Composing further (planner-window scaling,
// throughput-aware shrinkage) stays in compose.go/ctxplan/pace.go; this file never
// duplicates that math.
//
// SPEND (new here). Nothing in the runtime tracked a dollar ceiling before this
// issue — modelroute/cost.go prices a routing DECISION after the fact, but no axis
// let a user say "stop around $5". SpendCapCents is the one new leaf number this
// file adds (a spend axis with no runtime enforcement point yet, same posture
// ContextTokensCap had before #743 read it): carried on Envelope and echoed back in
// ParsedEnvelope for inspection, advisory until a debit path is wired to it.
//
// DETERMINISM. Parse is pure (string in, ParsedEnvelope-or-error out) — the same
// envelope string always parses to the same struct, and structs compare equal, so
// "inspect the parsed deterministic budget" (the issue's Done condition) is a real
// equality check a test can assert, never a narrated claim.

import (
	"fmt"
	"strconv"
	"strings"
	"time"
)

// EnvelopeUnbounded is the sentinel a user types to explicitly request "no limit"
// on an axis (mirrors Budget.Unbounded/TimeBudget.TimeUnbounded's -1 convention,
// surfaced here as the one string token both the token/turn axes and the CLI flag
// parser accept).
const EnvelopeUnbounded = "unbounded"

// Envelope is the user-stated budget goal: the plain, product-facing contract for
// "how much may this managed run cost, on every axis it might cost something."
// Every field is independently optional (zero = "the user expressed no opinion on
// this axis"), so a user may state just one axis (e.g. only WallClock) and get a
// deterministic parse that leaves every other axis unbounded — never a surprise
// cap on an axis nobody mentioned.
type Envelope struct {
	// Tokens caps total output tokens across the run. <0 (EnvelopeUnbounded) means
	// no cap; 0 means "not stated" (parses to Budget's Unbounded default); >0 is a
	// real ceiling.
	Tokens int `json:"tokens,omitempty"`
	// WallClock caps real elapsed time across the run's whole lineage. <=0 means
	// "not stated" (unbounded); this is the one axis with no "unbounded" string
	// form, since a zero/absent duration already means unbounded.
	WallClock time.Duration `json:"wall_clock,omitempty"`
	// Turns caps the number of model round-trips. Same tri-state convention as
	// Tokens: <0 explicit-unbounded, 0 not-stated, >0 a real ceiling.
	Turns int `json:"turns,omitempty"`
	// SpendCapCents caps the run's rough dollar cost, in integer cents (avoiding a
	// float money type). 0 means not stated. Advisory only today — see file header;
	// no runtime path debits it yet, but it round-trips through Parse/inspect so a
	// user's stated ceiling is never silently dropped.
	SpendCapCents int64 `json:"spend_cap_cents,omitempty"`
	// ThroughputFloor is the minimum tokens/sec the user expects this run to
	// sustain — the user-facing twin of Throughput.ExpectedTokensPerSec. 0 means
	// not stated (no expectation configured, exactly Throughput's zero-value
	// convention).
	ThroughputFloor float64 `json:"throughput_floor,omitempty"`
}

// IsZero reports whether the envelope states no opinion on any axis — the safe
// default a caller reads as "no envelope was requested" before doing any parsing
// work or attaching a budget to a run.
func (e Envelope) IsZero() bool {
	return e.Tokens == 0 && e.WallClock == 0 && e.Turns == 0 &&
		e.SpendCapCents == 0 && e.ThroughputFloor == 0
}

// ParsedEnvelope is the deterministic output a user (or a test) inspects: the
// Envelope they stated, plus the three runtime primitives it produces. It is the
// one artifact `fak session envelope` prints and the one shape a witness test
// compares for equality — never a narrated "budget looks right".
type ParsedEnvelope struct {
	Envelope   Envelope   `json:"envelope"`
	Budget     Budget     `json:"budget"`
	TimeBudget TimeBudget `json:"time_budget"`
	Pace       Pace       `json:"pace"`
}

// ToBudget projects the token/turn axes onto a Budget, in the exact Unbounded (-1)
// / not-configured (0 context) shape Decide/DebitUsage already consume. Context and
// clarification-query axes are left at their DefaultState zero (this envelope layer
// speaks the product-facing axes the issue names; a caller wanting a context cap
// too still sets Budget.ContextTokensLeft directly, unchanged).
func (e Envelope) ToBudget() Budget {
	b := Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded}
	if e.Tokens != 0 {
		b.TokensLeft = e.Tokens
	}
	if e.Turns != 0 {
		b.TurnsLeft = e.Turns
	}
	return b.withContextCap()
}

// ToTimeBudget projects the wall-clock axis onto a TimeBudget with WithLimit — a
// non-positive/absent WallClock yields the unbounded zero value, exactly
// TimeBudget's own "not configured" convention, so an envelope that never mentions
// wall-clock produces a TimeBudget byte-identical to NewTimeBudget().
func (e Envelope) ToTimeBudget() TimeBudget {
	return NewTimeBudget().WithLimit(e.WallClock)
}

// ToPace projects the throughput floor onto a Pace's expectation half. Pace itself
// carries no throughput field (see compose.go's file header: Throughput is
// deliberately a standalone type, not fields on Pace), so ToPace returns the zero
// Pace (no per-turn MaxTokensPerTurn/MinTurnGapMs opinion) — callers wanting the
// throughput floor read ToThroughput instead. Kept as a named method (rather than
// omitted) so ParsedEnvelope's shape is self-documenting: an envelope's per-turn
// pace opinion is always the runtime default unless a caller composes one
// separately.
func (e Envelope) ToPace() Pace { return Pace{} }

// ToThroughput projects the throughput floor onto a Throughput's expected-rate
// axis, leaving ObservedTokensPerSec at its zero ("no observation yet") — the
// runtime measures the observed rate; the envelope only states the floor it is
// judged against.
func (e Envelope) ToThroughput() Throughput {
	return Throughput{ExpectedTokensPerSec: e.ThroughputFloor}
}

// Parse folds the projections into the one deterministic artifact a caller
// inspects. It performs no I/O and no clock read (WallClock is a duration, not a
// start time — ToTimeBudget's Start(now) is the caller's job at the moment the run
// actually begins), so the same Envelope always parses to a byte-identical
// ParsedEnvelope.
func (e Envelope) Parse() ParsedEnvelope {
	return ParsedEnvelope{
		Envelope:   e,
		Budget:     e.ToBudget(),
		TimeBudget: e.ToTimeBudget(),
		Pace:       e.ToPace(),
	}
}

// ParseEnvelopeFlags parses the CLI's flat string-flag form into an Envelope — the
// one place the "unbounded" string token, a duration string ("10m"), and a dollar
// string ("$5.00" or "500c") are interpreted. Every argument is optional: an empty
// string means "not stated" for that axis, so a caller only sets the flags it read
// from the command line. A non-empty, unparsable value is a hard error (fail
// closed on a malformed user envelope rather than silently ignoring the axis).
func ParseEnvelopeFlags(tokens, wallClock, turns, spend, throughputFloor string) (Envelope, error) {
	var e Envelope
	var err error
	if e.Tokens, err = parseIntAxis(tokens); err != nil {
		return Envelope{}, fmt.Errorf("tokens: %w", err)
	}
	if wallClock != "" {
		if strings.EqualFold(wallClock, EnvelopeUnbounded) {
			e.WallClock = 0
		} else {
			d, err := time.ParseDuration(wallClock)
			if err != nil {
				return Envelope{}, fmt.Errorf("wall-clock: %w", err)
			}
			if d < 0 {
				return Envelope{}, fmt.Errorf("wall-clock: negative duration %q", wallClock)
			}
			e.WallClock = d
		}
	}
	if e.Turns, err = parseIntAxis(turns); err != nil {
		return Envelope{}, fmt.Errorf("turns: %w", err)
	}
	if spend != "" {
		cents, err := parseSpendCents(spend)
		if err != nil {
			return Envelope{}, fmt.Errorf("spend: %w", err)
		}
		e.SpendCapCents = cents
	}
	if throughputFloor != "" {
		f, err := strconv.ParseFloat(throughputFloor, 64)
		if err != nil {
			return Envelope{}, fmt.Errorf("throughput: %w", err)
		}
		if f < 0 {
			return Envelope{}, fmt.Errorf("throughput: negative rate %q", throughputFloor)
		}
		e.ThroughputFloor = f
	}
	return e, nil
}

// parseIntAxis parses a token/turns-style flag: "" is not-stated (0), the
// "unbounded" token is the explicit Unbounded (-1) sentinel, and anything else must
// be a base-10 integer greater than zero (a 0 or negative literal is ambiguous with
// the sentinel values and rejected rather than silently reinterpreted).
func parseIntAxis(s string) (int, error) {
	if s == "" {
		return 0, nil
	}
	if strings.EqualFold(s, EnvelopeUnbounded) {
		return Unbounded, nil
	}
	n, err := strconv.Atoi(s)
	if err != nil {
		return 0, fmt.Errorf("%q is not an integer or %q", s, EnvelopeUnbounded)
	}
	if n <= 0 {
		return 0, fmt.Errorf("%q must be a positive integer or %q", s, EnvelopeUnbounded)
	}
	return n, nil
}

// parseSpendCents parses a dollar-form spend cap: a leading "$" is optional, and
// the value is interpreted as dollars-and-cents ("5", "5.00", "$5.25") rounded to
// the nearest cent. A bare integer with no "$" and no decimal point is also
// accepted as dollars (spend is always stated in dollars at the CLI, never raw
// cents, so there is one unambiguous unit).
func parseSpendCents(s string) (int64, error) {
	s = strings.TrimSpace(s)
	s = strings.TrimPrefix(s, "$")
	if s == "" {
		return 0, fmt.Errorf("empty spend value")
	}
	dollars, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, fmt.Errorf("%q is not a dollar amount", s)
	}
	if dollars < 0 {
		return 0, fmt.Errorf("%q is negative", s)
	}
	cents := int64(dollars*100 + 0.5) // round half up; inputs are non-negative here
	return cents, nil
}
