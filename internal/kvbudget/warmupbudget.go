package kvbudget

// This file derives the admission token budget from a MEASURED warmup probe
// instead of a guessed constant (issue #5266, epic #2236; field-borrow from
// HuggingFace text-generation-inference v3). The sibling admit.go / prealloc.go
// folds spend an admission budget in KV blocks; this file answers the upstream
// question those folds assume already-answered: how big is that budget?
//
// The borrow: at warmup a backend runs the largest prefill it can and measures
// how much KV state actually fits — the usable bytes (or the fitted block
// count). fak then sets the admission token budget from that MEASURED figure,
// discounted by a small safety reserve, rather than from a static default. So
// the budget tracks real fitted room, and the shed boundary sheds at true
// room, not at an arbitrary number.
//
// Everything here is a deterministic integer fold. The measurement is an
// INJECTED value — the caller measured it during warmup — so there is no
// hardware, no network, and no wall clock in this file. Bad or empty
// measurements fail closed to a zero, typed-reason budget: never a huge or a
// negative budget.
//
// A compressed KV schedule (DeepSeek-V4 Flash, #13555) is not purely per-token:
// each stream also holds context-independent state — its window rows, the
// compressed rows' ceil slack, the compressor's in-flight buffer — however few
// tokens it has. Dividing the whole measurement by a per-token slope would hand
// that fixed state out as tokens, so WarmupCapacity can carry it per stream and
// the derive withholds it for every stream admission may run at once. With no
// fixed state declared the derive is the original fold, byte for byte.

import "math"

// DefaultReserveFraction is the safety reserve withheld from a measured usable
// figure before it is turned into a token budget — the mirror of TGI's 0.90
// wiggle room (provision against ~90% of what fit, hold back ~10%). A caller may
// pass its own fraction; this is the borrowed default.
const DefaultReserveFraction = 0.10

// The typed reasons a derive can fail closed with. An empty Reason (the zero
// value) means the budget was derived; a non-empty Reason means the derive
// failed closed and the budget is zero.
const (
	// ReasonNoMeasuredCapacity fails closed when the measured usable amount is
	// zero or negative — the probe reported no room, so there is nothing to
	// derive a budget from. The token here is "capacity" (usable room), which
	// does not name the guard root.
	ReasonNoMeasuredCapacity Reason = "no_measured_capacity"
	// ReasonInvalidUnitSize fails closed when the per-unit divisor (bytes per
	// token, or tokens per block) is zero or negative — a nonsense unit size the
	// budget cannot be divided by.
	ReasonInvalidUnitSize Reason = "invalid_unit_size"
	// ReasonReserveOutOfRange fails closed when the reserve fraction is not in
	// the half-open range [0, 1): a fraction of 1 or more would reserve the whole
	// measurement (or more), and a negative fraction would inflate it.
	ReasonReserveOutOfRange Reason = "reserve_fraction_out_of_range"
	// ReasonBelowOneUnit fails closed when a positive measurement, after the
	// reserve is applied, no longer holds even one whole token (or one whole
	// block) — the fitted room rounds down to a zero admittable budget. With
	// fixed per-stream state declared, it also covers a measurement that the
	// withheld fixed state consumes entirely.
	ReasonBelowOneUnit Reason = "measured_below_one_unit"
	// ReasonUnboundedStreamState fails closed when a WarmupCapacity declares
	// fixed per-stream state but no positive MaxConcurrentStreams: without a
	// stream bound the fixed state cannot be withheld, and deriving anyway would
	// over-admit.
	ReasonUnboundedStreamState Reason = "unbounded_fixed_stream_state"
)

// DerivedBudget is the outcome of turning a warmup measurement into an admission
// token budget. On success TokenBudget is the derived admittable tokens (> 0),
// Reason is empty, ReservedAmount records how much of the measurement the safety
// reserve withheld, and KeptAmount is the measurement left after the reserve. On
// a fail-closed TokenBudget is zero and Reason carries the typed cause.
type DerivedBudget struct {
	// TokenBudget is the derived admission budget in tokens (0 on fail-closed).
	TokenBudget int64
	// Reason is empty on success, else the typed fail-closed cause.
	Reason Reason
	// ReservedAmount is the measured amount withheld by the safety reserve, in
	// the measurement's own unit (bytes for WarmupCapacity, blocks for
	// WarmupBlockCapacity). Zero on fail-closed.
	ReservedAmount int64
	// KeptAmount is the measured amount left after the reserve, in the same unit.
	// Zero on fail-closed. With fixed per-stream state it is what is left after
	// the reserve AND FixedStateAmount — the room the token budget divides —
	// except when the fixed state consumes the whole kept measurement, where it
	// is the kept measurement itself.
	KeptAmount int64
	// FixedStateAmount is the bytes withheld for MaxConcurrentStreams ×
	// FixedBytesPerStream (saturating) before the per-token divide. Zero when no
	// fixed per-stream state is declared.
	FixedStateAmount int64
}

// Derived reports whether the budget was derived (a positive budget with an
// empty Reason) versus failed closed.
func (d DerivedBudget) Derived() bool { return d.Reason == ReasonAdmitted && d.TokenBudget > 0 }

// applyReserve withholds floor(amount × fraction) of a non-negative measured
// amount and returns the kept remainder plus the withheld reserve. The fraction
// is assumed already range-checked to [0, 1). Deterministic: for measured
// amounts well within float64's exact-integer range (KV byte counts are), the
// product and its floor are stable for a given input.
func applyReserve(amount int64, fraction float64) (kept, reserved int64) {
	reserved = int64(math.Floor(float64(amount) * fraction))
	if reserved < 0 {
		reserved = 0
	}
	if reserved > amount {
		reserved = amount
	}
	return amount - reserved, reserved
}

// validReserveFraction reports whether a reserve fraction is in the half-open
// range [0, 1) and is a real number. Anything else fails closed rather than
// deriving a nonsense budget.
func validReserveFraction(fraction float64) bool {
	return fraction >= 0 && fraction < 1 && !math.IsNaN(fraction)
}

// deriveBudget is the shared fold behind both measurement shapes: from a
// measured usable amount, a per-unit divisor, and a reserve fraction, derive the
// token budget as floor((usable − reserve) / perUnit) × tokensPerUnit. It fails
// closed with a typed Reason on a non-positive usable amount, a non-positive
// unit size, an out-of-range reserve, or a measurement that rounds below one
// whole unit after the reserve. Monotone in the usable amount; never negative.
func deriveBudget(usable, perUnit, tokensPerUnit int64, fraction float64) DerivedBudget {
	if perUnit <= 0 || tokensPerUnit <= 0 {
		return DerivedBudget{Reason: ReasonInvalidUnitSize}
	}
	if !validReserveFraction(fraction) {
		return DerivedBudget{Reason: ReasonReserveOutOfRange}
	}
	if usable <= 0 {
		return DerivedBudget{Reason: ReasonNoMeasuredCapacity}
	}
	kept, reserved := applyReserve(usable, fraction)
	units := kept / perUnit // floor division; kept ≥ 0, perUnit > 0
	if units <= 0 {
		return DerivedBudget{
			Reason:         ReasonBelowOneUnit,
			ReservedAmount: reserved,
			KeptAmount:     kept,
		}
	}
	return DerivedBudget{
		TokenBudget:    units * tokensPerUnit,
		ReservedAmount: reserved,
		KeptAmount:     kept,
	}
}

// WarmupCapacity carries a warmup probe measured in BYTES: the usable KV bytes
// that actually fit during warmup, plus the per-token KV footprint the budget is
// divided against. Both are INJECTED measurements — the caller measured them —
// so nothing here touches hardware.
type WarmupCapacity struct {
	// UsableBytes is the measured usable KV bytes that fit during warmup. A
	// zero or negative value fails the derive closed (no room measured).
	UsableBytes int64
	// BytesPerToken is the measured KV footprint of one token. A zero or
	// negative value fails the derive closed (no divisor). For a compressed
	// schedule it is the per-token slope bound, not an average.
	BytesPerToken int64
	// FixedBytesPerStream is the context-independent KV state one stream holds
	// however few tokens it has (a compressed schedule's window rows, ceil
	// slack, compressor in-flight buffer). Zero or negative means a per-token
	// layout: the derive is byte-identical to the original fold.
	FixedBytesPerStream int64
	// MaxConcurrentStreams is how many streams can hold that fixed state at once
	// (the admission max-num-seqs). Required (> 0) when FixedBytesPerStream > 0;
	// ignored otherwise.
	MaxConcurrentStreams int64
}

// DeriveTokenBudget turns the byte-measured warmup probe into an admission token
// budget: floor((UsableBytes − reserve) / BytesPerToken), with the safety
// reserve withheld first. A larger measurement yields a proportionally larger
// budget (monotone); a bigger reserve fraction shrinks it; a zero/negative
// measurement or unit fails closed to a zero, typed-reason budget; a positive
// measurement that holds under one token's bytes after the reserve fails closed
// with ReasonBelowOneUnit. Deterministic; no hardware, no clock.
//
// With FixedBytesPerStream > 0 the fixed state of MaxConcurrentStreams streams
// is withheld after the reserve and before the divide:
// floor((UsableBytes − reserve − MaxConcurrentStreams × FixedBytesPerStream) /
// BytesPerToken). See deriveWithFixedState.
func (c WarmupCapacity) DeriveTokenBudget(fraction float64) DerivedBudget {
	if c.FixedBytesPerStream <= 0 {
		// One token is the unit here, so tokens-per-unit is 1.
		return deriveBudget(c.UsableBytes, c.BytesPerToken, 1, fraction)
	}
	return c.deriveWithFixedState(fraction)
}

// deriveWithFixedState is the byte derive for a layout with fixed per-stream
// state. It validates in deriveBudget's order (unit size, reserve range,
// measured capacity), then fails closed on an unbounded stream count, then
// withholds the reserve and the saturating MaxConcurrentStreams ×
// FixedBytesPerStream before dividing the rest by BytesPerToken. Fixed state
// that meets or exceeds the kept measurement, or a remainder under one token,
// fails closed with ReasonBelowOneUnit. On success ReservedAmount + KeptAmount +
// FixedStateAmount == UsableBytes. Monotone in UsableBytes; never negative.
func (c WarmupCapacity) deriveWithFixedState(fraction float64) DerivedBudget {
	if c.BytesPerToken <= 0 {
		return DerivedBudget{Reason: ReasonInvalidUnitSize}
	}
	if !validReserveFraction(fraction) {
		return DerivedBudget{Reason: ReasonReserveOutOfRange}
	}
	if c.UsableBytes <= 0 {
		return DerivedBudget{Reason: ReasonNoMeasuredCapacity}
	}
	if c.MaxConcurrentStreams <= 0 {
		return DerivedBudget{Reason: ReasonUnboundedStreamState}
	}
	kept, reserved := applyReserve(c.UsableBytes, fraction)
	fixed := saturatingMul(c.MaxConcurrentStreams, c.FixedBytesPerStream)
	if fixed >= kept {
		return DerivedBudget{
			Reason:           ReasonBelowOneUnit,
			ReservedAmount:   reserved,
			KeptAmount:       kept,
			FixedStateAmount: fixed,
		}
	}
	room := kept - fixed
	units := room / c.BytesPerToken // floor division; room > 0, BytesPerToken > 0
	if units <= 0 {
		return DerivedBudget{
			Reason:           ReasonBelowOneUnit,
			ReservedAmount:   reserved,
			KeptAmount:       room,
			FixedStateAmount: fixed,
		}
	}
	return DerivedBudget{
		TokenBudget:      units,
		ReservedAmount:   reserved,
		KeptAmount:       room,
		FixedStateAmount: fixed,
	}
}

// saturatingMul returns a × b for positive a and b, clamped to math.MaxInt64
// instead of wrapping: an absurd stream count times a large fixed state must
// read as "consumes everything", never as a small or negative product.
func saturatingMul(a, b int64) int64 {
	if a > math.MaxInt64/b {
		return math.MaxInt64
	}
	return a * b
}

// WarmupBlockCapacity carries a warmup probe measured in KV BLOCKS: the count of
// blocks that fit during warmup, plus the tokens one block holds. Same contract
// as WarmupCapacity — an INJECTED measurement, no hardware.
type WarmupBlockCapacity struct {
	// FittedBlocks is the measured count of KV blocks that fit during warmup. A
	// zero or negative value fails the derive closed (no room measured).
	FittedBlocks int64
	// BlockTokens is the number of tokens one KV block holds. A zero or negative
	// value fails the derive closed (no divisor).
	BlockTokens int64
}

// DeriveTokenBudget turns the block-measured warmup probe into an admission
// token budget: (FittedBlocks − reserved blocks) × BlockTokens, with the safety
// reserve withheld from the block count first. Monotone in FittedBlocks; a
// bigger reserve shrinks it; a zero/negative count or block size fails closed; a
// count the reserve rounds below one whole block fails closed with
// ReasonBelowOneUnit. Deterministic; no hardware, no clock.
func (c WarmupBlockCapacity) DeriveTokenBudget(fraction float64) DerivedBudget {
	// The measured unit is one block, worth BlockTokens tokens; one block is the
	// smallest divisible unit, so per-unit is 1 block.
	return deriveBudget(c.FittedBlocks, 1, c.BlockTokens, fraction)
}
