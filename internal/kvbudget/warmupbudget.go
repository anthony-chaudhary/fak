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
	// block) — the fitted room rounds down to a zero admittable budget.
	ReasonBelowOneUnit Reason = "measured_below_one_unit"
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
	// Zero on fail-closed.
	KeptAmount int64
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
	// negative value fails the derive closed (no divisor).
	BytesPerToken int64
}

// DeriveTokenBudget turns the byte-measured warmup probe into an admission token
// budget: floor((UsableBytes − reserve) / BytesPerToken), with the safety
// reserve withheld first. A larger measurement yields a proportionally larger
// budget (monotone); a bigger reserve fraction shrinks it; a zero/negative
// measurement or unit fails closed to a zero, typed-reason budget; a positive
// measurement that holds under one token's bytes after the reserve fails closed
// with ReasonBelowOneUnit. Deterministic; no hardware, no clock.
func (c WarmupCapacity) DeriveTokenBudget(fraction float64) DerivedBudget {
	// One token is the unit here, so tokens-per-unit is 1.
	return deriveBudget(c.UsableBytes, c.BytesPerToken, 1, fraction)
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
