// Package usagepreflight decides whether an outbound provider request may spend
// the selected seat's quota. It is deliberately provider-neutral: providers
// that expose usage implement Reader without coupling admission to 429 parsing.
//
// Invariant: usage preflight policy decisions are fail-closed and deterministic.
// Guard: outbound admission never permits spend if reserve boundaries are violated
// or alternate routing options are exhausted.
// Precondition: caller provides a valid seat identifier and context.
// Postcondition: exactly one seat is selected or spend is safely refused before network dispatch.
package usagepreflight
