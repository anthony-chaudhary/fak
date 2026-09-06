// Package worktype defines the canonical work classes.
// This file bridges the worktype (work) and choicetriage (choices) taxonomies
// to unify the shared idea of fleet-owned work vs human-owned residual decisions
// and prevent taxonomy drift across the project.
package worktype

import (
	"errors"
	"strings"
)

// Discrete is an alias for DiscreteEpic to align the work taxonomy with choice
// triage projections.
const Discrete Class = DiscreteEpic

// Choice disposition constants matching the closed choicetriage vocabulary.
const (
	ChoiceTakeObvious   = "TAKE_OBVIOUS"
	ChoiceFreshContext  = "FRESH_CONTEXT"
	ChoiceFileTicket    = "FILE_TICKET"
	ChoiceHumanResidual = "HUMAN_RESIDUAL"
)

var (
	// ErrContradictoryOwnership indicates a choice disposition contradicts the ownership
	// model of a work class (e.g. HUMAN_RESIDUAL assigned to an automated optimization program).
	ErrContradictoryOwnership = errors.New("worktype: contradictory ownership between choice disposition and work class")

	// ErrUnknownDisposition indicates the choice disposition is not recognized by the taxonomy.
	ErrUnknownDisposition = errors.New("worktype: unknown choice disposition")

	// ErrUnknownWorkClass indicates the work class is not recognized by the taxonomy.
	ErrUnknownWorkClass = errors.New("worktype: unknown work class")
)

// ChoiceDispositions is the closed list of recognized choice dispositions.
var ChoiceDispositions = []string{
	ChoiceTakeObvious,
	ChoiceFreshContext,
	ChoiceFileTicket,
	ChoiceHumanResidual,
}

// ValidChoiceDisposition reports whether disposition is one of the closed choice dispositions.
func ValidChoiceDisposition(disposition string) bool {
	switch strings.ToUpper(strings.TrimSpace(disposition)) {
	case ChoiceTakeObvious, ChoiceFreshContext, ChoiceFileTicket, ChoiceHumanResidual:
		return true
	default:
		return false
	}
}

// Valid reports whether c is a recognized work class.
func (c Class) Valid() bool {
	switch c {
	case KernelOptimization, CacheOptimization, HumanOperatorEffectiveness, DiscreteEpic:
		return true
	default:
		return false
	}
}

// MapChoiceDisposition projects a choicetriage disposition token onto the canonical work Class.
// The four closed choicetriage dispositions are mapped as:
//   - "TAKE_OBVIOUS" -> Discrete, true
//   - "FRESH_CONTEXT" -> Discrete, true
//   - "FILE_TICKET" -> Discrete, true
//   - "HUMAN_RESIDUAL" -> HumanOperatorEffectiveness, true
//
// For any unknown disposition, it returns ("", false).
func MapChoiceDisposition(disposition string) (Class, bool) {
	switch strings.ToUpper(strings.TrimSpace(disposition)) {
	case ChoiceTakeObvious:
		return Discrete, true
	case ChoiceFreshContext:
		return Discrete, true
	case ChoiceFileTicket:
		return Discrete, true
	case ChoiceHumanResidual:
		return HumanOperatorEffectiveness, true
	default:
		return "", false
	}
}

// IsContradictoryOwnership reports whether a choice disposition contradicts the ownership
// model of a work class. Specifically, HUMAN_RESIDUAL contradicts automated optimization
// programs (KernelOptimization and CacheOptimization).
func IsContradictoryOwnership(disposition string, class Class) bool {
	d := strings.ToUpper(strings.TrimSpace(disposition))
	return d == ChoiceHumanResidual && (class == KernelOptimization || class == CacheOptimization)
}

// ValidateTaxonomyAlignment checks that a choice disposition and a work class do not
// contradict each other. Specifically:
//   - disposition must be a recognized choice disposition (TAKE_OBVIOUS, FRESH_CONTEXT, FILE_TICKET, HUMAN_RESIDUAL).
//   - class must be a valid work class (KernelOptimization, CacheOptimization, HumanOperatorEffectiveness, DiscreteEpic/Discrete).
//   - HUMAN_RESIDUAL must not be mapped to automated optimization programs (KernelOptimization or CacheOptimization),
//     returning ErrContradictoryOwnership.
func ValidateTaxonomyAlignment(disposition string, class Class) error {
	d := strings.ToUpper(strings.TrimSpace(disposition))
	if !ValidChoiceDisposition(d) {
		return ErrUnknownDisposition
	}

	if !class.Valid() {
		return ErrUnknownWorkClass
	}

	if IsContradictoryOwnership(d, class) {
		return ErrContradictoryOwnership
	}

	return nil
}
