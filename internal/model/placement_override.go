package model

import (
	"fmt"
	"regexp"
	"strings"
)

// placement_override.go — user-supplied per-tensor placement override for the CPU-offload
// split (the llama.cpp `-ot` / `--override-tensor` equivalent).
//
// The built-in split predicate (isExpertWeight, moe_offload.go) is all-or-nothing: it can only
// ever move the MoE experts to host RAM. An operator whose attention almost fits VRAM but wants
// one specific projection — or a non-standard tensor class in a new arch — pinned to host has no
// dial short of a Go code change. This file is that dial: an ORDERED, FIRST-MATCH-WINS list of
// {name-regex -> home} rules, consulted BEFORE the built-in default. It is a pure placement
// decision over tensor NAMES — no GPU, no network, no clock — so it introduces no new arithmetic;
// the only thing it can get wrong is the routing, which the witnesses pin directly.

// placementRule is one ordered override entry: a compiled tensor-name pattern and the home a
// matching tensor is pinned to (onHost true => host RAM, false => the device backend).
type placementRule struct {
	pattern *regexp.Regexp
	onHost  bool
}

// placementOverride is an ordered, first-match-wins list of name-regex rules. The zero value is an
// empty override that matches nothing (every tensor falls through to the built-in default).
type placementOverride struct {
	rules []placementRule
}

// resolve applies the ordered rules to a tensor name, FIRST-MATCH-WINS: it returns (home, true) for
// the first rule whose pattern matches, and (false, false) when no rule matches. Earlier rules
// shadow later ones, mirroring llama.cpp's `std::regex_search` loop with its first-match `break`.
func (o placementOverride) resolve(name string) (onHost bool, matched bool) {
	for _, r := range o.rules {
		if r.pattern.MatchString(name) {
			return r.onHost, true
		}
	}
	return false, false
}

// onHostWith folds this override IN FRONT of a fallback predicate, yielding the composite
// onHost(name) that splitKernel wants: the override's first match decides, otherwise the fallback
// (e.g. isExpertWeight) decides. An empty override returns the fallback unchanged, so wiring it in
// is a no-op until the operator actually supplies a rule.
func (o placementOverride) onHostWith(fallback func(string) bool) func(string) bool {
	if len(o.rules) == 0 {
		return fallback
	}
	return func(name string) bool {
		if onHost, matched := o.resolve(name); matched {
			return onHost
		}
		return fallback(name)
	}
}

// overrideParseError is the typed, fail-closed failure parsePlacementOverride returns for a
// malformed spec — a bad regex, an empty pattern, a missing or unknown target. Callers can match on
// it to distinguish operator-input errors from other failures; the parse NEVER silently drops a
// rule, so a typo cannot degrade into an unintended placement.
type overrideParseError struct {
	entry  string
	reason string
}

func (e *overrideParseError) Error() string {
	return fmt.Sprintf("placement override %q: %s", e.entry, e.reason)
}

// parsePlacementOverride parses a `pattern=target;pattern=target;...` spec (the CLI string form)
// into an ordered override, preserving entry order so first-match-wins reflects the operator's
// stated precedence. target is one of host|cpu|ram (pin to host RAM) or device|gpu (keep on the
// device). An empty or whitespace-only spec yields the empty override with a nil error. Any bad
// regex, empty pattern, or unknown/missing target fails closed with *overrideParseError and NO
// partial override.
func parsePlacementOverride(spec string) (placementOverride, error) {
	var o placementOverride
	if strings.TrimSpace(spec) == "" {
		return o, nil
	}
	for _, entry := range strings.Split(spec, ";") {
		entry = strings.TrimSpace(entry)
		if entry == "" {
			continue
		}
		eq := strings.LastIndex(entry, "=")
		if eq < 0 {
			return placementOverride{}, &overrideParseError{entry: entry, reason: "missing '=<target>'"}
		}
		pat := strings.TrimSpace(entry[:eq])
		tgt := strings.TrimSpace(entry[eq+1:])
		if pat == "" {
			return placementOverride{}, &overrideParseError{entry: entry, reason: "empty name pattern"}
		}
		onHost, ok := parsePlacementTarget(tgt)
		if !ok {
			return placementOverride{}, &overrideParseError{entry: entry, reason: fmt.Sprintf("unknown target %q (want host|cpu|ram|device|gpu)", tgt)}
		}
		re, err := regexp.Compile(pat)
		if err != nil {
			return placementOverride{}, &overrideParseError{entry: entry, reason: fmt.Sprintf("bad regex: %v", err)}
		}
		o.rules = append(o.rules, placementRule{pattern: re, onHost: onHost})
	}
	return o, nil
}

// parsePlacementTarget maps a target token to a home. host|cpu|ram => onHost true; device|gpu =>
// onHost false. Any other token is rejected (ok=false) so the parse fails closed rather than
// guessing a placement for an unrecognized destination.
func parsePlacementTarget(tgt string) (onHost bool, ok bool) {
	switch strings.ToLower(tgt) {
	case "host", "cpu", "ram":
		return true, true
	case "device", "gpu":
		return false, true
	default:
		return false, false
	}
}
