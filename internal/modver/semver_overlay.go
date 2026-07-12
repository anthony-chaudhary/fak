package modver

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// Declared is an opt-in, hand-declared contract version for a module that has
// graduated from pure derived-rev tracking to a real MAJOR.MINOR.PATCH
// compatibility contract — the handful of surfaces (abi, policy schema, ledger
// schemas) where MAJOR.MINOR semantics carry actual break/compat meaning.
//
// The overlay OVERLAYS the derived rev, it never replaces it. The package
// doctrine (rule 1, "derived, never declared" — see the package doc and
// docs/notes/VERSION-EVERYTHING-SPINE-2026-07-03.md) is that a hand-maintained
// per-module version would rot on this shared trunk; the derived rev stays the
// safety net. A Declared entry is a small, reconciled exception: the declared
// Semver states the contract, while the derived rev still renders as build
// metadata (v1.2.0+r47), so the module's real movement stays visible under the
// declared version and a declared version can never drift free of history.
type Declared struct {
	// Semver is the declared contract version, bare "MAJOR.MINOR.PATCH" (no
	// leading "v", no "+r" metadata — those are added only at render time).
	Semver string
	// SinceRev is the derived rev at which this contract version was cut — the
	// reconciliation anchor. A declared version is legitimate only if the module
	// has actually moved to (at least) SinceRev: declaring a bump the derived
	// history has not reached is the drift Validate rejects, which is what makes
	// "a declared bump must coincide with real movement" a checked fact.
	SinceRev int
}

// Overlay maps a module name (as keyed by moduleOf, e.g. "internal/abi") to its
// declared contract version. It is opt-in and deliberately tiny: every module
// not present here keeps the plain derived r<rev>+g<sha> stamp with no overlay.
type Overlay map[string]Declared

// GraduatedModules is the committed declared overlay: the modules that carry a
// real compatibility contract and have opted in to a declared MAJOR.MINOR.PATCH
// on top of their derived rev. It is validated against live derived history by
// TestGraduatedModulesReconcile — the #2501 witness — so a declaration here can
// never ship ahead of the module's actual movement.
var GraduatedModules = Overlay{
	// internal/abi is the engine ABI surface — a genuine compatibility contract
	// where MAJOR.MINOR breakage matters — so it graduates to a declared 1.0.0
	// contract from its first rev, reconciled against the derived rev at render
	// time (renders v1.0.0+r<rev>). SinceRev 1 anchors the contract at the
	// module's birth: every live module has rev >= 1, so this is the most
	// conservative honest anchor.
	"internal/abi": {Semver: "1.0.0", SinceRev: 1},
}

// parseSemver splits a bare "MAJOR.MINOR.PATCH" into its three non-negative
// integer components. Anything else — wrong field count, non-numeric, negative,
// a leading "v" — is rejected, since the overlay declares bare semver and the
// "v" prefix and "+r<rev>" metadata are added only by Rendered.
func parseSemver(s string) (major, minor, patch int, err error) {
	parts := strings.Split(s, ".")
	if len(parts) != 3 {
		return 0, 0, 0, fmt.Errorf("semver %q: want MAJOR.MINOR.PATCH", s)
	}
	var out [3]int
	for i, p := range parts {
		n, e := strconv.Atoi(p)
		if e != nil || n < 0 {
			return 0, 0, 0, fmt.Errorf("semver %q: field %d (%q) not a non-negative integer", s, i+1, p)
		}
		out[i] = n
	}
	return out[0], out[1], out[2], nil
}

// Rendered returns the overlay render for module m. If m carries a declared
// contract version, it renders "v<semver>+r<rev>" (e.g. v1.2.0+r47) — the
// declared contract with the module's CURRENT derived rev as build metadata, so
// a graduated module's real movement stays visible under its declared version.
// A module with no overlay entry falls back to the plain derived version
// m.Version() (r<rev>+g<sha>).
func (o Overlay) Rendered(m Module) string {
	if d, ok := o[m.Name]; ok {
		return fmt.Sprintf("v%s+r%d", d.Semver, m.Rev)
	}
	return m.Version()
}

// Validate reconciles the overlay against a derived Report and returns every
// violation found (nil if the overlay is clean). The rules make a declared
// version structurally unable to drift free of real movement:
//
//   - Semver must be a well-formed bare MAJOR.MINOR.PATCH.
//   - SinceRev must be positive (a contract is cut at a real rev, and every live
//     module has rev >= 1).
//   - The module must exist in the derived report — an overlay entry for a
//     module the history does not know is a typo or a deleted module.
//   - SinceRev must not exceed the module's derived rev: a declared version can
//     only be anchored at a rev the module has actually reached. Declaring a
//     bump ahead of derived movement is exactly the drift this overlay exists to
//     prevent, so it fails here ("a declared bump must coincide with real
//     movement").
//
// Validate runs at test time against a live Snapshot (the "validated against
// derived history" contract of #2501), so a bad declaration reds the build
// rather than shipping a fictional contract version. Violations are returned in
// deterministic (module-name) order.
func (o Overlay) Validate(rep Report) []error {
	byName := make(map[string]Module, len(rep.Modules))
	for _, m := range rep.Modules {
		byName[m.Name] = m
	}
	names := make([]string, 0, len(o))
	for name := range o {
		names = append(names, name)
	}
	sort.Strings(names)

	var errs []error
	for _, name := range names {
		d := o[name]
		if _, _, _, err := parseSemver(d.Semver); err != nil {
			errs = append(errs, fmt.Errorf("%s: %w", name, err))
		}
		if d.SinceRev < 1 {
			errs = append(errs, fmt.Errorf("%s: SinceRev %d must be >= 1", name, d.SinceRev))
		}
		m, ok := byName[name]
		if !ok {
			errs = append(errs, fmt.Errorf("%s: not in derived history (typo or deleted module)", name))
			continue
		}
		if d.SinceRev > m.Rev {
			errs = append(errs, fmt.Errorf("%s: declared SinceRev %d exceeds derived rev %d (declared ahead of real movement)", name, d.SinceRev, m.Rev))
		}
	}
	return errs
}
