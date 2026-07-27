package dispatchtick

import (
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// THE ON-DISK GRAMMAR FOR THE TWO FACTS A FINISHED SLOT CANNOT RECONSTRUCT.
//
// A witness sweep runs long after a worker exited. It can still read the log, the commit,
// and the test result — those are durable. Two things it CANNOT recover are exactly the two
// a capability grade needs:
//
//   - WHICH RUNG served the slot. The roster may have been edited since; a model id
//     re-bound from a laptop to a vendor account would silently re-attribute finished work.
//   - WHAT CLASS of work it was. Labels are mutable. An issue re-tagged tier/T0 after a
//     cheap model shipped it would hand that model frontier-tier evidence it never earned.
//
// Both are known at SPAWN and only at spawn, so both are written beside the log then, next
// to the .model / .account / .wave sidecars the tick already writes. The sweep reads back a
// point-in-time record rather than re-deriving a present-tense answer about a past event.
//
// Reading back is where a grammar is needed, because a sidecar is a file: it can be
// truncated by a crash mid-write, hand-edited, or left over from an older schema. Both
// parsers below therefore ALLOWLIST — they return the value only if it is one this package
// could itself have produced — and both answer with ok=false rather than a default, so an
// unreadable sidecar costs evidence instead of inventing it.
//
// The class allowlist carries a second, sharper job: it is derived from bucketWorkClass's
// values, which deliberately exclude modelroute.ClassSecurityRelease. That class's floor is
// what stops a cheap model serving push/delete/release work, and no dispatch label can name
// it (see workclass.go). Deriving the allowlist from the same table means a hand-written
// `.workclass` file holding "security-release" is refused by construction, not by a
// blocklist someone must remember to update.

// WorkClassSidecarSuffix records the work class a slot's outcome may be graded under,
// resolved at spawn from the issue's tier labels by the same parser that chose the slot's
// launch profile. Absent when nothing declared the class — an untriaged issue and a
// coordination slot both write no sidecar, which the fold counts as ungraded.
const WorkClassSidecarSuffix = ".workclass"

// SidecarPath is the one place a sidecar's path is derived from a worker log path: the log
// with its extension replaced by suffix. An empty log path yields an empty path so a caller
// with nothing to key on cannot write a bare ".zone" into the runs directory.
//
// filepath.Ext stops at the last path separator, so a dotted DIRECTORY (a run folder named
// "wave-1.2") cannot eat the stem — the property a hand-rolled strings.LastIndex(".") would
// get wrong, silently collapsing every slot in that wave onto one sidecar.
func SidecarPath(logPath, suffix string) string {
	logPath = strings.TrimSpace(logPath)
	if logPath == "" {
		return ""
	}
	return strings.TrimSuffix(logPath, filepath.Ext(logPath)) + suffix
}

// ZoneFromSidecar parses a .zone sidecar's bytes into a placement rung. ok is false for
// anything that is not one of the three known rungs, including empty and truncated content
// — never the device rung, which is the one wrong answer that would inflate the
// self-hosted share.
func ZoneFromSidecar(raw string) (modelroute.PlacementZone, bool) {
	z := modelroute.PlacementZone(strings.TrimSpace(raw))
	if !z.Valid() {
		return "", false
	}
	return z, true
}

// WorkClassFromSidecar parses a .workclass sidecar's bytes into a work class, accepting
// ONLY a class some launch bucket maps to. ok is false otherwise, so an unknown or edited
// value grades nothing rather than reaching modelroute.PolicyFor — which maps an
// unrecognized class to the T0 floor and would read the corruption as frontier-tier work.
func WorkClassFromSidecar(raw string) (modelroute.WorkClass, bool) {
	c := modelroute.WorkClass(strings.TrimSpace(raw))
	if c == "" {
		return "", false
	}
	for _, allowed := range bucketWorkClass {
		if c == allowed {
			return c, true
		}
	}
	return "", false
}

// ClassResolverFromSidecars adapts the classes a sweep scraped off disk — keyed by the
// worker log path each WitnessRecord already carries — into the Class hook
// TurnOutcomesFromWitness takes.
//
// A record with no entry classes empty, which is the honest answer for a slot whose class
// nothing declared, for one spawned before this sidecar existed, and for one whose sidecar
// failed the allowlist. All three cost evidence; none of them invent it.
func ClassResolverFromSidecars(byLog map[string]modelroute.WorkClass) func(WitnessRecord) modelroute.WorkClass {
	return func(r WitnessRecord) modelroute.WorkClass {
		if len(byLog) == 0 {
			return ""
		}
		return byLog[r.Log]
	}
}
