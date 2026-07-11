package fleetpane

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/choicetriage"
)

// health_triage.go applies the decenter-the-human doctrine at the fleet-pane
// worker-health seam. The pane's "replacement-needed" rollup (health.go) sums every
// worker the launcher MAY replace — dead, stale-transcript, auth-or-rate-blocked —
// into one number an operator reads as "workers you must relaunch." But those are
// not one job: the launcher auto-relaunches a dead or stale worker with no person in
// the loop, whereas an auth/rate-walled worker just re-hits its wall on relaunch —
// only a person re-authing the account clears it. And the class literally named
// "attention" is the monitor asking for a look, which under the doctrine is a
// fresh-context evaluation, not a reflexive page. Rendering all of them under one
// rollup trains an operator to babysit work the fleet already owns.
//
// PartitionWorkerHealth folds each non-healthy class's own remedy through
// internal/choicetriage and splits the operator-facing workers: only a
// HUMAN_RESIDUAL disposition (the account auth wall) waits on a person; the
// relaunchable and inspect-in-fresh-context classes are the fleet's to clear. It is
// the fleet-pane analogue of the resume-stopped and report-gate folds, and it soaks
// behind FAK_FLEETPANE_TRIAGE_GATE (read at the CLI): the default readout is
// unchanged until enforce, so the split is observable before it changes what an
// operator is told to do.

// healthOKClasses are the worker classes that are not an operator action at all: a
// healthy worker and one that finished cleanly. Every other class carries a remedy
// the fold triages, so these never appear in either partition bucket.
var healthOKClasses = map[string]bool{"healthy": true, "completed-final": true}

// workerClassRemedy maps a monitor worker-health class to the concrete remedy the
// triage fold reasons about, phrased as a choicetriage.Signal Action. A class the
// launcher relaunches names a runnable command (a fleet action -> TakeObvious); the
// auth-or-rate wall names account authority a person holds (-> HumanResidual); a
// class with no remedy in hand (the bare "attention" flag, or a class this build
// does not know) folds to a fresh-context evaluation rather than a page. The auth
// class is deliberately conservative: it conflates an auth wall (a person's re-auth)
// with a rate wall (self-clearing), and since a relaunch cannot tell them apart and
// re-hits an auth wall, the fold surfaces the whole class as the operator's — the
// same fail-toward-paging posture the resume-stopped fold takes for STOPPED_AUTH.
var workerClassRemedy = map[string]string{
	"dead":                 "`fak fleet` relaunches the dead worker",
	"stale-transcript":     "`fak fleet` relaunches the stale worker",
	"auth-or-rate-blocked": "re-authenticate the blocked account; a relaunch just re-hits the auth wall",
}

// WorkerBucket is one non-healthy class's operator-facing count plus the concrete
// next move the triage fold named for it. Order within a bucket follows the pane's
// canonical class order (healthClassOrder), then any unknown class sorted.
type WorkerBucket struct {
	Class   string `json:"class"`
	Count   int    `json:"count"`
	Resolve string `json:"resolve"`
}

// triageWorkerClass folds one worker-health class through the shared page-vs-act
// gate. The class name is the Detail (so "auth-or-rate-blocked" carries the AUTH
// authority signal), and its remedy is the Action a runnable relaunch surfaces.
func triageWorkerClass(class string) choicetriage.Verdict {
	return choicetriage.Triage(choicetriage.Signal{
		Source:   "fleetpane",
		Question: "worker class " + class,
		Detail:   class,
		Action:   workerClassRemedy[class],
	})
}

// orderedNonHealthyClasses returns the classes with a non-zero count that are an
// operator action (i.e. not healthy/completed), in the pane's canonical readout
// order first, then any unknown class sorted — mirroring WorkerHealthText so the
// split and the base health line cannot drift in ordering.
func orderedNonHealthyClasses(counts map[string]int) []string {
	out := make([]string, 0, len(counts))
	known := make(map[string]bool, len(healthClassOrder))
	for _, c := range healthClassOrder {
		known[c] = true
		if counts[c] > 0 && !healthOKClasses[c] {
			out = append(out, c)
		}
	}
	extras := make([]string, 0)
	for c, n := range counts {
		if !known[c] && n > 0 && !healthOKClasses[c] {
			extras = append(extras, c)
		}
	}
	sort.Strings(extras)
	return append(out, extras...)
}

// PartitionWorkerHealth splits the pane's non-healthy worker classes into the ones
// that genuinely wait on a person (the account auth wall) and the ones the fleet
// clears itself (a launcher relaunch or a fresh-context inspection). Pure and total:
// only classes with a non-zero count appear, healthy/completed workers never do, and
// an unavailable health summary (all-zero counts) yields two empty buckets.
func PartitionWorkerHealth(h WorkerHealth) (needHuman, fleetClears []WorkerBucket) {
	for _, class := range orderedNonHealthyClasses(h.Counts) {
		v := triageWorkerClass(class)
		b := WorkerBucket{Class: class, Count: h.Counts[class], Resolve: v.Resolve}
		if v.NeedsHuman {
			needHuman = append(needHuman, b)
		} else {
			fleetClears = append(fleetClears, b)
		}
	}
	return needHuman, fleetClears
}

// FleetTriageEnforced reports whether the decenter split is active for the given
// mode string. Only "enforce" (case-insensitive) flips the render; "warn", "" and
// anything else leave the pane's single replacement-needed rollup unchanged so the
// change can soak. Mirrors the enforce/warn switch every other seam reads.
func FleetTriageEnforced(mode string) bool {
	return strings.EqualFold(strings.TrimSpace(mode), "enforce")
}

// WorkerHealthTriageLine renders the decenter split as one extra readout line: the
// worker count that genuinely waits on a person (needs-you) vs the count the fleet
// clears itself (fleet-clears), each with its per-class breakdown. It returns "" for
// an unavailable or all-clear health summary (no witness, or nothing to split), so
// the caller appends nothing. Rendered only under FAK_FLEETPANE_TRIAGE_GATE=enforce.
func WorkerHealthTriageLine(h WorkerHealth) string {
	if !h.Available {
		return ""
	}
	needHuman, fleetClears := PartitionWorkerHealth(h)
	if len(needHuman) == 0 && len(fleetClears) == 0 {
		return ""
	}
	return "health-triage: needs-you=" + strconv.Itoa(sumBuckets(needHuman)) + fmtBucketClasses(needHuman) +
		" fleet-clears=" + strconv.Itoa(sumBuckets(fleetClears)) + fmtBucketClasses(fleetClears)
}

// sumBuckets totals the worker counts across a partition bucket.
func sumBuckets(bs []WorkerBucket) int {
	total := 0
	for _, b := range bs {
		total += b.Count
	}
	return total
}

// fmtBucketClasses renders " (class=count,...)" for a non-empty bucket, or "" when
// the bucket is empty, so the readout names which classes landed on each side.
func fmtBucketClasses(bs []WorkerBucket) string {
	if len(bs) == 0 {
		return ""
	}
	parts := make([]string, 0, len(bs))
	for _, b := range bs {
		parts = append(parts, b.Class+"="+strconv.Itoa(b.Count))
	}
	return " (" + strings.Join(parts, ",") + ")"
}

// TriageSelfcheck is the deterministic, no-I/O proof of the fleet-pane fold: an
// auth/rate-blocked worker waits on a person, dead/stale workers are the launcher's
// relaunch, the monitor's "attention" flag is a fresh-context evaluation (not a
// page), and PartitionWorkerHealth keeps only the auth wall on the human side while
// healthy/completed workers appear in neither bucket. It is the witness the fleet
// CLI surfaces as `fak fleetpane selfcheck`.
func TriageSelfcheck() error {
	h := WorkerHealth{Available: true, Counts: map[string]int{
		"healthy":              3,
		"completed-final":      2,
		"dead":                 2,
		"stale-transcript":     1,
		"auth-or-rate-blocked": 1,
		"attention":            1,
	}}
	needHuman, fleetClears := PartitionWorkerHealth(h)
	if len(needHuman) != 1 || needHuman[0].Class != "auth-or-rate-blocked" {
		return fmt.Errorf("only the account auth wall must wait on a person, got %v", needHuman)
	}
	if !needHuman[0].needsAuthorityResolve() {
		return fmt.Errorf("the auth wall's resolve must name the person's move, got %q", needHuman[0].Resolve)
	}
	// dead, stale-transcript, attention -> the fleet clears (relaunch or evaluate);
	// healthy + completed-final are not an operator action and appear in neither bucket.
	if got := bucketClasses(fleetClears); got != "attention,dead,stale-transcript" {
		return fmt.Errorf("dead+stale+attention must be the fleet's to clear, got %q", got)
	}
	// A measured all-healthy pane splits into nothing (no page, no fleet churn).
	if line := WorkerHealthTriageLine(WorkerHealth{Available: true, Counts: map[string]int{"healthy": 5, "completed-final": 1}}); line != "" {
		return fmt.Errorf("an all-healthy pane must not surface a triage split, got %q", line)
	}
	// An unavailable pane is an honest no-witness, not an all-zero split.
	if line := WorkerHealthTriageLine(WorkerHealth{Available: false}); line != "" {
		return fmt.Errorf("an unavailable health summary must not render a triage line, got %q", line)
	}
	if !FleetTriageEnforced("enforce") || FleetTriageEnforced("") || FleetTriageEnforced("warn") {
		return fmt.Errorf("FleetTriageEnforced must flip only on \"enforce\"")
	}
	return nil
}

// needsAuthorityResolve reports whether the bucket's resolve names the person's move
// (re-auth), the one authority a launcher relaunch cannot stand in for.
func (b WorkerBucket) needsAuthorityResolve() bool {
	return strings.Contains(strings.ToUpper(b.Resolve), "AUTH")
}

// bucketClasses joins a bucket's class names sorted, for a stable selfcheck compare
// independent of the readout order.
func bucketClasses(bs []WorkerBucket) string {
	names := make([]string, 0, len(bs))
	for _, b := range bs {
		names = append(names, b.Class)
	}
	sort.Strings(names)
	return strings.Join(names, ",")
}
