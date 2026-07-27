package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// `fak route --place --serving FILE` — handing the ladder a liveness snapshot
// (epic #5416, track H).
//
// The serving gate inside modelroute is complete and, until this file, unreachable.
// `PlaceWithServing` walks past candidates an operator's probes say are not answering,
// and `Place` delegates to it with an EMPTY report — so on every fleet alive today the
// gate is a structural no-op, because nothing in the repository produces a report and no
// surface accepted one. A gate nobody can hand an input to cannot be trusted on the day
// it matters; it can only be read.
//
//	fak route --accounts FILE --place --labels work_class=normal-impl --serving serving.json
//	fak route --accounts FILE --place --labels work_class=ultra-hard \
//	          --spawn-type explore --serving serving.json
//
// A file is the honest first producer, the same rung `--evidence` and `--outcomes` took
// for capability: an operator's own probe script writes what it saw, and fak places
// against it. It is deliberately not a probe of fak's own — placement must stay pure
// (same roster, same candidates, same report => same placement), and a CLI that dialled
// endpoints would be reading a clock and a network inside a decision that is asserted
// elsewhere to read neither.
//
// WHAT THIS SURFACE ADDS BEYOND LOADING A FILE is the join. A snapshot and a roster are
// validated apart and only ever meet here:
//
//   - `ServingReport.Validate` cannot see the roster, so it cannot know that an
//     observation names a model nothing binds.
//   - the ladder never asks about a model that is not a candidate, so an observation
//     under a misspelled id is honored *nowhere* — it gates nothing, silently, and the
//     run is byte-identical to one with no snapshot at all.
//
// The likeliest way to get there is filing a probe result under the model's UPSTREAM name
// (what the script dialled) instead of the routed id the roster binds — and that one typo
// fails in OPPOSITE directions depending on coverage, which is why both halves are
// reported rather than just the first:
//
//   - on a rung the snapshot does NOT cover, it fails OPEN. The observation gates
//     nothing, silence about the real candidate gates nothing, and the dead host keeps
//     taking work in a run byte-identical to one with no snapshot at all.
//   - INSIDE declared coverage it fails closed instead, and confusingly: the real
//     candidate is now silent on a rung the report claims to speak for, so rule 1 passes
//     it over as unknown. The operator reads a wall of `zone-serving-unknown` about hosts
//     that are up, with nothing on the ladder connecting it to the id they mistyped.
//
// Neither is legible from the ladder alone, so the summary names both: an observation
// nothing binds, and a covered candidate nothing observed. The same host appearing in
// both lines is the signature of exactly this mistake.
//
// The third shape only visible here is a report that declares a freshness bound with no
// as-of stamp to measure ages against — every observation unshowably fresh, so every
// covered candidate is skipped as stale. Fail-closed and correct; it just reads like a
// total outage.

// servingSummary is what the join found: the snapshot's own shape, and the three ways it
// can be at odds with the roster it is about to gate.
//
// It rides in `--json` under a pointer field for the same reason the spawn report does —
// an absent key means no snapshot was supplied, which must not be confusable with one
// that observed nothing.
type servingSummary struct {
	Path          string                     `json:"path"`
	AsOfUnix      int64                      `json:"as_of_unix,omitempty"`
	MaxAgeSeconds int64                      `json:"max_age_seconds,omitempty"`
	Covers        []modelroute.PlacementZone `json:"covers,omitempty"`
	// Observations is how many models the snapshot speaks about at all.
	Observations int `json:"observations"`
	// Gating is how many of those name a model this roster BINDS — the only ones the
	// ladder can ever consult, and therefore the real reach of the gate.
	Gating int `json:"gating"`
	// UnboundModels are observations under a model id the roster does not bind. Each
	// one is a probe result that will be honored nowhere; see the file header.
	UnboundModels []string `json:"unbound_models,omitempty"`
	// UnboundUpstream maps an unbound observation to the candidate(s) whose UPSTREAM
	// name it is. It turns the diagnosis into the fix for the one way this mistake
	// almost always happens: the probe filed its result under the name it dialled
	// rather than the routed id the ladder asks about. More than one candidate can
	// share an upstream name — two accounts serving the same weights — and that case
	// is worth naming rather than collapsing, because it is the reason a snapshot
	// cannot be keyed by upstream name at all: one observation cannot speak for two
	// hosts.
	UnboundUpstream map[string][]string `json:"unbound_upstream,omitempty"`
	// SilentModels are candidates on a rung the snapshot CLAIMS to speak for
	// and then says nothing about. Silence inside coverage is not health — those
	// candidates are passed over as unknown, so this is the set the snapshot removes
	// from the ladder without anything being reported down.
	SilentModels []string `json:"silent_models,omitempty"`
	// UncheckableFreshness is a report that declares a freshness bound and carries no
	// as-of stamp to measure ages against. Valid, and fail-closed by design: NO
	// observation in it can be shown fresh, so every covered candidate is skipped as
	// stale. It reads on the ladder exactly like a total outage.
	UncheckableFreshness bool `json:"uncheckable_freshness,omitempty"`
}

// loadServingReport reads an operator's liveness snapshot.
//
// Unknown fields are REFUSED rather than ignored, and that is the load-bearing one: a
// snapshot whose bound is misspelled `max_age_sec` still parses, still validates, and
// still looks like a well-formed report — with fail-closed freshness silently switched
// off. The same reading applies to every other key, so the whole shape is closed.
//
// A second JSON document after the first is refused for the neighbouring reason: two
// concatenated snapshots decode to the first one alone, and the operator would be gating
// on a report they can see is not the one they meant to hand over.
//
// A named file that cannot be read is an error, never an empty report. Absence of the
// FLAG means "no liveness signal"; absence of a file the operator NAMED means their
// probe did not run, and quietly placing work as if every host were healthy is the
// failure the gate exists to prevent.
func loadServingReport(path string) (modelroute.ServingReport, error) {
	var rep modelroute.ServingReport
	f, err := os.Open(path)
	if err != nil {
		return rep, fmt.Errorf("--serving %s: %w (a named snapshot that cannot be read is not an empty one; drop the flag to place with no liveness signal)", path, err)
	}
	defer f.Close()
	dec := json.NewDecoder(f)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&rep); err != nil {
		return modelroute.ServingReport{}, fmt.Errorf("--serving %s: %w (fields: schema, as_of_unix, max_age_seconds, covers, models)", path, err)
	}
	if dec.More() {
		return modelroute.ServingReport{}, fmt.Errorf("--serving %s: the file carries more than one JSON document; only the first would gate anything, so the rest would be silently discarded", path)
	}
	if err := rep.Validate(); err != nil {
		return modelroute.ServingReport{}, fmt.Errorf("--serving %s: %w", path, err)
	}
	return rep, nil
}

// summarizeServing joins the snapshot against the pool the ladder is about to walk.
//
// The roster is needed as well as the pool because a candidate's RUNG is a property of
// the account its model binds to, and coverage is declared per rung. A candidate whose
// binding does not resolve is left out of the coverage arithmetic rather than guessed
// at — Place is already fail-loud about it a moment later, and inventing a zone here
// would report a silence on a rung nobody established the candidate was on.
func summarizeServing(path string, rep modelroute.ServingReport, candidates []modelroute.Candidate, r modelroute.Roster) servingSummary {
	s := servingSummary{
		Path:          path,
		AsOfUnix:      rep.AsOfUnix,
		MaxAgeSeconds: rep.MaxAgeSeconds,
		Covers:        rep.Covers,
		Observations:  len(rep.Models),
		// Fail-closed freshness with nothing to measure against: see the field's doc.
		UncheckableFreshness: rep.MaxAgeSeconds > 0 && rep.AsOfUnix <= 0,
	}
	bound := make(map[string]bool, len(candidates))
	// upstream indexes the pool by the name each candidate is DIALLED under, which is
	// what a probe script has in hand and the ladder never asks about.
	upstream := map[string][]string{}
	for _, c := range candidates {
		bound[c.Model] = true
		t, err := r.Resolve(c.Model)
		if err != nil {
			continue
		}
		if u := strings.TrimSpace(t.UpstreamModel); u != "" {
			upstream[u] = append(upstream[u], c.Model)
		}
	}
	for model := range rep.Models {
		if bound[model] {
			s.Gating++
			continue
		}
		s.UnboundModels = append(s.UnboundModels, model)
		if hits := upstream[model]; len(hits) > 0 {
			if s.UnboundUpstream == nil {
				s.UnboundUpstream = map[string][]string{}
			}
			sort.Strings(hits)
			s.UnboundUpstream[model] = hits
		}
	}
	covered := map[modelroute.PlacementZone]bool{}
	for _, z := range rep.Covers {
		covered[z] = true
	}
	for _, c := range candidates {
		if _, observed := rep.Models[c.Model]; observed {
			continue
		}
		t, err := r.Resolve(c.Model)
		if err != nil {
			continue
		}
		if covered[t.Zone()] {
			s.SilentModels = append(s.SilentModels, c.Model)
		}
	}
	sort.Strings(s.UnboundModels)
	sort.Strings(s.SilentModels)
	return s
}

// printServingSnapshot renders the join for a human, under the placement it explains.
func printServingSnapshot(w io.Writer, s servingSummary) {
	fmt.Fprintf(w, "\nSERVING SNAPSHOT  (--serving %s)\n", s.Path)
	asOf := "(none — no clock to measure freshness against)"
	if s.AsOfUnix > 0 {
		asOf = fmt.Sprintf("%d", s.AsOfUnix)
	}
	bound := "(none declared — observations honored at any age)"
	if s.MaxAgeSeconds > 0 {
		bound = fmt.Sprintf("%ds", s.MaxAgeSeconds)
	}
	fmt.Fprintf(w, "  as of        %s      max age  %s\n", asOf, bound)
	covers := "(nothing — silence about a candidate gates nothing anywhere)"
	if len(s.Covers) > 0 {
		zones := make([]string, 0, len(s.Covers))
		for _, z := range s.Covers {
			zones = append(zones, string(z))
		}
		covers = strings.Join(zones, " ")
	}
	fmt.Fprintf(w, "  covers       %s\n", covers)
	fmt.Fprintf(w, "  observations %d, of which %d name a model this roster binds\n", s.Observations, s.Gating)
	if len(s.UnboundModels) > 0 {
		fmt.Fprintf(w, "  UNBOUND      %s\n", strings.Join(s.UnboundModels, " "))
		fmt.Fprintln(w, "               these observations gate NOTHING: the ladder only ever asks about")
		fmt.Fprintln(w, "               models the roster binds, so a probe filed under an id nothing binds")
		fmt.Fprintln(w, "               is honored nowhere and this run is identical to one with no snapshot.")
		fmt.Fprintln(w, "               Check the ids against the roster's bindings, not its upstream names.")
		hinted, width := 0, 0
		for _, m := range s.UnboundModels {
			if len(s.UnboundUpstream[m]) > 0 {
				hinted++
				if len(m) > width {
					width = len(m)
				}
			}
		}
		if hinted > 0 {
			fmt.Fprintln(w, "               Each of these is the UPSTREAM name of a candidate: the probe filed its")
			fmt.Fprintln(w, "               result under the name it dialled, and the ladder asks by routed id.")
			for _, m := range s.UnboundModels {
				hits := s.UnboundUpstream[m]
				if len(hits) == 0 {
					continue
				}
				fmt.Fprintf(w, "                 %-*s  ->  %s\n", width, m, strings.Join(hits, " "))
				if len(hits) > 1 {
					fmt.Fprintln(w, "               several accounts serve those weights, so one observation cannot")
					fmt.Fprintln(w, "               speak for all of them — observe each routed id separately.")
				}
			}
		}
	}
	if len(s.SilentModels) > 0 {
		fmt.Fprintf(w, "  SILENT       %s\n", strings.Join(s.SilentModels, " "))
		fmt.Fprintln(w, "               the snapshot claims to speak for these candidates' rungs and then says")
		fmt.Fprintln(w, "               nothing about them, so they are passed over as unknown rather than")
		fmt.Fprintln(w, "               assumed well. That is a gap in the probe, not an outage.")
	}
	if s.UncheckableFreshness {
		fmt.Fprintln(w, "  STALE-ALL    a freshness bound is declared and there is no as-of stamp to measure")
		fmt.Fprintln(w, "               ages against, so NO observation here can be shown fresh and every")
		fmt.Fprintln(w, "               covered candidate is skipped as stale. Fail-closed and working as")
		fmt.Fprintln(w, "               designed — it just reads like a total outage. Stamp as_of_unix.")
	}
}
