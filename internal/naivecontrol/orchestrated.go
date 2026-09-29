package naivecontrol

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"time"
)

// WitnessSidecar is the part of an orchestrated worker's
// .dispatch-runs/resolve-<N>-<stamp>.witness record (dispatchtick.WitnessRecord)
// the comparison reads. Its sha and claim are the orchestrated pipeline's report
// about itself, so they enter here as claims to re-verify, never as counts: a
// CLAIM_WITNESSED sidecar still has to pass the same ancestry check as a naive
// worker's receipt.
type WitnessSidecar struct {
	Issue int     `json:"issue"`
	Log   string  `json:"log"`
	SHA   *string `json:"sha"`
	Claim string  `json:"claim"`
}

// ParseWitnessSidecar decodes one sidecar.
func ParseWitnessSidecar(b []byte) (WitnessSidecar, error) {
	var w WitnessSidecar
	if err := json.Unmarshal(b, &w); err != nil {
		return w, err
	}
	if w.Issue <= 0 {
		return w, fmt.Errorf("sidecar has no issue number")
	}
	return w, nil
}

// OrchestratedWorker is one exited orchestrated worker: its sidecar and the trunk
// SHA it started from (the dispatcher's .basesha sidecar), "" when absent.
type OrchestratedWorker struct {
	Sidecar WitnessSidecar
	Base    string
}

// OrchestratedPicks maps exited orchestrated workers onto the naive arm's inputs.
// A worker is a pick (the dispatcher handed that issue to a worker), matching the
// naive arm, where each wave issue is a pick. The tick's lane_issue_count is a
// candidate pool the lane picker chose ONE target from, so it is deliberately not
// the counterpart. When several workers took the same issue the first one with a
// base wins, so the widest scan is kept.
func OrchestratedPicks(workers []OrchestratedWorker) ([]Pick, []Claim) {
	idx := map[int]int{}
	var picks []Pick
	var claims []Claim
	for _, w := range workers {
		n := w.Sidecar.Issue
		if n <= 0 {
			continue
		}
		base := strings.TrimSpace(w.Base)
		if i, ok := idx[n]; ok {
			if picks[i].Base == "" && base != "" {
				picks[i].Base = base
			}
		} else {
			idx[n] = len(picks)
			picks = append(picks, Pick{Issue: n, Base: base})
		}
		if w.Sidecar.SHA != nil && strings.TrimSpace(*w.Sidecar.SHA) != "" {
			claims = append(claims, Claim{Issue: n, SHA: strings.TrimSpace(*w.Sidecar.SHA), Source: SourceSelfReport})
		}
	}
	return picks, claims
}

var logStampRE = regexp.MustCompile(`^resolve-(\d+)-(\d{8}-\d{6})`)

// SpawnStamp reads the UTC spawn time a dispatcher encodes in a worker log or
// sidecar name (resolve-<issue>-YYYYMMDD-HHMMSS[-nonce].<ext>).
func SpawnStamp(name string) (time.Time, bool) {
	m := logStampRE.FindStringSubmatch(name)
	if m == nil {
		return time.Time{}, false
	}
	t, err := time.Parse("20060102-150405", m[2])
	if err != nil {
		return time.Time{}, false
	}
	return t.UTC(), true
}
