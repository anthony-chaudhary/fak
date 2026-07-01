package dispatchstatus

import (
	"math"
	"regexp"
	"sort"
	"strings"
	"time"
)

// LeaseRecord is one refs/fak/locks lease record, mirroring the JSON dict
// dispatch_status reads. AcquiredUnix/RenewedUnix are pointers so an absent field is
// distinguished from zero.
type LeaseRecord struct {
	ID           string   `json:"id"`
	TreeGlobs    []string `json:"tree_globs"`
	AcquiredUnix *int64   `json:"acquired_unix"`
	RenewedUnix  *int64   `json:"renewed_unix"`
	TTLSeconds   int64    `json:"ttl_seconds"`
	Holder       string   `json:"holder"`
	Generation   string   `json:"generation"`
}

// BacklogLane is one routed lane's tree + issue list from the dispatch router payload.
type BacklogLane struct {
	Tree   []string `json:"tree"`
	Issues []int    `json:"issues"`
}

// BacklogIssue is one routed issue row.
type BacklogIssue struct {
	Number     int    `json:"number"`
	Lane       string `json:"lane"`
	Confidence string `json:"confidence"`
}

// Backlog is the dispatch router payload leases are checked against. HasSkipped /
// HasError mirror the presence of the "_skipped" / "_error" keys the Python inspects to
// decide whether the candidate source is trustworthy.
type Backlog struct {
	Lanes      map[string]BacklogLane `json:"lanes"`
	Issues     []BacklogIssue         `json:"issues"`
	HasSkipped bool
	HasError   bool
}

// Candidate is one routed issue a lease may block, mirroring _backlog_candidates rows.
type Candidate struct {
	Issue      int      `json:"issue"`
	Lane       string   `json:"lane"`
	Confidence string   `json:"confidence,omitempty"`
	Tree       []string `json:"tree,omitempty"`
}

// LeaseRow is one classified lease for the status card.
type LeaseRow struct {
	ID                 string      `json:"id"`
	Lane               string      `json:"lane"`
	Holder             string      `json:"holder,omitempty"`
	Tree               []string    `json:"tree,omitempty"`
	AgeSeconds         *int64      `json:"age_seconds"`
	AgeMin             *float64    `json:"age_min"`
	TTLSeconds         int64       `json:"ttl_seconds"`
	ExpiresInSeconds   *int64      `json:"expires_in_seconds"`
	Generation         string      `json:"generation,omitempty"`
	Status             string      `json:"status"`
	BlocksCandidate    *bool       `json:"blocks_candidate"`
	BlockingCandidates []Candidate `json:"blocking_candidates"`
}

// LeaseState is the classified lease summary.
type LeaseState struct {
	Source                   string     `json:"source"`
	CandidateSourceAvailable bool       `json:"candidate_source_available"`
	CandidateCount           int        `json:"candidate_count"`
	ActiveCount              int        `json:"active_count"`
	ExpiredCount             int        `json:"expired_count"`
	BlockingCount            int        `json:"blocking_count"`
	Active                   []LeaseRow `json:"active"`
	Expired                  []LeaseRow `json:"expired"`
}

var laneNumSuffix = regexp.MustCompile(`-\d+$`)

// LeaseLane extracts the lane a "resolve-<lane>[-<n>]" lease id names, dropping a
// trailing numeric suffix; a non-resolve id is returned unchanged. Mirrors
// dispatch_status._lease_lane.
func LeaseLane(leaseID string) string {
	leaseID = strings.TrimSpace(leaseID)
	const prefix = "resolve-"
	if !strings.HasPrefix(leaseID, prefix) {
		return leaseID
	}
	lane := leaseID[len(prefix):]
	if laneNumSuffix.MatchString(lane) {
		lane = laneNumSuffix.ReplaceAllString(lane, "")
	}
	if lane == "" {
		return leaseID
	}
	return lane
}

func leaseActiveUnix(rec LeaseRecord) *int64 {
	a, r := rec.AcquiredUnix, rec.RenewedUnix
	if a == nil && r == nil {
		return nil
	}
	if a == nil {
		return r
	}
	if r == nil {
		return a
	}
	m := *a
	if *r > m {
		m = *r
	}
	return &m
}

func leaseExpired(rec LeaseRecord, nowUnix int64) bool {
	ttl := rec.TTLSeconds
	if ttl <= 0 {
		return false
	}
	active := leaseActiveUnix(rec)
	if active == nil {
		return false
	}
	return nowUnix >= *active+ttl
}

func stringList(v []string) []string {
	out := make([]string, 0, len(v))
	for _, x := range v {
		if strings.TrimSpace(x) != "" {
			out = append(out, x)
		}
	}
	return out
}

func normalizeTree(t string) string {
	t = strings.ReplaceAll(strings.TrimSpace(t), "\\", "/")
	t = strings.TrimPrefix(t, "./")
	t = strings.TrimRight(t, "/")
	for _, suffix := range []string{"/**", "/*"} {
		if strings.HasSuffix(t, suffix) {
			t = t[:len(t)-len(suffix)]
			break
		}
	}
	return strings.TrimRight(t, "/")
}

func cleanTree(tree []string) []string {
	var out []string
	for _, t := range stringList(tree) {
		if n := normalizeTree(t); n != "" {
			out = append(out, n)
		}
	}
	return out
}

func treeOverlapOne(a, b string) bool {
	a, b = normalizeTree(a), normalizeTree(b)
	if a == "" || b == "" {
		return true
	}
	if a == "**" || a == "**/*" || b == "**" || b == "**/*" {
		return true
	}
	return a == b || strings.HasPrefix(a, b+"/") || strings.HasPrefix(b, a+"/")
}

// TreesOverlap reports whether two file-tree glob sets could touch the same paths. An
// empty set means "unknown scope" and conservatively overlaps everything. Mirrors
// dispatch_status.trees_overlap.
func TreesOverlap(a, b []string) bool {
	ta, tb := cleanTree(a), cleanTree(b)
	if len(ta) == 0 || len(tb) == 0 {
		return true
	}
	for _, x := range ta {
		for _, y := range tb {
			if treeOverlapOne(x, y) {
				return true
			}
		}
	}
	return false
}

func backlogCandidates(backlog Backlog) []Candidate {
	type key struct {
		issue int
		lane  string
	}
	seen := map[key]bool{}
	var out []Candidate
	for _, row := range backlog.Issues {
		if row.Lane == "" || row.Number == 0 {
			continue
		}
		grp := backlog.Lanes[row.Lane]
		out = append(out, Candidate{
			Issue:      row.Number,
			Lane:       row.Lane,
			Confidence: row.Confidence,
			Tree:       stringList(grp.Tree),
		})
		seen[key{row.Number, row.Lane}] = true
	}
	// Deterministic lane order for the second pass (Go map iteration is random).
	lanes := make([]string, 0, len(backlog.Lanes))
	for lane := range backlog.Lanes {
		lanes = append(lanes, lane)
	}
	sort.Strings(lanes)
	for _, lane := range lanes {
		grp := backlog.Lanes[lane]
		for _, issue := range grp.Issues {
			if issue == 0 || seen[key{issue, lane}] {
				continue
			}
			out = append(out, Candidate{Issue: issue, Lane: lane, Tree: stringList(grp.Tree)})
			seen[key{issue, lane}] = true
		}
	}
	return out
}

// SummarizeLeases classifies refs/fak/locks records against the dispatch backlog for
// the status card. It is the Go port of dispatch_status.summarize_leases: active leases
// block a candidate only when their tree overlaps a routed issue's lane tree; expired
// records stay visible as reapable residue but never block. now is injected for
// determinism.
func SummarizeLeases(records []LeaseRecord, backlog Backlog, now time.Time) LeaseState {
	nowUnix := now.Unix()
	candidateSourceAvailable := !backlog.HasSkipped && !(backlog.HasError && len(backlog.Lanes) == 0)
	candidates := backlogCandidates(backlog)

	var active, expired []LeaseRow
	for _, rec := range records {
		leaseID := strings.TrimSpace(rec.ID)
		if leaseID == "" {
			continue
		}
		tree := stringList(rec.TreeGlobs)
		activeUnix := leaseActiveUnix(rec)
		var ageSeconds *int64
		var ageMin *float64
		if activeUnix != nil {
			age := nowUnix - *activeUnix
			if age < 0 {
				age = 0
			}
			ageSeconds = &age
			m := math.Round(float64(age)/60*10) / 10
			ageMin = &m
		}
		ttl := rec.TTLSeconds
		if ttl < 0 {
			ttl = 0
		}
		var expiresIn *int64
		if ttl > 0 && activeUnix != nil {
			e := *activeUnix + ttl - nowUnix
			expiresIn = &e
		}
		row := LeaseRow{
			ID:               leaseID,
			Lane:             LeaseLane(leaseID),
			Holder:           rec.Holder,
			Tree:             tree,
			AgeSeconds:       ageSeconds,
			AgeMin:           ageMin,
			TTLSeconds:       ttl,
			ExpiresInSeconds: expiresIn,
			Generation:       rec.Generation,
		}
		if leaseExpired(rec, nowUnix) {
			row.Status = "EXPIRED"
			bc := false
			row.BlocksCandidate = &bc
			row.BlockingCandidates = []Candidate{}
			expired = append(expired, row)
			continue
		}
		row.Status = "LIVE"
		if candidateSourceAvailable {
			var blockers []Candidate
			for _, c := range candidates {
				if TreesOverlap(tree, c.Tree) {
					blockers = append(blockers, c)
				}
			}
			blocks := len(blockers) > 0
			row.BlocksCandidate = &blocks
			row.BlockingCandidates = capCandidates(blockers, 8)
		} else {
			row.BlocksCandidate = nil
			row.BlockingCandidates = []Candidate{}
		}
		active = append(active, row)
	}

	sort.SliceStable(active, func(i, j int) bool {
		bi := active[i].BlocksCandidate != nil && *active[i].BlocksCandidate
		bj := active[j].BlocksCandidate != nil && *active[j].BlocksCandidate
		if bi != bj {
			return bi // blocking leases first
		}
		if active[i].Lane != active[j].Lane {
			return active[i].Lane < active[j].Lane
		}
		return active[i].ID < active[j].ID
	})
	sort.SliceStable(expired, func(i, j int) bool { return expired[i].ID < expired[j].ID })

	blocking := 0
	for _, r := range active {
		if r.BlocksCandidate != nil && *r.BlocksCandidate {
			blocking++
		}
	}
	return LeaseState{
		Source:                   "refs/fak/locks",
		CandidateSourceAvailable: candidateSourceAvailable,
		CandidateCount:           len(candidates),
		ActiveCount:              len(active),
		ExpiredCount:             len(expired),
		BlockingCount:            blocking,
		Active:                   active,
		Expired:                  capRows(expired, 8),
	}
}

func capCandidates(c []Candidate, n int) []Candidate {
	if len(c) > n {
		return c[:n]
	}
	return c
}

func capRows(r []LeaseRow, n int) []LeaseRow {
	if len(r) > n {
		return r[:n]
	}
	return r
}
