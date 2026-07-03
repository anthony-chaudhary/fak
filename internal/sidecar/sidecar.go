package sidecar

// sidecar.go is the pure, unit-tested surface of the sidecar pane v0 (#2215,
// epic #2209): ONE read-only fold over the four per-agent runtime planes —
// sessions (the B1 census over B2 descriptors), accounts (usable / throttled /
// blocked), lanes (dos-top occupancy, RENDERED not re-adjudicated), and
// context/cache posture (managed-cache posture + compaction/elision counters) —
// rendered IDENTICALLY on at least two surfaces (terminal text and Slack blocks)
// from ONE intermediate render model.
//
// The parity contract, stated as a type invariant rather than a review finding:
// both surfaces serialize the SAME model. There is exactly one function that
// produces the ordered field sequence — Pane.Sections() — and both RenderText and
// RenderSlack walk its output. So the core-field sequence a terminal reader sees
// and the sequence a Slack reader sees are the same object; a divergence would be
// a change to Sections(), which both renderers observe at once, not a drift one
// surface can acquire alone. CoreFields() exposes that sequence for the golden
// parity test to assert against both rendered forms.
//
// The sidecar is a VIEW: it renders adjudicated facts, it never re-adjudicates
// them. Lane occupancy is folded from a dos-top reading; it is not a fresh
// arbitration. Every plane carries a PROVENANCE label (WITNESSED / OBSERVED /
// UNMEASURED) so a reader weighs a fact without chasing its source, and a plane
// whose input was absent reads UNMEASURED — never a silent, fabricated GREEN.
//
// The fold is PURE: it takes already-collected typed inputs and returns one Pane.
// It runs no tool, reads no clock, and touches no disk — the live collectors live
// in cmd/fak/sidecar.go (the impure shell), exactly the execrollup / rollup.go
// split. That keeps the fold logic, the render model, and the parity test sharing
// one deterministic surface.

import (
	"fmt"
	"sort"
	"strings"
)

// Schema is the stable control-pane schema identifier for the sidecar pane.
const Schema = "fak.sidecar/v1"

// Provenance labels — the honesty discipline carried onto every plane. WITNESSED
// = read from an authored artifact (a config home census, a committed lease
// state). OBSERVED = a live reading relayed from a running source (a gateway
// posture probe, a dos-top occupancy snapshot). UNMEASURED = the plane's input
// was absent, so it carries no facts and cannot be certified.
const (
	Witnessed  = "WITNESSED"
	Observed   = "OBSERVED"
	Unmeasured = "UNMEASURED"
)

// Plane names — the four folds, in the stable order the pane renders them. This
// ordering is the spine of the parity contract: both surfaces render the planes
// in this sequence because both walk Sections(), which emits them in this order.
const (
	PlaneSessions = "sessions"
	PlaneAccounts = "accounts"
	PlaneLanes    = "lanes"
	PlanePosture  = "posture"
)

// planeOrder is the canonical plane sequence. Sections() walks it; changing it
// changes both renderers at once (the parity property).
var planeOrder = []string{PlaneSessions, PlaneAccounts, PlaneLanes, PlanePosture}

// SessionRow is one census row: an agent session joined across the four key
// spaces (session id, owning account, harness, disposition). It is a rendered
// descriptor, not a live process handle.
type SessionRow struct {
	Session     string `json:"session"`
	Account     string `json:"account,omitempty"`
	Harness     string `json:"harness,omitempty"`     // claude / codex / opencode / aider (harnessprofile Name)
	Disposition string `json:"disposition,omitempty"` // live / done / stopped / throttled …
}

// AccountRow is one account's usability posture: usable, throttled (rate-limited
// until reset), or blocked (auth/org disabled). The three-way split is the
// account-health fact an operator needs before resuming under it.
type AccountRow struct {
	Account string `json:"account"`
	State   string `json:"state"` // usable | throttled | blocked
	Detail  string `json:"detail,omitempty"`
}

// LaneRow is one lane's occupancy, rendered from a dos-top reading. Held=true
// means a live lease occupies it; the sidecar reports this, it does not decide it.
type LaneRow struct {
	Lane  string `json:"lane"`
	Kind  string `json:"kind,omitempty"` // cluster | keyword | global
	Held  bool   `json:"held"`
	Owner string `json:"owner,omitempty"`
}

// Posture is the context/cache posture for the joined session substrate: the
// managed-cache posture word plus the compaction/elision counters. Zeroes render
// as zeroes (a real reading of "nothing shed yet"), never as UNMEASURED — the
// plane is UNMEASURED only when Measured is false.
type Posture struct {
	CachePosture   string `json:"cache_posture,omitempty"` // e.g. "warm", "cold", "managed"
	Compactions    int    `json:"compactions"`
	Elisions       int    `json:"elisions"`
	SessionsJoined int    `json:"sessions_joined"`
}

// PlaneInput carries one plane's collected rows plus its measurement status. A
// collector that could not read its source sets Measured=false with a Note; the
// fold turns that into an UNMEASURED plane, never a silent empty pass.
type PlaneInput struct {
	Measured bool
	Note     string // why unmeasured, or a one-line source note when measured
}

// Inputs is the typed bundle the live collectors hand the fold. Each plane's rows
// travel with its PlaneInput status; a missing plane degrades to UNMEASURED, it
// does not panic.
type Inputs struct {
	Sessions      PlaneInput
	SessionRows   []SessionRow
	Accounts      PlaneInput
	AccountRows   []AccountRow
	Lanes         PlaneInput
	LaneRows      []LaneRow
	PostureStatus PlaneInput
	Posture       Posture

	Workspace   string
	Host        string
	GeneratedAt string
}

// Line is one rendered fact: a stable Key, its Value, and the plane's provenance
// label. Key is the parity anchor — it identifies the field independent of how a
// surface formats the value, so the golden test can assert the same ordered Key
// sequence appears in both rendered forms.
type Line struct {
	Key   string `json:"key"`
	Value string `json:"value"`
	Prov  string `json:"prov"`
}

// Section is one plane's rendered block: its name, its provenance label, a
// one-line summary, and the ordered facts. Both renderers consume Sections in the
// same order and each Section's Lines in the same order.
type Section struct {
	Plane   string `json:"plane"`
	Prov    string `json:"prov"`
	Summary string `json:"summary"`
	Lines   []Line `json:"lines"`
}

// Pane is one folded sidecar. It carries the control-pane spine (schema / ok /
// headline) plus the four plane folds and the identity fields. The rendered
// surfaces are produced only from Sections(), never from the raw inputs, so the
// two surfaces cannot diverge.
type Pane struct {
	Schema      string `json:"schema"`
	OK          bool   `json:"ok"`
	Headline    string `json:"headline"`
	Workspace   string `json:"workspace,omitempty"`
	Host        string `json:"host,omitempty"`
	GeneratedAt string `json:"generated_at,omitempty"`

	Sessions SessionsFold `json:"sessions"`
	Accounts AccountsFold `json:"accounts"`
	Lanes    LanesFold    `json:"lanes"`
	Posture  PostureFold  `json:"posture"`

	Unmeasured int `json:"unmeasured"`
}

// SessionsFold is the sessions plane after folding: the census rows, a live count,
// and the plane's provenance.
type SessionsFold struct {
	Measured bool         `json:"measured"`
	Prov     string       `json:"prov"`
	Note     string       `json:"note,omitempty"`
	Rows     []SessionRow `json:"rows"`
	Total    int          `json:"total"`
	Live     int          `json:"live"`
}

// AccountsFold is the accounts plane after folding: the usable/throttled/blocked
// tallies and the rows.
type AccountsFold struct {
	Measured  bool         `json:"measured"`
	Prov      string       `json:"prov"`
	Note      string       `json:"note,omitempty"`
	Rows      []AccountRow `json:"rows"`
	Usable    int          `json:"usable"`
	Throttled int          `json:"throttled"`
	Blocked   int          `json:"blocked"`
}

// LanesFold is the lanes plane after folding: the occupancy rows and the held
// count. Rendered from a dos-top reading — not re-adjudicated.
type LanesFold struct {
	Measured bool      `json:"measured"`
	Prov     string    `json:"prov"`
	Note     string    `json:"note,omitempty"`
	Rows     []LaneRow `json:"rows"`
	Held     int       `json:"held"`
	Free     int       `json:"free"`
}

// PostureFold is the context/cache posture plane after folding.
type PostureFold struct {
	Measured bool    `json:"measured"`
	Prov     string  `json:"prov"`
	Note     string  `json:"note,omitempty"`
	Posture  Posture `json:"posture"`
}

// Fold turns the collected inputs into one Pane. Deterministic and side-effect
// free; it reads no clock (GeneratedAt is carried on the Inputs).
func Fold(in Inputs) Pane {
	p := Pane{
		Schema:      Schema,
		Workspace:   in.Workspace,
		Host:        in.Host,
		GeneratedAt: in.GeneratedAt,
	}

	p.Sessions = foldSessions(in.Sessions, in.SessionRows)
	p.Accounts = foldAccounts(in.Accounts, in.AccountRows)
	p.Lanes = foldLanes(in.Lanes, in.LaneRows)
	p.Posture = foldPosture(in.PostureStatus, in.Posture)

	for _, measured := range []bool{p.Sessions.Measured, p.Accounts.Measured, p.Lanes.Measured, p.Posture.Measured} {
		if !measured {
			p.Unmeasured++
		}
	}
	p.OK = p.Unmeasured == 0
	p.Headline = headline(p)
	return p
}

func provOf(measured bool, whenMeasured string) string {
	if !measured {
		return Unmeasured
	}
	return whenMeasured
}

func foldSessions(in PlaneInput, rows []SessionRow) SessionsFold {
	f := SessionsFold{Measured: in.Measured, Note: in.Note, Prov: provOf(in.Measured, Witnessed)}
	if !in.Measured {
		return f
	}
	// Copy + stable-sort so the render order is deterministic regardless of the
	// collector's emission order (a parity prerequisite).
	f.Rows = append(f.Rows, rows...)
	sort.SliceStable(f.Rows, func(i, j int) bool {
		if f.Rows[i].Account != f.Rows[j].Account {
			return f.Rows[i].Account < f.Rows[j].Account
		}
		return f.Rows[i].Session < f.Rows[j].Session
	})
	f.Total = len(f.Rows)
	for _, r := range f.Rows {
		if strings.EqualFold(r.Disposition, "live") {
			f.Live++
		}
	}
	return f
}

func foldAccounts(in PlaneInput, rows []AccountRow) AccountsFold {
	f := AccountsFold{Measured: in.Measured, Note: in.Note, Prov: provOf(in.Measured, Witnessed)}
	if !in.Measured {
		return f
	}
	f.Rows = append(f.Rows, rows...)
	sort.SliceStable(f.Rows, func(i, j int) bool { return f.Rows[i].Account < f.Rows[j].Account })
	for _, r := range f.Rows {
		switch strings.ToLower(r.State) {
		case "throttled":
			f.Throttled++
		case "blocked":
			f.Blocked++
		default:
			f.Usable++
		}
	}
	return f
}

func foldLanes(in PlaneInput, rows []LaneRow) LanesFold {
	// Lane occupancy is OBSERVED (a live dos-top reading), not WITNESSED — the
	// pane renders it, it does not re-adjudicate the lease.
	f := LanesFold{Measured: in.Measured, Note: in.Note, Prov: provOf(in.Measured, Observed)}
	if !in.Measured {
		return f
	}
	f.Rows = append(f.Rows, rows...)
	sort.SliceStable(f.Rows, func(i, j int) bool { return f.Rows[i].Lane < f.Rows[j].Lane })
	for _, r := range f.Rows {
		if r.Held {
			f.Held++
		} else {
			f.Free++
		}
	}
	return f
}

func foldPosture(in PlaneInput, post Posture) PostureFold {
	// Posture is OBSERVED — a live reading of the gateway's managed-cache state.
	f := PostureFold{Measured: in.Measured, Note: in.Note, Prov: provOf(in.Measured, Observed)}
	if in.Measured {
		f.Posture = post
	}
	return f
}

// headline is the one-line "same pane, every surface" summary.
func headline(p Pane) string {
	var b strings.Builder
	b.WriteString("sidecar")
	if p.Sessions.Measured {
		fmt.Fprintf(&b, ": %d session(s) (%d live)", p.Sessions.Total, p.Sessions.Live)
	} else {
		b.WriteString(": sessions unmeasured")
	}
	if p.Accounts.Measured {
		fmt.Fprintf(&b, "; accounts %d usable / %d throttled / %d blocked",
			p.Accounts.Usable, p.Accounts.Throttled, p.Accounts.Blocked)
	}
	if p.Lanes.Measured {
		fmt.Fprintf(&b, "; %d/%d lane(s) held", p.Lanes.Held, p.Lanes.Held+p.Lanes.Free)
	}
	if p.Posture.Measured {
		fmt.Fprintf(&b, "; posture %s (%d compaction / %d elision)",
			dashIfEmpty(p.Posture.Posture.CachePosture), p.Posture.Posture.Compactions, p.Posture.Posture.Elisions)
	}
	if p.Unmeasured > 0 {
		fmt.Fprintf(&b, "; %d plane(s) unmeasured", p.Unmeasured)
	}
	b.WriteString(".")
	return b.String()
}

// Sections is the SINGLE producer of the ordered render model. Both RenderText
// and RenderSlack walk exactly this — that is the parity contract. Each Section's
// Lines carry a stable Key so the field sequence is identifiable independent of a
// surface's formatting.
func (p Pane) Sections() []Section {
	out := make([]Section, 0, len(planeOrder))
	for _, plane := range planeOrder {
		switch plane {
		case PlaneSessions:
			out = append(out, p.sessionsSection())
		case PlaneAccounts:
			out = append(out, p.accountsSection())
		case PlaneLanes:
			out = append(out, p.lanesSection())
		case PlanePosture:
			out = append(out, p.postureSection())
		}
	}
	return out
}

func (p Pane) sessionsSection() Section {
	s := Section{Plane: PlaneSessions, Prov: p.Sessions.Prov}
	if !p.Sessions.Measured {
		s.Summary = unmeasuredSummary(p.Sessions.Note)
		return s
	}
	s.Summary = fmt.Sprintf("%d session(s), %d live", p.Sessions.Total, p.Sessions.Live)
	for _, r := range p.Sessions.Rows {
		s.Lines = append(s.Lines, Line{
			Key:   "session/" + r.Session,
			Value: sessionValue(r),
			Prov:  p.Sessions.Prov,
		})
	}
	return s
}

func sessionValue(r SessionRow) string {
	parts := []string{r.Session}
	if r.Harness != "" {
		parts = append(parts, r.Harness)
	}
	if r.Account != "" {
		parts = append(parts, "acct="+r.Account)
	}
	if r.Disposition != "" {
		parts = append(parts, r.Disposition)
	}
	return strings.Join(parts, " · ")
}

func (p Pane) accountsSection() Section {
	s := Section{Plane: PlaneAccounts, Prov: p.Accounts.Prov}
	if !p.Accounts.Measured {
		s.Summary = unmeasuredSummary(p.Accounts.Note)
		return s
	}
	s.Summary = fmt.Sprintf("%d usable, %d throttled, %d blocked", p.Accounts.Usable, p.Accounts.Throttled, p.Accounts.Blocked)
	for _, r := range p.Accounts.Rows {
		val := r.Account + " · " + r.State
		if r.Detail != "" {
			val += " (" + r.Detail + ")"
		}
		s.Lines = append(s.Lines, Line{Key: "account/" + r.Account, Value: val, Prov: p.Accounts.Prov})
	}
	return s
}

func (p Pane) lanesSection() Section {
	s := Section{Plane: PlaneLanes, Prov: p.Lanes.Prov}
	if !p.Lanes.Measured {
		s.Summary = unmeasuredSummary(p.Lanes.Note)
		return s
	}
	s.Summary = fmt.Sprintf("%d held, %d free", p.Lanes.Held, p.Lanes.Free)
	for _, r := range p.Lanes.Rows {
		state := "free"
		if r.Held {
			state = "held"
			if r.Owner != "" {
				state += " by " + r.Owner
			}
		}
		val := r.Lane
		if r.Kind != "" {
			val += " (" + r.Kind + ")"
		}
		val += " · " + state
		s.Lines = append(s.Lines, Line{Key: "lane/" + r.Lane, Value: val, Prov: p.Lanes.Prov})
	}
	return s
}

func (p Pane) postureSection() Section {
	s := Section{Plane: PlanePosture, Prov: p.Posture.Prov}
	if !p.Posture.Measured {
		s.Summary = unmeasuredSummary(p.Posture.Note)
		return s
	}
	post := p.Posture.Posture
	s.Summary = fmt.Sprintf("cache %s, %d joined", dashIfEmpty(post.CachePosture), post.SessionsJoined)
	s.Lines = []Line{
		{Key: "posture/cache", Value: dashIfEmpty(post.CachePosture), Prov: p.Posture.Prov},
		{Key: "posture/compactions", Value: fmt.Sprintf("%d", post.Compactions), Prov: p.Posture.Prov},
		{Key: "posture/elisions", Value: fmt.Sprintf("%d", post.Elisions), Prov: p.Posture.Prov},
		{Key: "posture/joined", Value: fmt.Sprintf("%d", post.SessionsJoined), Prov: p.Posture.Prov},
	}
	return s
}

// CoreFields is the ordered core-field sequence the golden parity test asserts
// against both rendered surfaces. It is the flattened (plane, key) spine of
// Sections() — the machine-checkable statement that the two surfaces carry the
// identical fields in the identical order. Plane-header rows are included (keyed
// "@<plane>") so an empty/unmeasured plane still contributes an identical anchor
// to both surfaces.
func (p Pane) CoreFields() []string {
	var keys []string
	for _, sec := range p.Sections() {
		keys = append(keys, "@"+sec.Plane)
		for _, ln := range sec.Lines {
			keys = append(keys, ln.Key)
		}
	}
	return keys
}

func unmeasuredSummary(note string) string {
	if strings.TrimSpace(note) == "" {
		return "unmeasured"
	}
	return "unmeasured: " + note
}

func dashIfEmpty(s string) string {
	if strings.TrimSpace(s) == "" {
		return "-"
	}
	return s
}
