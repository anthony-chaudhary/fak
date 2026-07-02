package devexmeter

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
)

const (
	// FrictionSchema is the combined event-fold report schema.
	FrictionSchema = "fak.devexmeter.friction.v1"
	// ReadWasteSchema is the re-read waste report schema.
	ReadWasteSchema = "fak.devexmeter.read-waste.v1"
	// RefusalRetrySchema is the retry-after-refusal report schema.
	RefusalRetrySchema = "fak.devexmeter.refusal-retry.v1"
	// OnboardingSchema is the time-to-first-useful-action report schema.
	OnboardingSchema = "fak.devexmeter.onboarding.v1"
	// AbandonedSchema is the abandoned-turn report schema.
	AbandonedSchema = "fak.devexmeter.abandoned-turn.v1"

	ClassRetryAfterRefusal = "retry-after-refusal"
	ClassReReadWaste       = "re-read-waste"
	ClassHangRecovery      = "hang-recovery"
	ClassMisdiagnosedRed   = "misdiagnosed-red"
	ClassOnboardingReadIn  = "onboarding-read-in"
	ClassAbandonedTurn     = "abandoned-turn"

	CauseDefensive        = "defensive"
	CausePostCompaction   = "post-compaction"
	CausePostRevert       = "post-revert"
	CauseBlockedByLease   = "blocked-by-lease"
	CauseLoopedRefusal    = "looped-on-refusal"
	CauseNoProgressGiveUp = "no-progress-give-up"
)

// ToolEvent is the normalized, deterministic input row for the dev-ex meter.
// Adapters from transcripts, usage journals, and guard journals should map onto
// this shape, then call the pure Fold* functions below.
type ToolEvent struct {
	SessionID string `json:"session_id,omitempty"`
	Turn      int    `json:"turn,omitempty"`
	Seq       int    `json:"seq,omitempty"`
	Lane      string `json:"lane,omitempty"`

	Action string `json:"action,omitempty"` // read, edit, commit, test, ...
	Tool   string `json:"tool,omitempty"`   // agent-facing tool name, if distinct

	Path          string `json:"path,omitempty"`
	ContentDigest string `json:"content_digest,omitempty"`
	ArgsDigest    string `json:"args_digest,omitempty"`

	Outcome string `json:"outcome,omitempty"` // allow, deny, timeout, error, ...
	Verdict string `json:"verdict,omitempty"` // guard-style ALLOW/DENY fallback
	Reason  string `json:"reason,omitempty"`

	WasteClass string `json:"waste_class,omitempty"` // explicit class for known waste
	ReadCause  string `json:"read_cause,omitempty"`  // defensive/post-compaction/post-revert

	Tokens           int `json:"tokens,omitempty"`
	PromptTokens     int `json:"prompt_tokens,omitempty"`
	CompletionTokens int `json:"completion_tokens,omitempty"`

	AtMillis int64 `json:"at_millis,omitempty"`

	UsefulAction    bool `json:"useful_action,omitempty"`
	Edit            bool `json:"edit,omitempty"`
	Commit          bool `json:"commit,omitempty"`
	AfterCompaction bool `json:"after_compaction,omitempty"`
	AfterRevert     bool `json:"after_revert,omitempty"`
}

// CauseTotal is a ranked sub-bucket under a friction class.
type CauseTotal struct {
	Cause        string `json:"cause"`
	Events       int    `json:"events"`
	WastedTurns  int    `json:"wasted_turns"`
	WastedTokens int    `json:"wasted_tokens"`
}

// FrictionClassTotal is one ranked dev-ex friction class.
type FrictionClassTotal struct {
	Class        string       `json:"class"`
	Events       int          `json:"events"`
	WastedTurns  int          `json:"wasted_turns"`
	WastedTokens int          `json:"wasted_tokens"`
	Causes       []CauseTotal `json:"causes,omitempty"`
}

// FrictionReport is the parent fold: all child meters plus one worst-first
// ranking by wasted tokens, then wasted turns, then class name.
type FrictionReport struct {
	Schema          string               `json:"schema"`
	Events          int                  `json:"events"`
	Ranked          []FrictionClassTotal `json:"ranked"`
	Reads           ReadWasteReport      `json:"reads"`
	Refusals        RefusalRetryReport   `json:"refusals"`
	Onboarding      OnboardingReport     `json:"onboarding"`
	Abandoned       AbandonedReport      `json:"abandoned"`
	ExplicitOutcome []FrictionClassTotal `json:"explicit_outcome,omitempty"`
}

// ReadWasteFile is a repeated unchanged file read inside one session.
type ReadWasteFile struct {
	SessionID     string `json:"session_id,omitempty"`
	Path          string `json:"path"`
	ContentDigest string `json:"content_digest,omitempty"`
	Cause         string `json:"cause"`
	ReReads       int    `json:"re_reads"`
	WastedTurns   int    `json:"wasted_turns"`
	WastedTokens  int    `json:"wasted_tokens"`
}

// ReadWasteReport counts files read more than once with the same content hash.
type ReadWasteReport struct {
	Schema        string          `json:"schema"`
	Reads         int             `json:"reads"`
	ReReads       int             `json:"re_reads"`
	UnhashedReads int             `json:"unhashed_reads"`
	WastedTurns   int             `json:"wasted_turns"`
	WastedTokens  int             `json:"wasted_tokens"`
	ByCause       []CauseTotal    `json:"by_cause,omitempty"`
	Files         []ReadWasteFile `json:"files,omitempty"`
}

// RefusalRetry is the retry cost attributed to one refusal reason token.
type RefusalRetry struct {
	Reason       string `json:"reason"`
	RefusedCalls int    `json:"refused_calls"`
	Cleared      int    `json:"cleared"`
	Looped       int    `json:"looped"`
	NoRetry      int    `json:"no_retry"`
	RetryRows    int    `json:"retry_rows"`
	WastedTurns  int    `json:"wasted_turns"`
	WastedTokens int    `json:"wasted_tokens"`
}

// Retried is the number of episodes whose fate was decided by a later exact call.
func (r RefusalRetry) Retried() int { return r.Cleared + r.Looped }

// ClearRate is Cleared over decided retry episodes, in [0,1].
func (r RefusalRetry) ClearRate() float64 {
	if d := r.Retried(); d > 0 {
		return float64(r.Cleared) / float64(d)
	}
	return 0
}

// RefusalRetryReport attributes exact retry loops to the first refusal token.
type RefusalRetryReport struct {
	Schema       string         `json:"schema"`
	Rows         int            `json:"rows"`
	RefusedCalls int            `json:"refused_calls"`
	Unkeyed      int            `json:"unkeyed_refusals"`
	RetryRows    int            `json:"retry_rows"`
	WastedTurns  int            `json:"wasted_turns"`
	WastedTokens int            `json:"wasted_tokens"`
	ByReason     []RefusalRetry `json:"by_reason,omitempty"`
}

// LaneOnboarding is the first-useful-action tax for one lane.
type LaneOnboarding struct {
	Lane                      string  `json:"lane"`
	Sessions                  int     `json:"sessions"`
	SessionsWithoutAction     int     `json:"sessions_without_action"`
	TokensBeforeAction        int     `json:"tokens_before_action"`
	TurnsBeforeAction         int     `json:"turns_before_action"`
	MedianTokensBeforeAction  float64 `json:"median_tokens_before_action"`
	MedianTurnsBeforeAction   float64 `json:"median_turns_before_action"`
	MedianMillisToFirstAction float64 `json:"median_millis_to_first_action,omitempty"`
}

// OnboardingReport measures the cost before the first useful action.
type OnboardingReport struct {
	Schema                string           `json:"schema"`
	Sessions              int              `json:"sessions"`
	SessionsWithAction    int              `json:"sessions_with_action"`
	SessionsWithoutAction int              `json:"sessions_without_action"`
	TokensBeforeAction    int              `json:"tokens_before_action"`
	TurnsBeforeAction     int              `json:"turns_before_action"`
	ByLane                []LaneOnboarding `json:"by_lane,omitempty"`
}

// AbandonedReport classifies turns with no edit/commit artifact.
type AbandonedReport struct {
	Schema         string       `json:"schema"`
	Turns          int          `json:"turns"`
	AbandonedTurns int          `json:"abandoned_turns"`
	AbandonedRate  float64      `json:"abandoned_rate"`
	WastedTokens   int          `json:"wasted_tokens"`
	ByCause        []CauseTotal `json:"by_cause,omitempty"`
}

// FoldFriction folds all dev-ex child meters and returns a single ranked signal.
func FoldFriction(events []ToolEvent) FrictionReport {
	reads := FoldReadWaste(events)
	refusals := FoldRetryAfterRefusal(events)
	onboarding := FoldTimeToFirstUsefulAction(events)
	abandoned := FoldAbandonedTurns(events)
	explicit := FoldExplicitOutcomeWaste(events)

	b := newRankBuilder()
	for _, c := range reads.ByCause {
		b.addAggregate(ClassReReadWaste, c.Cause, c.Events, c.WastedTurns, c.WastedTokens)
	}
	for _, r := range refusals.ByReason {
		b.addAggregate(ClassRetryAfterRefusal, r.Reason, r.RetryRows, r.WastedTurns, r.WastedTokens)
	}
	for _, l := range onboarding.ByLane {
		b.addAggregate(ClassOnboardingReadIn, l.Lane, l.Sessions, l.TurnsBeforeAction, l.TokensBeforeAction)
	}
	for _, c := range abandoned.ByCause {
		b.addAggregate(ClassAbandonedTurn, c.Cause, c.Events, c.WastedTurns, c.WastedTokens)
	}
	for _, x := range explicit {
		if len(x.Causes) == 0 {
			b.addAggregate(x.Class, "", x.Events, x.WastedTurns, x.WastedTokens)
			continue
		}
		for _, c := range x.Causes {
			b.addAggregate(x.Class, c.Cause, c.Events, c.WastedTurns, c.WastedTokens)
		}
	}

	return FrictionReport{
		Schema:          FrictionSchema,
		Events:          len(events),
		Ranked:          b.ranked(),
		Reads:           reads,
		Refusals:        refusals,
		Onboarding:      onboarding,
		Abandoned:       abandoned,
		ExplicitOutcome: explicit,
	}
}

// FoldExplicitOutcomeWaste folds directly labeled waste plus outcome-derived
// hang and misdiagnosed-red events. The derived child meters handle re-reads,
// refusal retries, onboarding, and abandoned turns.
func FoldExplicitOutcomeWaste(events []ToolEvent) []FrictionClassTotal {
	b := newRankBuilder()
	for _, e := range orderedEvents(events) {
		class, cause := explicitWaste(e)
		if class == "" {
			continue
		}
		b.addAggregate(class, cause, 1, 1, tokenCost(e))
	}
	return b.ranked()
}

// FoldReadWaste counts repeated reads of the same path and content digest within
// a session. A blank content digest is not enough evidence for unchanged content,
// so it is counted as unhashed and excluded from waste.
func FoldReadWaste(events []ToolEvent) ReadWasteReport {
	rep := ReadWasteReport{Schema: ReadWasteSchema}
	seen := map[string]bool{}
	files := map[string]*readFileAcc{}
	causes := newCauseBuilder()

	for _, e := range orderedEvents(events) {
		if !isReadEvent(e) {
			continue
		}
		rep.Reads++
		if strings.TrimSpace(e.ContentDigest) == "" {
			rep.UnhashedReads++
			continue
		}
		path := strings.TrimSpace(e.Path)
		if path == "" {
			path = "(unknown)"
		}
		session := sessionID(e)
		key := session + "\x00" + path + "\x00" + e.ContentDigest
		if !seen[key] {
			seen[key] = true
			continue
		}

		cause := readWasteCause(e)
		tokens := tokenCost(e)
		rep.ReReads++
		rep.WastedTokens += tokens
		turn := turnKey(e)
		causes.add(cause, 1, turn, tokens)

		fileKey := key + "\x00" + cause
		f := files[fileKey]
		if f == nil {
			f = &readFileAcc{
				SessionID:     cleanSession(session),
				Path:          path,
				ContentDigest: e.ContentDigest,
				Cause:         cause,
				turns:         map[string]bool{},
			}
			files[fileKey] = f
		}
		f.ReReads++
		f.WastedTokens += tokens
		f.turns[turn] = true
	}

	rep.ByCause = causes.totals()
	for _, c := range rep.ByCause {
		rep.WastedTurns += c.WastedTurns
	}
	rep.Files = make([]ReadWasteFile, 0, len(files))
	for _, f := range files {
		rep.Files = append(rep.Files, ReadWasteFile{
			SessionID:     f.SessionID,
			Path:          f.Path,
			ContentDigest: f.ContentDigest,
			Cause:         f.Cause,
			ReReads:       f.ReReads,
			WastedTurns:   len(f.turns),
			WastedTokens:  f.WastedTokens,
		})
	}
	sort.Slice(rep.Files, func(i, j int) bool {
		a, b := rep.Files[i], rep.Files[j]
		if a.WastedTokens != b.WastedTokens {
			return a.WastedTokens > b.WastedTokens
		}
		if a.ReReads != b.ReReads {
			return a.ReReads > b.ReReads
		}
		if a.Path != b.Path {
			return a.Path < b.Path
		}
		return a.Cause < b.Cause
	})
	return rep
}

// FoldRetryAfterRefusal detects exact retries after a refusal, scoped to one
// session. Identity is (tool/action, args_digest); changed args are a repaired
// call and do not clear the original refusal.
func FoldRetryAfterRefusal(events []ToolEvent) RefusalRetryReport {
	rep := RefusalRetryReport{Schema: RefusalRetrySchema}
	byReason := map[string]*refusalAcc{}
	var currentSession string
	open := map[string]*retryEpisode{}
	haveSession := false

	closeAll := func(cleared bool) {
		for _, ep := range open {
			closeRetryEpisode(ep, cleared, byReason)
		}
		open = map[string]*retryEpisode{}
	}

	for _, e := range orderedEvents(events) {
		session := sessionID(e)
		if !haveSession {
			currentSession = session
			haveSession = true
		}
		if session != currentSession {
			closeAll(false)
			currentSession = session
		}
		refusal := isRefusalEvent(e)
		clear := isClearEvent(e)
		if !refusal && !clear {
			continue
		}
		rep.Rows++
		if strings.TrimSpace(e.ArgsDigest) == "" {
			if refusal {
				rep.Unkeyed++
			}
			continue
		}
		key := callIdentity(e)
		ep := open[key]
		switch {
		case refusal && ep == nil:
			open[key] = &retryEpisode{reason: refusalReason(e), turns: map[string]bool{}}
		case refusal:
			ep.retryRows++
			ep.wastedTokens += tokenCost(e)
			ep.turns[turnKey(e)] = true
		case clear && ep != nil:
			closeRetryEpisode(ep, true, byReason)
			delete(open, key)
		}
	}
	if haveSession {
		closeAll(false)
	}

	rep.ByReason = make([]RefusalRetry, 0, len(byReason))
	for _, r := range byReason {
		rep.ByReason = append(rep.ByReason, RefusalRetry{
			Reason:       r.Reason,
			RefusedCalls: r.RefusedCalls,
			Cleared:      r.Cleared,
			Looped:       r.Looped,
			NoRetry:      r.NoRetry,
			RetryRows:    r.RetryRows,
			WastedTurns:  len(r.turns),
			WastedTokens: r.WastedTokens,
		})
	}
	sort.Slice(rep.ByReason, func(i, j int) bool {
		a, b := rep.ByReason[i], rep.ByReason[j]
		if a.WastedTokens != b.WastedTokens {
			return a.WastedTokens > b.WastedTokens
		}
		if a.WastedTurns != b.WastedTurns {
			return a.WastedTurns > b.WastedTurns
		}
		if a.Looped != b.Looped {
			return a.Looped > b.Looped
		}
		return a.Reason < b.Reason
	})
	for _, r := range rep.ByReason {
		rep.RefusedCalls += r.RefusedCalls
		rep.RetryRows += r.RetryRows
		rep.WastedTurns += r.WastedTurns
		rep.WastedTokens += r.WastedTokens
	}
	return rep
}

// FoldTimeToFirstUsefulAction measures tokens and completed turns before the
// first edit/commit/useful action, split by lane.
func FoldTimeToFirstUsefulAction(events []ToolEvent) OnboardingReport {
	rep := OnboardingReport{Schema: OnboardingSchema}
	bySession := groupBySession(events)
	lanes := map[string]*laneAcc{}
	for _, sessionEvents := range bySession {
		if len(sessionEvents) == 0 {
			continue
		}
		rep.Sessions++
		lane := sessionLane(sessionEvents)
		acc := lanes[lane]
		if acc == nil {
			acc = &laneAcc{Lane: lane}
			lanes[lane] = acc
		}
		acc.Sessions++
		idx := firstUsefulIndex(sessionEvents)
		if idx < 0 {
			rep.SessionsWithoutAction++
			acc.SessionsWithoutAction++
			continue
		}
		rep.SessionsWithAction++
		action := sessionEvents[idx]
		tokens := 0
		turns := map[string]bool{}
		for i := 0; i < idx; i++ {
			e := sessionEvents[i]
			tokens += tokenCost(e)
			if completedBeforeAction(e, action) {
				turns[turnKey(e)] = true
			}
		}
		turnN := len(turns)
		rep.TokensBeforeAction += tokens
		rep.TurnsBeforeAction += turnN
		acc.TokensBeforeAction += tokens
		acc.TurnsBeforeAction += turnN
		acc.tokenSamples = append(acc.tokenSamples, tokens)
		acc.turnSamples = append(acc.turnSamples, turnN)
		if ms, ok := millisToAction(sessionEvents[0], action); ok {
			acc.millisSamples = append(acc.millisSamples, ms)
		}
	}
	rep.ByLane = make([]LaneOnboarding, 0, len(lanes))
	for _, l := range lanes {
		rep.ByLane = append(rep.ByLane, LaneOnboarding{
			Lane:                      l.Lane,
			Sessions:                  l.Sessions,
			SessionsWithoutAction:     l.SessionsWithoutAction,
			TokensBeforeAction:        l.TokensBeforeAction,
			TurnsBeforeAction:         l.TurnsBeforeAction,
			MedianTokensBeforeAction:  medianInts(l.tokenSamples),
			MedianTurnsBeforeAction:   medianInts(l.turnSamples),
			MedianMillisToFirstAction: medianInt64s(l.millisSamples),
		})
	}
	sort.Slice(rep.ByLane, func(i, j int) bool {
		a, b := rep.ByLane[i], rep.ByLane[j]
		if a.MedianTokensBeforeAction != b.MedianTokensBeforeAction {
			return a.MedianTokensBeforeAction > b.MedianTokensBeforeAction
		}
		if a.TokensBeforeAction != b.TokensBeforeAction {
			return a.TokensBeforeAction > b.TokensBeforeAction
		}
		return a.Lane < b.Lane
	})
	return rep
}

// FoldAbandonedTurns counts turns with no edit/commit artifact and classifies the
// likely cause from exact refusal loops and blocking reason tokens.
func FoldAbandonedTurns(events []ToolEvent) AbandonedReport {
	rep := AbandonedReport{Schema: AbandonedSchema}
	bySession := groupBySession(events)
	causes := newCauseBuilder()
	for _, sessionEvents := range bySession {
		turns := groupByTurn(sessionEvents)
		priorRefusals := map[string]bool{}
		for _, turnEvents := range turns {
			rep.Turns++
			looped := false
			blocked := false
			hasArtifact := false
			tokens := 0
			for _, e := range turnEvents {
				tokens += tokenCost(e)
				if isArtifactEvent(e) {
					hasArtifact = true
				}
				if isLeaseBlock(e) {
					blocked = true
				}
				if strings.TrimSpace(e.ArgsDigest) == "" {
					continue
				}
				id := callIdentity(e)
				if isRefusalEvent(e) {
					if priorRefusals[id] {
						looped = true
					}
					priorRefusals[id] = true
				}
				if isClearEvent(e) {
					delete(priorRefusals, id)
				}
			}
			if hasArtifact {
				continue
			}
			cause := CauseNoProgressGiveUp
			switch {
			case looped:
				cause = CauseLoopedRefusal
			case blocked:
				cause = CauseBlockedByLease
			}
			rep.AbandonedTurns++
			rep.WastedTokens += tokens
			causes.add(cause, 1, turnKey(turnEvents[0]), tokens)
		}
	}
	if rep.Turns > 0 {
		rep.AbandonedRate = round4(float64(rep.AbandonedTurns) / float64(rep.Turns))
	}
	rep.ByCause = causes.totals()
	return rep
}

// RenderFriction gives a compact human readout while keeping FoldFriction the
// primary machine interface.
func RenderFriction(rep FrictionReport) string {
	lines := []string{fmt.Sprintf("devex-friction: %d event(s), %d ranked class(es)", rep.Events, len(rep.Ranked))}
	if len(rep.Ranked) == 0 {
		return strings.Join(lines, "\n")
	}
	for _, r := range rep.Ranked {
		lines = append(lines, fmt.Sprintf("  %s: wasted_tokens=%d wasted_turns=%d events=%d", r.Class, r.WastedTokens, r.WastedTurns, r.Events))
	}
	return strings.Join(lines, "\n")
}

type readFileAcc struct {
	SessionID     string
	Path          string
	ContentDigest string
	Cause         string
	ReReads       int
	WastedTokens  int
	turns         map[string]bool
}

type retryEpisode struct {
	reason       string
	retryRows    int
	wastedTokens int
	turns        map[string]bool
}

type refusalAcc struct {
	Reason       string
	RefusedCalls int
	Cleared      int
	Looped       int
	NoRetry      int
	RetryRows    int
	WastedTokens int
	turns        map[string]bool
}

func closeRetryEpisode(ep *retryEpisode, cleared bool, byReason map[string]*refusalAcc) {
	r := byReason[ep.reason]
	if r == nil {
		r = &refusalAcc{Reason: ep.reason, turns: map[string]bool{}}
		byReason[ep.reason] = r
	}
	r.RefusedCalls++
	r.RetryRows += ep.retryRows
	r.WastedTokens += ep.wastedTokens
	for turn := range ep.turns {
		r.turns[turn] = true
	}
	switch {
	case cleared:
		r.Cleared++
	case ep.retryRows > 0:
		r.Looped++
	default:
		r.NoRetry++
	}
}

type laneAcc struct {
	Lane                  string
	Sessions              int
	SessionsWithoutAction int
	TokensBeforeAction    int
	TurnsBeforeAction     int
	tokenSamples          []int
	turnSamples           []int
	millisSamples         []int64
}

type rankBuilder struct {
	classes map[string]*rankAcc
}

type rankAcc struct {
	Class        string
	Events       int
	WastedTurns  int
	WastedTokens int
	causes       map[string]*CauseTotal
}

func newRankBuilder() *rankBuilder {
	return &rankBuilder{classes: map[string]*rankAcc{}}
}

func (b *rankBuilder) addAggregate(class, cause string, events, turns, tokens int) {
	class = normalizeClass(class)
	if class == "" || (turns == 0 && tokens == 0) {
		return
	}
	cause = cleanCauseLabel(cause)
	c := b.classes[class]
	if c == nil {
		c = &rankAcc{Class: class, causes: map[string]*CauseTotal{}}
		b.classes[class] = c
	}
	c.Events += events
	c.WastedTurns += turns
	c.WastedTokens += tokens
	sub := c.causes[cause]
	if sub == nil {
		sub = &CauseTotal{Cause: cause}
		c.causes[cause] = sub
	}
	sub.Events += events
	sub.WastedTurns += turns
	sub.WastedTokens += tokens
}

func (b *rankBuilder) ranked() []FrictionClassTotal {
	out := make([]FrictionClassTotal, 0, len(b.classes))
	for _, c := range b.classes {
		row := FrictionClassTotal{
			Class:        c.Class,
			Events:       c.Events,
			WastedTurns:  c.WastedTurns,
			WastedTokens: c.WastedTokens,
		}
		for _, cause := range c.causes {
			row.Causes = append(row.Causes, *cause)
		}
		sortCauseTotals(row.Causes)
		out = append(out, row)
	}
	sort.Slice(out, func(i, j int) bool {
		a, b := out[i], out[j]
		if a.WastedTokens != b.WastedTokens {
			return a.WastedTokens > b.WastedTokens
		}
		if a.WastedTurns != b.WastedTurns {
			return a.WastedTurns > b.WastedTurns
		}
		if a.Events != b.Events {
			return a.Events > b.Events
		}
		return a.Class < b.Class
	})
	return out
}

type causeBuilder struct {
	causes map[string]*causeAcc
}

type causeAcc struct {
	cause  string
	events int
	tokens int
	turns  map[string]bool
}

func newCauseBuilder() *causeBuilder {
	return &causeBuilder{causes: map[string]*causeAcc{}}
}

func (b *causeBuilder) add(cause string, events int, turn string, tokens int) {
	cause = normalizeCause(cause)
	c := b.causes[cause]
	if c == nil {
		c = &causeAcc{cause: cause, turns: map[string]bool{}}
		b.causes[cause] = c
	}
	c.events += events
	c.tokens += tokens
	if turn != "" {
		c.turns[turn] = true
	}
}

func (b *causeBuilder) totals() []CauseTotal {
	out := make([]CauseTotal, 0, len(b.causes))
	for _, c := range b.causes {
		out = append(out, CauseTotal{
			Cause:        c.cause,
			Events:       c.events,
			WastedTurns:  len(c.turns),
			WastedTokens: c.tokens,
		})
	}
	sortCauseTotals(out)
	return out
}

func sortCauseTotals(out []CauseTotal) {
	sort.Slice(out, func(i, j int) bool {
		a, b := out[i], out[j]
		if a.WastedTokens != b.WastedTokens {
			return a.WastedTokens > b.WastedTokens
		}
		if a.WastedTurns != b.WastedTurns {
			return a.WastedTurns > b.WastedTurns
		}
		if a.Events != b.Events {
			return a.Events > b.Events
		}
		return a.Cause < b.Cause
	})
}

func explicitWaste(e ToolEvent) (class, cause string) {
	if c := normalizeClass(e.WasteClass); c != "" {
		return c, explicitCause(e)
	}
	outcome := normalizeOutcome(e.Outcome)
	reason := reasonToken(e.Reason)
	switch {
	case outcome == "hang" || outcome == "timeout" || outcome == "deadline-exceeded" ||
		strings.Contains(reason, "HANG") || strings.Contains(reason, "TIMEOUT"):
		return ClassHangRecovery, explicitCause(e)
	case outcome == ClassMisdiagnosedRed || strings.Contains(strings.ReplaceAll(reason, "_", "-"), "MISDIAGNOSED-RED"):
		return ClassMisdiagnosedRed, explicitCause(e)
	default:
		return "", ""
	}
}

func explicitCause(e ToolEvent) string {
	if r := strings.TrimSpace(e.Reason); r != "" {
		return r
	}
	if o := strings.TrimSpace(e.Outcome); o != "" {
		return o
	}
	if t := strings.TrimSpace(e.Tool); t != "" {
		return t
	}
	return "unspecified"
}

func isReadEvent(e ToolEvent) bool {
	a := normalizeAction(e.Action)
	t := normalizeAction(e.Tool)
	return a == "read" || t == "read"
}

func isArtifactEvent(e ToolEvent) bool {
	a := normalizeAction(e.Action)
	t := normalizeAction(e.Tool)
	return e.Edit || e.Commit || a == "edit" || a == "commit" || t == "edit" || t == "commit"
}

func isUsefulAction(e ToolEvent) bool {
	return e.UsefulAction || isArtifactEvent(e)
}

func isRefusalEvent(e ToolEvent) bool {
	v := normalizeOutcome(firstNonblank(e.Verdict, e.Outcome))
	switch v {
	case "deny", "denied", "refusal", "refused", "quarantine", "blocked", "result-deny":
		return true
	default:
		return false
	}
}

func isClearEvent(e ToolEvent) bool {
	v := normalizeOutcome(firstNonblank(e.Verdict, e.Outcome))
	switch v {
	case "allow", "allowed", "transform", "transformed", "ok", "success", "succeeded":
		return true
	default:
		return false
	}
}

func isLeaseBlock(e ToolEvent) bool {
	reason := reasonToken(e.Reason)
	return strings.Contains(reason, "LEASE") ||
		strings.Contains(reason, "COLLISION") ||
		strings.Contains(reason, "LOCK") ||
		strings.Contains(reason, "MERGE-IN-PROGRESS")
}

func readWasteCause(e ToolEvent) string {
	if c := normalizeCause(e.ReadCause); c != "" && c != "unspecified" {
		return c
	}
	reason := reasonToken(e.Reason)
	switch {
	case e.AfterCompaction || strings.Contains(reason, "COMPACTION"):
		return CausePostCompaction
	case e.AfterRevert || strings.Contains(reason, "REVERT"):
		return CausePostRevert
	default:
		return CauseDefensive
	}
}

func refusalReason(e ToolEvent) string {
	if r := strings.TrimSpace(e.Reason); r != "" {
		return r
	}
	return "(blank)"
}

func tokenCost(e ToolEvent) int {
	if e.Tokens > 0 {
		return e.Tokens
	}
	n := 0
	if e.PromptTokens > 0 {
		n += e.PromptTokens
	}
	if e.CompletionTokens > 0 {
		n += e.CompletionTokens
	}
	return n
}

func orderedEvents(events []ToolEvent) []ToolEvent {
	type indexed struct {
		event ToolEvent
		index int
	}
	rows := make([]indexed, len(events))
	for i, e := range events {
		rows[i] = indexed{event: e, index: i}
	}
	sort.SliceStable(rows, func(i, j int) bool {
		a, b := rows[i].event, rows[j].event
		if sessionID(a) != sessionID(b) {
			return sessionID(a) < sessionID(b)
		}
		if a.Turn != b.Turn {
			return a.Turn < b.Turn
		}
		if a.Seq != b.Seq {
			return a.Seq < b.Seq
		}
		return rows[i].index < rows[j].index
	})
	out := make([]ToolEvent, len(rows))
	for i, row := range rows {
		out[i] = row.event
	}
	return out
}

func groupBySession(events []ToolEvent) [][]ToolEvent {
	ordered := orderedEvents(events)
	var out [][]ToolEvent
	for _, e := range ordered {
		session := sessionID(e)
		if len(out) == 0 || sessionID(out[len(out)-1][0]) != session {
			out = append(out, []ToolEvent{e})
			continue
		}
		out[len(out)-1] = append(out[len(out)-1], e)
	}
	return out
}

func groupByTurn(events []ToolEvent) [][]ToolEvent {
	if len(events) == 0 {
		return nil
	}
	var out [][]ToolEvent
	for _, e := range events {
		if len(out) == 0 || e.Turn != out[len(out)-1][0].Turn {
			out = append(out, []ToolEvent{e})
			continue
		}
		out[len(out)-1] = append(out[len(out)-1], e)
	}
	return out
}

func sessionLane(events []ToolEvent) string {
	for _, e := range events {
		if lane := strings.TrimSpace(e.Lane); lane != "" {
			return lane
		}
	}
	return "(unknown)"
}

func firstUsefulIndex(events []ToolEvent) int {
	for i, e := range events {
		if isUsefulAction(e) {
			return i
		}
	}
	return -1
}

func completedBeforeAction(e, action ToolEvent) bool {
	if action.Turn == 0 {
		return e.Turn == 0
	}
	return e.Turn < action.Turn
}

func millisToAction(first, action ToolEvent) (int64, bool) {
	if first.AtMillis == 0 || action.AtMillis == 0 || action.AtMillis < first.AtMillis {
		return 0, false
	}
	return action.AtMillis - first.AtMillis, true
}

func medianInts(xs []int) float64 {
	if len(xs) == 0 {
		return 0
	}
	ys := append([]int(nil), xs...)
	sort.Ints(ys)
	n := len(ys)
	if n%2 == 1 {
		return float64(ys[n/2])
	}
	return round4(float64(ys[n/2-1]+ys[n/2]) / 2)
}

func medianInt64s(xs []int64) float64 {
	if len(xs) == 0 {
		return 0
	}
	ys := append([]int64(nil), xs...)
	sort.Slice(ys, func(i, j int) bool { return ys[i] < ys[j] })
	n := len(ys)
	if n%2 == 1 {
		return float64(ys[n/2])
	}
	return round4(float64(ys[n/2-1]+ys[n/2]) / 2)
}

func callIdentity(e ToolEvent) string {
	return strings.ToLower(strings.TrimSpace(firstNonblank(e.Tool, e.Action))) + "\x00" + strings.TrimSpace(e.ArgsDigest)
}

func sessionID(e ToolEvent) string {
	if s := strings.TrimSpace(e.SessionID); s != "" {
		return s
	}
	return "(session)"
}

func cleanSession(s string) string {
	if s == "(session)" {
		return ""
	}
	return s
}

func turnKey(e ToolEvent) string {
	return sessionID(e) + "\x00" + strconv.Itoa(e.Turn)
}

func firstNonblank(a, b string) string {
	if strings.TrimSpace(a) != "" {
		return a
	}
	return b
}

func normalizeAction(s string) string {
	s = strings.TrimSpace(strings.ToLower(s))
	s = strings.ReplaceAll(s, "_", "-")
	return s
}

func normalizeOutcome(s string) string {
	return normalizeAction(s)
}

func normalizeCause(s string) string {
	if c := normalizeClass(s); c != "" {
		return c
	}
	return "unspecified"
}

func cleanCauseLabel(s string) string {
	if s = strings.TrimSpace(s); s != "" {
		return s
	}
	return "unspecified"
}

func reasonToken(s string) string {
	s = strings.TrimSpace(strings.ToUpper(s))
	s = strings.ReplaceAll(s, "_", "-")
	return s
}
