// Package incidentrsi classifies operational incidents and decides when a
// trusted development checkout should launch an RSI investigation.
package incidentrsi

import (
	"crypto/sha256"
	"encoding/hex"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode"
)

// Source identifies an adapter-defined incident source.
type Source string

const (
	SourceUnexpectedHook   Source = "unexpected_hook"
	SourceGatewayTransport Source = "gateway_transport"
)

// ErrorClass and CauseIdentity are adapter-owned structural identities, not error text.
type ErrorClass string
type CauseIdentity string

// Action is the next safe action for an incident observation.
type Action string

const (
	ActionObserve Action = "observe"
	ActionLaunch  Action = "launch"
	ActionUpdate  Action = "update"
	ActionDoctor  Action = "doctor"
	ActionNoop    Action = "noop"
)

const doctorRecommendation = "run the built-in doctor and review its redacted report before filing an issue"

// Input contains only structural incident identity. Callers must classify an
// error before this boundary; raw messages, arguments, prompts, paths,
// hostnames, tokens, and environment values do not belong in these fields.
type Input struct {
	Source        Source
	Operation     string
	ErrorClass    ErrorClass
	CauseIdentity CauseIdentity
	OccurredAt    time.Time
	Developer     bool
	Expected      bool
}

// Config controls burst coalescing and retained incident state.
type Config struct {
	Threshold       int
	Cooldown        time.Duration
	MaxFingerprints int
	StaleAfter      time.Duration
}

// DefaultConfig is conservative enough for unattended hooks while keeping
// attacker-controlled distinct identities bounded.
func DefaultConfig() Config {
	return Config{
		Threshold:       3,
		Cooldown:        30 * time.Minute,
		MaxFingerprints: 1024,
		StaleAfter:      24 * time.Hour,
	}
}

// Decision describes the single action selected for an observation.
type Decision struct {
	Action         Action
	Count          int
	Fingerprint    string
	Reason         string
	NextEligibleAt time.Time
	Recommendation string
}

type incidentState struct {
	count          int
	lastSeen       time.Time
	nextEligibleAt time.Time
}

// Kernel serializes decisions so a threshold crossing produces exactly one
// launch even when identical observations arrive concurrently.
type Kernel struct {
	mu     sync.Mutex
	config Config
	states map[string]incidentState
}

// New returns a decision kernel with invalid or missing limits replaced by
// safe defaults.
func New(config Config) *Kernel {
	defaults := DefaultConfig()
	if config.Threshold <= 0 {
		config.Threshold = defaults.Threshold
	}
	if config.Cooldown <= 0 {
		config.Cooldown = defaults.Cooldown
	}
	if config.MaxFingerprints <= 0 {
		config.MaxFingerprints = defaults.MaxFingerprints
	}
	if config.StaleAfter <= 0 {
		config.StaleAfter = defaults.StaleAfter
	}
	return &Kernel{config: config, states: make(map[string]incidentState)}
}

// Observe classifies one occurrence and updates debounce state when automatic
// RSI is eligible.
func (k *Kernel) Observe(input Input) Decision {
	fingerprint := Fingerprint(input)
	if !validSource(input.Source) {
		return Decision{Action: ActionNoop, Fingerprint: fingerprint, Reason: "unsupported source class"}
	}
	if !input.Developer {
		return Decision{
			Action:         ActionDoctor,
			Count:          1,
			Fingerprint:    fingerprint,
			Reason:         "automatic RSI is limited to trusted development checkouts",
			Recommendation: doctorRecommendation,
		}
	}
	if input.Expected {
		return Decision{Action: ActionNoop, Fingerprint: fingerprint, Reason: "expected failure or known remedy"}
	}

	now := input.OccurredAt
	if now.IsZero() {
		now = time.Now().UTC()
	}

	k.mu.Lock()
	defer k.mu.Unlock()

	k.prune(now)
	state, exists := k.states[fingerprint]
	if !exists {
		k.makeRoom()
	}
	state.count++
	state.lastSeen = now

	decision := Decision{
		Action:      ActionObserve,
		Count:       state.count,
		Fingerprint: fingerprint,
		Reason:      "below launch threshold",
	}
	if state.count >= k.config.Threshold {
		switch {
		case state.nextEligibleAt.IsZero() || !now.Before(state.nextEligibleAt):
			state.nextEligibleAt = now.Add(k.config.Cooldown)
			decision.Action = ActionLaunch
			decision.Reason = "launch threshold reached"
			decision.NextEligibleAt = state.nextEligibleAt
		default:
			decision.Action = ActionUpdate
			decision.Reason = "identical incident coalesced during cooldown"
			decision.NextEligibleAt = state.nextEligibleAt
		}
	}
	k.states[fingerprint] = state
	return decision
}

// Fingerprint returns a content-free digest over normalized structural fields.
func Fingerprint(input Input) string {
	fields := []string{
		"incidentrsi-v1",
		normalize(string(input.Source)),
		normalize(input.Operation),
		normalize(string(input.ErrorClass)),
		normalize(string(input.CauseIdentity)),
	}
	h := sha256.New()
	for _, field := range fields {
		// A NUL separator is unambiguous because normalize never emits NUL.
		h.Write([]byte(field))
		h.Write([]byte{0})
	}
	return "irsi-v1-" + hex.EncodeToString(h.Sum(nil))
}

func validSource(source Source) bool {
	switch Source(normalize(string(source))) {
	case SourceUnexpectedHook, SourceGatewayTransport:
		return true
	default:
		return false
	}
}

func normalize(value string) string {
	var b strings.Builder
	separator := false
	for _, r := range strings.TrimSpace(value) {
		switch {
		case unicode.IsLetter(r), unicode.IsDigit(r):
			if separator && b.Len() > 0 {
				b.WriteByte('_')
			}
			separator = false
			b.WriteRune(unicode.ToLower(r))
		default:
			separator = true
		}
	}
	if b.Len() == 0 {
		return "unknown"
	}
	return b.String()
}

func (k *Kernel) prune(now time.Time) {
	for fingerprint, state := range k.states {
		if now.Sub(state.lastSeen) > k.config.StaleAfter {
			delete(k.states, fingerprint)
		}
	}
}

func (k *Kernel) makeRoom() {
	if len(k.states) < k.config.MaxFingerprints {
		return
	}
	type candidate struct {
		fingerprint string
		lastSeen    time.Time
	}
	candidates := make([]candidate, 0, len(k.states))
	for fingerprint, state := range k.states {
		candidates = append(candidates, candidate{fingerprint: fingerprint, lastSeen: state.lastSeen})
	}
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].lastSeen.Equal(candidates[j].lastSeen) {
			return candidates[i].fingerprint < candidates[j].fingerprint
		}
		return candidates[i].lastSeen.Before(candidates[j].lastSeen)
	})
	delete(k.states, candidates[0].fingerprint)
}
