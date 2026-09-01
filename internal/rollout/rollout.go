package rollout

import (
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
)

const cohortBuckets = 10_000

// Generation identifies an immutable build. ID is its operator-facing name;
// Digest binds the name to exact artifact bytes.
type Generation struct {
	ID     string
	Digest string
}

func (g Generation) isZero() bool { return g.ID == "" && g.Digest == "" }

func (g Generation) validate(label string) error {
	if g.ID == "" || g.Digest == "" {
		return fmt.Errorf("%s generation requires ID and digest", label)
	}
	return nil
}

// State is the complete admission policy for new sessions. Existing sessions
// keep the generation they were assigned; changing State only affects future
// Select calls.
type State struct {
	Stable                Generation
	Canary                Generation
	ExposureBasisPoints   int
	ExposureConcurrentCap int
	CohortSalt            string
	AdmissionsPaused      bool
	ExposureKilled        bool
	LastKnownGood         Generation
	RejectedGenerationIDs map[string]bool
	RejectedDigests       map[string]bool
}

// Reason explains why Select chose a generation.
type Reason string

const (
	ReasonCanarySelected      Reason = "canary_cohort"
	ReasonStableNoCanary      Reason = "stable_no_canary"
	ReasonStablePaused        Reason = "stable_admissions_paused"
	ReasonStableCanaryKilled  Reason = "stable_canary_killed"
	ReasonStableZeroExposure  Reason = "stable_zero_canary_percentage"
	ReasonStableExposureCap   Reason = "stable_canary_cap_reached"
	ReasonStableOutsideCohort Reason = "stable_cohort"
)

// Selection is the immutable generation assignment for a new session.
type Selection struct {
	Generation Generation
	Canary     bool
	Reason     Reason
}

// Validate rejects states that could make admission ambiguous or unbounded.
func (s State) Validate() error {
	if err := s.Stable.validate("stable"); err != nil {
		return err
	}
	if err := s.LastKnownGood.validate("last-known-good"); err != nil {
		return err
	}
	if s.ExposureBasisPoints < 0 || s.ExposureBasisPoints > cohortBuckets {
		return fmt.Errorf("canary basis points must be between 0 and %d", cohortBuckets)
	}
	if s.ExposureConcurrentCap < 0 {
		return errors.New("canary concurrent cap cannot be negative")
	}

	if s.Canary.isZero() {
		if s.ExposureBasisPoints != 0 || s.ExposureConcurrentCap != 0 || s.CohortSalt != "" {
			return errors.New("canary controls require a canary generation")
		}
		return nil
	}
	if err := s.Canary.validate("canary"); err != nil {
		return err
	}
	if s.Canary.ID == s.Stable.ID || s.Canary.Digest == s.Stable.Digest {
		return errors.New("canary must differ from stable generation and digest")
	}
	if s.Canary.ID == s.LastKnownGood.ID || s.Canary.Digest == s.LastKnownGood.Digest {
		return errors.New("canary must differ from last-known-good generation and digest")
	}
	if s.RejectedGenerationIDs[s.Canary.ID] || s.RejectedDigests[s.Canary.Digest] {
		return errors.New("canary generation or digest was previously rejected")
	}
	if s.ExposureBasisPoints > 0 {
		if s.ExposureConcurrentCap == 0 {
			return errors.New("positive canary percentage requires a positive concurrent cap")
		}
		if s.CohortSalt == "" {
			return errors.New("positive canary percentage requires a cohort salt")
		}
	}
	return nil
}

// Stage returns a copy of State with canary admission configured. A rejected
// generation name or digest cannot be staged again under a different alias.
func (s State) Stage(canary Generation, basisPoints, concurrentCap int, cohortSalt string) (State, error) {
	next := s
	next.Canary = canary
	next.ExposureBasisPoints = basisPoints
	next.ExposureConcurrentCap = concurrentCap
	next.CohortSalt = cohortSalt
	next.AdmissionsPaused = false
	next.ExposureKilled = false
	if err := next.Validate(); err != nil {
		return State{}, err
	}
	return next, nil
}

// Select deterministically assigns a new session. Percentage eligibility and
// the absolute concurrent cap must both allow canary admission.
func (s State) Select(sessionID string, activeCanaryCount int) (Selection, error) {
	if err := s.Validate(); err != nil {
		return Selection{}, err
	}
	if sessionID == "" {
		return Selection{}, errors.New("session ID is required")
	}
	if activeCanaryCount < 0 {
		return Selection{}, errors.New("active canary count cannot be negative")
	}

	stable := func(reason Reason) Selection {
		return Selection{Generation: s.Stable, Reason: reason}
	}
	if s.Canary.isZero() {
		if s.ExposureKilled {
			return stable(ReasonStableCanaryKilled), nil
		}
		return stable(ReasonStableNoCanary), nil
	}
	if s.AdmissionsPaused {
		return stable(ReasonStablePaused), nil
	}
	if s.ExposureKilled {
		return stable(ReasonStableCanaryKilled), nil
	}
	if s.ExposureBasisPoints == 0 {
		return stable(ReasonStableZeroExposure), nil
	}
	if activeCanaryCount >= s.ExposureConcurrentCap {
		return stable(ReasonStableExposureCap), nil
	}
	if cohortBucket(sessionID, s.CohortSalt) >= s.ExposureBasisPoints {
		return stable(ReasonStableOutsideCohort), nil
	}
	return Selection{Generation: s.Canary, Canary: true, Reason: ReasonCanarySelected}, nil
}

// Abort rejects the canary, stops canary admission, and restores stable to the
// last-known-good generation. Rejection survives future Stage calls.
func (s State) Abort() (State, error) {
	if err := s.Validate(); err != nil {
		return State{}, err
	}
	if s.Canary.isZero() {
		return State{}, errors.New("cannot abort without a canary generation")
	}

	next := s
	next.RejectedGenerationIDs = cloneSet(s.RejectedGenerationIDs)
	next.RejectedDigests = cloneSet(s.RejectedDigests)
	next.RejectedGenerationIDs[s.Canary.ID] = true
	next.RejectedDigests[s.Canary.Digest] = true
	next.Stable = s.LastKnownGood
	next.Canary = Generation{}
	next.ExposureBasisPoints = 0
	next.ExposureConcurrentCap = 0
	next.CohortSalt = ""
	next.AdmissionsPaused = false
	next.ExposureKilled = true
	if err := next.Validate(); err != nil {
		return State{}, err
	}
	return next, nil
}

// Promote makes the canary stable and preserves the prior stable generation as
// the last-known-good rollback point.
func (s State) Promote() (State, error) {
	if err := s.Validate(); err != nil {
		return State{}, err
	}
	if s.Canary.isZero() {
		return State{}, errors.New("cannot promote without a canary generation")
	}
	if s.AdmissionsPaused || s.ExposureKilled {
		return State{}, errors.New("cannot promote a paused or killed canary")
	}

	next := s
	next.LastKnownGood = s.Stable
	next.Stable = s.Canary
	next.Canary = Generation{}
	next.ExposureBasisPoints = 0
	next.ExposureConcurrentCap = 0
	next.CohortSalt = ""
	next.AdmissionsPaused = false
	next.ExposureKilled = false
	if err := next.Validate(); err != nil {
		return State{}, err
	}
	return next, nil
}

func cohortBucket(sessionID, salt string) int {
	sum := sha256.Sum256([]byte(salt + "\x00" + sessionID))
	return int(binary.BigEndian.Uint64(sum[:8]) % cohortBuckets)
}

func cloneSet(src map[string]bool) map[string]bool {
	dst := make(map[string]bool, len(src)+1)
	for value, rejected := range src {
		if rejected {
			dst[value] = true
		}
	}
	return dst
}
