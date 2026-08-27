package bgloop

import (
	"context"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dormancy"
)

// Job is the scheduler's lower-tier view of a persisted loop job.
type Job interface {
	JobID() string
}

// DutyCycle is the scheduler's lower-tier view of a recurring on-window.
type DutyCycle interface {
	Active(time.Time) bool
	NextOn(time.Time) (time.Time, error)
}

// StoredWake carries one armed job and its durable alarm state.
type StoredWake struct {
	Job    Job
	WakeAt time.Time
	Duty   DutyCycle
}

// RegistryStore isolates durable scheduling from the loop registry's storage
// representation. loopmgr provides the production adapter.
type RegistryStore interface {
	Validate(path string) error
	LookupArmed(path, jobID string) (Job, bool, error)
	SetWake(path, jobID string, at time.Time, duty DutyCycle, updatedAt time.Time) error
	ArmedWakes(path string) ([]StoredWake, error)
	MoveWake(path, jobID string, at, updatedAt time.Time) error
	ClearWake(path, jobID string, updatedAt time.Time) error
}

// Admission is the content-free result of a rehydration policy decision.
type Admission struct {
	Admitted  bool
	RefusedBy string
	Detail    string
}

// RehydrationGate is the lower-tier policy seam used before firing a wake.
type RehydrationGate interface {
	Admit(context.Context, dormancy.Horizon) Admission
}
