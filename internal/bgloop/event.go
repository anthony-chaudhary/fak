package bgloop

import (
	"context"
	"errors"
	"fmt"
	"hash/fnv"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dormancy"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/rehydrate"
)

var ErrUnauthenticatedEvent = errors.New("bgloop: unauthenticated wake event")

type Event struct {
	JobID             string
	Principal         string
	SignatureVerified bool
	Horizon           dormancy.Horizon
}

type WakeRequest struct {
	Job       loopmgr.Job
	Cause     string
	DueAt     time.Time
	Principal string
}

type EventScheduler struct {
	RegistryPath string
	Gate         *rehydrate.Gate
	Clock        Clock
	JitterWindow time.Duration
	Admit        func(context.Context, WakeRequest) (func(), error)
	Fire         func(context.Context, loopmgr.Job) error
}

func (s EventScheduler) DeliverEvent(ctx context.Context, event Event) (rehydrate.Admission, error) {
	if !event.SignatureVerified || strings.TrimSpace(event.Principal) == "" {
		return rehydrate.Admission{}, ErrUnauthenticatedEvent
	}
	registry, err := loopmgr.LoadRegistry(s.RegistryPath)
	if err != nil {
		return rehydrate.Admission{}, err
	}
	job, ok := registry.Get(event.JobID)
	if !ok || !job.State.Armed() {
		return rehydrate.Admission{}, fmt.Errorf("bgloop: dormant job %q is not armed", event.JobID)
	}
	admission := s.Gate.Admit(ctx, event.Horizon)
	if !admission.Admitted {
		return admission, fmt.Errorf("bgloop: rehydration refused by %s: %s", admission.RefusedBy, admission.Detail)
	}
	request := WakeRequest{Job: job, Cause: "authenticated_event", DueAt: s.clock().Now(), Principal: event.Principal}
	if s.Admit != nil {
		release, err := s.Admit(ctx, request)
		if err != nil {
			return admission, fmt.Errorf("bgloop: runtime admission: %w", err)
		}
		if release != nil {
			defer release()
		}
	}
	if s.Fire == nil {
		return admission, errors.New("bgloop: nil event callback")
	}
	if err := s.Fire(ctx, job); err != nil {
		return admission, err
	}
	return admission, nil
}

func (s EventScheduler) ScheduleWakeCohort(jobIDs []string, wakeAt time.Time) ([]time.Time, error) {
	if len(jobIDs) == 0 {
		return nil, nil
	}
	if s.JitterWindow < 0 {
		return nil, errors.New("bgloop: jitter window must not be negative")
	}
	ids := append([]string(nil), jobIDs...)
	sort.Slice(ids, func(i, j int) bool {
		hi, hj := stableWakeHash(ids[i]), stableWakeHash(ids[j])
		if hi == hj {
			return ids[i] < ids[j]
		}
		return hi < hj
	})
	seen := make(map[string]struct{}, len(ids))
	deadlines := make([]time.Time, 0, len(ids))
	durable, err := NewDurableScheduler(s.RegistryPath, s.clock(), func(context.Context, loopmgr.Job) error { return nil })
	if err != nil {
		return nil, err
	}
	for rank, id := range ids {
		id = strings.TrimSpace(id)
		if id == "" {
			return nil, errors.New("bgloop: empty cohort job id")
		}
		if _, duplicate := seen[id]; duplicate {
			return nil, fmt.Errorf("bgloop: duplicate cohort job %q", id)
		}
		seen[id] = struct{}{}
		offset := time.Duration(0)
		if len(ids) > 1 {
			offset = time.Duration(int64(s.JitterWindow) * int64(rank) / int64(len(ids)-1))
		}
		deadline := wakeAt.Add(offset)
		if err := durable.SleepUntil(id, deadline, nil); err != nil {
			return nil, err
		}
		deadlines = append(deadlines, deadline)
	}
	sort.Slice(deadlines, func(i, j int) bool { return deadlines[i].Before(deadlines[j]) })
	return deadlines, nil
}

func (s EventScheduler) clock() Clock {
	if s.Clock != nil {
		return s.Clock
	}
	return WallClock{}
}
func stableWakeHash(id string) uint64 {
	h := fnv.New64a()
	_, _ = h.Write([]byte(id))
	return h.Sum64()
}
