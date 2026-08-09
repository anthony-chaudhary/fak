package microagent

import (
	"errors"
	"sort"
	"time"
)

type TenantEnvelope struct {
	Tenant         string
	Weight         int
	MaxQueued      int
	MaxConcurrent  int
	MaxSpendMicros int64
	RatePerMinute  int
}

type TenantTask struct {
	ID          string
	Tenant      string
	Deadline    time.Time
	CostMicros  int64
	Cancelled   bool
	Interactive bool
	EnqueuedAt  time.Time
}

type FairnessSnapshot struct {
	Scheduled   map[string]int   `json:"scheduled"`
	Cancelled   map[string]int   `json:"cancelled"`
	SpendMicros map[string]int64 `json:"spend_micros"`
	MaxLag      map[string]int   `json:"max_service_lag"`
}

type TenantQueue struct {
	envelopes  map[string]TenantEnvelope
	queues     map[string][]TenantTask
	served     map[string]int
	spend      map[string]int64
	cancelled  map[string]int
	rateWindow map[string][]time.Time
}

func NewTenantQueue(envelopes []TenantEnvelope) (*TenantQueue, error) {
	s := &TenantQueue{envelopes: map[string]TenantEnvelope{}, queues: map[string][]TenantTask{}, served: map[string]int{}, spend: map[string]int64{}, cancelled: map[string]int{}, rateWindow: map[string][]time.Time{}}
	for _, e := range envelopes {
		if e.Tenant == "" || e.Weight <= 0 || e.MaxQueued <= 0 || e.MaxConcurrent <= 0 {
			return nil, errors.New("microagent: tenant envelope fields must be positive")
		}
		if _, ok := s.envelopes[e.Tenant]; ok {
			return nil, errors.New("microagent: duplicate tenant")
		}
		s.envelopes[e.Tenant] = e
	}
	if len(s.envelopes) == 0 {
		return nil, errors.New("microagent: at least one tenant is required")
	}
	return s, nil
}

func (s *TenantQueue) Submit(t TenantTask) error {
	e, ok := s.envelopes[t.Tenant]
	if !ok {
		return errors.New("microagent: unknown tenant")
	}
	if t.ID == "" {
		return errors.New("microagent: task ID required")
	}
	if t.Cancelled {
		s.cancelled[t.Tenant]++
		return nil
	}
	if len(s.queues[t.Tenant]) >= e.MaxQueued {
		return errors.New("microagent: tenant queue envelope exhausted")
	}
	if e.MaxSpendMicros > 0 && s.spend[t.Tenant]+t.CostMicros > e.MaxSpendMicros {
		return errors.New("microagent: tenant spend envelope exhausted")
	}
	s.queues[t.Tenant] = append(s.queues[t.Tenant], t)
	return nil
}

func (s *TenantQueue) Next(now time.Time) (TenantTask, bool) {
	type candidate struct {
		tenant      string
		score       float64
		deadline    time.Time
		interactive bool
	}
	var cs []candidate
	for tenant, q := range s.queues {
		if len(q) == 0 {
			continue
		}
		e := s.envelopes[tenant]
		cutoff := now.Add(-time.Minute)
		kept := s.rateWindow[tenant][:0]
		for _, at := range s.rateWindow[tenant] {
			if at.After(cutoff) {
				kept = append(kept, at)
			}
		}
		s.rateWindow[tenant] = kept
		if e.RatePerMinute > 0 && len(kept) >= e.RatePerMinute {
			continue
		}
		head := q[0]
		cs = append(cs, candidate{tenant: tenant, score: float64(s.served[tenant]+1) / float64(e.Weight), deadline: head.Deadline, interactive: head.Interactive})
	}
	if len(cs) == 0 {
		return TenantTask{}, false
	}
	sort.Slice(cs, func(i, j int) bool {
		if cs[i].score != cs[j].score {
			return cs[i].score < cs[j].score
		}
		if cs[i].interactive != cs[j].interactive {
			return cs[i].interactive
		}
		if !cs[i].deadline.Equal(cs[j].deadline) {
			if cs[i].deadline.IsZero() {
				return false
			}
			if cs[j].deadline.IsZero() {
				return true
			}
			return cs[i].deadline.Before(cs[j].deadline)
		}
		return cs[i].tenant < cs[j].tenant
	})
	tenant := cs[0].tenant
	t := s.queues[tenant][0]
	s.queues[tenant] = s.queues[tenant][1:]
	s.served[tenant]++
	s.spend[tenant] += t.CostMicros
	s.rateWindow[tenant] = append(s.rateWindow[tenant], now)
	return t, true
}

func (s *TenantQueue) Snapshot() FairnessSnapshot {
	o := FairnessSnapshot{Scheduled: map[string]int{}, Cancelled: map[string]int{}, SpendMicros: map[string]int64{}, MaxLag: map[string]int{}}
	maxNorm := 0.0
	for tenant, e := range s.envelopes {
		norm := float64(s.served[tenant]) / float64(e.Weight)
		if norm > maxNorm {
			maxNorm = norm
		}
	}
	for tenant, e := range s.envelopes {
		o.Scheduled[tenant] = s.served[tenant]
		o.Cancelled[tenant] = s.cancelled[tenant]
		o.SpendMicros[tenant] = s.spend[tenant]
		lag := int(maxNorm*float64(e.Weight)) - s.served[tenant]
		if lag < 0 {
			lag = 0
		}
		o.MaxLag[tenant] = lag
	}
	return o
}
