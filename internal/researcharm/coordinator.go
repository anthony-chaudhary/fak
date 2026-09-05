package researcharm

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"net/http"
	"sort"
	"sync"
	"time"
)

// RequestLease is granted upon request admission and MUST be released via Done when the request finishes.
type RequestLease struct {
	RequestID     string
	ArmID         string
	CallerPID     int
	CallerProcess string
	StartedAt     time.Time
	c             *Coordinator
	once          sync.Once
}

// Done signals completion of the request, updating token and error metrics.
func (rl *RequestLease) Done(tokens int, err error) {
	if rl == nil || rl.c == nil {
		return
	}
	rl.once.Do(func() {
		rl.c.finishRequest(rl.RequestID, rl.ArmID, tokens, err)
	})
}

// Coordinator manages research project arms, request attribution, concurrency limits, and leases.
type Coordinator struct {
	mu                    sync.RWMutex
	arms                  map[string]*armState
	inflight              map[string]*InflightRequest
	leases                map[string]*LeaseInfo
	defaultMaxConcurrency int
	enforceLeases         bool
}

type armState struct {
	id             string
	group          string
	displayName    string
	maxConcurrency int
	activeRequests int
	totalRequests  int64
	totalTokens    int64
	errorCount     int64
	recentPIDs     []int
	lastSeen       time.Time
}

// NewCoordinator creates a new research arm coordinator.
// defaultMaxConcurrency sets the default concurrency ceiling per arm (0 = unlimited).
func NewCoordinator(defaultMaxConcurrency int) *Coordinator {
	return &Coordinator{
		arms:                  make(map[string]*armState),
		inflight:              make(map[string]*InflightRequest),
		leases:                make(map[string]*LeaseInfo),
		defaultMaxConcurrency: defaultMaxConcurrency,
	}
}

// SetEnforceLeases toggles whether arms are required to hold a lease before being admitted.
func (c *Coordinator) SetEnforceLeases(enforce bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.enforceLeases = enforce
}

// Admit checks whether an incoming request from the given origin may proceed.
func (c *Coordinator) Admit(ctx context.Context, r *http.Request, endpoint string, traceID string) (*RequestLease, error) {
	origin := ExtractOrigin(r, traceID)
	reqID := randomHex(8)
	now := time.Now()

	c.mu.Lock()
	defer c.mu.Unlock()

	c.pruneExpiredLeasesLocked(now)

	// 1. Check exclusive leases held by other arms.
	for _, l := range c.leases {
		if l.Mode == LeaseModeExclusive && l.ArmID != origin.ArmID {
			return nil, fmt.Errorf("%w (held by arm %q, PID %d)", ErrExclusiveLeaseHeld, l.ArmID, l.HolderPID)
		}
	}

	// 2. Check lease enforcement if required.
	var activeLease *LeaseInfo
	for _, l := range c.leases {
		if l.ArmID == origin.ArmID {
			activeLease = l
			break
		}
	}
	if c.enforceLeases && activeLease == nil {
		return nil, fmt.Errorf("%w for arm %q", ErrLeaseRequired, origin.ArmID)
	}

	// 3. Find or register arm.
	arm, ok := c.arms[origin.ArmID]
	if !ok {
		arm = &armState{
			id:             origin.ArmID,
			group:          origin.ArmGroup,
			displayName:    origin.ArmID,
			maxConcurrency: c.defaultMaxConcurrency,
			lastSeen:       now,
		}
		c.arms[origin.ArmID] = arm
	}
	arm.lastSeen = now

	// 4. Determine effective concurrency ceiling.
	effectiveLimit := arm.maxConcurrency
	if activeLease != nil && activeLease.Concurrency > 0 {
		effectiveLimit = activeLease.Concurrency
	}

	// 5. Check concurrency limit.
	if effectiveLimit > 0 && arm.activeRequests >= effectiveLimit {
		return nil, fmt.Errorf("%w: arm %q at %d/%d active requests",
			ErrArmConcurrencyExceeded, origin.ArmID, arm.activeRequests, effectiveLimit)
	}

	// 6. Record admission.
	arm.activeRequests++
	if origin.CallerPID > 0 {
		addRecentPID(arm, origin.CallerPID)
	}

	inflight := &InflightRequest{
		RequestID:     reqID,
		ArmID:         origin.ArmID,
		ArmGroup:      origin.ArmGroup,
		CallerPID:     origin.CallerPID,
		CallerProcess: origin.CallerProcess,
		Endpoint:      endpoint,
		TraceID:       traceID,
		StartedAt:     now,
		RemoteAddr:    origin.RemoteAddr,
		UserAgent:     origin.UserAgent,
	}
	c.inflight[reqID] = inflight

	return &RequestLease{
		RequestID:     reqID,
		ArmID:         origin.ArmID,
		CallerPID:     origin.CallerPID,
		CallerProcess: origin.CallerProcess,
		StartedAt:     now,
		c:             c,
	}, nil
}

func (c *Coordinator) finishRequest(reqID, armID string, tokens int, err error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	delete(c.inflight, reqID)

	if arm, ok := c.arms[armID]; ok {
		if arm.activeRequests > 0 {
			arm.activeRequests--
		}
		arm.totalRequests++
		if tokens > 0 {
			arm.totalTokens += int64(tokens)
		}
		if err != nil {
			arm.errorCount++
		}
		arm.lastSeen = time.Now()
	}
}

// AcquireLease attempts to acquire or renew a lease for an arm.
func (c *Coordinator) AcquireLease(req LeaseRequest) (*LeaseInfo, error) {
	if req.ArmID == "" {
		return nil, fmt.Errorf("researcharm: arm_id required")
	}
	if req.Mode == "" {
		req.Mode = LeaseModeShared
	}
	if req.TTL <= 0 {
		req.TTL = 5 * time.Minute
	}

	now := time.Now()
	c.mu.Lock()
	defer c.mu.Unlock()

	c.pruneExpiredLeasesLocked(now)

	// If exclusive requested, ensure no other arm holds an active lease or has in-flight requests.
	if req.Mode == LeaseModeExclusive {
		for _, l := range c.leases {
			if l.ArmID != req.ArmID {
				return nil, fmt.Errorf("%w: active lease held by arm %q", ErrExclusiveLeaseHeld, l.ArmID)
			}
		}
		for _, inf := range c.inflight {
			if inf.ArmID != req.ArmID {
				return nil, fmt.Errorf("researcharm: cannot acquire exclusive lease while arm %q has in-flight requests", inf.ArmID)
			}
		}
	} else {
		// If shared requested, ensure no exclusive lease is held by another arm.
		for _, l := range c.leases {
			if l.Mode == LeaseModeExclusive && l.ArmID != req.ArmID {
				return nil, fmt.Errorf("%w: exclusive lease held by arm %q", ErrExclusiveLeaseHeld, l.ArmID)
			}
		}
	}

	leaseID := "lease-" + randomHex(8)
	token := randomHex(16)

	info := &LeaseInfo{
		ID:          leaseID,
		ArmID:       req.ArmID,
		HolderPID:   req.HolderPID,
		Mode:        req.Mode,
		Concurrency: req.Concurrency,
		Token:       token,
		CreatedAt:   now,
		ExpiresAt:   now.Add(req.TTL),
	}

	// Replace any previous lease for this arm
	for k, l := range c.leases {
		if l.ArmID == req.ArmID {
			delete(c.leases, k)
		}
	}
	c.leases[leaseID] = info

	// Ensure arm is registered
	if _, ok := c.arms[req.ArmID]; !ok {
		c.arms[req.ArmID] = &armState{
			id:             req.ArmID,
			group:          deriveGroup(req.ArmID),
			displayName:    req.ArmID,
			maxConcurrency: c.defaultMaxConcurrency,
			lastSeen:       now,
		}
	}

	// Return a copy with token visible to the caller
	res := *info
	return &res, nil
}

// ReleaseLease releases an existing lease.
func (c *Coordinator) ReleaseLease(leaseID, token string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	l, ok := c.leases[leaseID]
	if !ok {
		// Look up by arm ID if not found by lease ID
		for id, lease := range c.leases {
			if lease.ArmID == leaseID {
				l = lease
				leaseID = id
				ok = true
				break
			}
		}
	}
	if !ok {
		return ErrLeaseNotFound
	}

	if token != "" && l.Token != token {
		return ErrInvalidLeaseToken
	}

	delete(c.leases, leaseID)
	return nil
}

// SetLimit updates the max concurrency limit for an arm.
func (c *Coordinator) SetLimit(armID string, maxConcurrency int) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	arm, ok := c.arms[armID]
	if !ok {
		arm = &armState{
			id:          armID,
			group:       deriveGroup(armID),
			displayName: armID,
			lastSeen:    time.Now(),
		}
		c.arms[armID] = arm
	}
	arm.maxConcurrency = maxConcurrency
	return nil
}

// Snapshot returns an immutable point-in-time snapshot of arms, active requests, and leases.
func (c *Coordinator) Snapshot() Snapshot {
	c.mu.RLock()
	defer c.mu.RUnlock()

	now := time.Now()
	var arms []ArmInfo
	for _, a := range c.arms {
		info := ArmInfo{
			ID:             a.id,
			Group:          a.group,
			DisplayName:    a.displayName,
			MaxConcurrency: a.maxConcurrency,
			ActiveRequests: a.activeRequests,
			TotalRequests:  a.totalRequests,
			TotalTokens:    a.totalTokens,
			ErrorCount:     a.errorCount,
			RecentPIDs:     append([]int(nil), a.recentPIDs...),
			LastSeen:       a.lastSeen,
		}
		for _, l := range c.leases {
			if l.ArmID == a.id && now.Before(l.ExpiresAt) {
				cp := *l
				cp.Token = "" // scrub token in public snapshot
				info.ActiveLease = &cp
				break
			}
		}
		arms = append(arms, info)
	}

	sort.Slice(arms, func(i, j int) bool {
		if arms[i].ActiveRequests != arms[j].ActiveRequests {
			return arms[i].ActiveRequests > arms[j].ActiveRequests
		}
		return arms[i].TotalRequests > arms[j].TotalRequests
	})

	var inflight []InflightRequest
	for _, inf := range c.inflight {
		inflight = append(inflight, *inf)
	}
	sort.Slice(inflight, func(i, j int) bool {
		return inflight[i].StartedAt.Before(inflight[j].StartedAt)
	})

	var leases []LeaseInfo
	for _, l := range c.leases {
		if now.Before(l.ExpiresAt) {
			cp := *l
			cp.Token = ""
			leases = append(leases, cp)
		}
	}

	return Snapshot{
		Timestamp:     now,
		TotalInflight: len(inflight),
		TotalArms:     len(arms),
		Arms:          arms,
		Inflight:      inflight,
		ActiveLeases:  leases,
	}
}

// ActiveInflight returns all current in-flight requests.
func (c *Coordinator) ActiveInflight() []InflightRequest {
	c.mu.RLock()
	defer c.mu.RUnlock()

	var out []InflightRequest
	for _, inf := range c.inflight {
		out = append(out, *inf)
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].StartedAt.Before(out[j].StartedAt)
	})
	return out
}

func (c *Coordinator) pruneExpiredLeasesLocked(now time.Time) {
	for id, l := range c.leases {
		if !l.ExpiresAt.IsZero() && now.After(l.ExpiresAt) {
			delete(c.leases, id)
		}
	}
}

func addRecentPID(a *armState, pid int) {
	for _, p := range a.recentPIDs {
		if p == pid {
			return
		}
	}
	a.recentPIDs = append(a.recentPIDs, pid)
	if len(a.recentPIDs) > 8 {
		a.recentPIDs = a.recentPIDs[len(a.recentPIDs)-8:]
	}
}

func randomHex(bytesLen int) string {
	b := make([]byte, bytesLen)
	if _, err := rand.Read(b); err != nil {
		return fmt.Sprintf("%x", time.Now().UnixNano())
	}
	return hex.EncodeToString(b)
}
