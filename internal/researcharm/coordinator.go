package researcharm

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
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
	store                 *leaseStore
	closed                bool
	storeErr              error
	admissionReady        bool

	// admitWait is the bounded wait the gateway passes to AdmitWithWait and the base for
	// the Retry-After hint. 0 preserves the historical instant shed. See saturation.go.
	admitWait time.Duration
	// queueDepth bounds how many admissions may WAIT per arm (0 = unbounded). Beyond it a
	// new arrival is shed at once rather than joining an unbounded backlog.
	queueDepth int
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
	waiters        []*armWaiter // queued admissions blocked on the concurrency limit

	// Admission-queue accounting (saturation.go): how many requests entered the queue,
	// how many of those were admitted after waiting, how many were shed for capacity, and
	// the measured average wait behind the Retry-After hint.
	waitedTotal       int64
	admittedAfterWait int64
	shedsTotal        int64
	waitEMAMicros     int64
}

// armWaiter is one queued admission blocked on an arm's concurrency limit. The
// channel has capacity 1: a signal means "re-check admission", and a waiter that
// already holds a pending signal is not double-signaled (non-blocking send).
type armWaiter struct {
	ch chan struct{}
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

// DefaultMaxConcurrency returns the ceiling a newly registered arm inherits
// when it holds no lease of its own (0 = unlimited). It is the configured
// ship value, distinct from any per-arm override set by SetLimit or carried by
// an active lease; the awareness surfaces read it so an operator can see what
// an arm will get before it first appears.
func (c *Coordinator) DefaultMaxConcurrency() int {
	if c == nil {
		return 0
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.defaultMaxConcurrency
}

// Admit checks whether an incoming request from the given origin may proceed.
// It refuses immediately when the arm is at its concurrency ceiling; callers
// that want bounded queueing instead use AdmitWithWait.
func (c *Coordinator) Admit(ctx context.Context, r *http.Request, endpoint string, traceID string) (*RequestLease, error) {
	return c.AdmitWithWait(ctx, r, endpoint, traceID, 0)
}

// AdmitWithWait admits like Admit but, when the arm is at its concurrency
// ceiling, queues behind it instead of refusing instantly. Waiters are woken
// event-driven — a slot release, a limit/lease change, a lease expiring by
// TTL, or coordinator close — and re-check the FULL admission conditions on
// every wake, so a spurious or stale signal simply loops back to the check.
// While parked, each waiter's timer additionally fires at the sooner of its
// remaining budget and the next lease expiry, so a ceiling that rises by
// passive lease expiry is observed without polling. ctx must be non-nil.
// maxWait <= 0 preserves the legacy immediate refusal. Timeout or context
// cancellation returns a *SaturationError wrapping ErrArmConcurrencyExceeded
// (errors.Is keeps working, and a caller's context error stays reachable too),
// stating how long it waited — see saturation.go.
func (c *Coordinator) AdmitWithWait(ctx context.Context, r *http.Request, endpoint string, traceID string, maxWait time.Duration) (*RequestLease, error) {
	if maxWait <= 0 {
		return c.admitNow(ctx, r, endpoint, traceID)
	}
	origin := ExtractOrigin(r, traceID)
	budgetEnd := time.Now().Add(maxWait)
	timer := time.NewTimer(0)
	defer timer.Stop()

	waiter := &armWaiter{ch: make(chan struct{}, 1)}
	registered := false
	queued := false     // whether this request ever actually waited (vs. admitted on the first try)
	var start time.Time // when this request entered the queue (for the measured wait)
	for {
		c.mu.Lock()
		lease, err := c.tryAdmitLocked(origin, randomHex(8), time.Now(), endpoint, traceID)
		if err == nil || !errors.Is(err, ErrArmConcurrencyExceeded) {
			if registered {
				c.removeWaiterLocked(origin.ArmID, waiter)
			}
			if lease != nil && queued {
				c.recordWaitLocked(origin.ArmID, time.Since(start))
			}
			c.mu.Unlock()
			return lease, err
		}
		if !registered {
			arm, ok := c.arms[origin.ArmID]
			if !ok {
				// tryAdmitLocked registers the arm before the ceiling check, so this is
				// unreachable in practice; fall through to the immediate refusal rather
				// than parking a request with no queue to wake it.
				c.mu.Unlock()
				return nil, err
			}
			// The queue is BOUNDED: past its depth a new arrival is shed at once rather
			// than joining an unbounded backlog, which would convert a capacity signal
			// into unbounded latency.
			if c.queueDepth > 0 && len(arm.waiters) >= c.queueDepth {
				active, limit, wq := c.armLoadLocked(origin.ArmID)
				arm.shedsTotal++
				c.mu.Unlock()
				return nil, &SaturationError{
					ArmID: origin.ArmID, Active: active, Limit: limit, Queued: wq,
					Full: true, QueueBound: c.queueDepth,
				}
			}
			if !queued {
				start = time.Now()
				queued = true
				arm.waitedTotal++
			}
			arm.waiters = append(arm.waiters, waiter)
			registered = true
		}
		// Park until the sooner of the budget end and the next lease expiry:
		// an expiry can raise the effective ceiling even with no other event.
		parkUntil := budgetEnd
		for _, l := range c.leases {
			if !l.ExpiresAt.IsZero() && l.ExpiresAt.Before(parkUntil) {
				parkUntil = l.ExpiresAt
			}
		}
		c.mu.Unlock()

		if !timer.Reset(time.Until(parkUntil)) {
			// The previous tick was already consumed by the select below; drain
			// the stale value so it cannot fire spuriously on this iteration.
			<-timer.C
		}

		select {
		case <-waiter.ch:
			// Woken: loop back and re-check everything under the lock.
		case <-ctx.Done():
			c.mu.Lock()
			c.removeWaiterLocked(origin.ArmID, waiter)
			if arm, ok := c.arms[origin.ArmID]; ok {
				arm.shedsTotal++
			}
			sat := c.saturationNow(origin.ArmID, time.Since(start), ctx.Err())
			c.mu.Unlock()
			return nil, sat
		case <-timer.C:
			if time.Now().Before(budgetEnd) {
				// A lease expired: re-check (the prune inside tryAdmitLocked
				// applies it) and re-park with a fresh horizon.
				continue
			}
			c.mu.Lock()
			c.removeWaiterLocked(origin.ArmID, waiter)
			if arm, ok := c.arms[origin.ArmID]; ok {
				arm.shedsTotal++
			}
			sat := c.saturationNow(origin.ArmID, time.Since(start), nil)
			c.mu.Unlock()
			return nil, sat
		}
	}
}

// admitNow is the legacy immediate-refusal admission.
func (c *Coordinator) admitNow(ctx context.Context, r *http.Request, endpoint string, traceID string) (*RequestLease, error) {
	origin := ExtractOrigin(r, traceID)

	c.mu.Lock()
	defer c.mu.Unlock()

	return c.tryAdmitLocked(origin, randomHex(8), time.Now(), endpoint, traceID)
}

// tryAdmitLocked runs the full admission check and records the admission when
// it passes. Callers hold c.mu. Capacity exhaustion is reported as
// ErrArmConcurrencyExceeded so AdmitWithWait can distinguish "queue me" from
// "refuse me"; every other error is operator state (closed, lease rules, store
// unavailable) that waiting cannot cure.
func (c *Coordinator) tryAdmitLocked(origin Origin, reqID string, now time.Time, endpoint, traceID string) (*RequestLease, error) {
	if err := c.availableLocked(); err != nil {
		return nil, err
	}
	if c.store != nil && !c.admissionReady {
		return nil, fmt.Errorf("researcharm: durable admission requires initial exclusive lease")
	}
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
		return nil, &SaturationError{
			ArmID:  origin.ArmID,
			Active: arm.activeRequests,
			Limit:  effectiveLimit,
			Queued: len(arm.waiters),
		}
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
		c.wakeArmWaiters(armID) // a slot just freed; queued admissions re-check
	}
}

// wakeArmWaiters signals every queued admission on armID to re-check. Callers
// hold c.mu; sends are non-blocking into cap-1 channels, so a waiter that
// already holds a pending signal is not double-signaled. A wake is only ever a
// hint — each waiter re-runs the full admission check under the lock.
func (c *Coordinator) wakeArmWaiters(armID string) {
	arm, ok := c.arms[armID]
	if !ok {
		return
	}
	for _, w := range arm.waiters {
		select {
		case w.ch <- struct{}{}:
		default:
		}
	}
}

// removeWaiter unregisters a queued admission. Safe to call when it was never
// registered or was already removed.
func (c *Coordinator) removeWaiter(armID string, w *armWaiter) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.removeWaiterLocked(armID, w)
}

func (c *Coordinator) removeWaiterLocked(armID string, w *armWaiter) {
	arm, ok := c.arms[armID]
	if !ok {
		return
	}
	for i, cand := range arm.waiters {
		if cand == w {
			arm.waiters = append(arm.waiters[:i], arm.waiters[i+1:]...)
			return
		}
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
	if req.Mode != LeaseModeShared && req.Mode != LeaseModeExclusive {
		return nil, fmt.Errorf("researcharm: invalid lease mode %q", req.Mode)
	}
	if req.HolderPID < 0 || req.Concurrency < 0 {
		return nil, fmt.Errorf("researcharm: negative lease PID or concurrency")
	}
	if req.TTL <= 0 {
		req.TTL = 5 * time.Minute
	}

	now := time.Now()
	c.mu.Lock()
	defer c.mu.Unlock()

	if err := c.availableLocked(); err != nil {
		return nil, err
	}
	if c.store != nil && !c.admissionReady && req.Mode != LeaseModeExclusive {
		return nil, fmt.Errorf("researcharm: initial durable lease must be exclusive")
	}
	candidate := c.liveLeasesLocked(now)

	// If exclusive requested, ensure no other arm holds an active lease or has in-flight requests.
	if req.Mode == LeaseModeExclusive {
		for _, l := range candidate {
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
		for _, l := range candidate {
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
	for k, l := range candidate {
		if l.ArmID == req.ArmID {
			delete(candidate, k)
		}
	}
	candidate[leaseID] = info
	if err := c.commitLeasesLocked(candidate, true); err != nil {
		return nil, err
	}

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

	c.wakeArmWaiters(req.ArmID) // a lease can raise (or lower) the effective ceiling

	// Return a copy with token visible to the caller
	res := *info
	return &res, nil
}

// ReleaseLease releases an existing lease.
func (c *Coordinator) ReleaseLease(leaseID, token string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if err := c.availableLocked(); err != nil {
		return err
	}
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

	if (c.store != nil || token != "") && l.Token != token {
		return ErrInvalidLeaseToken
	}

	candidate := c.liveLeasesLocked(time.Now())
	delete(candidate, leaseID)
	if err := c.commitLeasesLocked(candidate, c.admissionReady); err != nil {
		return err
	}
	c.wakeArmWaiters(l.ArmID) // dropping a low-concurrency lease can raise the ceiling
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
	c.wakeArmWaiters(armID) // the ceiling may have risen; queued admissions re-check
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
			ID:                a.id,
			Group:             a.group,
			DisplayName:       a.displayName,
			MaxConcurrency:    a.maxConcurrency,
			ActiveRequests:    a.activeRequests,
			WaitingRequests:   len(a.waiters),
			WaitedTotal:       a.waitedTotal,
			AdmittedAfterWait: a.admittedAfterWait,
			ShedsTotal:        a.shedsTotal,
			TotalRequests:     a.totalRequests,
			TotalTokens:       a.totalTokens,
			ErrorCount:        a.errorCount,
			RecentPIDs:        append([]int(nil), a.recentPIDs...),
			LastSeen:          a.lastSeen,
		}
		if a.waitEMAMicros > 0 {
			info.AvgQueueWait = time.Duration(a.waitEMAMicros) * time.Microsecond
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

	status := ""
	if c.store != nil {
		status = "ready"
		if !c.admissionReady {
			status = "bootstrap"
		}
		if c.storeErr != nil {
			status = "unavailable"
		}
		if c.closed {
			status = "closed"
		}
	}
	return Snapshot{
		DurableStatus:         status,
		Timestamp:             now,
		TotalInflight:         len(inflight),
		TotalArms:             len(arms),
		DefaultMaxConcurrency: c.defaultMaxConcurrency,
		Arms:                  arms,
		Inflight:              inflight,
		ActiveLeases:          leases,
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
	var expiredArms []string
	for id, l := range c.leases {
		if !l.ExpiresAt.IsZero() && now.After(l.ExpiresAt) {
			delete(c.leases, id)
			expiredArms = append(expiredArms, l.ArmID)
		}
	}
	for _, armID := range expiredArms {
		c.wakeArmWaiters(armID) // an expiry can raise the effective ceiling
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

// Close stops admission before releasing the process-lifetime store lock.
// Queued admissions are woken so they observe the closed state and exit
// instead of waiting out their timers.
func (c *Coordinator) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil
	}
	c.closed = true
	for id := range c.arms {
		c.wakeArmWaiters(id)
	}
	if c.store != nil {
		return c.store.close()
	}
	return nil
}

func (c *Coordinator) availableLocked() error {
	if c.closed {
		return fmt.Errorf("researcharm: coordinator closed")
	}
	if c.storeErr != nil {
		return fmt.Errorf("researcharm: lease store unavailable: %w", c.storeErr)
	}
	return nil
}

func (c *Coordinator) liveLeasesLocked(now time.Time) map[string]*LeaseInfo {
	out := make(map[string]*LeaseInfo, len(c.leases))
	for id, lease := range c.leases {
		if now.Before(lease.ExpiresAt) {
			cp := *lease
			out[id] = &cp
		}
	}
	return out
}

func (c *Coordinator) commitLeasesLocked(candidate map[string]*LeaseInfo, ready bool) error {
	if c.store != nil {
		uncertain, err := c.store.save(candidate, ready)
		if err != nil {
			if uncertain {
				c.storeErr = err
			}
			return fmt.Errorf("researcharm: persist lease mutation: %w", err)
		}
	}
	c.leases = candidate
	c.admissionReady = ready
	return nil
}
