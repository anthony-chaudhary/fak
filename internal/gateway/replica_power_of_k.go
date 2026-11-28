// Adapted from vllm-project/vllm vllm/v1/engine/core_client.py:1811-1872
// Reimplemented cleanly in Go against ReplicaDispatch's PickPolicy seam.

package gateway

import (
	"errors"
	"fmt"
	"sort"
	"time"
)

const (
	// DefaultPowerOfKK is the default number of HRW-ordered candidates sampled
	// per routing decision (power-of-two choices).
	DefaultPowerOfKK = 2

	// DefaultPowerOfKNumGateways is the default gateway multiplier for the
	// local in-flight floor max(localInflight * numGateways, waiting + running).
	DefaultPowerOfKNumGateways = 1

	// DefaultPowerOfKStaleWindow is the default freshness cutoff after which
	// engine-reported waiting+running counts expire and fall back to the local floor.
	DefaultPowerOfKStaleWindow = 5 * time.Second
)

// ReplicaLoadReport carries engine-reported waiting and running request counts
// together with the timestamp when the observation was recorded.
type ReplicaLoadReport struct {
	Waiting    int
	Running    int
	ReportedAt time.Time
}

// PowerOfKConfig configures PowerOfKPolicy. Zero fields take safe defaults from
// DefaultPowerOfKConfig; negative numeric fields are rejected by NewPowerOfKPolicy.
type PowerOfKConfig struct {
	K           int
	NumGateways int
	StaleWindow time.Duration
	Report      func(name string) (ReplicaLoadReport, bool)
	Now         func() time.Time
}

// DefaultPowerOfKConfig returns the standing defaults (K=2, NumGateways=1,
// StaleWindow=5s).
func DefaultPowerOfKConfig() PowerOfKConfig {
	return PowerOfKConfig{
		K:           DefaultPowerOfKK,
		NumGateways: DefaultPowerOfKNumGateways,
		StaleWindow: DefaultPowerOfKStaleWindow,
	}
}

// PowerOfKDecision captures the primary least-loaded choice among the k sampled
// candidates together with the runner-up failover candidate when >=2 candidates
// were sampled.
type PowerOfKDecision struct {
	Primary      PlannerReplica
	RunnerUp     PlannerReplica
	HasRunnerUp  bool
	PrimaryLoad  int
	RunnerUpLoad int
}

// PowerOfKPolicy samples up to k admissible candidates in deterministic HRW order,
// scores each with max(localInflight*numGateways, reportedWaiting+reportedRunning)
// (ignoring reported counts older than StaleWindow), picks the least-loaded
// candidate, and retains the runner-up as the failover target.
type PowerOfKPolicy struct {
	k           int
	numGateways int
	staleWindow time.Duration
	report      func(name string) (ReplicaLoadReport, bool)
	now         func() time.Time
}

// NewPowerOfKPolicy validates cfg and constructs a PowerOfKPolicy.
func NewPowerOfKPolicy(cfg PowerOfKConfig) (*PowerOfKPolicy, error) {
	if cfg.K < 0 {
		return nil, fmt.Errorf("gateway: power-of-k K must be >= 0, got %d", cfg.K)
	}
	if cfg.NumGateways < 0 {
		return nil, fmt.Errorf("gateway: power-of-k NumGateways must be >= 0, got %d", cfg.NumGateways)
	}
	if cfg.StaleWindow < 0 {
		return nil, fmt.Errorf("gateway: power-of-k StaleWindow must be >= 0, got %s", cfg.StaleWindow)
	}
	def := DefaultPowerOfKConfig()
	if cfg.K == 0 {
		cfg.K = def.K
	}
	if cfg.NumGateways == 0 {
		cfg.NumGateways = def.NumGateways
	}
	if cfg.StaleWindow == 0 {
		cfg.StaleWindow = def.StaleWindow
	}
	now := cfg.Now
	if now == nil {
		now = time.Now
	}
	return &PowerOfKPolicy{
		k:           cfg.K,
		numGateways: cfg.NumGateways,
		staleWindow: cfg.StaleWindow,
		report:      cfg.Report,
		now:         now,
	}, nil
}

// PowerOfKLoadEstimate computes max(localInflight*numGateways, reportedWaiting+reportedRunning),
// ignoring absent, invalid, or stale (> staleWindow) reports. usedReport indicates
// whether a fresh report exceeded the local in-flight floor.
func PowerOfKLoadEstimate(localInflight, numGateways int, report ReplicaLoadReport, hasReport bool, now time.Time, staleWindow time.Duration) (estimate int, usedReport bool) {
	if localInflight < 0 {
		localInflight = 0
	}
	if numGateways <= 0 {
		numGateways = DefaultPowerOfKNumGateways
	}
	if staleWindow <= 0 {
		staleWindow = DefaultPowerOfKStaleWindow
	}
	if now.IsZero() {
		now = time.Now()
	}

	floor := saturatingMulNonnegative(localInflight, numGateways)
	if !hasReport || report.Waiting < 0 || report.Running < 0 || report.ReportedAt.IsZero() {
		return floor, false
	}
	if now.Sub(report.ReportedAt) > staleWindow {
		return floor, false
	}
	reported := saturatingNonnegativeAdd(report.Waiting, report.Running)
	if reported > floor {
		return reported, true
	}
	return floor, false
}

func saturatingMulNonnegative(a, b int) int {
	if a <= 0 || b <= 0 {
		return 0
	}
	maxInt := maxIntValue()
	if a > maxInt/b {
		return maxInt
	}
	return a * b
}

// Pick implements PickPolicy by returning the primary choice from PickWithRunnerUp.
func (p *PowerOfKPolicy) Pick(candidates []PlannerReplica, prefix []string, load func(name string) int) (PlannerReplica, bool) {
	d, ok := p.PickWithRunnerUp(candidates, prefix, load)
	if !ok {
		return PlannerReplica{}, false
	}
	return d.Primary, true
}

func (p *PowerOfKPolicy) pickWithRunnerUp(candidates []PlannerReplica, prefix []string, load func(name string) int) (primary PlannerReplica, runnerUp PlannerReplica, hasRunnerUp bool, ok bool) {
	d, ok := p.PickWithRunnerUp(candidates, prefix, load)
	if !ok {
		return PlannerReplica{}, PlannerReplica{}, false, false
	}
	return d.Primary, d.RunnerUp, d.HasRunnerUp, true
}

// PickWithRunnerUp samples up to K candidates in deterministic HRW order, picks the
// least loaded by PowerOfKLoadEstimate, and returns the second-best sampled candidate
// as RunnerUp when at least two candidates are sampled.
func (p *PowerOfKPolicy) PickWithRunnerUp(candidates []PlannerReplica, prefix []string, load func(name string) int) (PowerOfKDecision, bool) {
	if p == nil || p.k <= 0 || p.numGateways <= 0 || len(candidates) == 0 {
		return PowerOfKDecision{}, false
	}
	if load == nil {
		load = func(string) int { return 0 }
	}
	nowFn := p.now
	if nowFn == nil {
		nowFn = time.Now
	}
	now := nowFn()

	type hrwCandidate struct {
		replica PlannerReplica
		score   uint64
	}
	seen := make(map[string]struct{}, len(candidates))
	ordered := make([]hrwCandidate, 0, len(candidates))
	key := replicaRendezvousKey(prefix)
	for _, c := range candidates {
		if c.Name == "" {
			continue
		}
		if _, dup := seen[c.Name]; dup {
			continue
		}
		seen[c.Name] = struct{}{}
		ordered = append(ordered, hrwCandidate{
			replica: c,
			score:   hrwScore(key, c.Name),
		})
	}
	if len(ordered) == 0 {
		return PowerOfKDecision{}, false
	}

	sort.Slice(ordered, func(i, j int) bool {
		if ordered[i].score != ordered[j].score {
			return ordered[i].score > ordered[j].score
		}
		return ordered[i].replica.Name < ordered[j].replica.Name
	})

	sampleN := p.k
	if sampleN > len(ordered) {
		sampleN = len(ordered)
	}

	type scoredCandidate struct {
		replica  PlannerReplica
		estimate int
		local    int
		hrwRank  int
	}
	sampled := make([]scoredCandidate, sampleN)
	for i := 0; i < sampleN; i++ {
		c := ordered[i].replica
		local := load(c.Name)
		if local < 0 {
			local = 0
		}
		var rep ReplicaLoadReport
		var hasRep bool
		if p.report != nil {
			rep, hasRep = p.report(c.Name)
		}
		est, _ := PowerOfKLoadEstimate(local, p.numGateways, rep, hasRep, now, p.staleWindow)
		sampled[i] = scoredCandidate{
			replica:  c,
			estimate: est,
			local:    local,
			hrwRank:  i,
		}
	}

	sort.Slice(sampled, func(i, j int) bool {
		if sampled[i].estimate != sampled[j].estimate {
			return sampled[i].estimate < sampled[j].estimate
		}
		if sampled[i].local != sampled[j].local {
			return sampled[i].local < sampled[j].local
		}
		return sampled[i].hrwRank < sampled[j].hrwRank
	})

	out := PowerOfKDecision{
		Primary:     sampled[0].replica,
		PrimaryLoad: sampled[0].estimate,
	}
	if len(sampled) >= 2 {
		out.RunnerUp = sampled[1].replica
		out.RunnerUpLoad = sampled[1].estimate
		out.HasRunnerUp = true
	}
	return out, true
}

// WithPowerOfKPolicy attaches a PowerOfKPolicy to ReplicaDispatch, failing closed
// on a nil dispatch or invalid configuration without altering existing policy state.
func (r *ReplicaDispatch) WithPowerOfKPolicy(cfg PowerOfKConfig) (*ReplicaDispatch, error) {
	if r == nil {
		return nil, errors.New("gateway: cannot attach power-of-k policy to nil replica dispatch")
	}
	p, err := NewPowerOfKPolicy(cfg)
	if err != nil {
		return nil, err
	}
	r.WithPickPolicy(p)
	return r, nil
}
