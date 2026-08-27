package gateway

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

const hedgeReceiptSchema = "fak-gateway-hedge-receipt/1"

var hedgeFamilySequence atomic.Uint64

// HedgePolicy is the default-off admission contract for one delayed buffered hedge.
// Enabling it declares that calls reaching this router are read-only and provider
// cancellation/duplicate-request semantics are understood by the operator. The
// router still rejects tool-bearing, non-deterministic, mismatched-model, and
// capacity-ineligible calls.
type HedgePolicy struct {
	Enabled             bool
	PolicyVersion       string
	ProviderContract    string
	ReadOnly            bool
	Deterministic       bool
	Delay               time.Duration
	Deadline            time.Duration
	DrainTimeout        time.Duration
	DuplicateWorkBudget int
	SpareCapacity       func() bool
	// Validate applies the caller's accepted-output contract. The built-in
	// checks still reject nil, empty, and tool-call-dropped completions.
	Validate func(*agent.Completion) error
	Observe  func(HedgeReceipt)
}

// HedgeAttemptReceipt describes one physical attempt without exposing response bodies.
type HedgeAttemptReceipt struct {
	Replica       string
	Opened        bool
	StartedAt     time.Time
	FinishedAt    time.Time
	ValidatedAt   time.Time
	TerminalClass string
	Usage         agent.Usage
}

// HedgeReceipt joins both physical attempts into one truthful request-family record.
type HedgeReceipt struct {
	Schema                        string
	RequestFamilyID               string
	PolicyVersion                 string
	RequestedMode                 string
	RealizedMode                  string
	EligibilityVerdict            string
	AbstentionReason              string
	Model                         string
	ProviderContract              string
	Delay                         time.Duration
	Deadline                      time.Duration
	Primary                       HedgeAttemptReceipt
	Hedge                         HedgeAttemptReceipt
	Winner                        string
	FirstValid                    bool
	CancellationRequested         bool
	LocalTransportAcknowledgement bool
	DrainDuration                 time.Duration
	DrainOutcome                  string
	LoserResultChannelClosed      bool
	DuplicatePromptWork           int
	ProviderWorkAfterCancel       string
	CancelledBilling              string
}

func hedgeIneligibility(policy *HedgePolicy, replicas int, tools []agent.ToolDef, opts []agent.SampleOpt) string {
	switch {
	case !policy.Enabled:
		return "policy_disabled"
	case policy.Delay <= 0:
		return "delay_not_positive"
	case policy.Deadline <= 0:
		return "deadline_not_positive"
	case policy.Deadline <= policy.Delay:
		return "deadline_not_after_delay"
	case policy.DuplicateWorkBudget != 1:
		return "physical_attempt_budget_not_one"
	case strings.TrimSpace(policy.ProviderContract) == "":
		return "provider_contract_unknown"
	case !policy.ReadOnly:
		return "not_declared_read_only"
	case !policy.Deterministic:
		return "not_declared_deterministic"
	case len(tools) != 0:
		return "tool_catalog_present"
	case replicas < 2:
		return "insufficient_replicas"
	case policy.SpareCapacity == nil:
		return "capacity_gate_missing"
	}
	params := agent.SampleParams{}
	for _, opt := range opts {
		opt(&params)
	}
	if params.Temperature == nil || *params.Temperature != 0 {
		return "temperature_not_explicit_zero"
	}
	return ""
}

func (r *ReplicaRouter) observeHedgeAbstention(primary PlannerReplica, reason string) {
	policy := r.Hedge
	if policy == nil || policy.Observe == nil {
		return
	}
	policy.Observe(HedgeReceipt{
		Schema: hedgeReceiptSchema, RequestFamilyID: fmt.Sprintf("hedge-%d", hedgeFamilySequence.Add(1)),
		PolicyVersion: policy.PolicyVersion, RequestedMode: "selective_delayed_hedge", RealizedMode: "primary_only",
		EligibilityVerdict: "abstain", AbstentionReason: reason, Model: primary.Planner.Model(),
		ProviderContract: policy.ProviderContract, Delay: policy.Delay, Deadline: policy.Deadline,
		Primary:                 HedgeAttemptReceipt{Replica: primary.Name, Opened: true, StartedAt: time.Now()},
		ProviderWorkAfterCancel: "unknown", CancelledBilling: "unknown",
	})
}

type hedgeResult struct {
	index      int
	completion *agent.Completion
	err        error
	finished   time.Time
}

func (r *ReplicaRouter) completeHedged(ctx context.Context, primary reservedPlannerReplica, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	policy := r.Hedge
	receipt := HedgeReceipt{
		Schema:                  hedgeReceiptSchema,
		RequestFamilyID:         fmt.Sprintf("hedge-%d", hedgeFamilySequence.Add(1)),
		PolicyVersion:           policy.PolicyVersion,
		RequestedMode:           "selective_delayed_hedge",
		RealizedMode:            "primary_only",
		EligibilityVerdict:      "eligible",
		Model:                   primary.replica.Planner.Model(),
		ProviderContract:        policy.ProviderContract,
		Delay:                   policy.Delay,
		Deadline:                policy.Deadline,
		ProviderWorkAfterCancel: "unknown",
		CancelledBilling:        "unknown",
	}
	emit := func() {
		if policy.Observe != nil {
			policy.Observe(receipt)
		}
	}

	familyCtx, cancelFamily := context.WithTimeout(ctx, policy.Deadline)
	defer cancelFamily()
	pctx, pcancel := context.WithCancel(familyCtx)
	defer pcancel()
	result := make(chan hedgeResult, 2) // attempts may finish after the bounded drain
	receipt.Primary = HedgeAttemptReceipt{Replica: primary.replica.Name, Opened: true, StartedAt: time.Now()}
	go runHedgeAttempt(pctx, 0, primary, result, messages, tools, opts...)

	timer := time.NewTimer(policy.Delay)
	defer timer.Stop()
	select {
	case res := <-result:
		finishAttempt(&receipt.Primary, res, policy)
		if policy.validCompletion(res.completion, res.err) {
			receipt.Winner = "primary"
			receipt.FirstValid = true
			receipt.AbstentionReason = "primary_completed_before_delay"
			emit()
			return res.completion, nil
		}
		receipt.EligibilityVerdict = "abstain"
		receipt.AbstentionReason = "primary_terminal_before_delay"
		emit()
		return nil, typedHedgeError(res)
	case <-timer.C:
	case <-familyCtx.Done():
		receipt.EligibilityVerdict = "abstain"
		receipt.AbstentionReason = "deadline_before_hedge"
		receipt.Primary.TerminalClass = classifyHedgeError(familyCtx.Err())
		emit()
		return nil, familyCtx.Err()
	}

	awaitPrimary := func(reason string) (*agent.Completion, error) {
		receipt.EligibilityVerdict = "abstain"
		receipt.AbstentionReason = reason
		select {
		case res := <-result:
			finishAttempt(&receipt.Primary, res, policy)
			emit()
			return res.completion, res.err
		case <-familyCtx.Done():
			receipt.Primary.TerminalClass = classifyHedgeError(familyCtx.Err())
			emit()
			return nil, familyCtx.Err()
		}
	}

	if policy.SpareCapacity != nil && !policy.SpareCapacity() {
		return awaitPrimary("no_spare_capacity")
	}

	var alternate reservedPlannerReplica
	if r.membership == nil {
		if _, decodeAware := r.policy.(decodeFootprintPickPolicy); decodeAware {
			var err error
			alternate, err = r.reserveOnEngineWithDecode(
				primary.prefix,
				map[string]struct{}{primary.replica.Name: {}},
				"",
				decodeFootprintRouteRequest{ExpectedOutputTokens: primary.decodeDecision.RequestedOutputTokens},
			)
			if err != nil {
				return awaitPrimary("no_distinct_admissible_replica")
			}
		} else {
			repl, ok := r.pickDistinctReplica(primary.replica.Name)
			if !ok {
				return awaitPrimary("no_distinct_admissible_replica")
			}
			alternate = reservedPlannerReplica{replica: repl, prefix: primary.prefix}
		}
	} else {
		var err error
		alternate, err = r.reserveOnEngineWithDecode(
			primary.prefix,
			map[string]struct{}{primary.replica.Name: {}},
			primary.reservation.Engine(),
			decodeFootprintRouteRequest{ExpectedOutputTokens: primary.decodeDecision.RequestedOutputTokens},
		)
		if err != nil {
			return awaitPrimary("no_distinct_admissible_replica")
		}
	}
	if alternate.replica.Planner.Model() != primary.replica.Planner.Model() {
		alternate.Release()
		return awaitPrimary("model_contract_mismatch")
	}

	hctx, hcancel := context.WithCancel(familyCtx)
	defer hcancel()
	receipt.RealizedMode = "hedged"
	receipt.DuplicatePromptWork = 1
	receipt.Hedge = HedgeAttemptReceipt{Replica: alternate.replica.Name, Opened: true, StartedAt: time.Now()}
	go runHedgeAttempt(hctx, 1, alternate, result, messages, tools, opts...)

	var failures [2]hedgeResult
	completed := 0
	for completed < 2 {
		select {
		case res := <-result:
			completed++
			attempt := &receipt.Primary
			if res.index == 1 {
				attempt = &receipt.Hedge
			}
			finishAttempt(attempt, res, policy)
			if !policy.validCompletion(res.completion, res.err) {
				failures[res.index] = res
				continue
			}
			receipt.FirstValid = true
			if res.index == 0 {
				receipt.Winner = "primary"
				hcancel()
			} else {
				receipt.Winner = "hedge"
				pcancel()
			}
			receipt.CancellationRequested = completed < 2
			if completed < 2 {
				drainLoser(result, policy.DrainTimeout, policy, &receipt)
			}
			emit()
			return res.completion, nil
		case <-familyCtx.Done():
			pcancel()
			hcancel()
			receipt.CancellationRequested = true
			receipt.DrainOutcome = "family_deadline"
			emit()
			return nil, familyCtx.Err()
		}
	}
	emit()
	return nil, fmt.Errorf("hedged completion failed: primary=%s; hedge=%s", classifyHedgeResult(failures[0], policy), classifyHedgeResult(failures[1], policy))
}

func drainLoser(result <-chan hedgeResult, timeout time.Duration, policy *HedgePolicy, receipt *HedgeReceipt) {
	if timeout <= 0 {
		timeout = 10 * time.Millisecond
	}
	started := time.Now()
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	select {
	case loser := <-result:
		receipt.DrainOutcome = "completed"
		receipt.LoserResultChannelClosed = true
		receipt.LocalTransportAcknowledgement = errors.Is(loser.err, context.Canceled)
		if loser.index == 0 {
			finishAttempt(&receipt.Primary, loser, policy)
		} else {
			finishAttempt(&receipt.Hedge, loser, policy)
		}
	case <-timer.C:
		receipt.DrainOutcome = "timeout"
	}
	receipt.DrainDuration = time.Since(started)
}

func runHedgeAttempt(ctx context.Context, index int, route reservedPlannerReplica, ch chan<- hedgeResult, m []agent.Message, t []agent.ToolDef, o ...agent.SampleOpt) {
	c, e := route.replica.Planner.Complete(ctx, m, t, o...)
	route.finish(ctx, c, e, false)
	ch <- hedgeResult{index: index, completion: c, err: e, finished: time.Now()}
}

func (p *HedgePolicy) validCompletion(c *agent.Completion, e error) bool {
	if e != nil || c == nil || c.ToolCallsDropped || strings.TrimSpace(c.Message.Content) == "" {
		return false
	}
	return p.Validate == nil || p.Validate(c) == nil
}

func finishAttempt(a *HedgeAttemptReceipt, r hedgeResult, policy *HedgePolicy) {
	a.FinishedAt = r.finished
	a.ValidatedAt = time.Now()
	a.TerminalClass = classifyHedgeResult(r, policy)
	if r.completion != nil {
		a.Usage = r.completion.Usage
	}
}

func classifyHedgeResult(r hedgeResult, policy *HedgePolicy) string {
	if r.err != nil {
		return classifyHedgeError(r.err)
	}
	if policy == nil {
		policy = &HedgePolicy{}
	}
	if policy.validCompletion(r.completion, nil) {
		return "valid"
	}
	return "invalid_completion"
}

func classifyHedgeError(err error) string {
	switch {
	case errors.Is(err, context.Canceled):
		return "canceled"
	case errors.Is(err, context.DeadlineExceeded):
		return "deadline"
	case err != nil:
		return "error"
	default:
		return "unknown"
	}
}

func typedHedgeError(r hedgeResult) error {
	if r.err != nil {
		return fmt.Errorf("primary terminal before hedge: %s: %w", classifyHedgeError(r.err), r.err)
	}
	return errors.New("primary terminal before hedge: invalid_completion")
}
