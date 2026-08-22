package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchauto"
	"github.com/anthony-chaudhary/fak/internal/dispatchcache"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

const (
	dispatchAutoPhaseBuild        = "build"
	dispatchAutoPhaseBacklogFetch = "backlog-fetch"
	dispatchAutoPhaseRanking      = "ranking"
	dispatchAutoPhasePricing      = "pricing"
	dispatchAutoPhaseOutput       = "render"

	dispatchAutoPhaseTimeoutCode = "DISPATCH_AUTO_PHASE_TIMEOUT"
	dispatchAutoTotalTimeoutCode = "DISPATCH_AUTO_TOTAL_TIMEOUT"
	dispatchAutoPhaseErrorCode   = "DISPATCH_AUTO_PHASE_ERROR"
	dispatchAutoProbeErrorCode   = "DISPATCH_AUTO_PROBE_ERROR"

	dispatchAutoBacklogLimit = 100000
)

var dispatchAutoPhaseOrder = []string{
	dispatchAutoPhaseBuild,
	dispatchAutoPhaseBacklogFetch,
	dispatchAutoPhaseRanking,
	dispatchAutoPhasePricing,
	dispatchAutoPhaseOutput,
}

type dispatchAutoBudgets struct {
	Build        time.Duration
	BacklogFetch time.Duration
	Ranking      time.Duration
	Pricing      time.Duration
	Render       time.Duration
	Total        time.Duration
}

func defaultDispatchAutoBudgets() dispatchAutoBudgets {
	return dispatchAutoBudgets{
		Build:        20 * time.Second,
		BacklogFetch: 45 * time.Second,
		Ranking:      5 * time.Second,
		Pricing:      10 * time.Second,
		Render:       2 * time.Second,
		Total:        60 * time.Second,
	}
}

func (b dispatchAutoBudgets) validate() error {
	for name, budget := range b.byPhase() {
		if budget <= 0 {
			return fmt.Errorf("%s budget must be positive", name)
		}
	}
	if b.Total <= 0 {
		return errors.New("total budget must be positive")
	}
	return nil
}

func (b dispatchAutoBudgets) byPhase() map[string]time.Duration {
	return map[string]time.Duration{
		dispatchAutoPhaseBuild:        b.Build,
		dispatchAutoPhaseBacklogFetch: b.BacklogFetch,
		dispatchAutoPhaseRanking:      b.Ranking,
		dispatchAutoPhasePricing:      b.Pricing,
		dispatchAutoPhaseOutput:       b.Render,
	}
}

func (b dispatchAutoBudgets) receipt() map[string]any {
	phases := map[string]int64{}
	for name, budget := range b.byPhase() {
		phases[name] = budget.Milliseconds()
	}
	return map[string]any{
		"scope":     "dry-run-admission",
		"total_ms":  b.Total.Milliseconds(),
		"phases_ms": phases,
	}
}

type dispatchAutoPhaseTiming struct {
	Phase     string `json:"phase"`
	Status    string `json:"status"`
	ElapsedMS int64  `json:"elapsed_ms"`
	BudgetMS  int64  `json:"budget_ms"`
}

type dispatchAutoError struct {
	Code      string `json:"code"`
	Phase     string `json:"phase"`
	Message   string `json:"message"`
	Timeout   bool   `json:"timeout"`
	Partial   bool   `json:"partial"`
	ElapsedMS int64  `json:"elapsed_ms"`
	BudgetMS  int64  `json:"budget_ms"`
}

type dispatchAutoPhaseValue[T any] struct {
	value T
	err   error
}

func runDispatchAutoPhase[T any](total context.Context, name string, budget time.Duration, fn func(context.Context) (T, error)) (T, dispatchAutoPhaseTiming, *dispatchAutoError) {
	var zero T
	started := time.Now()
	ctx, cancel := context.WithTimeout(total, budget)
	defer cancel()
	result := make(chan dispatchAutoPhaseValue[T], 1)
	go func() {
		value, err := fn(ctx)
		result <- dispatchAutoPhaseValue[T]{value: value, err: err}
	}()

	timing := dispatchAutoPhaseTiming{Phase: name, BudgetMS: budget.Milliseconds()}
	select {
	case got := <-result:
		timing.ElapsedMS = elapsedDispatchAutoMS(started)
		if ctx.Err() != nil || errors.Is(got.err, context.DeadlineExceeded) || errors.Is(got.err, context.Canceled) {
			timing, phaseErr := dispatchAutoTimeout(total, timing, firstError(got.err, ctx.Err()))
			return got.value, timing, phaseErr
		}
		if got.err == nil {
			timing.Status = "ok"
			return got.value, timing, nil
		}
		timing.Status = "error"
		return got.value, timing, &dispatchAutoError{
			Code: dispatchAutoPhaseErrorCode, Phase: name, Message: got.err.Error(), Partial: true,
			ElapsedMS: timing.ElapsedMS, BudgetMS: timing.BudgetMS,
		}
	case <-ctx.Done():
		timing.ElapsedMS = elapsedDispatchAutoMS(started)
		timing, phaseErr := dispatchAutoTimeout(total, timing, ctx.Err())
		return zero, timing, phaseErr
	}
}

func dispatchAutoTimeout(total context.Context, timing dispatchAutoPhaseTiming, cause error) (dispatchAutoPhaseTiming, *dispatchAutoError) {
	timing.Status = "timeout"
	code := dispatchAutoPhaseTimeoutCode
	if errors.Is(total.Err(), context.DeadlineExceeded) {
		code = dispatchAutoTotalTimeoutCode
	}
	return timing, &dispatchAutoError{
		Code: code, Phase: timing.Phase,
		Message: fmt.Sprintf("%s exceeded its %dms budget: %v", timing.Phase, timing.BudgetMS, cause),
		Timeout: true, Partial: true, ElapsedMS: timing.ElapsedMS, BudgetMS: timing.BudgetMS,
	}
}

func firstError(errs ...error) error {
	for _, err := range errs {
		if err != nil {
			return err
		}
	}
	return errors.New("phase canceled")
}

func elapsedDispatchAutoMS(started time.Time) int64 {
	ms := time.Since(started).Milliseconds()
	if ms < 1 {
		return 1
	}
	return ms
}

func dispatchAutoCompleteTimings(timings []dispatchAutoPhaseTiming, budgets dispatchAutoBudgets) []dispatchAutoPhaseTiming {
	byPhase := make(map[string]dispatchAutoPhaseTiming, len(timings))
	for _, timing := range timings {
		byPhase[timing.Phase] = timing
	}
	out := make([]dispatchAutoPhaseTiming, 0, len(dispatchAutoPhaseOrder))
	for _, phase := range dispatchAutoPhaseOrder {
		if timing, ok := byPhase[phase]; ok {
			out = append(out, timing)
			continue
		}
		out = append(out, dispatchAutoPhaseTiming{
			Phase: phase, Status: "skipped", BudgetMS: budgets.byPhase()[phase].Milliseconds(),
		})
	}
	return out
}

type dispatchAutoBuildResult struct {
	Tree     dispatchtick.TreeCheck
	Evidence dispatchTreeBuildEvidence
}

var dispatchAutoBuildProbe = probeDispatchAutoBuild

func probeDispatchAutoBuild(ctx context.Context, root string) (dispatchAutoBuildResult, error) {
	tree, evidence := dispatchProbeTreeBuildContext(ctx, root)
	result := dispatchAutoBuildResult{Tree: tree, Evidence: evidence}
	if err := ctx.Err(); err != nil {
		return result, err
	}
	if tree.Poisoned {
		return result, fmt.Errorf("committed build failed: %s", firstDispatchAutoNonEmpty(tree.Error, tree.Package))
	}
	if tree.Error != "" {
		return result, errors.New(tree.Error)
	}
	return result, nil
}

func firstDispatchAutoNonEmpty(values ...string) string {
	for _, value := range values {
		if value = strings.TrimSpace(value); value != "" {
			return value
		}
	}
	return "unknown build failure"
}

type dispatchAutoBacklogResult struct {
	Fetched       dispatchFetchedBacklog
	ProjectFields map[int]dispatchtick.ProjectIssueFields
}

var dispatchAutoBacklogProbe = probeDispatchAutoBacklog

func probeDispatchAutoBacklog(ctx context.Context, root string) (dispatchAutoBacklogResult, error) {
	taxonomy, err := dispatchLaneTaxonomyContext(ctx, root)
	if err != nil {
		return dispatchAutoBacklogResult{}, err
	}
	issues, err := dispatchFetchBacklogIncrementalContext(ctx, root, dispatchAutoBacklogLimit, time.Now())
	if err != nil {
		return dispatchAutoBacklogResult{}, err
	}
	fields := dispatchFetchProjectFieldsContext(ctx, root)
	if err := ctx.Err(); err != nil {
		return dispatchAutoBacklogResult{}, err
	}
	return dispatchAutoBacklogResult{
		Fetched: dispatchFetchedBacklog{
			Taxonomy: taxonomy, Issues: issues, IssueLimit: dispatchAutoBacklogLimit,
			CacheKey: dispatchcache.Key(root, "", dispatchAutoBacklogLimit),
		},
		ProjectFields: fields,
	}, nil
}

var dispatchAutoRankingProbe = probeDispatchAutoRanking

func probeDispatchAutoRanking(ctx context.Context, root string, backlog dispatchAutoBacklogResult) (dispatchtick.RouterPayload, error) {
	if err := ctx.Err(); err != nil {
		return dispatchtick.RouterPayload{}, err
	}
	router := dispatchRankFetchedBacklog(root, backlog.Fetched)
	router = dispatchFinalizeRankedBacklog(root, router, backlog.ProjectFields)
	if err := ctx.Err(); err != nil {
		return router, err
	}
	if reason := dispatchAutoRouterProbeError(router); reason != "" {
		return router, errors.New(reason)
	}
	return router, nil
}

type dispatchAutoPricingResult struct {
	Input     dispatchauto.Input
	Plan      dispatchauto.Plan
	Notes     []string
	Errors    []dispatchAutoError
	Preflight map[string]any
}

var dispatchAutoPricingProbe = probeDispatchAutoPricing

func probeDispatchAutoPricing(
	ctx context.Context,
	root string,
	stderr io.Writer,
	maxWorkers int,
	workKind, backend, lane string,
	excluded []string,
	requiredWorkers, contextTokens int,
	tree dispatchtick.TreeCheck,
	router dispatchtick.RouterPayload,
) (dispatchAutoPricingResult, error) {
	product := dispatchtick.ProductForBackend(backend)
	pf, _, err := dispatchPreflightTimedWithTree(root, stderr, maxWorkers, workKind, product, &tree)
	result := dispatchAutoPricingResult{Preflight: pf}
	if err != nil {
		return result, err
	}
	if err := ctx.Err(); err != nil {
		return result, err
	}
	var preflightSeatFree *int
	if terms, ok := pf["cap_terms"].(map[string]any); ok {
		result.Input.EffectiveCap = dispatchMapInt(terms, "effective_cap")
	}
	result.Input.LiveWorkers = dispatchMapInt(pf, "live")
	if seat, ok := pf["seat"].(map[string]any); ok {
		preflightSeatFree = intPtrFromAny(seat["free"])
	}
	if verdict := dispatchMapString(pf, "verdict"); verdict != "" {
		result.Notes = append(result.Notes, "preflight: "+verdict)
	}

	rows, rosterErr := dispatchReadAccountRoster(root)
	if rosterErr != nil {
		message := "account roster probe failed: " + rosterErr.Error()
		result.Notes = append(result.Notes, message)
		result.Errors = append(result.Errors, dispatchAutoError{
			Code: dispatchAutoProbeErrorCode, Phase: dispatchAutoPhasePricing, Message: message, Partial: true,
		})
		return result, errors.New(message)
	}
	leases := dispatchLiveSeatLeases(filepath.Join(root, dispatchtick.RunsDirName))
	pool := dispatchtick.BuildSeatPool(rows, leases, product)
	alloc := dispatchtick.AllocateWave(dispatchtick.AccountWaveInput{
		Rows: rows, Leases: leases, Count: pool.FreeSeats, WorkKind: workKind, Product: product,
	})
	freeSlots := alloc.Granted
	if preflightSeatFree != nil && *preflightSeatFree < freeSlots {
		freeSlots = *preflightSeatFree
	}
	result.Input.DistinctPools = dispatchAutoAccountCapacity(result.Input.LiveWorkers, freeSlots)
	if pool.TotalSeats > 0 {
		result.Notes = append(result.Notes, fmt.Sprintf("accounts: %d/%d compatible session slot(s) free for %s", freeSlots, pool.TotalSeats, workKind))
	}
	if freeSlots == 0 && strings.TrimSpace(alloc.Reason) != "" {
		result.Notes = append(result.Notes, "accounts: "+strings.TrimSpace(alloc.Reason))
	}
	result.Input.ReadyWork = dispatchAutoReadyWork(router, lane, excluded)
	result.Input.RequiredWorkers = requiredWorkers
	result.Input.SharedContextTokens = contextTokens
	result.Plan = dispatchauto.PlanAuto(result.Input)
	return result, ctx.Err()
}
