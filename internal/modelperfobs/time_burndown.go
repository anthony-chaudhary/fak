package modelperfobs

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/nativeperf"
)

const (
	TimeBurndownSchema            = "fak-time-burndown/1"
	RequestPhaseSchema            = "fak-request-phases/1"
	BackendAttributionUnavailable = "backend-does-not-expose-dispatch-attribution"
)

var ErrInvalidTimeBurndown = errors.New("invalid time burndown evidence")

// RequestPhaseReceipt binds complete native accounting to one API observation.
type RequestPhaseReceipt struct {
	Schema    string                  `json:"schema"`
	RequestID string                  `json:"request_id"`
	Model     string                  `json:"model"`
	Backend   string                  `json:"backend"`
	Receipt   nativeperf.PhaseReceipt `json:"receipt"`
}

// TimeBurndownPhase belongs to one accounting plane. API spans have no observed
// active/wait split; native percentages use only the native wall denominator.
type TimeBurndownPhase struct {
	Phase    string   `json:"phase"`
	WallMS   float64  `json:"wall_ms"`
	Percent  float64  `json:"percent"`
	ActiveMS *float64 `json:"active_ms"`
	WaitMS   *float64 `json:"wait_ms"`
}

type TimeBurndownGroup struct {
	Backend                string              `json:"backend"`
	Model                  string              `json:"model"`
	CacheState             string              `json:"cache_state"`
	Requests               int                 `json:"requests"`
	Samples                int                 `json:"samples"`
	FailedRequests         int                 `json:"failed_requests"`
	WarmupExcluded         int                 `json:"warmup_excluded"`
	NativeRequests         int                 `json:"native_requests"`
	TTFTSamples            int                 `json:"ttft_samples"`
	DurationP50MS          float64             `json:"duration_p50_ms"`
	TTFTP50MS              float64             `json:"ttft_p50_ms"`
	APIWallMS              float64             `json:"api_wall_ms"`
	NativeCoveredAPIWallMS float64             `json:"native_covered_api_wall_ms"`
	NativeWallMS           float64             `json:"native_wall_ms"`
	OutsideNativeMS        float64             `json:"outside_native_ms"`
	Attribution            string              `json:"attribution"`
	AttributionUnavailable string              `json:"attribution_unavailable,omitempty"`
	APIRank                []TimeBurndownPhase `json:"api_rank"`
	NativeRank             []TimeBurndownPhase `json:"native_rank"`
}

type TimeBurndown struct {
	Schema           string              `json:"schema"`
	Requests         int                 `json:"requests"`
	IncludedRequests int                 `json:"included_requests"`
	WarmupExcluded   int                 `json:"warmup_excluded"`
	FailedRequests   int                 `json:"failed_requests"`
	NativeRequests   int                 `json:"native_requests"`
	Groups           []TimeBurndownGroup `json:"groups"`
}

// ParseCacheState admits declared experimental arms, never inferring cache reuse.
func ParseCacheState(raw string) (string, error) {
	state := strings.TrimSpace(raw)
	switch state {
	case "":
		return "unknown", nil
	case "cold", "warm", "steady", "unknown":
		return state, nil
	default:
		return "", fmt.Errorf("%w: cache state must be cold, warm, steady, or unknown", ErrInvalidTimeBurndown)
	}
}

// ReadRequestPhaseReceipts reads JSONL envelopes without accepting unknown fields
// or trailing JSON values. Receipt identities are checked against observations by
// ReduceTimeBurndown.
func ReadRequestPhaseReceipts(r io.Reader) ([]RequestPhaseReceipt, error) {
	var rows []RequestPhaseReceipt
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 64*1024), 16*1024*1024)
	for scanner.Scan() {
		var row RequestPhaseReceipt
		decoder := json.NewDecoder(strings.NewReader(scanner.Text()))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&row); err != nil {
			return nil, fmt.Errorf("%w: native receipt %d: %v", ErrInvalidTimeBurndown, len(rows)+1, err)
		}
		var trailing any
		if err := decoder.Decode(&trailing); err != io.EOF {
			return nil, fmt.Errorf("%w: native receipt %d has trailing JSON", ErrInvalidTimeBurndown, len(rows)+1)
		}
		if err := validateRequestPhaseReceipt(row); err != nil {
			return nil, err
		}
		rows = append(rows, row)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidTimeBurndown, err)
	}
	return rows, nil
}

func validateRequestPhaseReceipt(row RequestPhaseReceipt) error {
	if row.Schema != RequestPhaseSchema || strings.TrimSpace(row.RequestID) == "" || strings.TrimSpace(row.Model) == "" || strings.TrimSpace(row.Backend) == "" {
		return fmt.Errorf("%w: native receipt requires schema and request/model/backend identities", ErrInvalidTimeBurndown)
	}
	if err := row.Receipt.Validate(0); err != nil {
		return fmt.Errorf("%w: native receipt %q: %v", ErrInvalidTimeBurndown, row.RequestID, err)
	}
	return nil
}

type burndownGroupAccumulator struct {
	group     TimeBurndownGroup
	durations []float64
	ttfts     []float64
	api       map[string]TimeBurndownPhase
	native    map[string]TimeBurndownPhase
}

// ReduceTimeBurndown ranks cumulative request time, excluding declared warmups
// and failed samples. API and native planes reconcile independently.
func ReduceTimeBurndown(rows []Observation, native []RequestPhaseReceipt) (TimeBurndown, error) {
	report := TimeBurndown{Schema: TimeBurndownSchema, Requests: len(rows), Groups: []TimeBurndownGroup{}}
	observations := make(map[string]Observation, len(rows))
	states := make([]string, len(rows))
	for i, row := range rows {
		if row.Schema != Schema || strings.TrimSpace(row.RequestID) == "" || strings.TrimSpace(row.Backend) == "" {
			return TimeBurndown{}, fmt.Errorf("%w: observation %d requires schema, request ID, and backend", ErrInvalidTimeBurndown, i+1)
		}
		if _, exists := observations[row.RequestID]; exists {
			return TimeBurndown{}, fmt.Errorf("%w: duplicate observation %q", ErrInvalidTimeBurndown, row.RequestID)
		}
		state, err := ParseCacheState(row.CacheState)
		if err != nil {
			return TimeBurndown{}, err
		}
		if row.CacheStateSource != "" && row.CacheStateSource != "declared" && row.CacheStateSource != "unknown" {
			return TimeBurndown{}, fmt.Errorf("%w: invalid cache state source", ErrInvalidTimeBurndown)
		}
		if row.CacheStateSource == "unknown" && state != "unknown" {
			return TimeBurndown{}, fmt.Errorf("%w: unknown cache source contradicts declared state", ErrInvalidTimeBurndown)
		}
		for _, duration := range []float64{row.DurationMS, row.TTFTMS, row.TPOTMS, row.InterChunkP50MS, row.InterChunkP95MS} {
			if !finiteNonnegative(duration) {
				return TimeBurndown{}, fmt.Errorf("%w: observation %q has non-finite or negative duration", ErrInvalidTimeBurndown, row.RequestID)
			}
		}
		if row.TTFTMS > row.DurationMS {
			return TimeBurndown{}, fmt.Errorf("%w: observation %q TTFT exceeds request wall", ErrInvalidTimeBurndown, row.RequestID)
		}
		states[i] = state
		observations[row.RequestID] = row
	}
	joins := make(map[string]nativeperf.PhaseReceipt, len(native))
	for _, row := range native {
		if err := validateRequestPhaseReceipt(row); err != nil {
			return TimeBurndown{}, err
		}
		if _, exists := joins[row.RequestID]; exists {
			return TimeBurndown{}, fmt.Errorf("%w: duplicate native receipt %q", ErrInvalidTimeBurndown, row.RequestID)
		}
		obs, exists := observations[row.RequestID]
		if !exists || obs.Model != row.Model || obs.Backend != row.Backend {
			return TimeBurndown{}, fmt.Errorf("%w: native receipt %q does not match observation identity", ErrInvalidTimeBurndown, row.RequestID)
		}
		if millis(row.Receipt.Wall) > obs.DurationMS {
			return TimeBurndown{}, fmt.Errorf("%w: native receipt %q exceeds API request wall", ErrInvalidTimeBurndown, row.RequestID)
		}
		joins[row.RequestID] = row.Receipt
	}
	type groupKey struct{ backend, model, cache string }
	groups := make(map[groupKey]*burndownGroupAccumulator)
	order := make([]int, len(rows))
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(i, j int) bool { return rows[order[i]].RequestID < rows[order[j]].RequestID })
	for _, i := range order {
		row := rows[i]
		key := groupKey{row.Backend, row.Model, states[i]}
		acc := groups[key]
		if acc == nil {
			acc = &burndownGroupAccumulator{group: TimeBurndownGroup{Backend: key.backend, Model: key.model, CacheState: key.cache}, api: make(map[string]TimeBurndownPhase), native: make(map[string]TimeBurndownPhase)}
			groups[key] = acc
		}
		g := &acc.group
		if row.Warmup {
			g.WarmupExcluded++
			report.WarmupExcluded++
			continue
		}
		g.Requests++
		if row.Error != "" || row.Status < 200 || row.Status >= 400 {
			g.FailedRequests++
			report.FailedRequests++
			continue
		}
		g.Samples++
		report.IncludedRequests++
		g.APIWallMS += row.DurationMS
		acc.durations = append(acc.durations, row.DurationMS)
		if row.Streaming && row.TTFTMS > 0 {
			g.TTFTSamples++
			acc.ttfts = append(acc.ttfts, row.TTFTMS)
			addBurndownPhase(acc.api, "before_first_output", row.TTFTMS, nil, nil)
			addBurndownPhase(acc.api, "after_first_output", row.DurationMS-row.TTFTMS, nil, nil)
		} else {
			addBurndownPhase(acc.api, "unpartitioned_request", row.DurationMS, nil, nil)
		}
		if receipt, exists := joins[row.RequestID]; exists {
			g.NativeRequests++
			report.NativeRequests++
			g.NativeCoveredAPIWallMS += row.DurationMS
			g.NativeWallMS += millis(receipt.Wall)
			for _, phase := range receipt.Phases {
				active, wait := millis(phase.Exclusive.Active), millis(phase.Exclusive.Wait)
				addBurndownPhase(acc.native, string(phase.Phase), millis(phase.Exclusive.Wall), &active, &wait)
			}
		}
	}
	for _, acc := range groups {
		g := &acc.group
		g.DurationP50MS = percentile(acc.durations, .5)
		g.TTFTP50MS = percentile(acc.ttfts, .5)
		g.OutsideNativeMS = g.NativeCoveredAPIWallMS - g.NativeWallMS
		g.Attribution = "api_only"
		g.AttributionUnavailable = BackendAttributionUnavailable
		if g.NativeRequests > 0 {
			g.Attribution = "partial"
			if g.NativeRequests == g.Samples {
				g.Attribution = "native"
				g.AttributionUnavailable = ""
			}
		}
		g.APIRank = rankBurndownPhases(acc.api, g.APIWallMS)
		g.NativeRank = rankBurndownPhases(acc.native, g.NativeWallMS)
		for _, total := range []float64{g.APIWallMS, g.NativeCoveredAPIWallMS, g.NativeWallMS, g.OutsideNativeMS} {
			if !finiteNonnegative(total) {
				return TimeBurndown{}, fmt.Errorf("%w: cumulative duration overflow", ErrInvalidTimeBurndown)
			}
		}
		for _, phase := range append(append([]TimeBurndownPhase(nil), g.APIRank...), g.NativeRank...) {
			if !finiteNonnegative(phase.WallMS) || !finiteNonnegative(phase.Percent) ||
				(phase.ActiveMS != nil && (!finiteNonnegative(*phase.ActiveMS) || !finiteNonnegative(*phase.WaitMS))) {
				return TimeBurndown{}, fmt.Errorf("%w: cumulative phase duration overflow", ErrInvalidTimeBurndown)
			}
		}
		report.Groups = append(report.Groups, *g)
	}
	sort.Slice(report.Groups, func(i, j int) bool {
		a, b := report.Groups[i], report.Groups[j]
		if a.Backend != b.Backend {
			return a.Backend < b.Backend
		}
		if a.Model != b.Model {
			return a.Model < b.Model
		}
		return a.CacheState < b.CacheState
	})
	return report, nil
}

func finiteNonnegative(value float64) bool {
	return value >= 0 && !math.IsNaN(value) && !math.IsInf(value, 0)
}

func addBurndownPhase(phases map[string]TimeBurndownPhase, name string, wall float64, active, wait *float64) {
	phase := phases[name]
	phase.Phase = name
	phase.WallMS += wall
	if active != nil {
		if phase.ActiveMS == nil {
			phase.ActiveMS = new(float64)
			phase.WaitMS = new(float64)
		}
		*phase.ActiveMS += *active
		*phase.WaitMS += *wait
	}
	phases[name] = phase
}

func rankBurndownPhases(phases map[string]TimeBurndownPhase, denominator float64) []TimeBurndownPhase {
	rank := make([]TimeBurndownPhase, 0, len(phases))
	for _, phase := range phases {
		if denominator > 0 {
			phase.Percent = phase.WallMS / denominator * 100
		}
		rank = append(rank, phase)
	}
	sort.Slice(rank, func(i, j int) bool {
		if rank[i].WallMS != rank[j].WallMS {
			return rank[i].WallMS > rank[j].WallMS
		}
		return rank[i].Phase < rank[j].Phase
	})
	return rank
}

func WriteTimeBurndownMarkdown(w io.Writer, report TimeBurndown) error {
	if _, err := fmt.Fprintf(w, "# Request time burndown\n\nRequests: %d; included: %d; warmups excluded: %d; failures excluded: %d; native receipts: %d.\n\nCache states are declared experimental arms; they do not prove cache residency or reuse. API spans include network, queueing, and backend work. API active/wait and tool spans are unavailable.\n", report.Requests, report.IncludedRequests, report.WarmupExcluded, report.FailedRequests, report.NativeRequests); err != nil {
		return err
	}
	for _, group := range report.Groups {
		if _, err := fmt.Fprintf(w, "\n## %s / %s / %s\n\nSamples: %d; failures: %d; warmups excluded: %d; native coverage: %d/%d. Duration p50: %.3f ms; TTFT p50: %.3f ms (%d samples). Attribution: %s.\n", markdownCell(group.Backend), markdownCell(group.Model), group.CacheState, group.Samples, group.FailedRequests, group.WarmupExcluded, group.NativeRequests, group.Samples, group.DurationP50MS, group.TTFTP50MS, group.TTFTSamples, group.Attribution); err != nil {
			return err
		}
		if group.AttributionUnavailable != "" {
			if _, err := fmt.Fprintf(w, "\nNative attribution unavailable for uncovered samples: `%s`.\n", group.AttributionUnavailable); err != nil {
				return err
			}
		}
		if _, err := fmt.Fprintln(w, "\n| Plane | Phase | Cumulative ms | Share | Active ms | Wait ms |\n| --- | --- | ---: | ---: | ---: | ---: |"); err != nil {
			return err
		}
		for _, plane := range []struct {
			name   string
			phases []TimeBurndownPhase
		}{{"api", group.APIRank}, {"native", group.NativeRank}} {
			for _, phase := range plane.phases {
				active, wait := "unavailable", "unavailable"
				if phase.ActiveMS != nil {
					active = fmt.Sprintf("%.3f", *phase.ActiveMS)
					wait = fmt.Sprintf("%.3f", *phase.WaitMS)
				}
				if _, err := fmt.Fprintf(w, "| %s | %s | %.3f | %.2f%% | %s | %s |\n", plane.name, phase.Phase, phase.WallMS, phase.Percent, active, wait); err != nil {
					return err
				}
			}
		}
		if _, err := fmt.Fprintf(w, "\nAPI denominator: %.3f ms. Native denominator: %.3f ms. Matched API wall: %.3f ms = native %.3f ms + outside_native %.3f ms. These planes overlap and must not be added.\n", group.APIWallMS, group.NativeWallMS, group.NativeCoveredAPIWallMS, group.NativeWallMS, group.OutsideNativeMS); err != nil {
			return err
		}
	}
	return nil
}

func markdownCell(value string) string {
	return strings.NewReplacer("\n", " ", "\r", " ", "|", "\\|", "`", "'").Replace(value)
}

// WriteTimeBurndownPrometheus aggregates identities away before rendering the
// fixed cache-state/phase/plane label vocabulary. It exports no request IDs.
func WriteTimeBurndownPrometheus(w io.Writer, report TimeBurndown) error {
	type metricKey struct{ cache, plane, phase, kind string }
	totals := make(map[metricKey]float64)
	counts := make(map[string][4]int)
	for _, group := range report.Groups {
		state, err := ParseCacheState(group.CacheState)
		if err != nil {
			return err
		}
		count := counts[state]
		count[0] += group.Samples
		count[1] += group.WarmupExcluded
		count[2] += group.FailedRequests
		count[3] += group.NativeRequests
		counts[state] = count
		for _, plane := range []struct {
			name   string
			phases []TimeBurndownPhase
		}{{"api", group.APIRank}, {"native", group.NativeRank}} {
			for _, phase := range plane.phases {
				if !knownBurndownPhase(plane.name, phase.Phase) || !finiteNonnegative(phase.WallMS) {
					return fmt.Errorf("%w: unbounded phase or invalid timing", ErrInvalidTimeBurndown)
				}
				totals[metricKey{state, plane.name, phase.Phase, "wall"}] += phase.WallMS / 1000
				if plane.name == "native" && phase.ActiveMS != nil && phase.WaitMS != nil {
					if !finiteNonnegative(*phase.ActiveMS) || !finiteNonnegative(*phase.WaitMS) {
						return fmt.Errorf("%w: invalid native active/wait timing", ErrInvalidTimeBurndown)
					}
					totals[metricKey{state, plane.name, phase.Phase, "active"}] += *phase.ActiveMS / 1000
					totals[metricKey{state, plane.name, phase.Phase, "wait"}] += *phase.WaitMS / 1000
				}
			}
		}
		if !finiteNonnegative(group.OutsideNativeMS) {
			return fmt.Errorf("%w: invalid outside native duration", ErrInvalidTimeBurndown)
		}
		totals[metricKey{state, "outside_native", "outside_native", "wall"}] += group.OutsideNativeMS / 1000
	}
	for _, total := range totals {
		if !finiteNonnegative(total) {
			return fmt.Errorf("%w: cumulative metric duration overflow", ErrInvalidTimeBurndown)
		}
	}
	if _, err := fmt.Fprint(w, "# HELP fak_time_burndown_seconds Snapshot of cumulative successful request seconds; API and native planes overlap.\n# TYPE fak_time_burndown_seconds gauge\n"); err != nil {
		return err
	}
	keys := make([]metricKey, 0, len(totals))
	for key := range totals {
		keys = append(keys, key)
	}
	sort.Slice(keys, func(i, j int) bool {
		a, b := keys[i], keys[j]
		if a.cache != b.cache {
			return a.cache < b.cache
		}
		if a.plane != b.plane {
			return a.plane < b.plane
		}
		if a.phase != b.phase {
			return a.phase < b.phase
		}
		return a.kind < b.kind
	})
	for _, key := range keys {
		if _, err := fmt.Fprintf(w, "fak_time_burndown_seconds{cache_state=%q,plane=%q,phase=%q,kind=%q} %.9g\n", key.cache, key.plane, key.phase, key.kind, totals[key]); err != nil {
			return err
		}
	}
	for _, metric := range []struct {
		name  string
		index int
	}{{"samples", 0}, {"warmups_excluded", 1}, {"failures_excluded", 2}, {"native_samples", 3}} {
		if _, err := fmt.Fprintf(w, "# TYPE fak_time_burndown_%s gauge\n", metric.name); err != nil {
			return err
		}
		for _, state := range []string{"cold", "warm", "steady", "unknown"} {
			if count, ok := counts[state]; ok {
				if _, err := fmt.Fprintf(w, "fak_time_burndown_%s{cache_state=%q} %d\n", metric.name, state, count[metric.index]); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

func knownBurndownPhase(plane, phase string) bool {
	if plane == "api" {
		return phase == "before_first_output" || phase == "after_first_output" || phase == "unpartitioned_request"
	}
	for _, known := range nativeperf.Phases() {
		if phase == string(known) {
			return true
		}
	}
	return false
}
