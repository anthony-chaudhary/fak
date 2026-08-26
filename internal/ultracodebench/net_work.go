package ultracodebench

import (
	"errors"
	"fmt"
	"sort"
)

const NetWorkCampaignSchema = "fak-ultracode-net-work-campaign/1"
const NetWorkReportSchema = "fak-ultracode-net-work-report/1"

var netWorkCells = []string{"no-reuse", "prefix-only", "scope-only", "combined"}
var requiredWorkComponents = []string{"model_prefill", "model_decode", "cache_operations", "tool_work", "routing", "orchestration"}

type NetWorkEnvelope struct {
	Model           string `json:"model"`
	Runtime         string `json:"runtime"`
	Tokenizer       string `json:"tokenizer"`
	TaskDigest      string `json:"task_digest"`
	CachePosture    string `json:"cache_posture"`
	CampaignVersion string `json:"campaign_version"`
}

type NetWorkCampaign struct {
	Schema             string          `json:"schema"`
	EvidenceKind       string          `json:"evidence_kind"`
	CapturedAt         string          `json:"captured_at"`
	SourceReceipt      string          `json:"source_receipt"`
	Envelope           NetWorkEnvelope `json:"envelope"`
	OutcomeDigest      string          `json:"accepted_outcome_digest"`
	OrderPolicy        string          `json:"order_policy"`
	MinimumRepetitions int             `json:"minimum_repetitions"`
	Cells              []NetWorkCell   `json:"cells"`
}

type NetWorkCell struct {
	Width       int                 `json:"width"`
	Treatment   string              `json:"treatment"`
	Repetitions []NetWorkRepetition `json:"repetitions"`
}

type NetWorkRepetition struct {
	Receipt       string                      `json:"receipt"`
	Accepted      bool                        `json:"accepted"`
	OutcomeDigest string                      `json:"outcome_digest"`
	InputTokens   int64                       `json:"input_tokens"`
	CachedTokens  int64                       `json:"cached_tokens"`
	WallLatencyNS int64                       `json:"wall_latency_ns"`
	Components    map[string]NetWorkComponent `json:"components"`
}

type NetWorkComponent struct {
	Controller    string `json:"controller"`
	DurationNS    int64  `json:"duration_ns"`
	Authoritative bool   `json:"authoritative"`
}

type NetWorkReport struct {
	Schema        string               `json:"schema"`
	Verdict       string               `json:"verdict"`
	Reason        string               `json:"reason,omitempty"`
	EvidenceKind  string               `json:"evidence_kind"`
	SourceReceipt string               `json:"source_receipt"`
	Envelope      NetWorkEnvelope      `json:"envelope"`
	Widths        []NetWorkWidthReport `json:"widths"`
	HillClimb     NetWorkHillClimb     `json:"hill_climb"`
	ReplayCommand string               `json:"replay_command"`
}

type NetWorkWidthReport struct {
	Width                  int                 `json:"width"`
	Verdict                string              `json:"verdict"`
	Reason                 string              `json:"reason,omitempty"`
	Cells                  []NetWorkCellReport `json:"cells"`
	CombinedNetWorkDeltaNS int64               `json:"combined_net_work_delta_ns,omitempty"`
	CombinedWallDeltaNS    int64               `json:"combined_wall_delta_ns,omitempty"`
	CombinedTokenDelta     int64               `json:"combined_token_delta,omitempty"`
}

type NetWorkCellReport struct {
	Treatment             string                            `json:"treatment"`
	Repetitions           int                               `json:"repetitions"`
	InputTokensMedian     int64                             `json:"input_tokens_median"`
	CachedTokensMedian    int64                             `json:"cached_tokens_median"`
	AccountedWorkMedianNS int64                             `json:"accounted_work_median_ns"`
	WallLatencyMedianNS   int64                             `json:"wall_latency_median_ns"`
	WallLatencyP95NS      int64                             `json:"wall_latency_p95_ns"`
	Components            map[string]NetWorkComponentReport `json:"components"`
}

type NetWorkComponentReport struct {
	Controller string `json:"controller"`
	MedianNS   int64  `json:"median_ns"`
	P95NS      int64  `json:"p95_ns"`
}

type NetWorkHillClimb struct {
	ChosenWidth int    `json:"chosen_width"`
	StopWidth   int    `json:"stop_width,omitempty"`
	StopReason  string `json:"stop_reason,omitempty"`
}

// EvaluateNetWorkCampaign compares controlled cells in common duration units. Token
// counts remain diagnostics and can never override a measured net-work loss.
func EvaluateNetWorkCampaign(c NetWorkCampaign) (NetWorkReport, error) {
	if c.Schema != NetWorkCampaignSchema {
		return NetWorkReport{}, fmt.Errorf("schema %q, want %q", c.Schema, NetWorkCampaignSchema)
	}
	if c.SourceReceipt == "" {
		return NetWorkReport{}, errors.New("source receipt is required")
	}
	if c.Envelope.Model == "" || c.Envelope.Runtime == "" || c.Envelope.Tokenizer == "" || c.Envelope.TaskDigest == "" || c.Envelope.CachePosture == "" || c.Envelope.CampaignVersion == "" {
		return NetWorkReport{}, errors.New("complete production envelope is required")
	}
	minimum := c.MinimumRepetitions
	if minimum < 3 {
		minimum = 3
	}
	grouped := map[int]map[string]NetWorkCell{}
	for _, cell := range c.Cells {
		if cell.Width <= 0 {
			return NetWorkReport{}, errors.New("width must be positive")
		}
		if grouped[cell.Width] == nil {
			grouped[cell.Width] = map[string]NetWorkCell{}
		}
		if _, exists := grouped[cell.Width][cell.Treatment]; exists {
			return NetWorkReport{}, fmt.Errorf("duplicate width %d treatment %s", cell.Width, cell.Treatment)
		}
		grouped[cell.Width][cell.Treatment] = cell
	}
	widths := make([]int, 0, len(grouped))
	for w := range grouped {
		widths = append(widths, w)
	}
	sort.Ints(widths)
	report := NetWorkReport{Schema: NetWorkReportSchema, Verdict: "GAIN", EvidenceKind: c.EvidenceKind, SourceReceipt: c.SourceReceipt, Envelope: c.Envelope, ReplayCommand: "fak ultracode bench --net-work <campaign.json> --json"}
	stopped := false
	for _, width := range widths {
		wr := NetWorkWidthReport{Width: width, Verdict: "GAIN"}
		cells := grouped[width]
		for _, treatment := range netWorkCells {
			cell, ok := cells[treatment]
			if !ok {
				wr.Verdict, wr.Reason = "ABSTAIN", "missing treatment "+treatment
				continue
			}
			cr, reason := summarizeNetWorkCell(cell, c.OutcomeDigest, minimum)
			wr.Cells = append(wr.Cells, cr)
			if reason != "" && wr.Reason == "" {
				wr.Verdict, wr.Reason = "ABSTAIN", reason
			}
		}
		if wr.Verdict != "ABSTAIN" {
			base, combined := wr.Cells[0], wr.Cells[3]
			wr.CombinedNetWorkDeltaNS = combined.AccountedWorkMedianNS - base.AccountedWorkMedianNS
			wr.CombinedWallDeltaNS = combined.WallLatencyMedianNS - base.WallLatencyMedianNS
			wr.CombinedTokenDelta = combined.InputTokensMedian - base.InputTokensMedian
			if wr.CombinedNetWorkDeltaNS >= 0 || wr.CombinedWallDeltaNS >= 0 {
				wr.Verdict, wr.Reason = "LOSS", "combined cell has an equal-outcome net loss"
			}
		}
		report.Widths = append(report.Widths, wr)
		if !stopped {
			if wr.Verdict == "GAIN" {
				report.HillClimb.ChosenWidth = width
			} else {
				stopped = true
				report.HillClimb.StopWidth = width
				report.HillClimb.StopReason = wr.Reason
			}
		}
	}
	if len(widths) == 0 {
		report.Verdict, report.Reason = "ABSTAIN", "no measured widths"
	}
	for _, wr := range report.Widths {
		if wr.Verdict == "ABSTAIN" {
			report.Verdict, report.Reason = "ABSTAIN", "one or more widths lack authoritative equal-outcome telemetry"
			break
		}
		if wr.Verdict == "LOSS" && report.Verdict == "GAIN" {
			report.Verdict = "LOSS"
		}
	}
	return report, nil
}

func summarizeNetWorkCell(cell NetWorkCell, outcome string, minimum int) (NetWorkCellReport, string) {
	cr := NetWorkCellReport{Treatment: cell.Treatment, Repetitions: len(cell.Repetitions), Components: map[string]NetWorkComponentReport{}}
	if len(cell.Repetitions) < minimum {
		return cr, fmt.Sprintf("%s has %d repetitions; need %d to expose startup noise", cell.Treatment, len(cell.Repetitions), minimum)
	}
	inputs, cached, walls, totals := []int64{}, []int64{}, []int64{}, []int64{}
	componentValues := map[string][]int64{}
	controllers := map[string]string{}
	for _, rep := range cell.Repetitions {
		if rep.Receipt == "" {
			return cr, cell.Treatment + " has a repetition without a source receipt"
		}
		if !rep.Accepted || rep.OutcomeDigest == "" || rep.OutcomeDigest != outcome {
			return cr, cell.Treatment + " has an unequal or unverified outcome"
		}
		if rep.WallLatencyNS <= 0 {
			return cr, cell.Treatment + " lacks authoritative wall latency"
		}
		var total int64
		for _, name := range requiredWorkComponents {
			component, ok := rep.Components[name]
			if !ok || !component.Authoritative || component.Controller == "" || component.DurationNS < 0 {
				return cr, cell.Treatment + " lacks authoritative " + name + " telemetry"
			}
			if prior := controllers[name]; prior != "" && prior != component.Controller {
				return cr, cell.Treatment + " changes controller for " + name
			}
			controllers[name] = component.Controller
			componentValues[name] = append(componentValues[name], component.DurationNS)
			total += component.DurationNS
		}
		inputs = append(inputs, rep.InputTokens)
		cached = append(cached, rep.CachedTokens)
		walls = append(walls, rep.WallLatencyNS)
		totals = append(totals, total)
	}
	cr.InputTokensMedian = quantile(inputs, 0.5)
	cr.CachedTokensMedian = quantile(cached, 0.5)
	cr.AccountedWorkMedianNS = quantile(totals, 0.5)
	cr.WallLatencyMedianNS = quantile(walls, 0.5)
	cr.WallLatencyP95NS = quantile(walls, 0.95)
	for _, name := range requiredWorkComponents {
		cr.Components[name] = NetWorkComponentReport{Controller: controllers[name], MedianNS: quantile(componentValues[name], 0.5), P95NS: quantile(componentValues[name], 0.95)}
	}
	return cr, ""
}

func quantile(values []int64, q float64) int64 {
	copyValues := append([]int64(nil), values...)
	sort.Slice(copyValues, func(i, j int) bool { return copyValues[i] < copyValues[j] })
	index := int(float64(len(copyValues)-1)*q + 0.999999999)
	return copyValues[index]
}
