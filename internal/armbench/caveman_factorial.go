package armbench

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const CavemanFactorialSchema = "fak-armbench-caveman-factorial/1"

type FactorialTreatment string

const (
	TreatmentPassthrough FactorialTreatment = "passthrough"
	TreatmentCompress    FactorialTreatment = "tool-result-compression"
	TreatmentShed        FactorialTreatment = "context-shedding"
	TreatmentBoth        FactorialTreatment = "compression+shedding"
	TreatmentTuned       FactorialTreatment = "tuned-bundle"
)

type FactorialOptions struct {
	OutputDir string
	Pressures []int
	BaseURL   string
	APIKey    string
	Model     string
	InputDir  string
}

type FactorialStage struct {
	Name         string `json:"name"`
	CPUTimeNS    int64  `json:"cpu_time_ns"`
	BytesBefore  int    `json:"bytes_before"`
	BytesAfter   int    `json:"bytes_after"`
	TokensBefore int    `json:"tokens_before_estimated"`
	TokensAfter  int    `json:"tokens_after_estimated"`
}

type FactorialCell struct {
	Style                    string             `json:"style"`
	Treatment                FactorialTreatment `json:"treatment"`
	Workload                 string             `json:"workload"`
	Pressure                 int                `json:"pressure"`
	Turns                    int                `json:"turns"`
	OutputTokensEstimated    int                `json:"answer_output_tokens_estimated"`
	OutputTokensProvider     *int64             `json:"answer_output_tokens_provider"`
	ProviderOutput           string             `json:"provider_output,omitempty"`
	ProviderInputTokens      *int64             `json:"provider_input_tokens"`
	ProviderCacheReadTokens  *int64             `json:"provider_cache_read_tokens"`
	ProviderCacheWriteTokens *int64             `json:"provider_cache_write_tokens"`
	RetainedFacts            int                `json:"retained_facts"`
	TotalFacts               int                `json:"total_facts"`
	Quality                  float64            `json:"quality"`
	Stages                   []FactorialStage   `json:"stages"`
	providerContext          string
}

type InteractionRow struct {
	Workload        string             `json:"workload"`
	Pressure        int                `json:"pressure"`
	Treatment       FactorialTreatment `json:"treatment"`
	InputBytesDID   int                `json:"input_bytes_difference_in_differences"`
	OutputTokensDID int                `json:"output_tokens_difference_in_differences_estimated"`
	QualityDID      float64            `json:"quality_difference_in_differences"`
	Classification  string             `json:"classification"`
}

type FrontierPoint struct {
	Style                 string             `json:"style"`
	Treatment             FactorialTreatment `json:"treatment"`
	Workload              string             `json:"workload"`
	Pressure              int                `json:"pressure"`
	InputBytes            int                `json:"input_bytes"`
	OutputTokensEstimated int                `json:"output_tokens_estimated"`
	Quality               float64            `json:"quality"`
}

type FactorialManifest struct {
	Schema           string           `json:"schema"`
	Comparator       string           `json:"comparator"`
	EvidenceClass    string           `json:"evidence_class"`
	TokenMethod      string           `json:"token_method"`
	DisabledFeatures []string         `json:"disabled_features"`
	ProviderEndpoint string           `json:"provider_endpoint,omitempty"`
	ProviderModel    string           `json:"provider_model,omitempty"`
	Pressures        []int            `json:"pressures"`
	Cells            []FactorialCell  `json:"cells"`
	QualityFrontier  []FrontierPoint  `json:"quality_frontier"`
	Interactions     []InteractionRow `json:"interaction_effects"`
	Conclusion       string           `json:"conclusion"`
}

func RunCavemanFactorial(o FactorialOptions) (FactorialManifest, error) {
	if len(o.Pressures) == 0 {
		o.Pressures = []int{1, 4, 12}
	}
	if len(o.Pressures) < 3 {
		return FactorialManifest{}, fmt.Errorf("factorial requires at least three pressure levels")
	}
	for _, p := range o.Pressures {
		if p < 1 {
			return FactorialManifest{}, fmt.Errorf("pressure must be positive: %d", p)
		}
	}
	styles := []string{"normal", "caveman"}
	treatments := []FactorialTreatment{TreatmentPassthrough, TreatmentCompress, TreatmentShed, TreatmentBoth, TreatmentTuned}
	workloads := []string{"native-one-shot", "multi-turn-growing-tool-results"}
	comparator := "JuliusBrussee/caveman@" + CavemanRevision
	if o.InputDir != "" {
		if _, err := verifyCavemanInputs(o.InputDir); err != nil {
			return FactorialManifest{}, err
		}
	}
	m := FactorialManifest{Schema: CavemanFactorialSchema, Comparator: comparator, EvidenceClass: "deterministic-transform-receipt", TokenMethod: "UTF-8 bytes/4 ceiling; provider token fields intentionally null without provider receipts", DisabledFeatures: []string{"routing", "policy", "semantic-response-reuse"}, Pressures: append([]int(nil), o.Pressures...)}
	for _, w := range workloads {
		for _, p := range o.Pressures {
			for _, s := range styles {
				for _, t := range treatments {
					cell, err := factorialCell(s, t, w, p)
					if err != nil {
						return FactorialManifest{}, err
					}
					if o.BaseURL != "" {
						if err := fillFactorialProvider(context.Background(), &cell, o); err != nil {
							return FactorialManifest{}, err
						}
					}
					m.Cells = append(m.Cells, cell)
				}
			}
		}
	}
	m.QualityFrontier = factorialFrontier(m.Cells)
	m.Interactions = factorialInteractions(m.Cells)
	m.Conclusion = "Deterministic receipts establish transformation costs, retained-fact quality, and byte-level interactions only. Provider input/cache tokens and efficiency remain NOT-YET until a live manifest supplies provider usage."
	if o.BaseURL != "" {
		m.EvidenceClass = "live-provider-and-deterministic-transform-receipt"
		m.ProviderEndpoint = sanitizeEndpoint(o.BaseURL)
		m.ProviderModel = o.Model
		m.Conclusion = "Live provider usage and outputs are captured per cell; transformation CPU/bytes are local receipts. Cache conclusions require non-zero provider cache fields."
	}
	if o.OutputDir != "" {
		if err := os.MkdirAll(o.OutputDir, 0755); err != nil {
			return FactorialManifest{}, err
		}
		b, _ := json.MarshalIndent(m, "", "  ")
		b = append(b, '\n')
		if err := os.WriteFile(filepath.Join(o.OutputDir, "manifest.json"), b, 0644); err != nil {
			return FactorialManifest{}, err
		}
	}
	return m, nil
}

func factorialCell(style string, treatment FactorialTreatment, workload string, pressure int) (FactorialCell, error) {
	msgs, totalFacts, turns := factorialMessages(workload, pressure)
	root := map[string]any{"system": []any{map[string]any{"type": "text", "text": factorialSystem(style)}}, "messages": msgs}
	raw, err := json.Marshal(root)
	if err != nil {
		return FactorialCell{}, err
	}
	cell := FactorialCell{Style: style, Treatment: treatment, Workload: workload, Pressure: pressure, Turns: turns, TotalFacts: totalFacts}
	current := raw
	addStage := func(name string, enabled bool, transform func([]byte) ([]byte, error)) error {
		before := current
		start := time.Now()
		after := before
		var e error
		if enabled {
			after, e = transform(before)
		}
		elapsed := time.Since(start).Nanoseconds()
		if elapsed < 1 {
			elapsed = 1
		}
		if e != nil {
			return e
		}
		cell.Stages = append(cell.Stages, FactorialStage{Name: name, CPUTimeNS: elapsed, BytesBefore: len(before), BytesAfter: len(after), TokensBefore: estimateTokens(before), TokensAfter: estimateTokens(after)})
		current = after
		return nil
	}
	compress := treatment == TreatmentCompress || treatment == TreatmentBoth || treatment == TreatmentTuned
	shed := treatment == TreatmentShed || treatment == TreatmentBoth || treatment == TreatmentTuned
	if err := addStage("tool-result-compression", compress, func(b []byte) ([]byte, error) {
		out, _, e := transformAnthropicRequest(b, ManagedToggles{ToolResultCompression: true})
		return out, e
	}); err != nil {
		return cell, err
	}
	if err := addStage("context-shedding", shed, func(b []byte) ([]byte, error) {
		out, _, e := transformAnthropicRequest(b, ManagedToggles{ContextShedding: true})
		return out, e
	}); err != nil {
		return cell, err
	}
	if err := addStage("tuned-cache-layout", treatment == TreatmentTuned, func(b []byte) ([]byte, error) {
		out, _, e := transformAnthropicRequest(b, ManagedToggles{SharedPrefixProviderCache: true})
		return out, e
	}); err != nil {
		return cell, err
	}
	cell.providerContext = string(current)
	cell.RetainedFacts = countFacts(current, totalFacts)
	cell.Quality = float64(cell.RetainedFacts) / float64(totalFacts)
	answer := "The retained task facts are complete and the requested action is ready."
	if style == "caveman" {
		answer = "Facts kept. Action ready."
	}
	cell.OutputTokensEstimated = estimateTokens([]byte(answer))
	return cell, nil
}

func factorialMessages(workload string, pressure int) ([]any, int, int) {
	if workload == "native-one-shot" {
		return []any{map[string]any{"role": "user", "content": "You are presenting to an audience. Explain a difficult technical idea clearly, preserve FACT-000, and finish with one action."}}, 1, 1
	}
	msgs := []any{map[string]any{"role": "user", "content": "Stable setup: preserve every FACT marker and finish with one action."}}
	for i := 0; i < pressure; i++ {
		fact := fmt.Sprintf("FACT-%03d", i)
		payload := fact + ": " + strings.Repeat(fmt.Sprintf("tool observation %03d; ", i), 240)
		msgs = append(msgs, map[string]any{"role": "assistant", "content": []any{map[string]any{"type": "tool_use", "id": fmt.Sprintf("tool-%03d", i), "name": "inspect"}}}, map[string]any{"role": "user", "content": []any{map[string]any{"type": "tool_result", "tool_use_id": fmt.Sprintf("tool-%03d", i), "content": payload}}})
	}
	msgs = append(msgs, map[string]any{"role": "user", "content": fmt.Sprintf("Now synthesize all %d FACT markers and give the action.", pressure)})
	return msgs, pressure, pressure + 2
}
func factorialSystem(style string) string {
	if style == "caveman" {
		return "Apply the pinned Caveman answer style: telegraphic, terse, no lost facts."
	}
	return "Answer normally and preserve all task facts."
}
func estimateTokens(b []byte) int { return (len(b) + 3) / 4 }
func countFacts(b []byte, n int) int {
	c := 0
	for i := 0; i < n; i++ {
		if strings.Contains(string(b), fmt.Sprintf("FACT-%03d", i)) {
			c++
		}
	}
	return c
}
func finalBytes(c FactorialCell) int {
	if len(c.Stages) == 0 {
		return 0
	}
	return c.Stages[len(c.Stages)-1].BytesAfter
}
func factorialFrontier(cells []FactorialCell) []FrontierPoint {
	var out []FrontierPoint
	for _, c := range cells {
		dom := false
		for _, x := range cells {
			if x.Style != c.Style || x.Workload != c.Workload || x.Pressure != c.Pressure {
				continue
			}
			if finalBytes(x) <= finalBytes(c) && x.OutputTokensEstimated <= c.OutputTokensEstimated && x.Quality >= c.Quality && (finalBytes(x) < finalBytes(c) || x.OutputTokensEstimated < c.OutputTokensEstimated || x.Quality > c.Quality) {
				dom = true
				break
			}
		}
		if !dom {
			out = append(out, FrontierPoint{c.Style, c.Treatment, c.Workload, c.Pressure, finalBytes(c), c.OutputTokensEstimated, c.Quality})
		}
	}
	return out
}
func factorialInteractions(cells []FactorialCell) []InteractionRow {
	lookup := map[string]FactorialCell{}
	for _, c := range cells {
		lookup[fmt.Sprintf("%s/%d/%s/%s", c.Workload, c.Pressure, c.Style, c.Treatment)] = c
	}
	var out []InteractionRow
	for _, w := range []string{"native-one-shot", "multi-turn-growing-tool-results"} {
		for _, p := range uniquePressures(cells) {
			baseN := lookup[fmt.Sprintf("%s/%d/normal/%s", w, p, TreatmentPassthrough)]
			baseC := lookup[fmt.Sprintf("%s/%d/caveman/%s", w, p, TreatmentPassthrough)]
			for _, t := range []FactorialTreatment{TreatmentCompress, TreatmentShed, TreatmentBoth, TreatmentTuned} {
				n := lookup[fmt.Sprintf("%s/%d/normal/%s", w, p, t)]
				c := lookup[fmt.Sprintf("%s/%d/caveman/%s", w, p, t)]
				ib := (finalBytes(c) - finalBytes(baseC)) - (finalBytes(n) - finalBytes(baseN))
				ot := (c.OutputTokensEstimated - baseC.OutputTokensEstimated) - (n.OutputTokensEstimated - baseN.OutputTokensEstimated)
				q := (c.Quality - baseC.Quality) - (n.Quality - baseN.Quality)
				class := "redundant"
				if q < 0 {
					class = "harmful"
				} else if ib < 0 || ot < 0 || q > 0 {
					class = "complementary"
				}
				out = append(out, InteractionRow{w, p, t, ib, ot, q, class})
			}
		}
	}
	return out
}
func uniquePressures(cells []FactorialCell) []int {
	m := map[int]bool{}
	for _, c := range cells {
		m[c.Pressure] = true
	}
	var out []int
	for p := range m {
		out = append(out, p)
	}
	sort.Ints(out)
	return out
}

func fillFactorialProvider(ctx context.Context, cell *FactorialCell, o FactorialOptions) error {
	if o.APIKey == "" || o.Model == "" {
		return fmt.Errorf("live factorial requires API key and model")
	}
	requestContext := fmt.Sprintf("Workload receipt follows. Preserve and enumerate every FACT marker, then give one action. Style=%s.\n%s", cell.Style, cell.providerContext)
	body, _ := json.Marshal(map[string]any{"model": o.Model, "temperature": 0, "max_tokens": 512, "messages": []any{map[string]any{"role": "system", "content": factorialSystem(cell.Style)}, map[string]any{"role": "user", "content": requestContext}}})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(o.BaseURL, "/")+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+o.APIKey)
	req.Header.Set("Content-Type", "application/json")
	resp, err := (&http.Client{Timeout: 5 * time.Minute}).Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode/100 != 2 {
		return fmt.Errorf("provider %s: %s", resp.Status, string(raw))
	}
	var root struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Usage struct {
			Prompt     int64 `json:"prompt_tokens"`
			Completion int64 `json:"completion_tokens"`
			Details    struct {
				Cached int64 `json:"cached_tokens"`
			} `json:"prompt_tokens_details"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(raw, &root); err != nil {
		return err
	}
	if len(root.Choices) == 0 {
		return fmt.Errorf("provider returned no choice")
	}
	cell.ProviderInputTokens = &root.Usage.Prompt
	cell.ProviderCacheReadTokens = &root.Usage.Details.Cached
	zero := int64(0)
	cell.ProviderCacheWriteTokens = &zero
	cell.OutputTokensProvider = &root.Usage.Completion
	cell.ProviderOutput = root.Choices[0].Message.Content
	cell.RetainedFacts = countFacts([]byte(cell.ProviderOutput), cell.TotalFacts)
	cell.Quality = float64(cell.RetainedFacts) / float64(cell.TotalFacts)
	return nil
}
