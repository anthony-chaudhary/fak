package readmenext

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"
)

const (
	// DefaultHeroSweepDataPath is the canonical hero sweep dataset path.
	DefaultHeroSweepDataPath = "tools/hero_sweep.data.json"
	// DefaultHeroSweepSVGPath is the generated SVG hero visual path.
	DefaultHeroSweepSVGPath = "visuals/61-hero-benchmark-sweep.svg"
	// DefaultHeroSweepPNGPath is the rasterized PNG hero visual path.
	DefaultHeroSweepPNGPath = "visuals/61-hero-benchmark-sweep.png"
	// DefaultMinDeltaPct is the default materiality threshold (5%) for updating the hero.
	DefaultMinDeltaPct = 5.0
)

var (
	reISODateTrailing = regexp.MustCompile(`\b\d{4}-\d{2}-\d{2}\s*$`)
	reDisallowedClaim = regexp.MustCompile(`(?i)\[(\s*SIMULATED\s*|MODELED|THEORETICAL)\]|\b(SIMULATED|MODELED|THEORETICAL)\b`)
)

// HeroSweepFile mirrors tools/hero_sweep.data.json.
type HeroSweepFile struct {
	About    string            `json:"_about,omitempty"`
	Meta     HeroSweepMeta     `json:"meta"`
	Kicker   string            `json:"kicker"`
	Title    string            `json:"title"`
	Subtitle string            `json:"subtitle"`
	Legend   []HeroSweepLegend `json:"legend"`
	Panels   []HeroSweepPanel  `json:"panels"`
	Footer   string            `json:"footer"`
	Desc     string            `json:"desc"`
}

// HeroSweepMeta holds metadata for the hero sweep grid.
type HeroSweepMeta struct {
	Title     string `json:"title"`
	AsOf      string `json:"as_of"`
	OutSVG    string `json:"out_svg"`
	Authority string `json:"authority"`
	Receipts  string `json:"receipts"`
}

// HeroSweepLegend describes a legend entry in the hero sweep grid.
type HeroSweepLegend struct {
	Label string `json:"label"`
	Color string `json:"color"`
}

// HeroSweepPanel describes one benchmark panel in the hero sweep grid.
type HeroSweepPanel struct {
	Name          string         `json:"name"`
	Pillar        string         `json:"pillar"`
	Foot          string         `json:"foot"`
	Commit        string         `json:"commit"`
	Callout       string         `json:"callout,omitempty"`
	LogMultiplier string         `json:"log_multiplier,omitempty"`
	LowerIsBetter bool           `json:"lower_is_better,omitempty"`
	Capability    bool           `json:"capability,omitempty"`
	Fence         bool           `json:"fence,omitempty"`
	YMax          float64        `json:"y_max"`
	Bars          []HeroSweepBar `json:"bars"`
}

// HeroSweepBar describes one bar inside a hero sweep panel.
type HeroSweepBar struct {
	Who     string  `json:"who"`
	Badge   string  `json:"badge"`
	V       float64 `json:"v"`
	Label   string  `json:"label"`
	Subject bool    `json:"subject,omitempty"`
}

// HeroRecordInput describes a newly beaten benchmark record candidate.
type HeroRecordInput struct {
	Panel            string  `json:"panel"`
	Value            float64 `json:"value"`
	Label            string  `json:"label,omitempty"`
	Commit           string  `json:"commit"`
	ReceiptPath      string  `json:"receipt_path"`
	Date             string  `json:"date,omitempty"`
	Foot             string  `json:"foot,omitempty"`
	Callout          string  `json:"callout,omitempty"`
	Desc             string  `json:"desc,omitempty"`
	MinDeltaPct      float64 `json:"min_delta_pct,omitempty"`
	ReadmeOld        string  `json:"readme_old,omitempty"`
	ReadmeNew        string  `json:"readme_new,omitempty"`
	Platform         string  `json:"platform,omitempty"`
	HardwareRow      string  `json:"hardware_row,omitempty"`
	AllowNonMaterial bool    `json:"allow_non_material,omitempty"`
}

// RecordOutcome reports the evaluation and applied updates for a record candidate.
type RecordOutcome struct {
	Panel         string   `json:"panel"`
	Material      bool     `json:"material"`
	Beat          bool     `json:"beat"`
	Skipped       bool     `json:"skipped"`
	SkipReason    string   `json:"skip_reason,omitempty"`
	PriorValue    float64  `json:"prior_value"`
	NewValue      float64  `json:"new_value"`
	DeltaPct      float64  `json:"delta_pct"`
	LowerIsBetter bool     `json:"lower_is_better"`
	FlippedFence  bool     `json:"flipped_fence"`
	UpdatedFiles  []string `json:"updated_files,omitempty"`
	NextActions   []string `json:"next_actions,omitempty"`
}

// ValidateHeroRecordInput checks provenance, date, and honesty constraints on a candidate.
func ValidateHeroRecordInput(cand HeroRecordInput, repoRoot string) error {
	if strings.TrimSpace(cand.Panel) == "" {
		return errors.New("panel is required")
	}
	if math.IsNaN(cand.Value) || math.IsInf(cand.Value, 0) || cand.Value <= 0 {
		return fmt.Errorf("value must be a finite positive number, got %v", cand.Value)
	}
	if strings.TrimSpace(cand.Commit) == "" {
		return errors.New("commit is required")
	}
	receipt := strings.TrimSpace(cand.ReceiptPath)
	if receipt == "" {
		return errors.New("receipt_path is required")
	}
	cleaned := filepath.Clean(receipt)
	if filepath.IsAbs(receipt) || strings.HasPrefix(cleaned, "..") {
		return fmt.Errorf("receipt_path %q must be a relative path within the repository", receipt)
	}
	if repoRoot != "" {
		full := filepath.Join(repoRoot, cleaned)
		if _, err := os.Stat(full); err != nil {
			if os.IsNotExist(err) {
				return fmt.Errorf("receipt_path %q does not exist relative to repoRoot: %s", receipt, full)
			}
			return fmt.Errorf("failed to stat receipt_path %q: %w", receipt, err)
		}
	}

	if cand.Date != "" {
		if _, err := time.Parse("2006-01-02", cand.Date); err != nil {
			return fmt.Errorf("invalid date %q: expected YYYY-MM-DD", cand.Date)
		}
	}

	for _, field := range []struct {
		name string
		val  string
	}{
		{"foot", cand.Foot},
		{"callout", cand.Callout},
		{"readme_new", cand.ReadmeNew},
		{"hardware_row", cand.HardwareRow},
	} {
		if reDisallowedClaim.MatchString(field.val) {
			return fmt.Errorf("%s cannot contain simulated, modeled, or theoretical claims: %q", field.name, field.val)
		}
	}

	if cand.ReadmeNew != "" {
		if err := validateSOTALaw(&CandidateFragment{
			CandidateContent: cand.ReadmeNew,
			LawsChecklist:    LawsChecklist{SOTAComparison: true},
		}); err != nil {
			return fmt.Errorf("readme_new failed SOTA law check: %w", err)
		}
	}

	if (cand.ReadmeOld != "" && cand.ReadmeNew == "") || (cand.ReadmeOld == "" && cand.ReadmeNew != "") {
		return errors.New("readme_old and readme_new must be provided together")
	}
	if (cand.Platform != "" && cand.HardwareRow == "") || (cand.Platform == "" && cand.HardwareRow != "") {
		return errors.New("platform and hardware_row must be provided together")
	}
	if cand.HardwareRow != "" && !strings.HasPrefix(strings.TrimSpace(cand.HardwareRow), "|") {
		return fmt.Errorf("hardware_row must be a markdown table row starting with '|': %q", cand.HardwareRow)
	}

	return nil
}

func findPanelIndex(panels []HeroSweepPanel, name string) int {
	want := strings.TrimSpace(name)
	for i, p := range panels {
		if p.Name == want {
			return i
		}
	}
	for i, p := range panels {
		if strings.EqualFold(strings.TrimSpace(p.Name), want) {
			return i
		}
	}
	return -1
}

func findFakBarIndex(panel HeroSweepPanel) int {
	for i, b := range panel.Bars {
		if strings.HasPrefix(strings.ToLower(strings.TrimSpace(b.Who)), "fak") && b.Subject {
			return i
		}
	}
	for i, b := range panel.Bars {
		if strings.HasPrefix(strings.ToLower(strings.TrimSpace(b.Who)), "fak") {
			return i
		}
	}
	return -1
}

func findSubjectBarIndex(panel HeroSweepPanel) int {
	for i, b := range panel.Bars {
		if b.Subject {
			return i
		}
	}
	return -1
}

// EvaluateRecordBeat determines whether cand beats the prior record materially.
func EvaluateRecordBeat(sweep *HeroSweepFile, cand HeroRecordInput) (*RecordOutcome, error) {
	if sweep == nil {
		return nil, errors.New("hero sweep data is nil")
	}
	panelIdx := findPanelIndex(sweep.Panels, cand.Panel)
	if panelIdx < 0 {
		return nil, fmt.Errorf("panel %q not found in hero sweep", cand.Panel)
	}
	panel := sweep.Panels[panelIdx]
	fakIdx := findFakBarIndex(panel)
	if fakIdx < 0 {
		return nil, fmt.Errorf("panel %q has no fak bar", panel.Name)
	}

	priorVal := panel.Bars[fakIdx].V
	newVal := cand.Value
	minDelta := cand.MinDeltaPct
	if minDelta <= 0 {
		minDelta = DefaultMinDeltaPct
	}

	var beat bool
	var deltaPct float64
	if panel.LowerIsBetter {
		beat = newVal < priorVal
		if priorVal > 0 {
			deltaPct = (priorVal - newVal) / priorVal * 100.0
		}
	} else {
		beat = newVal > priorVal
		if priorVal > 0 {
			deltaPct = (newVal - priorVal) / priorVal * 100.0
		}
	}

	flippedFence := false
	if panel.Fence {
		leaderIdx := findSubjectBarIndex(panel)
		if leaderIdx >= 0 && leaderIdx != fakIdx {
			leaderVal := panel.Bars[leaderIdx].V
			if panel.LowerIsBetter {
				flippedFence = newVal < leaderVal
			} else {
				flippedFence = newVal > leaderVal
			}
		}
	}

	material := beat && (deltaPct >= minDelta || flippedFence)
	outcome := &RecordOutcome{
		Panel:         panel.Name,
		Material:      material,
		Beat:          beat,
		PriorValue:    priorVal,
		NewValue:      newVal,
		DeltaPct:      deltaPct,
		LowerIsBetter: panel.LowerIsBetter,
		FlippedFence:  flippedFence,
	}

	if !beat {
		outcome.SkipReason = "new value does not beat prior record"
	} else if !material {
		outcome.SkipReason = fmt.Sprintf("improvement %.2f%% is below materiality floor %.2f%%", deltaPct, minDelta)
	}

	return outcome, nil
}

func formatBarLabel(oldLabel string, newVal float64) string {
	idx := strings.IndexFunc(oldLabel, func(r rune) bool {
		return (r < '0' || r > '9') && r != ',' && r != '.'
	})
	suffix := ""
	if idx >= 0 {
		suffix = oldLabel[idx:]
	}
	decimals := 0
	if dotIdx := strings.IndexByte(oldLabel, '.'); dotIdx >= 0 {
		end := dotIdx + 1
		for end < len(oldLabel) && oldLabel[end] >= '0' && oldLabel[end] <= '9' {
			end++
		}
		decimals = end - (dotIdx + 1)
	}
	formattedNum := formatFloatWithCommas(newVal, decimals)
	return formattedNum + suffix
}

func formatFloatWithCommas(v float64, decimals int) string {
	s := strconv.FormatFloat(v, 'f', decimals, 64)
	parts := strings.SplitN(s, ".", 2)
	intPart := parts[0]
	neg := false
	if strings.HasPrefix(intPart, "-") {
		neg = true
		intPart = intPart[1:]
	}
	if len(intPart) > 3 {
		var buf strings.Builder
		rem := len(intPart) % 3
		if rem > 0 {
			buf.WriteString(intPart[:rem])
			if len(intPart) > rem {
				buf.WriteByte(',')
			}
		}
		for i := rem; i < len(intPart); i += 3 {
			if i > rem {
				buf.WriteByte(',')
			}
			buf.WriteString(intPart[i : i+3])
		}
		intPart = buf.String()
	}
	if neg {
		intPart = "-" + intPart
	}
	if len(parts) == 2 {
		return intPart + "." + parts[1]
	}
	return intPart
}

func appendReceiptIfMissing(receipts, receiptPath string) string {
	norm := filepath.ToSlash(strings.TrimSpace(receiptPath))
	if strings.HasSuffix(norm, "/RESULTS.md") {
		norm = strings.TrimSuffix(norm, "/RESULTS.md")
	}
	if norm == "" {
		return receipts
	}
	for _, part := range strings.Split(receipts, ";") {
		if strings.TrimSpace(part) == norm {
			return receipts
		}
	}
	if strings.TrimSpace(receipts) == "" {
		return norm
	}
	return norm + "; " + receipts
}

func marshalJSONPretty(v any) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// ApplyRecordUpdate evaluates a record candidate and atomically updates hero sweep data,
// README.md, and optionally hardware-latest.json + docs/benchmarks/README.md.
func ApplyRecordUpdate(repoRoot string, cand HeroRecordInput, dryRun bool) (*RecordOutcome, error) {
	if err := ValidateHeroRecordInput(cand, repoRoot); err != nil {
		return nil, err
	}

	dateStr := cand.Date
	if dateStr == "" {
		dateStr = time.Now().Format("2006-01-02")
	}

	sweepPath := filepath.Join(repoRoot, DefaultHeroSweepDataPath)
	sweepBytes, err := os.ReadFile(sweepPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read %s: %w", DefaultHeroSweepDataPath, err)
	}
	var sweep HeroSweepFile
	if err := json.Unmarshal(sweepBytes, &sweep); err != nil {
		return nil, fmt.Errorf("failed to parse %s: %w", DefaultHeroSweepDataPath, err)
	}

	outcome, err := EvaluateRecordBeat(&sweep, cand)
	if err != nil {
		return nil, err
	}
	if !outcome.Material && !cand.AllowNonMaterial {
		outcome.Skipped = true
		return outcome, nil
	}

	panelIdx := findPanelIndex(sweep.Panels, cand.Panel)
	panel := &sweep.Panels[panelIdx]
	fakIdx := findFakBarIndex(*panel)
	oldLabel := panel.Bars[fakIdx].Label
	newLabel := cand.Label
	if strings.TrimSpace(newLabel) == "" {
		newLabel = formatBarLabel(oldLabel, cand.Value)
	}

	panel.Bars[fakIdx].V = cand.Value
	panel.Bars[fakIdx].Label = newLabel
	panel.Commit = cand.Commit
	if cand.Foot != "" {
		panel.Foot = cand.Foot
	}
	if cand.Callout != "" {
		panel.Callout = cand.Callout
	}
	if !panel.LowerIsBetter && panel.YMax > 0 && cand.Value > 0.90*panel.YMax {
		panel.YMax = math.Ceil(cand.Value * 1.15)
	}
	if outcome.FlippedFence {
		leaderIdx := findSubjectBarIndex(*panel)
		if leaderIdx >= 0 {
			panel.Bars[leaderIdx].Subject = false
		}
		panel.Bars[fakIdx].Subject = true
		panel.Fence = false
		if panel.Pillar == "FENCE" {
			panel.Pillar = "SERVING"
		}
	}

	sweep.Meta.AsOf = dateStr
	sweep.Meta.Receipts = appendReceiptIfMissing(sweep.Meta.Receipts, cand.ReceiptPath)
	if reISODateTrailing.MatchString(sweep.Kicker) {
		sweep.Kicker = reISODateTrailing.ReplaceAllString(sweep.Kicker, dateStr)
	}
	if cand.Desc != "" {
		sweep.Desc = cand.Desc
	} else if oldLabel != "" && strings.Contains(sweep.Desc, oldLabel) {
		sweep.Desc = strings.Replace(sweep.Desc, oldLabel, newLabel, 1)
	}

	newSweepBytes, err := marshalJSONPretty(&sweep)
	if err != nil {
		return nil, fmt.Errorf("failed to encode %s: %w", DefaultHeroSweepDataPath, err)
	}

	readmePath := filepath.Join(repoRoot, DefaultReadmePath)
	readmeOrig, err := os.ReadFile(readmePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read %s: %w", DefaultReadmePath, err)
	}
	readmeUpdated := string(readmeOrig)
	if cand.ReadmeOld != "" && cand.ReadmeNew != "" {
		if !strings.Contains(readmeUpdated, cand.ReadmeOld) {
			return nil, fmt.Errorf("readme_old text not found in %s: %q", DefaultReadmePath, cand.ReadmeOld)
		}
		readmeUpdated = strings.Replace(readmeUpdated, cand.ReadmeOld, cand.ReadmeNew, 1)
	}
	if reVerifiedMarker.MatchString(readmeUpdated) {
		readmeUpdated = reVerifiedMarker.ReplaceAllString(readmeUpdated, "${1}"+dateStr)
	}

	type pendingWrite struct {
		relPath string
		absPath string
		orig    []byte
		next    []byte
	}
	writes := []pendingWrite{
		{DefaultHeroSweepDataPath, sweepPath, sweepBytes, newSweepBytes},
		{DefaultReadmePath, readmePath, readmeOrig, []byte(readmeUpdated)},
	}

	if cand.Platform != "" && cand.HardwareRow != "" {
		hwJSONPath := filepath.Join(repoRoot, DefaultHardwareJSONPath)
		hwJSONOrig, err := os.ReadFile(hwJSONPath)
		if err != nil {
			return nil, fmt.Errorf("failed to read %s: %w", DefaultHardwareJSONPath, err)
		}
		var manifest HardwareLatestManifest
		if err := json.Unmarshal(hwJSONOrig, &manifest); err != nil {
			return nil, fmt.Errorf("failed to parse %s: %w", DefaultHardwareJSONPath, err)
		}
		if manifest.Platforms == nil {
			manifest.Platforms = make(map[string]*HardwarePlatformEntry)
		}
		manifest.AsOf = dateStr
		manifest.Platforms[cand.Platform] = &HardwarePlatformEntry{
			Observed: dateStr,
			Detail:   filepath.ToSlash(strings.TrimSpace(cand.ReceiptPath)),
			Row:      strings.TrimSpace(cand.HardwareRow),
		}
		newHWJSON, err := marshalJSONPretty(&manifest)
		if err != nil {
			return nil, fmt.Errorf("failed to encode %s: %w", DefaultHardwareJSONPath, err)
		}

		hwReadmePath := filepath.Join(repoRoot, DefaultHardwareReadmePath)
		hwReadmeOrig, err := os.ReadFile(hwReadmePath)
		if err != nil {
			return nil, fmt.Errorf("failed to read %s: %w", DefaultHardwareReadmePath, err)
		}
		hwReadmeUpdated, _, err := PreviewNext(string(hwReadmeOrig), []*CandidateFragment{
			{
				Topic:            "record-" + cand.Panel,
				TargetSection:    TargetHardwareTable,
				CandidateContent: strings.TrimSpace(cand.HardwareRow),
				Date:             dateStr,
			},
		})
		if err != nil {
			return nil, fmt.Errorf("failed to update %s: %w", DefaultHardwareReadmePath, err)
		}

		writes = append(writes,
			pendingWrite{DefaultHardwareJSONPath, hwJSONPath, hwJSONOrig, newHWJSON},
			pendingWrite{DefaultHardwareReadmePath, hwReadmePath, hwReadmeOrig, []byte(hwReadmeUpdated)},
		)
	}

	for _, w := range writes {
		outcome.UpdatedFiles = append(outcome.UpdatedFiles, w.relPath)
	}
	outcome.NextActions = []string{
		"python tools/hero_sweep_gen.py",
		"bash visuals/render-hero-pngs.sh",
		"python tools/readme_freshness_audit.py",
	}

	if dryRun {
		return outcome, nil
	}

	written := 0
	for _, w := range writes {
		if err := os.WriteFile(w.absPath, w.next, 0644); err != nil {
			for i := written - 1; i >= 0; i-- {
				_ = os.WriteFile(writes[i].absPath, writes[i].orig, 0644)
			}
			return nil, fmt.Errorf("failed to write %s: %w", w.relPath, err)
		}
		written++
	}

	return outcome, nil
}
