package bench

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// ActivationPatchCase is one deterministic clean/corrupt activation pair and its predicted
// positive target direction. Layer is the causal L3 site; ControlLayer must differ.
type ActivationPatchCase struct {
	ID           string    `json:"id"`
	Layer        int       `json:"layer"`
	ControlLayer int       `json:"control_layer"`
	Clean        []float32 `json:"clean"`
	Corrupt      []float32 `json:"corrupt"`
	Target       []float32 `json:"target"`
	MinEffect    float64   `json:"min_effect"`
	MaxControl   float64   `json:"max_control"`
}

// ActivationPatchRow is the auditable clean/corrupt/patched/control output record.
type ActivationPatchRow struct {
	ID              string  `json:"id"`
	CleanScore      float64 `json:"clean_score"`
	CorruptScore    float64 `json:"corrupt_score"`
	PatchedScore    float64 `json:"patched_score"`
	ControlScore    float64 `json:"control_score"`
	PatchEffect     float64 `json:"patch_effect"`
	ControlEffect   float64 `json:"control_effect"`
	SignCorrect     bool    `json:"sign_correct"`
	AboveThreshold  bool    `json:"above_threshold"`
	ControlSpecific bool    `json:"control_specific"`
}

// RunActivationPatchL3 captures each clean L3 activation, injects it into the matched
// corrupt activation, and compares the predicted-target shift with a random-site control.
func RunActivationPatchL3(cases []ActivationPatchCase) ([]ActivationPatchRow, error) {
	rows := make([]ActivationPatchRow, 0, len(cases))
	for _, c := range cases {
		if c.ID == "" || c.Layer < 0 || c.ControlLayer < 0 || c.Layer == c.ControlLayer {
			return nil, fmt.Errorf("invalid activation patch case %q", c.ID)
		}
		if len(c.Clean) == 0 || len(c.Clean) != len(c.Corrupt) || len(c.Clean) != len(c.Target) {
			return nil, fmt.Errorf("case %s: activation shapes differ", c.ID)
		}
		patch, err := model.NewActivationPatch(c.Layer)
		if err != nil {
			return nil, err
		}
		patch.Capture(c.Layer, c.Clean)
		patched := append([]float32(nil), c.Corrupt...)
		if changed, err := patch.Inject(c.Layer, patched); err != nil || !changed {
			return nil, fmt.Errorf("case %s: L3 patch changed=%v: %w", c.ID, changed, err)
		}
		control := append([]float32(nil), c.Corrupt...)
		if changed, err := patch.Inject(c.ControlLayer, control); err != nil || changed {
			return nil, fmt.Errorf("case %s: control patch changed=%v: %w", c.ID, changed, err)
		}
		cleanScore := cosine(c.Clean, c.Target)
		corruptScore := cosine(c.Corrupt, c.Target)
		patchedScore := cosine(patched, c.Target)
		controlScore := cosine(control, c.Target)
		patchEffect := patchedScore - corruptScore
		controlEffect := controlScore - corruptScore
		row := ActivationPatchRow{ID: c.ID, CleanScore: cleanScore, CorruptScore: corruptScore, PatchedScore: patchedScore, ControlScore: controlScore, PatchEffect: patchEffect, ControlEffect: controlEffect, SignCorrect: patchEffect > 0, AboveThreshold: patchEffect >= c.MinEffect, ControlSpecific: math.Abs(controlEffect) <= c.MaxControl}
		if !row.SignCorrect || !row.AboveThreshold || !row.ControlSpecific {
			return nil, fmt.Errorf("case %s: causal gate failed: effect=%.6f control=%.6f", c.ID, patchEffect, controlEffect)
		}
		rows = append(rows, row)
	}
	if len(rows) == 0 {
		return nil, fmt.Errorf("empty activation patch set")
	}
	return rows, nil
}

// ActivationPatchJSONL renders stable one-record-per-case witness output.
func ActivationPatchJSONL(rows []ActivationPatchRow) ([]byte, error) {
	var b strings.Builder
	enc := json.NewEncoder(&b)
	enc.SetEscapeHTML(false)
	for _, row := range rows {
		if err := enc.Encode(row); err != nil {
			return nil, err
		}
	}
	return []byte(b.String()), nil
}

func cosine(a, b []float32) float64 {
	var dot, aa, bb float64
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		dot += x * y
		aa += x * x
		bb += y * y
	}
	if aa == 0 || bb == 0 {
		return 0
	}
	return dot / math.Sqrt(aa*bb)
}
