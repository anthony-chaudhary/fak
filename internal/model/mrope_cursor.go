package model

import (
	"errors"
	"fmt"
)

// ErrMRoPEDisabled is the typed refusal emitted when 3-axis M-RoPE is disabled on a vision sequence.
var ErrMRoPEDisabled = errors.New("model: M-RoPE position input required for vision sequences; flattening refused")

// MRoPEAxisPosition records 3D rotary positions across temporal, height, and width axes.
type MRoPEAxisPosition struct {
	Temporal int `json:"temporal"`
	Height   int `json:"height"`
	Width    int `json:"width"`
}

// VisionImageGrid describes the spatial patch layout of an image.
type VisionImageGrid struct {
	GridHeight int `json:"grid_height"`
	GridWidth  int `json:"grid_width"`
}

// MRoPECursor maintains the 3-axis rotary coordinates for a request across prefill and decode.
type MRoPECursor struct {
	Current    MRoPEAxisPosition `json:"current"`
	FlatLength int               `json:"flat_length"`
	DeltaT     int               `json:"delta_t"`
	DeltaH     int               `json:"delta_h"`
	DeltaW     int               `json:"delta_w"`
	Enabled    bool              `json:"enabled"`
	HasVision  bool              `json:"has_vision"`
}

// NewMRoPECursor creates an M-RoPE cursor initialized at coordinate (0, 0, 0).
func NewMRoPECursor(enabled bool) *MRoPECursor {
	return &MRoPECursor{
		Current:    MRoPEAxisPosition{Temporal: 0, Height: 0, Width: 0},
		FlatLength: 0,
		Enabled:    enabled,
		HasVision:  false,
	}
}

// AdvanceText advances positions for a run of text tokens.
func (c *MRoPECursor) AdvanceText(numTokens int) ([]MRoPEAxisPosition, error) {
	if numTokens < 0 {
		return nil, fmt.Errorf("numTokens must be non-negative, got %d", numTokens)
	}
	if numTokens == 0 {
		return nil, nil
	}

	positions := make([]MRoPEAxisPosition, numTokens)
	for i := 0; i < numTokens; i++ {
		positions[i] = c.Current
		c.Current.Temporal++
		c.Current.Height++
		c.Current.Width++
		c.FlatLength++
	}
	c.updateDeltas()
	return positions, nil
}

// AdvanceImage advances positions for a 2D image grid of patch tokens.
// If M-RoPE is disabled, it returns ErrMRoPEDisabled instead of silently flattening.
func (c *MRoPECursor) AdvanceImage(grid VisionImageGrid) ([]MRoPEAxisPosition, error) {
	if !c.Enabled {
		return nil, ErrMRoPEDisabled
	}
	if grid.GridHeight <= 0 || grid.GridWidth <= 0 {
		return nil, fmt.Errorf("invalid image grid dimensions: %dx%d", grid.GridHeight, grid.GridWidth)
	}

	c.HasVision = true
	totalPatches := grid.GridHeight * grid.GridWidth
	positions := make([]MRoPEAxisPosition, totalPatches)

	startT := c.Current.Temporal
	startH := c.Current.Height
	startW := c.Current.Width

	idx := 0
	for r := 0; r < grid.GridHeight; r++ {
		for col := 0; col < grid.GridWidth; col++ {
			positions[idx] = MRoPEAxisPosition{
				Temporal: startT,
				Height:   startH + r,
				Width:    startW + col,
			}
			idx++
		}
	}

	// Post-image cursor: temporal advances by 1 frame; height/width advance by spatial dimensions
	c.Current.Temporal = startT + 1
	c.Current.Height = startH + grid.GridHeight
	c.Current.Width = startW + grid.GridWidth
	c.FlatLength += totalPatches

	c.updateDeltas()
	return positions, nil
}

// DecodeStep returns the 3-axis position for the next single decode row and advances the cursor.
func (c *MRoPECursor) DecodeStep() (MRoPEAxisPosition, error) {
	if c.HasVision && !c.Enabled {
		return MRoPEAxisPosition{}, ErrMRoPEDisabled
	}

	pos := c.Current
	c.Current.Temporal++
	c.Current.Height++
	c.Current.Width++
	c.FlatLength++
	c.updateDeltas()

	return pos, nil
}

func (c *MRoPECursor) updateDeltas() {
	c.DeltaT = c.Current.Temporal - c.FlatLength
	c.DeltaH = c.Current.Height - c.FlatLength
	c.DeltaW = c.Current.Width - c.FlatLength
}
