package model

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
)

// DirectionContrast is one matched positive/negative residual pair captured at
// the same layer and site. FitDirectionReader averages each side before fitting.
type DirectionContrast struct {
	Positive []float32
	Negative []float32
}

// DirectionReader is the model-independent read/write coordinate consumed by a
// residual-state operator. Implementations expose fit metadata, project a state,
// and steer only at the layer where the direction was fitted.
type DirectionReader interface {
	Name() string
	Layer() int
	Scale() float32
	Read(hidden []float32) (float32, error)
	Steer(layer int, alpha float32, hidden []float32) error
}

// FittedDirection is a named, reloadable unit direction. Scale is the L2 norm
// of the unnormalized difference-of-means and records the fit's natural size;
// Vector is normalized so Read and Steer use coefficient units consistently.
type FittedDirection struct {
	Schema   string    `json:"schema"`
	Concept  string    `json:"name"`
	AtLayer  int       `json:"layer"`
	FitScale float32   `json:"scale"`
	Vector   []float32 `json:"direction"`
}

const directionReaderSchema = "fak.direction-reader.v1"

var (
	ErrDirectionName  = errors.New("model: direction name is empty")
	ErrDirectionLayer = errors.New("model: direction layer is invalid")
	ErrDirectionFit   = errors.New("model: direction contrast set is invalid")
	ErrDirectionWidth = errors.New("model: hidden width does not match direction")
)

// FitDirectionReader fits a named unit direction from matched contrast pairs.
// It is deterministic for a fixed ordered fixture set and performs no random fitting.
func FitDirectionReader(name string, layer int, pairs []DirectionContrast) (*FittedDirection, error) {
	name = strings.TrimSpace(name)
	if name == "" {
		return nil, ErrDirectionName
	}
	if layer < 0 {
		return nil, ErrDirectionLayer
	}
	if len(pairs) == 0 || len(pairs[0].Positive) == 0 {
		return nil, ErrDirectionFit
	}
	width := len(pairs[0].Positive)
	positive := make([][]float32, len(pairs))
	negative := make([][]float32, len(pairs))
	for i, pair := range pairs {
		if len(pair.Positive) != width || len(pair.Negative) != width {
			return nil, ErrDirectionFit
		}
		positive[i], negative[i] = pair.Positive, pair.Negative
	}
	unit := VerbalizableDirection(positive, negative)
	if len(unit) == 0 {
		return nil, ErrDirectionFit
	}
	var rawNorm float64
	for j := 0; j < width; j++ {
		var delta float32
		for i := range pairs {
			delta += (pairs[i].Positive[j] - pairs[i].Negative[j]) / float32(len(pairs))
		}
		rawNorm += float64(delta * delta)
	}
	fitted := &FittedDirection{
		Schema:   directionReaderSchema,
		Concept:  name,
		AtLayer:  layer,
		FitScale: float32(math.Sqrt(rawNorm)),
		Vector:   append([]float32(nil), unit...),
	}
	if err := validateFittedDirection(fitted); err != nil {
		return nil, err
	}
	return fitted, nil
}

func (d *FittedDirection) Name() string {
	if d == nil {
		return ""
	}
	return d.Concept
}
func (d *FittedDirection) Layer() int {
	if d == nil {
		return -1
	}
	return d.AtLayer
}
func (d *FittedDirection) Scale() float32 {
	if d == nil {
		return 0
	}
	return d.FitScale
}

func (d *FittedDirection) Read(hidden []float32) (float32, error) {
	if d == nil || len(d.Vector) == 0 || len(hidden) != len(d.Vector) {
		return 0, ErrDirectionWidth
	}
	return DirectionProjection(hidden, d.Vector), nil
}

func (d *FittedDirection) Steer(layer int, alpha float32, hidden []float32) error {
	if d == nil || layer != d.AtLayer {
		return fmt.Errorf("%w: got %d want %d", ErrDirectionLayer, layer, d.Layer())
	}
	if len(hidden) != len(d.Vector) {
		return ErrDirectionWidth
	}
	for i := range hidden {
		hidden[i] += alpha * d.Vector[i]
	}
	return nil
}

// MarshalDirectionReader emits the stable public representation consumed by
// LoadDirectionReader. The concrete type remains hidden behind the Reader seam.
func MarshalDirectionReader(reader DirectionReader) ([]byte, error) {
	d, ok := reader.(*FittedDirection)
	if !ok || validateFittedDirection(d) != nil {
		return nil, ErrDirectionFit
	}
	return json.Marshal(d)
}

// LoadDirectionReader validates and reloads a serialized named direction.
func LoadDirectionReader(data []byte) (DirectionReader, error) {
	var d FittedDirection
	if err := json.Unmarshal(data, &d); err != nil {
		return nil, fmt.Errorf("model: decode direction reader: %w", err)
	}
	if err := validateFittedDirection(&d); err != nil {
		return nil, err
	}
	return &d, nil
}

func validateFittedDirection(d *FittedDirection) error {
	if d == nil || d.Schema != directionReaderSchema || strings.TrimSpace(d.Concept) == "" || d.AtLayer < 0 || d.FitScale <= 0 || math.IsNaN(float64(d.FitScale)) || math.IsInf(float64(d.FitScale), 0) || len(d.Vector) == 0 {
		return ErrDirectionFit
	}
	var norm float64
	for _, value := range d.Vector {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return ErrDirectionFit
		}
		norm += float64(value * value)
	}
	if math.Abs(math.Sqrt(norm)-1) > 1e-4 {
		return ErrDirectionFit
	}
	return nil
}
