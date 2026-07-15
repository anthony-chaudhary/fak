package model

import (
	"encoding/binary"
	"math"
	"os"
	"strconv"
	"strings"
)

// DirectionSteer is a default-off residual-stream injection at one layer and
// absolute decode position. Direction must be unit length; Apply adds Alpha*d.
type DirectionSteer struct {
	Layer     int
	Position  int
	Alpha     float32
	Direction []float32
}

func (s DirectionSteer) Armed() bool {
	return s.Alpha != 0 && len(s.Direction) > 0 && s.Layer >= 0 && s.Position >= 0
}

func (s DirectionSteer) Apply(layer, position int, hidden []float32) bool {
	if !s.Armed() || layer != s.Layer || position != s.Position || len(hidden) != len(s.Direction) {
		return false
	}
	for i := range hidden {
		hidden[i] += s.Alpha * s.Direction[i]
	}
	return true
}

func VerbalizableDirection(positive, negative [][]float32) []float32 {
	if len(positive) == 0 || len(negative) == 0 || len(positive[0]) == 0 {
		return nil
	}
	n := len(positive[0])
	d := make([]float32, n)
	for _, row := range positive {
		if len(row) != n {
			return nil
		}
		for i := range d {
			d[i] += row[i] / float32(len(positive))
		}
	}
	for _, row := range negative {
		if len(row) != n {
			return nil
		}
		for i := range d {
			d[i] -= row[i] / float32(len(negative))
		}
	}
	var norm float64
	for _, v := range d {
		norm += float64(v * v)
	}
	if norm == 0 {
		return nil
	}
	scale := float32(1 / math.Sqrt(norm))
	for i := range d {
		d[i] *= scale
	}
	return d
}

func DirectionProjection(hidden, direction []float32) float32 {
	if len(hidden) != len(direction) {
		return 0
	}
	var x float32
	for i := range hidden {
		x += hidden[i] * direction[i]
	}
	return x
}
func DirectionCosine(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
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

func directionSteerFromEnv() DirectionSteer {
	path := strings.TrimSpace(os.Getenv("FAK_HIDDEN_STEER"))
	if path == "" {
		return DirectionSteer{}
	}
	b, err := os.ReadFile(path)
	if err != nil || len(b)%4 != 0 {
		return DirectionSteer{}
	}
	d := make([]float32, len(b)/4)
	for i := range d {
		d[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	layer, e1 := strconv.Atoi(strings.TrimSpace(os.Getenv("FAK_HIDDEN_STEER_LAYER")))
	pos, e2 := strconv.Atoi(strings.TrimSpace(os.Getenv("FAK_HIDDEN_STEER_POS")))
	alpha, e3 := strconv.ParseFloat(strings.TrimSpace(os.Getenv("FAK_HIDDEN_STEER_ALPHA")), 32)
	if e1 != nil || e2 != nil || e3 != nil {
		return DirectionSteer{}
	}
	return DirectionSteer{Layer: layer, Position: pos, Alpha: float32(alpha), Direction: d}
}
