package model

import (
	"reflect"
	"testing"
)

func TestResidualHookOffByDefault(t *testing.T) {
	m := &Model{}
	if m.Cfg.EnableResidualHook {
		t.Fatal("residual hook gate is on in zero-value config")
	}
	fired := 0
	m.SetResidualHook(func(int, []float32) { fired++ })
	if !m.ResidualHookSet() {
		t.Fatal("ResidualHookSet=false after installation")
	}
	x := []float32{1, 2}
	composeBlockAtLayer(0, PreNorm, x, identityNorm(), identityNorm(), 1e-5, m.Cfg, zeroSublayer, zeroSublayer)
	if fired != 0 || !reflect.DeepEqual(x, []float32{1, 2}) {
		t.Fatalf("default-off hook changed execution: fired=%d x=%v", fired, x)
	}
}

func TestResidualHook(t *testing.T) {
	for _, topology := range []BlockTopology{PreNorm, PostNorm, SandwichNorm, ParallelResidual} {
		t.Run(topology.String(), func(t *testing.T) {
			m := &Model{Cfg: Config{EnableResidualHook: true}}
			var layers []int
			m.SetResidualHook(func(layer int, _ []float32) { layers = append(layers, layer) })
			x := []float32{1, 2}
			want := append([]float32(nil), x...)
			for layer := 0; layer < 3; layer++ {
				composeBlockAtLayer(layer, topology, x, identityNorm(), identityNorm(), 1e-5, m.Cfg, zeroSublayer, zeroSublayer)
			}
			if !reflect.DeepEqual(layers, []int{0, 1, 2}) {
				t.Fatalf("hook layers=%v, want [0 1 2]", layers)
			}
			if !reflect.DeepEqual(x, want) {
				t.Fatalf("identity hook changed residual/logit input: got %v want %v", x, want)
			}
		})
	}
}

func zeroSublayer(in []float32) []float32 { return make([]float32, len(in)) }

func identityNorm() normWeights {
	ones := []float32{1, 1}
	return normWeights{pre: ones, post: ones}
}
