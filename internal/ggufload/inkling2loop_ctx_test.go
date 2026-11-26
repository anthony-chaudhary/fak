package ggufload

import "testing"

func TestInkling2ResidentCtxFromEnv(t *testing.T) {
	cases := []struct {
		name string
		env  map[string]string
		want int
	}{
		{"unset keeps bench default", nil, 4096},
		{"agentic size", map[string]string{"FAK_INKLING2_CTX": "16384"}, 16384},
		{"whitespace tolerated", map[string]string{"FAK_INKLING2_CTX": " 8192 "}, 8192},
		{"garbage falls back", map[string]string{"FAK_INKLING2_CTX": "big"}, 4096},
		{"below floor falls back", map[string]string{"FAK_INKLING2_CTX": "64"}, 4096},
		{"explicit floor is kept", map[string]string{"FAK_INKLING2_CTX": "512"}, 512},
		{"between fallback ceilings", map[string]string{"FAK_INKLING2_CTX": "28672"}, 28672},
		{"64K reaches device admission", map[string]string{"FAK_INKLING2_CTX": "65536"}, 65536},
		{"128K reaches device admission", map[string]string{"FAK_INKLING2_CTX": "131072"}, 131072},
		{"above parse ceiling clamps", map[string]string{"FAK_INKLING2_CTX": "2000000"}, 1 << 20},
		// past the 24576 unknown-free ceiling the request reaches the KV fit check
		{"large request reaches the fit check", map[string]string{"FAK_INKLING2_CTX": "32768"}, 32768},
		{"past the shared-memory score row", map[string]string{"FAK_INKLING2_CTX": "24576"}, 24576},
		{"former shared-memory bound", map[string]string{"FAK_INKLING2_CTX": "23552"}, 23552},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := inkling2ResidentCtxFrom(func(k string) (string, bool) { v, ok := c.env[k]; return v, ok })
			if got != c.want {
				t.Fatalf("got %d want %d", got, c.want)
			}
		})
	}
}

func TestInkling2ResidentCtxMaxPerBackend(t *testing.T) {
	if Inkling2ResidentCtxMaxCUDA != 24576 || Inkling2ResidentCtxMaxMetal != 32768 {
		t.Fatalf("bounds cuda %d metal %d", Inkling2ResidentCtxMaxCUDA, Inkling2ResidentCtxMaxMetal)
	}
	for backend, want := range map[string]int{"metal": 32768, "cuda": 24576, "cpu": 24576, "": 24576} {
		if got := Inkling2ResidentCtxMax(backend); got != want {
			t.Errorf("max(%q) = %d, want %d", backend, got, want)
		}
	}
	if Inkling2ResidentCtxDefault < Inkling2ResidentCtxMin || Inkling2ResidentCtxDefault > Inkling2ResidentCtxMaxCUDA {
		t.Fatalf("default %d outside %d..%d", Inkling2ResidentCtxDefault, Inkling2ResidentCtxMin, Inkling2ResidentCtxMaxCUDA)
	}
}

func TestInkling2ResidentCtxBuildBound(t *testing.T) {
	limit := Inkling2ResidentCtxMax(inkling2ResidentBackend)
	if Inkling2LoopResidentContext != inkling2ResidentCtx || inkling2ResidentCtx > limit || inkling2ResidentCtx < Inkling2ResidentCtxMin {
		t.Fatalf("exported %d, driver %d, %s bound %d", Inkling2LoopResidentContext, inkling2ResidentCtx, inkling2ResidentBackend, limit)
	}
}
