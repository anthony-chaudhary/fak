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
		{"above ceiling clamps", map[string]string{"FAK_INKLING2_CTX": "1000000"}, 23552},
		{"unproven 32768 clamps", map[string]string{"FAK_INKLING2_CTX": "32768"}, 23552},
		{"largest prefill-safe size", map[string]string{"FAK_INKLING2_CTX": "23552"}, 23552},
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
