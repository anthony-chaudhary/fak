package engine

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=100ms lane=default
func TestRecurrentBoundaryReplacementRetainsCapturedOwner(t *testing.T) {
	for _, tokens := range [][]int{{7, 8, 9}, {1, 2}} {
		t.Run("boundary", func(t *testing.T) {
			m := rpHybridModel()
			c := NewRecurrentPrefixCache(1)
			defer c.Clear()
			old := rpSnapshot(t, m)
			if err := c.Store("turn", []int{1, 2, 3}, old); err != nil {
				t.Fatal(err)
			}
			session := &model.Session{M: m, Cache: model.NewKVCache(m.Cfg)}
			session.Prefill(tokens)
			fresh, err := session.PrefixSnapshot()
			if err != nil || fresh == nil {
				t.Fatalf("snapshot: %v", err)
			}
			input := append([]int(nil), tokens...)
			if err := c.storeBoundary("turn", input, fresh); err != nil {
				t.Fatal(err)
			}
			input[0] = 42 // Cache must own its token identity.
			prompt := append(append([]int(nil), tokens...), 4)
			matched, got, hit := c.Lookup("turn", prompt)
			if !hit || matched != len(tokens) || got != fresh {
				t.Fatalf("replacement lost captured owner: %d %p %v", matched, got, hit)
			}
			defer got.Close()
			sess := &model.Session{M: m, Cache: model.NewKVCache(m.Cfg)}
			if err := got.Restore(sess); err != nil {
				t.Fatalf("fresh boundary was closed: %v", err)
			}
			stats := c.Stats()
			if stats.Puts != 2 || stats.Refusals != 1 || stats.Evictions != 1 || stats.Hits != 1 {
				t.Fatalf("accounting changed: %+v", stats)
			}
		})
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestRecurrentBoundaryReplacementKeepsPublicStoreStrict(t *testing.T) {
	m := rpHybridModel()
	c := NewRecurrentPrefixCache(1)
	defer c.Clear()
	old := rpSnapshot(t, m)
	if err := c.Store("turn", []int{1, 2, 3}, old); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		tokens []int
		want   error
	}{{[]int{1, 2}, ErrRecurrentPrefixBackwardRewind}, {[]int{7, 8, 9}, ErrRecurrentPrefixDivergence}} {
		if err := c.Store("turn", tc.tokens, rpSnapshot(t, m)); !errors.Is(err, tc.want) {
			t.Fatalf("strict Store: %v, want %v", err, tc.want)
		}
	}
	_, got, hit := c.Lookup("turn", []int{1, 2, 3, 4})
	if !hit || got != old {
		t.Fatal("public refusal replaced old boundary")
	}
	got.Close()
}
