package polymodel

import (
	"errors"
	"reflect"
	"testing"
)

// fak-test:runtime fast est=10ms
func TestPoolAdmissionVictims(t *testing.T) {
	// The optional API assertion also gives the incumbent a runtime missing-
	// capability witness, without changing its production admission algorithm.
	preview := func(t *testing.T, p *Pool, m Model) ([]ModelID, error) {
		t.Helper()
		v, ok := any(p).(interface {
			AdmissionVictims(Model) ([]ModelID, error)
		})
		if !ok {
			t.Fatal("Pool cannot preview exact admission victims without mutation")
		}
		return v.AdmissionVictims(m)
	}
	copyPool := func(p *Pool) *Pool {
		copy := *p
		copy.models = make(map[ModelID]*entry, len(p.models))
		for id, e := range p.models {
			value := *e
			copy.models[id] = &value
		}
		return &copy
	}
	check := func(t *testing.T, p *Pool, m Model, want []ModelID, wantErr error) {
		t.Helper()
		before := copyPool(p)
		original := make(map[ModelID]*entry, len(p.models))
		for id, e := range p.models {
			original[id] = e
		}
		got, err := preview(t, p, m)
		if !errors.Is(err, wantErr) || !reflect.DeepEqual(got, want) {
			t.Fatalf("preview(%+v) = %v, %v; want %v, %v", m, got, err, want, wantErr)
		}
		if !reflect.DeepEqual(p, before) {
			t.Fatalf("preview mutated live pool: before=%+v after=%+v", before, p)
		}
		for id, e := range original {
			if p.models[id] != e {
				t.Fatalf("preview replaced live entry %q", id)
			}
		}
		actual, actualErr := before.Admit(m)
		if !errors.Is(err, actualErr) || !reflect.DeepEqual(got, actual) {
			t.Fatalf("preview %v, %v disagrees with Admit %v, %v", got, err, actual, actualErr)
		}
		// Returned slices are caller-owned, never a live eviction plan.
		if len(got) > 0 {
			got[0] = "caller-modified"
			again, _ := preview(t, p, m)
			if !reflect.DeepEqual(again, actual) {
				t.Fatalf("caller modified future preview: %v", again)
			}
		}
	}
	admit := func(t *testing.T, p *Pool, models ...Model) {
		t.Helper()
		for _, m := range models {
			if _, err := p.Admit(m); err != nil {
				t.Fatal(err)
			}
		}
	}
	t.Run("room existing and refusal", func(t *testing.T) {
		p := NewPool(12)
		admit(t, p, Model{ID: "a", WeightBytes: 4})
		check(t, p, Model{ID: "b", WeightBytes: 8}, nil, nil)
		check(t, p, Model{ID: "b", WeightBytes: -1}, nil, nil)
		check(t, p, Model{ID: "a", WeightBytes: 100, Pinned: true}, nil, nil)
		check(t, p, Model{}, nil, ErrEmptyID)
		check(t, p, Model{ID: "b", WeightBytes: 13}, nil, ErrTooLarge)
		admit(t, p, Model{ID: "pin", WeightBytes: 8, Pinned: true})
		check(t, p, Model{ID: "b", WeightBytes: 8}, nil, ErrPinnedNoRoom)
	})
	t.Run("touch and repin preserve exact LRU", func(t *testing.T) {
		p := NewPool(12)
		admit(t, p, Model{ID: "a", WeightBytes: 4}, Model{ID: "b", WeightBytes: 4}, Model{ID: "c", WeightBytes: 4})
		p.Touch("a")
		p.Evict("b")
		admit(t, p, Model{ID: "b", WeightBytes: 4, Pinned: true})
		check(t, p, Model{ID: "d", WeightBytes: 8}, []ModelID{"c", "a"}, nil)
	})
	t.Run("zero-byte victims and deterministic ties", func(t *testing.T) {
		p := NewPool(8)
		admit(t, p, Model{ID: "zero"}, Model{ID: "b", WeightBytes: 4}, Model{ID: "a", WeightBytes: 4})
		p.models["a"].used = p.models["b"].used
		check(t, p, Model{ID: "new", WeightBytes: 8}, []ModelID{"zero", "a", "b"}, nil)
	})
	t.Run("int64 boundary", func(t *testing.T) {
		const limit = int64(^uint64(0) >> 1)
		p := NewPool(limit)
		admit(t, p, Model{ID: "a", WeightBytes: limit - 1}, Model{ID: "b", WeightBytes: 1})
		check(t, p, Model{ID: "new", WeightBytes: limit}, []ModelID{"a", "b"}, nil)
	})
	t.Run("preview is not reservation", func(t *testing.T) {
		p := NewPool(8)
		admit(t, p, Model{ID: "a", WeightBytes: 4}, Model{ID: "b", WeightBytes: 4})
		m := Model{ID: "new", WeightBytes: 4}
		check(t, p, m, []ModelID{"a"}, nil)
		p.Touch("a")
		check(t, p, m, []ModelID{"b"}, nil)
		if _, err := p.Resize(4); err != nil {
			t.Fatal(err)
		}
		check(t, p, m, []ModelID{"a"}, nil)
	})
}
