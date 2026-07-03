package procguard

import (
	"reflect"
	"testing"
)

func TestEnvSlice(t *testing.T) {
	t.Run("nil map yields empty non-nil slice", func(t *testing.T) {
		got := EnvSlice(nil)
		if got == nil {
			t.Fatalf("EnvSlice(nil) = nil, want empty non-nil slice")
		}
		if len(got) != 0 {
			t.Fatalf("EnvSlice(nil) = %v, want empty", got)
		}
	})

	t.Run("empty map yields empty non-nil slice", func(t *testing.T) {
		got := EnvSlice(map[string]string{})
		if got == nil || len(got) != 0 {
			t.Fatalf("EnvSlice(empty) = %v, want empty non-nil slice", got)
		}
	})

	t.Run("entries are KEY=VALUE and sorted", func(t *testing.T) {
		env := map[string]string{
			"PATH": "/usr/bin",
			"A":    "1",
			"HOME": "/root",
			"Z":    "last",
		}
		want := []string{"A=1", "HOME=/root", "PATH=/usr/bin", "Z=last"}
		got := EnvSlice(env)
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("EnvSlice = %v, want %v", got, want)
		}
	})

	t.Run("order is deterministic across calls", func(t *testing.T) {
		env := map[string]string{"B": "2", "C": "3", "A": "1", "D": "4", "E": "5"}
		first := EnvSlice(env)
		for i := 0; i < 20; i++ {
			if got := EnvSlice(env); !reflect.DeepEqual(got, first) {
				t.Fatalf("iteration %d: EnvSlice = %v, want stable %v", i, got, first)
			}
		}
	})

	t.Run("empty value keeps trailing equals form", func(t *testing.T) {
		got := EnvSlice(map[string]string{"EMPTY": ""})
		want := []string{"EMPTY="}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("EnvSlice = %v, want %v", got, want)
		}
	})
}
