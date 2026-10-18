package model

import "testing"

// fak-test:runtime fast est=5ms lane=default
func TestV41LogitScaleBindingReplacementIsCopyLocal(t *testing.T) {
	calls := 0
	ordinary := func(row []float32, factor float32) ([]float32, error) {
		calls++
		return []float32{row[0] * factor}, nil
	}
	owner := &Session{}
	original := &v41LogitScaleBinding{ordinary: ordinary, owner: owner}
	state := v41ForwardState{logitScaleBinding: original}
	copied := state
	state.setLogitScale(state.logitScale())
	if state.logitScaleBinding == original || state.logitScaleBinding.owner != nil || copied.logitScaleBinding != original || copied.logitScaleBinding.owner != owner {
		t.Fatal("replacement inherited authority or mutated copied binding")
	}
	got, err := state.logitScale()([]float32{3}, 2)
	if err != nil || len(got) != 1 || got[0] != 6 || calls != 1 {
		t.Fatal("ordinary replacement changed callback")
	}
	state.setLogitScale(nil)
	if state.logitScale() != nil || copied.logitScale() == nil {
		t.Fatal("nil replacement mutated copied binding")
	}
	var absent *v41ForwardState
	if absent.logitScale() != nil {
		t.Fatal("nil state acquired callback")
	}
}
