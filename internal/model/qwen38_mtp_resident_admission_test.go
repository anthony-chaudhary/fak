package model

import (
	"errors"
	"testing"
)

// fak-test:runtime fast est=200ms lane=default
func TestQwen38MTPResidentQ4KMHeadAdmitsDraftSession(t *testing.T) {
	q4, ref := qwen38MTPQ4KTestModels(t)
	for name := range qwen38MTPQ4KMArtifactMatrixTypes {
		if _, ok := q4.manifest[name]; ok {
			t.Fatalf("fixture keeps %s in the F32 manifest; the resident route must not", name)
		}
	}
	for label, m := range map[string]*Model{"resident Q4_K_M": q4, "F32 reference": ref} {
		mode, err := m.Qwen35MTPMode(false)
		if err != nil || !mode.Eligible || !mode.Enabled {
			t.Fatalf("%s: Qwen35MTPMode = %+v, %v; want eligible and enabled", label, mode, err)
		}
		if err := m.validateQwen35MTPForwardTensors(); err != nil {
			t.Fatalf("%s: forward tensor validation: %v", label, err)
		}
		if mode, err := m.Qwen35MTPMode(true); err != nil || !mode.Eligible || mode.Enabled {
			t.Fatalf("%s: disabled Qwen35MTPMode = %+v, %v; want eligible and not enabled", label, mode, err)
		}
	}
}

// fak-test:runtime fast est=200ms lane=default
func TestQwen38MTPResidentQ4KMIncompleteHeadStaysRefused(t *testing.T) {
	q4, _ := qwen38MTPQ4KTestModels(t)
	delete(q4.q4kw, "mtp.layers.0.self_attn.q_proj.weight")
	mode, err := q4.Qwen35MTPMode(false)
	var incomplete *Qwen35MTPIncompleteError
	if !errors.As(err, &incomplete) || mode.Enabled {
		t.Fatalf("Qwen35MTPMode = %+v, %v; want *Qwen35MTPIncompleteError and not enabled", mode, err)
	}
	if err := q4.validateQwen35MTPForwardTensors(); err == nil {
		t.Fatal("forward tensor validation admitted a resident head missing q_proj")
	}
}
