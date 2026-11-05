package main

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelengine"
)

// fak-test:runtime fast est=2s lane=default
func TestServePDRoles(t *testing.T) {
	m := fakmodel.NewSynthetic(modelengine.SyntheticConfig())

	// drainRequest drains Tokens() to channel close, then checks Result().
	drainRequest := func(t *testing.T, r abi.EngineRequest) {
		t.Helper()
		for range r.Tokens() {
		}
		if _, err := r.Result(); err != nil {
			t.Fatalf("Result() = %v, want nil", err)
		}
	}

	pdRequest := func() modelengine.NativePDRequest {
		return modelengine.NativePDRequest{
			Call:             &abi.ToolCall{Tool: "tool", Args: abi.Ref{Kind: abi.RefInline, Inline: []byte("{}")}},
			Prompt:           []int{3, 7, 11, 5},
			Taint:            abi.TaintTrusted,
			Scope:            abi.ScopeFleet,
			AdmissionVerdict: cachemeta.AdmissionAllow,
		}
	}

	t.Run("flag-surface", func(t *testing.T) {
		fs, sf := newServeFlagSet()
		if err := fs.Parse([]string{"--role", "prefill"}); err != nil {
			t.Fatalf("Parse(--role prefill): %v", err)
		}
		if *sf.pdRole != "prefill" {
			t.Fatalf("*sf.pdRole = %q after --role prefill, want %q", *sf.pdRole, "prefill")
		}
		if err := fs.Parse([]string{"--role", "decode"}); err != nil {
			t.Fatalf("Parse(--role decode): %v", err)
		}
		if *sf.pdRole != "decode" {
			t.Fatalf("*sf.pdRole = %q after --role decode, want %q", *sf.pdRole, "decode")
		}
	})

	t.Run("resolver-closed-set", func(t *testing.T) {
		role, err := resolveServePDRole("")
		if err != nil || role != modelengine.NativePDRole("") {
			t.Fatalf("resolveServePDRole(\"\") = (%q, %v), want (%q, nil)", role, err, modelengine.NativePDRole(""))
		}
		role, err = resolveServePDRole("prefill")
		if err != nil || role != modelengine.NativePDRolePrefill {
			t.Fatalf("resolveServePDRole(\"prefill\") = (%q, %v), want (%q, nil)", role, err, modelengine.NativePDRolePrefill)
		}
		role, err = resolveServePDRole("decode")
		if err != nil || role != modelengine.NativePDRoleDecode {
			t.Fatalf("resolveServePDRole(\"decode\") = (%q, %v), want (%q, nil)", role, err, modelengine.NativePDRoleDecode)
		}
		role, err = resolveServePDRole(" Decode ")
		if err != nil || role != modelengine.NativePDRoleDecode {
			t.Fatalf("resolveServePDRole(\" Decode \") = (%q, %v), want (%q, nil) after normalization", role, err, modelengine.NativePDRoleDecode)
		}
		_, err = resolveServePDRole("bogus")
		if !errors.Is(err, ErrServeRoleUnknown) {
			t.Fatalf("resolveServePDRole(\"bogus\") err = %v, want errors.Is(_, ErrServeRoleUnknown)", err)
		}
	})

	t.Run("prefill-role-host", func(t *testing.T) {
		host, err := newServePDRoleHost(modelengine.NativePDRolePrefill, m, 2)
		if err != nil {
			t.Fatalf("newServePDRoleHost(prefill): %v", err)
		}
		defer host.Close()

		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		admission, err := host.AdmitPrefill(ctx, pdRequest())
		if err != nil {
			t.Fatalf("AdmitPrefill: %v", err)
		}
		if admission.PrefillWorker == "" || admission.DecodeWorker == "" || admission.PrefillWorker == admission.DecodeWorker {
			t.Fatalf("workers not split: prefill=%q decode=%q", admission.PrefillWorker, admission.DecodeWorker)
		}
		drainRequest(t, admission.Request)

		if got := host.Cluster().PrefillWorker().Stats().PrefillRequests; got != 1 {
			t.Fatalf("PrefillWorker().Stats().PrefillRequests = %d, want 1", got)
		}
		decode, ok := host.Cluster().DecodeWorker(admission.DecodeWorker)
		if !ok {
			t.Fatalf("DecodeWorker(%q) not found", admission.DecodeWorker)
		}
		stats := decode.Stats()
		if stats.DecodeImports != 1 || stats.RePrefillOnDecode != 0 {
			t.Fatalf("decode stats = %+v, want DecodeImports=1 and RePrefillOnDecode=0", stats)
		}

		sess := m.NewSession()
		prompt := []int{1, 2, 3}
		logits := sess.Prefill(prompt)
		if _, err := host.AdmitImported(ctx, nil, modelengine.ImportedSequence{Session: sess, Logits: logits, Prompt: prompt, Tool: "tool"}); !errors.Is(err, ErrServeRoleMismatch) {
			t.Fatalf("AdmitImported on prefill host err = %v, want errors.Is(_, ErrServeRoleMismatch)", err)
		}
	})

	t.Run("decode-role-host", func(t *testing.T) {
		host, err := newServePDRoleHost(modelengine.NativePDRoleDecode, m, 2)
		if err != nil {
			t.Fatalf("newServePDRoleHost(decode): %v", err)
		}
		defer host.Close()

		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		sess := m.NewSession()
		prompt := []int{1, 2, 3}
		logits := sess.Prefill(prompt)
		req, err := host.AdmitImported(ctx, nil, modelengine.ImportedSequence{Session: sess, Logits: logits, Prompt: prompt, Tool: "tool"})
		if err != nil {
			t.Fatalf("AdmitImported: %v", err)
		}
		if req == nil {
			t.Fatal("AdmitImported returned nil EngineRequest")
		}
		drainRequest(t, req)

		if _, err := host.AdmitPrefill(ctx, pdRequest()); !errors.Is(err, ErrServeRoleMismatch) {
			t.Fatalf("AdmitPrefill on decode host err = %v, want errors.Is(_, ErrServeRoleMismatch)", err)
		}
	})

	t.Run("constructor-guards", func(t *testing.T) {
		if _, err := newServePDRoleHost(modelengine.NativePDRole("bogus"), m, 1); !errors.Is(err, ErrServeRoleUnknown) {
			t.Fatalf("newServePDRoleHost(bogus) err = %v, want errors.Is(_, ErrServeRoleUnknown)", err)
		}
		if _, err := newServePDRoleHost(modelengine.NativePDRolePrefill, nil, 1); !errors.Is(err, ErrServeRoleRequiresModel) {
			t.Fatalf("newServePDRoleHost(prefill, nil) err = %v, want errors.Is(_, ErrServeRoleRequiresModel)", err)
		}
	})

	t.Run("close-idempotent", func(t *testing.T) {
		host, err := newServePDRoleHost(modelengine.NativePDRolePrefill, m, 1)
		if err != nil {
			t.Fatalf("newServePDRoleHost(prefill): %v", err)
		}
		host.Close()
		host.Close()
	})
}
