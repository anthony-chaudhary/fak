package model

import (
	"encoding/json"
	"strings"
	"testing"
)

func qwen35StateIdentityFixture(t *testing.T, authority string) Qwen35MetalStateIdentityReceipt {
	t.Helper()
	ids := make([]int, 32)
	for i := range ids {
		ids[i] = i*17 + 3
	}
	accounting := qwen35MetalStateIdentityAccounting{}
	if authority == Qwen35MetalStateAuthoritySequence {
		accounting = qwen35MetalStateIdentityAccounting{
			GDNSnapshotOps: 1, GDNSeedOps: 1,
			GDNStateD2HBytes: 24, GDNStateH2DBytes: 24,
		}
	}
	receipt, err := buildQwen35MetalStateIdentityReceipt(
		strings.Repeat("a", 64), ids, authority,
		[]qwen35MetalStateDigestSource{
			{Layer: 0, Role: Qwen35MetalStateRoleKRaw, Values: []float32{1, 2}},
			{Layer: 0, Role: Qwen35MetalStateRoleKPost, Values: []float32{3, 4}},
			{Layer: 0, Role: Qwen35MetalStateRoleV, Values: []float32{5, 6}},
			{Layer: 1, Role: Qwen35MetalStateRoleGDNConv, Values: []float32{7, 8}},
			{Layer: 1, Role: Qwen35MetalStateRoleGDNRecurrent, Values: []float32{9, 10, 11, 12}},
		},
		accounting,
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateQwen35MetalStateIdentityReceipt(receipt); err != nil {
		t.Fatalf("fixture validation: %v", err)
	}
	return receipt
}

func TestQwen35MetalStateIdentityMutationMatrixFailsClosed(t *testing.T) {
	base := qwen35StateIdentityFixture(t, Qwen35MetalStateAuthoritySequence)
	cases := map[string]func(*Qwen35MetalStateIdentityReceipt){
		"owner":         func(r *Qwen35MetalStateIdentityReceipt) { r.OwnerGeneration = strings.Repeat("b", 64) },
		"token-lineage": func(r *Qwen35MetalStateIdentityReceipt) { r.TokenLineageSHA256 = strings.Repeat("b", 64) },
		"layer":         func(r *Qwen35MetalStateIdentityReceipt) { r.States[0].Layer++ },
		"role":          func(r *Qwen35MetalStateIdentityReceipt) { r.States[0].Role = Qwen35MetalStateRoleV },
		"length":        func(r *Qwen35MetalStateIdentityReceipt) { r.States[0].Elements++ },
		"digest":        func(r *Qwen35MetalStateIdentityReceipt) { r.States[0].SHA256 = strings.Repeat("b", 64) },
		"binding":       func(r *Qwen35MetalStateIdentityReceipt) { r.BindingSHA256 = strings.Repeat("b", 64) },
		"snapshot-ops":  func(r *Qwen35MetalStateIdentityReceipt) { r.GDNSnapshotOps++ },
		"seed-bytes":    func(r *Qwen35MetalStateIdentityReceipt) { r.GDNStateH2DBytes++ },
		"digest-bytes":  func(r *Qwen35MetalStateIdentityReceipt) { r.DigestInputBytes++ },
		"missing":       func(r *Qwen35MetalStateIdentityReceipt) { r.States = r.States[:len(r.States)-1] },
		"duplicate":     func(r *Qwen35MetalStateIdentityReceipt) { r.States[1] = r.States[0] },
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			got := cloneQwen35MetalStateIdentityReceipt(base)
			mutate(&got)
			if err := ValidateQwen35MetalStateIdentityReceipt(got); err == nil {
				t.Fatalf("mutation %q validated: %+v", name, got)
			}
		})
	}
}

func TestQwen35MetalStateIdentityDigestDetectsWithinArmMutation(t *testing.T) {
	before := qwen35StateIdentityFixture(t, Qwen35MetalStateAuthorityControl)
	ids := make([]int, 32)
	for i := range ids {
		ids[i] = i*17 + 3
	}
	after, err := buildQwen35MetalStateIdentityReceipt(
		before.OwnerGeneration, ids, Qwen35MetalStateAuthorityControl,
		[]qwen35MetalStateDigestSource{
			{Layer: 0, Role: Qwen35MetalStateRoleKRaw, Values: []float32{1, 2}},
			{Layer: 0, Role: Qwen35MetalStateRoleKPost, Values: []float32{3, 4}},
			{Layer: 0, Role: Qwen35MetalStateRoleV, Values: []float32{5, 6}},
			{Layer: 1, Role: Qwen35MetalStateRoleGDNConv, Values: []float32{7, 8}},
			{Layer: 1, Role: Qwen35MetalStateRoleGDNRecurrent, Values: []float32{9, 10, 11, 99}},
		},
		qwen35MetalStateIdentityAccounting{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if before.OwnerGeneration != after.OwnerGeneration || before.TokenLineageSHA256 != after.TokenLineageSHA256 {
		t.Fatal("within-arm mutation changed owner or token-lineage binding")
	}
	if before.States[4].SHA256 == after.States[4].SHA256 || before.BindingSHA256 == after.BindingSHA256 {
		t.Fatal("canonical digest did not detect recurrent-state mutation")
	}
}

func TestQwen35MetalStateIdentityPublicJSONOmitsNativeAndTensorDetails(t *testing.T) {
	receipt := qwen35StateIdentityFixture(t, Qwen35MetalStateAuthoritySequence)
	encoded, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	public := string(encoded)
	for _, forbidden := range []string{`"handle"`, `"pointer"`, `"path"`, `"tensor"`, `"values"`, `"chunks"`} {
		if strings.Contains(public, forbidden) {
			t.Fatalf("public state identity exposed forbidden field %s: %s", forbidden, public)
		}
	}
}

func TestQwen35MetalStateIdentitySessionIsolationResetAndOmission(t *testing.T) {
	ids := make([]int, 32)
	first, err := newQwen35MetalStateIdentityObservation(ids)
	if err != nil {
		t.Fatal(err)
	}
	second, err := newQwen35MetalStateIdentityObservation(ids)
	if err != nil {
		t.Fatal(err)
	}
	if first.ownerGeneration == second.ownerGeneration {
		t.Fatal("distinct sessions reused an opaque owner generation")
	}
	a := &Session{qwen35MetalStateIdentity: first}
	b := &Session{qwen35MetalStateIdentity: second}
	a.ResetQwen35MetalStateIdentityReceipt()
	if _, ok := a.Qwen35MetalStateIdentityReceipt(); ok {
		t.Fatal("reset session retained state identity")
	}
	if b.qwen35MetalStateIdentity == nil || b.qwen35MetalStateIdentity.ownerGeneration != second.ownerGeneration {
		t.Fatal("reset crossed into another session")
	}
	if _, ok := new(Session).Qwen35MetalStateIdentityReceipt(); ok {
		t.Fatal("default session exposed state identity")
	}
}

func TestQwen35MetalStateIdentityUnsupportedOmission(t *testing.T) {
	s := NewSynthetic(qwen35HybridQ4KTestCfg()).NewSession()
	if err := s.EnableQwen35MetalStateIdentityReceipt(make([]int, 32)); err == nil {
		t.Fatal("non-Metal session admitted state-identity observation")
	}
	if _, ok := s.Qwen35MetalStateIdentityReceipt(); ok {
		t.Fatal("unsupported session exposed state identity")
	}
}
