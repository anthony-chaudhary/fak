package model

import "testing"

type decodeHandoffTestBackend struct {
	*residentGDNBackend
	blockCalls int
	mixerCalls int
}

func newDecodeHandoffTestBackend() *decodeHandoffTestBackend {
	return &decodeHandoffTestBackend{residentGDNBackend: newResidentGDNBackend()}
}

func (b *decodeHandoffTestBackend) Qwen35MetalDecodeBlock(_ *Session, _ int, x []float32) ([]float32, qwen35DecodeBlockReceipt, bool, error) {
	b.blockCalls++
	return append([]float32(nil), x...), qwen35DecodeBlockReceipt{}, true, nil
}

func (b *decodeHandoffTestBackend) Qwen35MetalDecodeMixer(_ *Session, _ int, x []float32) ([]float32, qwen35DecodeMixerReceipt, bool, error) {
	b.mixerCalls++
	return append([]float32(nil), x...), qwen35DecodeMixerReceipt{}, true, nil
}

func promotedDecodeHandoffTestSession(t *testing.T) (*Session, *decodeHandoffTestBackend) {
	t.Helper()
	backend := newDecodeHandoffTestBackend()
	s := residentTestSession(t, backend)
	if used, err := s.promoteQwen35MetalGDNDecode(residentSnapshots(s)); err != nil || !used {
		t.Fatalf("promote resident GDN: used=%v err=%v", used, err)
	}
	t.Cleanup(s.Close)
	return s, backend
}

func TestQwen35DecodeHandoffAutoPreservesBlockFirstSelection(t *testing.T) {
	s, backend := promotedDecodeHandoffTestSession(t)
	if _, _, accepted, err := s.tryQwen35MetalDecodeBlock(0, []float32{1}); err != nil || !accepted {
		t.Fatalf("AUTO block: accepted=%v err=%v", accepted, err)
	}
	got := s.Qwen35DecodeHandoffReceipt()
	want := Qwen35DecodeHandoffReceipt{Mode: Qwen35DecodeHandoffAuto, BlockAcceptedCalls: 1}
	if got != want || backend.blockCalls != 1 || backend.mixerCalls != 0 {
		t.Fatalf("AUTO receipt/backend=%+v %d/%d, want %+v 1/0", got, backend.blockCalls, backend.mixerCalls, want)
	}
}

func TestQwen35DecodeHandoffControlSuppressesBlockAndMixer(t *testing.T) {
	s, backend := promotedDecodeHandoffTestSession(t)
	if err := s.SetQwen35DecodeHandoffMode(Qwen35DecodeHandoffControl); err != nil {
		t.Fatal(err)
	}
	if _, _, accepted, err := s.tryQwen35MetalDecodeBlock(0, []float32{1}); accepted || err != nil {
		t.Fatalf("CONTROL block: accepted=%v err=%v", accepted, err)
	}
	if _, _, accepted, err := s.tryQwen35MetalDecodeMixer(0, []float32{1}); accepted || err != nil {
		t.Fatalf("CONTROL mixer: accepted=%v err=%v", accepted, err)
	}
	if _, accepted, err := s.tryQwen35MetalGDNDecode(0, nil, nil, nil, nil, nil, nil, nil, nil, 0); err != nil || !accepted {
		t.Fatalf("CONTROL resident GDN: accepted=%v err=%v", accepted, err)
	}
	got := s.Qwen35DecodeHandoffReceipt()
	want := Qwen35DecodeHandoffReceipt{Mode: Qwen35DecodeHandoffControl, ResidentGDNAcceptedCalls: 1}
	if got != want || backend.blockCalls != 0 || backend.mixerCalls != 0 {
		t.Fatalf("CONTROL receipt/backend=%+v %d/%d, want %+v 0/0", got, backend.blockCalls, backend.mixerCalls, want)
	}
}

func TestQwen35DecodeHandoffMixerSuppressesBlockAndCountsMixer(t *testing.T) {
	s, backend := promotedDecodeHandoffTestSession(t)
	if err := s.SetQwen35DecodeHandoffMode(Qwen35DecodeHandoffMixer); err != nil {
		t.Fatal(err)
	}
	if _, _, accepted, err := s.tryQwen35MetalDecodeBlock(0, []float32{1}); accepted || err != nil {
		t.Fatalf("MIXER block: accepted=%v err=%v", accepted, err)
	}
	if _, _, accepted, err := s.tryQwen35MetalDecodeMixer(0, []float32{1}); err != nil || !accepted {
		t.Fatalf("MIXER mixer: accepted=%v err=%v", accepted, err)
	}
	got := s.Qwen35DecodeHandoffReceipt()
	want := Qwen35DecodeHandoffReceipt{Mode: Qwen35DecodeHandoffMixer, MixerAcceptedCalls: 1}
	if got != want || backend.blockCalls != 0 || backend.mixerCalls != 1 {
		t.Fatalf("MIXER receipt/backend=%+v %d/%d, want %+v 0/1", got, backend.blockCalls, backend.mixerCalls, want)
	}
}

func TestQwen35DecodeHandoffResetRestoresAutoWithZeroCounts(t *testing.T) {
	s, _ := promotedDecodeHandoffTestSession(t)
	if err := s.SetQwen35DecodeHandoffMode(Qwen35DecodeHandoffControl); err != nil {
		t.Fatal(err)
	}
	if _, accepted, err := s.tryQwen35MetalGDNDecode(0, nil, nil, nil, nil, nil, nil, nil, nil, 0); err != nil || !accepted {
		t.Fatalf("CONTROL resident GDN: accepted=%v err=%v", accepted, err)
	}
	if got := s.Qwen35DecodeHandoffReceipt(); got.ResidentGDNAcceptedCalls != 1 {
		t.Fatalf("receipt before reset = %+v, want one resident GDN call", got)
	}

	s.ResetQwen35MetalGDNDecode()
	want := Qwen35DecodeHandoffReceipt{Mode: Qwen35DecodeHandoffAuto}
	if got := s.Qwen35DecodeHandoffReceipt(); got != want {
		t.Fatalf("receipt after reset = %+v, want %+v", got, want)
	}
}

func TestQwen35DecodeHandoffGradedModesRequireSequenceAndValidateCounts(t *testing.T) {
	plain := NewSynthetic(qwen35HybridTestCfg()).NewSession()
	defer plain.Close()
	for _, mode := range []Qwen35DecodeHandoffMode{Qwen35DecodeHandoffControl, Qwen35DecodeHandoffMixer} {
		if err := plain.SetQwen35DecodeHandoffMode(mode); err == nil {
			t.Fatalf("graded mode %s accepted without sequence owner", mode)
		}
	}

	valid := []Qwen35DecodeHandoffReceipt{
		{Mode: Qwen35DecodeHandoffAuto},
		{Mode: Qwen35DecodeHandoffControl, ResidentGDNAcceptedCalls: 1},
		{Mode: Qwen35DecodeHandoffMixer, MixerAcceptedCalls: 1},
	}
	for _, receipt := range valid {
		if err := ValidateQwen35DecodeHandoffReceipt(receipt); err != nil {
			t.Fatalf("valid receipt %+v rejected: %v", receipt, err)
		}
	}
	invalid := []Qwen35DecodeHandoffReceipt{
		{Mode: Qwen35DecodeHandoffControl},
		{Mode: Qwen35DecodeHandoffControl, BlockAcceptedCalls: 1, ResidentGDNAcceptedCalls: 1},
		{Mode: Qwen35DecodeHandoffMixer},
		{Mode: Qwen35DecodeHandoffMixer, MixerAcceptedCalls: 1, ResidentGDNAcceptedCalls: 1},
	}
	for _, receipt := range invalid {
		if err := ValidateQwen35DecodeHandoffReceipt(receipt); err == nil {
			t.Fatalf("invalid receipt %+v accepted", receipt)
		}
	}
}
