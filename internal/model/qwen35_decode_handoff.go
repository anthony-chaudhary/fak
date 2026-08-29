package model

import (
	"fmt"
	"strings"
)

// Qwen35DecodeHandoffMode selects only the decode boundary after the Qwen
// whole-sequence prefill owner has promoted its recurrent state. AUTO preserves
// the production block-first hierarchy; the other modes exist solely so
// modelbench can isolate the M3 projection-to-GDN handoff.
type Qwen35DecodeHandoffMode string

const (
	Qwen35DecodeHandoffAuto    Qwen35DecodeHandoffMode = "AUTO"
	Qwen35DecodeHandoffControl Qwen35DecodeHandoffMode = "CONTROL"
	Qwen35DecodeHandoffMixer   Qwen35DecodeHandoffMode = "MIXER"
)

func (m Qwen35DecodeHandoffMode) String() string {
	if m == "" {
		return string(Qwen35DecodeHandoffAuto)
	}
	return string(m)
}

func (m Qwen35DecodeHandoffMode) valid() bool {
	return m == Qwen35DecodeHandoffAuto || m == Qwen35DecodeHandoffControl || m == Qwen35DecodeHandoffMixer
}

// Set implements flag.Value while retaining a closed, model-owned mode set.
func (m *Qwen35DecodeHandoffMode) Set(value string) error {
	parsed := Qwen35DecodeHandoffMode(strings.ToUpper(strings.TrimSpace(value)))
	if !parsed.valid() {
		return fmt.Errorf("Qwen decode handoff mode %q is invalid; want AUTO, CONTROL, or MIXER", value)
	}
	*m = parsed
	return nil
}

// Qwen35DecodeHandoffReceipt is one session's accepted decode-route accounting.
// Counts advance only after the corresponding operation accepts ownership.
type Qwen35DecodeHandoffReceipt struct {
	Mode                     Qwen35DecodeHandoffMode `json:"mode"`
	BlockAcceptedCalls       uint64                  `json:"block_accepted_calls"`
	MixerAcceptedCalls       uint64                  `json:"mixer_accepted_calls"`
	ResidentGDNAcceptedCalls uint64                  `json:"resident_gdn_accepted_calls"`
}

// SetQwen35DecodeHandoffMode configures the benchmark-only ablation before
// decode begins. Graded modes require an already-admitted sequence owner so a
// caller cannot label an ordinary host path CONTROL or MIXER.
func (s *Session) SetQwen35DecodeHandoffMode(mode Qwen35DecodeHandoffMode) error {
	if !mode.valid() {
		return fmt.Errorf("model: invalid Qwen decode handoff mode %q", mode)
	}
	if s == nil || s.qwen35HAL == nil {
		if mode == Qwen35DecodeHandoffAuto {
			return nil
		}
		return fmt.Errorf("model: Qwen decode handoff mode %s requires an admitted sequence owner", mode)
	}
	q := s.qwen35HAL
	if mode != Qwen35DecodeHandoffAuto && !q.sequenceAccepted && !q.decodeAccepted {
		return fmt.Errorf("model: Qwen decode handoff mode %s requires the sequence path enabled", mode)
	}
	if q.decodeHandoff.BlockAcceptedCalls != 0 || q.decodeHandoff.MixerAcceptedCalls != 0 || q.decodeHandoff.ResidentGDNAcceptedCalls != 0 {
		return fmt.Errorf("model: Qwen decode handoff mode cannot change after decode execution")
	}
	q.decodeHandoff = Qwen35DecodeHandoffReceipt{Mode: mode}
	return nil
}

func (s *Session) qwen35DecodeHandoffMode() Qwen35DecodeHandoffMode {
	if s == nil || s.qwen35HAL == nil || s.qwen35HAL.decodeHandoff.Mode == "" {
		return Qwen35DecodeHandoffAuto
	}
	return s.qwen35HAL.decodeHandoff.Mode
}

func (s *Session) recordQwen35DecodeBlockAccepted() {
	s.qwen35HAL.decodeHandoff.BlockAcceptedCalls++
}

func (s *Session) recordQwen35DecodeMixerAccepted() {
	s.qwen35HAL.decodeHandoff.MixerAcceptedCalls++
}

func (s *Session) recordQwen35ResidentGDNAccepted() {
	s.qwen35HAL.decodeHandoff.ResidentGDNAcceptedCalls++
}

// Qwen35DecodeHandoffReceipt returns an immutable session-local snapshot.
func (s *Session) Qwen35DecodeHandoffReceipt() Qwen35DecodeHandoffReceipt {
	if s == nil || s.qwen35HAL == nil {
		return Qwen35DecodeHandoffReceipt{Mode: Qwen35DecodeHandoffAuto}
	}
	receipt := s.qwen35HAL.decodeHandoff
	if receipt.Mode == "" {
		receipt.Mode = Qwen35DecodeHandoffAuto
	}
	return receipt
}

// ValidateQwen35DecodeHandoffReceipt enforces the two attribution-bearing
// routes. AUTO is observational and preserves whatever production hierarchy
// accepted the call; CONTROL and MIXER are exact experiment contracts.
func ValidateQwen35DecodeHandoffReceipt(receipt Qwen35DecodeHandoffReceipt) error {
	if !receipt.Mode.valid() {
		return fmt.Errorf("Qwen decode handoff receipt has invalid mode %q", receipt.Mode)
	}
	switch receipt.Mode {
	case Qwen35DecodeHandoffAuto:
		return nil
	case Qwen35DecodeHandoffControl:
		if receipt.BlockAcceptedCalls != 0 || receipt.MixerAcceptedCalls != 0 || receipt.ResidentGDNAcceptedCalls == 0 {
			return fmt.Errorf("Qwen decode handoff CONTROL requires block=0 mixer=0 resident_gdn>0, got %+v", receipt)
		}
	case Qwen35DecodeHandoffMixer:
		if receipt.BlockAcceptedCalls != 0 || receipt.MixerAcceptedCalls == 0 || receipt.ResidentGDNAcceptedCalls != 0 {
			return fmt.Errorf("Qwen decode handoff MIXER requires block=0 mixer>0 resident_gdn=0, got %+v", receipt)
		}
	}
	return nil
}
