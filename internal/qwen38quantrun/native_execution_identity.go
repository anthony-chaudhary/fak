package qwen38quantrun

import (
	"errors"
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/qwen38quant"
)

const (
	fakNativeRequestEngine  = "inkernel"
	fakNativeRequestPlanner = "inkernel"
	fakNativeRequestOwner   = "fak"
)

// FakNativeRequestIdentity binds the campaign-level fak-native classification
// to the runtime and ownership identity emitted by a real in-kernel request.
// The two identity layers deliberately retain their original spellings.
type FakNativeRequestIdentity struct {
	CampaignEngine  string
	ArmEngine       string
	ExpectedBackend string
	Receipt         *model.NativeInferenceReceipt
}

// ValidateFakNativeRequestIdentity rejects any request evidence that does not
// carry the exact fak-owned in-kernel tuple. It validates; it never normalizes
// or rewrites a receipt engine into the campaign classification.
func ValidateFakNativeRequestIdentity(identity FakNativeRequestIdentity) error {
	if identity.CampaignEngine != qwen38quant.EngineFakNative {
		return fmt.Errorf("fak-native request campaign engine %q, want %q", identity.CampaignEngine, qwen38quant.EngineFakNative)
	}
	if identity.ArmEngine != qwen38quant.EngineFakNative {
		return fmt.Errorf("fak-native request arm engine %q, want %q", identity.ArmEngine, qwen38quant.EngineFakNative)
	}
	if strings.TrimSpace(identity.ExpectedBackend) == "" {
		return errors.New("fak-native request expected backend is required")
	}
	if identity.Receipt == nil {
		return errors.New("fak-native request receipt is required")
	}
	receipt := identity.Receipt
	if receipt.Engine != fakNativeRequestEngine {
		return fmt.Errorf("fak-native request receipt engine %q, want %q", receipt.Engine, fakNativeRequestEngine)
	}
	if receipt.Planner != fakNativeRequestPlanner {
		return fmt.Errorf("fak-native request receipt planner %q, want %q", receipt.Planner, fakNativeRequestPlanner)
	}
	if receipt.Owner != fakNativeRequestOwner {
		return fmt.Errorf("fak-native request receipt owner %q, want %q", receipt.Owner, fakNativeRequestOwner)
	}
	if receipt.Backend != identity.ExpectedBackend {
		return fmt.Errorf("fak-native request receipt backend %q, want %q", receipt.Backend, identity.ExpectedBackend)
	}
	if strings.TrimSpace(receipt.ForwardPath) == "" {
		return errors.New("fak-native request receipt forward path is required")
	}
	if receipt.FallbackActive {
		return errors.New("fak-native request receipt reports active fallback")
	}
	return nil
}

// FakNativePhysicalIdentity binds the campaign-level fak-native classification
// to a canonical raw modelbench Qwen3.8 Vulkan physical receipt. This schema is
// intentionally distinct from model.NativeInferenceReceipt.
type FakNativePhysicalIdentity struct {
	CampaignEngine  string
	ArmEngine       string
	ExpectedBackend string
	Receipt         *compute.Qwen38VulkanDecodeReceipt
}

// ValidateFakNativePhysicalIdentity delegates physical evidence validation to
// the canonical receipt and additionally binds it to the declared arm backend.
func ValidateFakNativePhysicalIdentity(identity FakNativePhysicalIdentity) error {
	if identity.CampaignEngine != qwen38quant.EngineFakNative {
		return fmt.Errorf("fak-native physical campaign engine %q, want %q", identity.CampaignEngine, qwen38quant.EngineFakNative)
	}
	if identity.ArmEngine != qwen38quant.EngineFakNative {
		return fmt.Errorf("fak-native physical arm engine %q, want %q", identity.ArmEngine, qwen38quant.EngineFakNative)
	}
	if strings.TrimSpace(identity.ExpectedBackend) == "" {
		return errors.New("fak-native physical expected backend is required")
	}
	if identity.Receipt == nil {
		return errors.New("fak-native physical receipt is required")
	}
	if err := identity.Receipt.Validate(); err != nil {
		return fmt.Errorf("fak-native physical receipt: %w", err)
	}
	if identity.Receipt.Backend != identity.ExpectedBackend || identity.Receipt.Engine.Backend != identity.ExpectedBackend {
		return fmt.Errorf("fak-native physical receipt backend %q/%q, want %q", identity.Receipt.Backend, identity.Receipt.Engine.Backend, identity.ExpectedBackend)
	}
	return nil
}
