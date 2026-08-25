package modelroute

import "testing"

func TestProviderServiceTierContractAndReceipt(t *testing.T) {
	openai, ok := LookupProviderContract("openai")
	if !ok {
		t.Fatal("openai contract missing")
	}
	wire, receipt, err := BindServiceTier(openai, ServiceTierRequest{Mode: ServiceModeFast, Policy: ServiceRequire})
	if err != nil || wire != "priority" {
		t.Fatalf("bind openai fast wire=%q receipt=%+v err=%v", wire, receipt, err)
	}
	receipt, err = RealizeServiceTier(receipt, ServiceModeFast, 0)
	if err != nil || receipt.Requested != ServiceModeFast || receipt.Realized != ServiceModeFast || receipt.CacheInvalidated {
		t.Fatalf("realize fast receipt=%+v err=%v", receipt, err)
	}

	anthropic, ok := LookupProviderContract("anthropic")
	if !ok {
		t.Fatal("anthropic contract missing")
	}
	wire, receipt, err = BindServiceTier(anthropic, ServiceTierRequest{Mode: ServiceModeFast, Policy: ServiceAllowDeclaredDowngrade})
	if err != nil || wire != "auto" {
		t.Fatalf("bind anthropic fast wire=%q err=%v", wire, err)
	}
	receipt, err = RealizeServiceTier(receipt, ServiceModeStandard, 73)
	if err != nil || receipt.Realized != ServiceModeStandard || receipt.DowngradeReason == "" || !receipt.CacheInvalidated || receipt.CacheRewarmTokens != 73 {
		t.Fatalf("downgrade receipt=%+v err=%v", receipt, err)
	}
}

func TestProviderServiceTierPoliciesAndUnknownReadback(t *testing.T) {
	openai, _ := LookupProviderContract("openai")
	_, _, err := BindServiceTier(openai, ServiceTierRequest{Mode: ServiceModeFast, Policy: ServiceStandardOnly})
	if err == nil {
		t.Fatal("standard_only accepted fast")
	}
	_, receipt, err := BindServiceTier(openai, ServiceTierRequest{Mode: ServiceModeFast, Policy: ServiceRequire})
	if err != nil {
		t.Fatal(err)
	}
	if _, err = RealizeServiceTier(receipt, ServiceModeUnknown, 0); err == nil {
		t.Fatal("require accepted unknown readback")
	}
	unknown := ProviderContract{Provider: "unknown"}
	if _, _, err = BindServiceTier(unknown, ServiceTierRequest{Mode: ServiceModeFast}); err == nil {
		t.Fatal("unknown capability accepted")
	}
}

func TestProviderServiceTierMetadataSupportedOnly(t *testing.T) {
	openai, _ := LookupProviderContract("openai")
	modes, rows := SupportedServiceTierMetadata(openai)
	if len(modes) != 2 || modes[0] != "standard" || modes[1] != "fast" || rows[1]["wire_value"] != "priority" {
		t.Fatalf("modes=%v rows=%v", modes, rows)
	}
	modes, rows = SupportedServiceTierMetadata(ProviderContract{})
	if len(modes) != 0 || len(rows) != 0 {
		t.Fatalf("unknown metadata modes=%v rows=%v", modes, rows)
	}
}
