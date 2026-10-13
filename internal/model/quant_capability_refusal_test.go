package model

import (
	"errors"
	"reflect"
	"testing"
)

// fak-test:runtime fast est=1ms
func TestQuantCapabilityRefusalRoutesEveryBand(t *testing.T) {
	bands := []QuantWeightBand{
		{Name: "resident-q4", QuantType: "Q4_K"},
		{Name: "legacy-q5", QuantType: "Q5_K"},
		{Name: "embedding-iq", QuantType: "IQ3_XXS", CPUOnly: true},
		{Name: "resident-iq", QuantType: "IQ3_XXS"},
		{Name: "unknown-iq", QuantType: "IQ2_S", CPUOnly: true, UnknownRole: true},
	}
	routes, err := QuantCapabilityCensus("metal", bands)
	var refusal *QuantCapabilityRefusal
	if !errors.As(err, &refusal) {
		t.Fatalf("error type = %T, want *QuantCapabilityRefusal", err)
	}
	if want := []string{"IQ2_S", "IQ3_XXS"}; !reflect.DeepEqual(refusal.Unsupported, want) {
		t.Fatalf("unsupported = %v, want %v", refusal.Unsupported, want)
	}
	wantRoutes := []QuantBandDispatch{
		{Name: "resident-q4", QuantType: "Q4_K", Backend: "metal"},
		{Name: "legacy-q5", QuantType: "Q5_K", Backend: "cpu"},
		{Name: "embedding-iq", QuantType: "IQ3_XXS", Backend: "cpu"},
		{Name: "resident-iq", QuantType: "IQ3_XXS", Backend: "unselected", Refused: true},
		{Name: "unknown-iq", QuantType: "IQ2_S", Backend: "unselected", Refused: true},
	}
	if !reflect.DeepEqual(routes, wantRoutes) {
		t.Fatalf("routes = %+v, want %+v", routes, wantRoutes)
	}

	cpuRoutes, err := QuantCapabilityCensus("cpu", bands)
	if err != nil {
		t.Fatal(err)
	}
	for _, route := range cpuRoutes {
		if route.Backend != "cpu" || route.Refused {
			t.Errorf("CPU route = %+v, want admitted CPU", route)
		}
	}
}
