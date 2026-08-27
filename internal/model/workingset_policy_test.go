package model

import "testing"

func TestPlaceWorkingSetsBoundsPollutionAndFairness(t *testing.T) {
	sets := []WorkingSet{{"a-hot", "a", 100, 100, 100, 10}, {"a-cold", "a", 100, 1, 500, 10}, {"b-hot", "b", 100, 50, 100, 10}, {"b-cold", "b", 100, 2, 400, 10}}
	r, err := PlaceWorkingSets(sets, 200)
	if err != nil {
		t.Fatal(err)
	}
	if r.Engine != "fak-native" || len(r.Resident) != 2 || r.Resident[0] != "a-hot" || r.Resident[1] != "b-hot" || r.MinTenantShare != .5 || r.EvictionAmplificationBytes != 900 || r.ReloadBytesPerAccepted != 22.5 {
		t.Fatalf("decision=%+v", r)
	}
}
func TestPlaceWorkingSetsRejectsInvalid(t *testing.T) {
	if _, err := PlaceWorkingSets([]WorkingSet{{ID: "bad", Tenant: "x", Bytes: -1}}, 1); err == nil {
		t.Fatal("invalid set admitted")
	}
}
