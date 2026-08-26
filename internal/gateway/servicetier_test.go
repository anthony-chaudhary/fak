package gateway

import (
	"encoding/json"
	"net/http/httptest"
	"testing"
)

func TestModelsPublishesSupportedServiceTiers(t *testing.T) {
	rec := httptest.NewRecorder()
	(&Server{model: "gpt-test"}).handleModels(rec, httptest.NewRequest("GET", "/v1/models", nil))
	var body struct {
		Models []struct {
			Additional []string `json:"additional_speed_tiers"`
			Tiers      []struct {
				ID          string `json:"id"`
				Name        string `json:"name"`
				Description string `json:"description"`
			} `json:"service_tiers"`
		} `json:"models"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if len(body.Models) != 1 || len(body.Models[0].Additional) != 2 || body.Models[0].Additional[1] != "fast" {
		t.Fatalf("metadata=%+v body=%s", body, rec.Body.String())
	}
	wantIDs := []string{"default", "priority"}
	wantNames := []string{"Standard", "Fast"}
	if len(body.Models[0].Tiers) != len(wantIDs) {
		t.Fatalf("service_tiers=%+v body=%s", body.Models[0].Tiers, rec.Body.String())
	}
	for i, tier := range body.Models[0].Tiers {
		if tier.ID != wantIDs[i] || tier.Name != wantNames[i] || tier.Description == "" {
			t.Fatalf("service_tiers[%d]=%+v body=%s", i, tier, rec.Body.String())
		}
	}

	var rawBody struct {
		Models []struct {
			Tiers []map[string]json.RawMessage `json:"service_tiers"`
		} `json:"models"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &rawBody); err != nil {
		t.Fatal(err)
	}
	for i, tier := range rawBody.Models[0].Tiers {
		if _, ok := tier["mode"]; ok {
			t.Fatalf("service_tiers[%d] leaked internal mode: %s", i, rec.Body.String())
		}
		if _, ok := tier["wire_value"]; ok {
			t.Fatalf("service_tiers[%d] leaked provider wire_value: %s", i, rec.Body.String())
		}
	}
}
