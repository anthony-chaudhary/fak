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
			Additional []string            `json:"additional_speed_tiers"`
			Tiers      []map[string]string `json:"service_tiers"`
		} `json:"models"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if len(body.Models) != 1 || len(body.Models[0].Additional) != 2 || body.Models[0].Additional[1] != "fast" || body.Models[0].Tiers[1]["wire_value"] != "priority" {
		t.Fatalf("metadata=%+v body=%s", body, rec.Body.String())
	}
}
