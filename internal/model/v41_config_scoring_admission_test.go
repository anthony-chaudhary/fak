package model

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

// fak-test:justify why=contract when=changed:internal/model/v41_config.go
// fak-test:runtime fast est=5ms lane=default
func TestV41ConfigRejectsIgnoredScoringSelector(t *testing.T) {
	raw, _ := readDeepSeekV41Config(t)
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		t.Fatal(err)
	}
	var text map[string]json.RawMessage
	if err := json.Unmarshal(root["text_config"], &text); err != nil {
		t.Fatal(err)
	}
	for _, scoring := range []string{"softmax", "sqrtsoftplus", ""} {
		t.Run(scoring, func(t *testing.T) {
			if scoring == "" {
				delete(text, "scoring_func")
			} else {
				text["scoring_func"], _ = json.Marshal(scoring)
			}
			var err error
			root["text_config"], err = json.Marshal(text)
			if err != nil {
				t.Fatal(err)
			}
			mutated, err := json.Marshal(root)
			if err != nil {
				t.Fatal(err)
			}
			var cfg Config
			err = json.Unmarshal(mutated, &cfg)
			if scoring == "softmax" {
				if !errors.Is(err, ErrV41ConfigAdmission) || !strings.Contains(err.Error(), "scoring_func") {
					t.Fatalf("incompatible scoring selector: got %v, want scoring_func admission error", err)
				}
			} else if err != nil {
				t.Fatalf("supported/default scoring selector rejected: %v", err)
			}
		})
	}
}
