package sessionintent

import (
	"encoding/json"
	"testing"
	"time"
)

func TestMarshalJSONEscapesHookActionsAndOmitsUnusedRecurrenceForm(t *testing.T) {
	i := baseIntent()
	i.Recurrence = &Recurrence{Cron: "0 9 * * 1", Timezone: "America/Los_Angeles", OverlapPolicy: "forbid", MisfirePolicy: "skip"}
	i.Hooks = []Hook{{Event: "on_start", Action: "notify\noperator", Timeout: time.Second, FailurePolicy: "continue"}}
	got, err := json.Marshal(i)
	if err != nil {
		t.Fatal(err)
	}
	if !json.Valid(got) {
		t.Fatalf("invalid JSON: %q", got)
	}
	var decoded map[string]any
	if err := json.Unmarshal(got, &decoded); err != nil {
		t.Fatal(err)
	}
	recurrence := decoded["recurrence"].(map[string]any)
	if _, exists := recurrence["every"]; exists {
		t.Fatalf("cron recurrence carried every: %s", got)
	}
	hook := decoded["hooks"].([]any)[0].(map[string]any)
	if hook["action"] != "notify\noperator" {
		t.Fatalf("action did not round-trip: %#v", hook["action"])
	}
}
