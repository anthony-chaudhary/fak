package strictjson

import (
	"encoding/json"
	"testing"
)

func TestNumberValue(t *testing.T) {
	value, err := NumberValue([]byte(`{"n":5}`))
	if err != nil {
		t.Fatal(err)
	}
	if got := value.(map[string]any)["n"]; got != json.Number("5") {
		t.Fatalf("number=%T(%v)", got, got)
	}
	if _, err := NumberValue([]byte(`{} {}`)); err == nil {
		t.Fatal("accepted trailing value")
	}
}
