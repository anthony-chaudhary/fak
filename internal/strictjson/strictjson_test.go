package strictjson

import (
	"strings"
	"testing"
)

func TestDecode(t *testing.T) {
	var dst struct {
		Name string `json:"name"`
	}
	if err := Decode([]byte(`{"name":"ok"}`), &dst, "multiple JSON values"); err != nil || dst.Name != "ok" {
		t.Fatalf("Decode valid = %+v, %v", dst, err)
	}
	if err := Decode([]byte(`{"unknown":1}`), &dst, "multiple JSON values"); err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("unknown field error = %v", err)
	}
	if err := Decode([]byte(`{"name":"a"} {}`), &dst, "trailing document"); err == nil || err.Error() != "trailing document" {
		t.Fatalf("trailing error = %v", err)
	}
}
