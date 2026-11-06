package servicespec

import (
	"reflect"
	"strings"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestParseSpecRequiresOneDocument(t *testing.T) {
	t.Parallel()
	want := mustParse(t, fullFixture)
	got, err := ParseSpec([]byte(" \t\r\n" + fullFixture + "\n\r\t "))
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("JSON whitespace changed normalized spec: got=%+v err=%v want=%+v", got, err, want)
	}
	for _, tail := range []string{
		`{"desired":"desired-stopped"}`,
		`{}`,
		`null`,
		`42`,
		`]`,
		`garbage`,
	} {
		t.Run(tail, func(t *testing.T) {
			got, err := ParseSpec([]byte(fullFixture + "\n" + tail))
			if got != nil || err == nil || !strings.HasPrefix(err.Error(), "servicespec:") {
				t.Fatalf("trailing %q returned spec=%+v err=%v, want nil spec and contextual refusal", tail, got, err)
			}
		})
	}
	unknown := strings.TrimSuffix(fullFixture, "}") + `,"unexpected":true}`
	if got, err := ParseSpec([]byte(unknown)); got != nil || err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("unknown field returned spec=%+v err=%v, want unchanged refusal", got, err)
	}
}
