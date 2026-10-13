package strictjson

import (
	"bytes"
	"testing"
)

// fak-test:runtime fast est=100ms lane=default
func TestRepairClosedJSON(t *testing.T) {
	valid := []byte(" \n{\"message\":\"keep ,} and \\\" escapes\",\"items\":[1,2]}\t")
	got, ok := RepairClosedJSON(valid)
	if !ok || !bytes.Equal(got, valid) {
		t.Fatalf("valid identity = %q, %v", got, ok)
	}

	for _, tc := range []struct {
		name  string
		input string
		want  string
	}{
		{"object trailing comma", `{"a":1,}`, `{"a":1}`},
		{"nested trailing commas", `{"a":[1,2,],"b":{"c":3,},}`, `{"a":[1,2],"b":{"c":3}}`},
		{"closed json fence", "```json\n{\"a\":1,}\n```", `{"a":1}`},
		{"closed tool fence", "```tool_call\n[{\"name\":\"Read\",},]\n```", `[{"name":"Read"}]`},
		{"commas in strings", `{"text":",} ,]",}`, `{"text":",} ,]"}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := RepairClosedJSON([]byte(tc.input))
			if !ok || string(got) != tc.want {
				t.Fatalf("RepairClosedJSON(%q) = %q, %v; want %q, true", tc.input, got, ok, tc.want)
			}
		})
	}

	for _, input := range []string{
		`{,}`,
		`[,]`,
		"```json\n{\"a\":1,}",
		"prose {\"a\":1,} trailing",
		`{"a":1,`,
		`{"a":"cut,}`,
		`{"a":[1,2},`,
		`[1,,]`,
	} {
		t.Run("reject "+input, func(t *testing.T) {
			got, ok := RepairClosedJSON([]byte(input))
			if ok || string(got) != input {
				t.Fatalf("RepairClosedJSON(%q) = %q, %v; want original, false", input, got, ok)
			}
		})
	}
}
