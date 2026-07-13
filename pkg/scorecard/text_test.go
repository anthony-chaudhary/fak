package scorecard

import "testing"

func TestValueTextPreservesScorecardWireForms(t *testing.T) {
	tests := []struct {
		name string
		in   any
		want string
	}{
		{name: "string stays unquoted", in: "ready", want: "ready"},
		{name: "positive int", in: 42, want: "42"},
		{name: "negative int", in: -7, want: "-7"},
		{name: "boolean uses JSON", in: true, want: "true"},
		{name: "nil uses JSON", in: nil, want: "null"},
		{name: "object uses JSON", in: map[string]int{"score": 9}, want: `{"score":9}`},
		{name: "marshal failure stays empty", in: make(chan int), want: ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ValueText(tt.in); got != tt.want {
				t.Fatalf("ValueText(%#v) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}
