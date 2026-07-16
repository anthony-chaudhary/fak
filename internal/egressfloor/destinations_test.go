package egressfloor

import (
	"reflect"
	"sort"
	"testing"
)

// TestDestinationsExtractsHosts pins the extraction contract the egresslist layer relies
// on: whole-value destination args and shell-embedded curl/wget targets both surface as
// normalized, de-duplicated bare hosts, and a call with no destination yields nil.
func TestDestinationsExtractsHosts(t *testing.T) {
	cases := []struct {
		name string
		args map[string]any
		want []string
	}{
		{
			name: "webfetch url",
			args: map[string]any{"url": "https://Docs.Example.com:443/guide"},
			want: []string{"docs.example.com"},
		},
		{
			name: "bare host with port",
			args: map[string]any{"endpoint": "internal.corp:8443"},
			want: []string{"internal.corp"},
		},
		{
			name: "shell curl and wget",
			args: map[string]any{"command": "curl -s https://a.example/x && wget http://b.example/y"},
			want: []string{"a.example", "b.example"},
		},
		{
			name: "dedup across args",
			args: map[string]any{"url": "https://dup.example/a", "target_url": "https://dup.example/b"},
			want: []string{"dup.example"},
		},
		{
			name: "no destination",
			args: map[string]any{"content": "just some text mentioning example.com in prose"},
			want: nil,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := Destinations("AnyTool", tc.args)
			// dedup-order is first-seen, but map iteration over multiple keys is
			// unordered, so compare as sets.
			gs := append([]string(nil), got...)
			ws := append([]string(nil), tc.want...)
			sort.Strings(gs)
			sort.Strings(ws)
			if len(gs) == 0 && len(ws) == 0 {
				return
			}
			if !reflect.DeepEqual(gs, ws) {
				t.Fatalf("Destinations = %v, want %v", got, tc.want)
			}
		})
	}
}
