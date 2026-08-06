package agent

import "testing"

func TestNativeRouteLabelsCarryCallSignals(t *testing.T) {
	for _, tc := range []struct {
		name      string
		meta      map[string]string
		principal string
		want      map[string]string
	}{
		{name: "canonical sensitivity", meta: map[string]string{"readOnlyHint": "true", "sensitivity": "pii"}, principal: "acme", want: map[string]string{"read_only": "true", "sensitivity": "pii", "tenant": "acme"}},
		{name: "compatibility sensitivity", meta: map[string]string{"readOnlyHint": "false", "data_sensitivity": "tenant"}, want: map[string]string{"read_only": "false", "sensitivity": "tenant"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := nativeRouteLabels(tc.meta, tc.principal)
			if len(got) != len(tc.want) {
				t.Fatalf("labels = %#v, want %#v", got, tc.want)
			}
			for key, want := range tc.want {
				if got[key] != want {
					t.Fatalf("label %q = %q, want %q (all labels: %#v)", key, got[key], want, got)
				}
			}
		})
	}
}
