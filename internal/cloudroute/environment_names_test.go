package cloudroute

import (
	"reflect"
	"runtime"
	"sort"
	"strings"
	"testing"
)

// Pure environment snapshots; no process environment or credentials are read.
// fak-test:runtime fast est=1ms lane=default
func TestRouteWaiverAndCredentialNamesFollowHost(t *testing.T) {
	t.Parallel()
	for _, route := range []struct {
		kind                 Kind
		selector, credential string
	}{
		{KindBedrock, "CLAUDE_CODE_USE_BEDROCK", "AWS_SESSION_TOKEN"},
		{KindVertex, "CLAUDE_CODE_USE_VERTEX", "GOOGLE_APPLICATION_CREDENTIALS"},
	} {
		for _, spelling := range []struct {
			name    string
			rewrite func(string) string
			matches bool
		}{
			{"canonical", func(s string) string { return s }, true},
			{"lowercase", strings.ToLower, runtime.GOOS == "windows"},
			{"leading space", func(s string) string { return " " + s }, false},
			{"trailing space", func(s string) string { return s + " " }, false},
		} {
			t.Run(string(route.kind)+"/"+spelling.name, func(t *testing.T) {
				env := []string{
					route.selector + "= 1 ",
					spelling.rewrite(WaiverKey) + "= 1 ",
					spelling.rewrite(route.credential) + "=synthetic-fixture",
				}
				r, ok := Detect(env)
				if !ok || r.Kind != route.kind || r.Selector != route.selector || r.Waived != spelling.matches {
					t.Fatalf("Detect = %+v, %v; want route %s with waived=%v", r, ok, route.kind, spelling.matches)
				}
				wantObserved := []string{route.selector}
				var wantCredentials []string
				if spelling.matches {
					wantObserved = append(wantObserved, route.credential)
					wantCredentials = []string{route.credential}
				}
				sort.Strings(wantObserved)
				if !reflect.DeepEqual(r.Observed, wantObserved) {
					t.Fatalf("Observed = %v, want %v", r.Observed, wantObserved)
				}
				if got := r.CredentialNames(env); !reflect.DeepEqual(got, wantCredentials) {
					t.Fatalf("CredentialNames = %v, want %v", got, wantCredentials)
				}
			})
		}
	}
}
