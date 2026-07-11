package main

import "testing"

func TestParseDispatchProjectFields(t *testing.T) {
	raw := []byte(`{"data":{"repositoryOwner":{"projectV2":{"items":{"nodes":[{"content":{"number":4175},"fieldValues":{"nodes":[{"name":"P1","field":{"name":"Priority"}},{"name":"Ready","field":{"name":"Status"}}]}},{"content":{"number":9},"fieldValues":{"nodes":[{"name":"Done","field":{"name":"Status"}}]}}]}}}}}`)
	got := parseDispatchProjectFields(raw)
	if got[4175].Priority != "P1" || got[4175].Status != "Ready" || got[9].Status != "Done" {
		t.Fatalf("fields=%v", got)
	}
}

func TestParseDispatchProjectFieldsMalformedAbstains(t *testing.T) {
	if got := parseDispatchProjectFields([]byte(`{`)); got != nil {
		t.Fatalf("got=%v want nil", got)
	}
}
