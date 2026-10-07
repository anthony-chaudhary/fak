package agent

import (
	"context"
	"errors"
	"testing"
)

// TestCompleteStreamInBandErrorIsTyped pins the closed contract callers branch on: the
// sentinel via errors.Is and the normalized code via errors.As.
//
// fak-test:runtime fast est=50ms
func TestCompleteStreamInBandErrorIsTyped(t *testing.T) {
	cases := []struct {
		name, frame, wantCode string
	}{
		{"string_code", inBandTruncatedFrame, "upstream_truncated"},
		{"numeric_code", "data: {\"error\":{\"type\":\"server_error\",\"code\":502,\"message\":\"x\"}}\n\n", "502"},
		{"bare_string", "data: {\"error\":\"x\"}\n\n", ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			body := "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"},\"finish_reason\":null}]}\n\n" +
				tc.frame + "data: [DONE]\n\n"
			srv, _ := countingSSEServer(t, body)
			p := NewHTTPPlanner(srv.URL, "m", "")
			_, err := p.CompleteStream(context.Background(), nil, []Message{{Role: RoleUser, Content: "hi"}}, nil)
			if !errors.Is(err, ErrUpstreamStreamError) {
				t.Fatalf("errors.Is(err, ErrUpstreamStreamError) = false; err = %v", err)
			}
			var se *UpstreamStreamError
			if !errors.As(err, &se) {
				t.Fatalf("errors.As(err, *UpstreamStreamError) = false; err = %v", err)
			}
			if se.Code != tc.wantCode {
				t.Fatalf("code = %q, want %q", se.Code, tc.wantCode)
			}
		})
	}
}
