package h2hbench

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// fak-test:runtime fast est=1s lane=default
func TestRequestedSpeculativeMatrixEvidenceAndGrouping(t *testing.T) {
	var caps []any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		caps = append(caps, body["speculative.n_max"])
		w.Header().Set("Trailer", "X-Fak-Speculative")
		fmt.Fprintln(w, `data: {"choices":[{"delta":{"content":"hello"}}],"usage":{"prompt_tokens":8,"completion_tokens":4},"timings":{"draft_n":3,"draft_n_accepted":2}}`)
		fmt.Fprintln(w, "data: [DONE]")
		w.Header().Set("X-Fak-Speculative", "mtp-vulkan")
	}))
	defer server.Close()
	cfg := Config{RunID: "matrix", Model: "test", Arms: []Arm{{Name: "fak", Role: RoleFak, BaseURL: server.URL}}, DecodeTokens: 4, SpeculativeModes: []string{"0", "2", "server-default"}}
	rows, err := Run(context.Background(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 3 || len(caps) != 3 || caps[0] != float64(0) || caps[1] != float64(2) || caps[2] != nil {
		t.Fatalf("requested caps: %v, rows=%d", caps, len(rows))
	}
	for _, row := range rows {
		if row.SpeculationEvidence != "draft-observed" || row.Server.DraftAccepted != 2 || row.SpeculativeTrailer != "mtp-vulkan" {
			t.Fatalf("missing same-response evidence: %+v", row)
		}
	}
	if cells := Summarize(rows).Cells; len(cells) != 3 {
		t.Fatalf("modes pooled: %+v", cells)
	}
	losses, wins := Compare([]Cell{{Arm: "fak", Role: RoleFak, Scenario: ScenarioDecode, Size: 4, N: 1, DecodeP50: 10, RequestedSpeculation: "0"}, {Arm: "llama", Role: RoleBaseline, Scenario: ScenarioDecode, Size: 4, N: 1, DecodeP50: 100, RequestedSpeculation: "2"}})
	if len(losses)+len(wins) != 0 {
		t.Fatal("unlike requested modes compared")
	}
	cfg.SpeculativeModes = nil
	rows, err = Run(context.Background(), cfg)
	if err != nil || len(rows) != 1 || caps[len(caps)-1] != nil {
		t.Fatalf("library default changed: %v %v", rows, err)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestSpeculativeUnknownAndModeValidation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, `data: {"choices":[{"delta":{"content":"ok"}}]}`)
		fmt.Fprintln(w, "data: [DONE]")
	}))
	defer server.Close()
	rows, err := Run(context.Background(), Config{Arms: []Arm{{BaseURL: server.URL}}, DecodeTokens: 4, SpeculativeModes: []string{"0", "2"}})
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range rows {
		if r.SpeculationEvidence != "unknown" {
			t.Fatalf("unwitnessed speculation: %+v", r)
		}
	}
	for _, raw := range []string{"", "-1", "65", "0,0", "serial"} {
		if _, err := ParseSpeculativeModes(raw); err == nil {
			t.Errorf("accepted %q", raw)
		}
	}
}

// fak-test:runtime fast est=1s lane=default
func TestIncompleteDraftCountersStayUnknown(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, `data: {"choices":[{"delta":{"content":"ok"}}],"timings":{"draft_n":3}}`)
		fmt.Fprintln(w, "data: [DONE]")
	}))
	defer server.Close()
	rows, err := Run(context.Background(), Config{Arms: []Arm{{BaseURL: server.URL}}, DecodeTokens: 4})
	if err != nil || len(rows) != 1 || rows[0].SpeculationEvidence != "unknown" {
		t.Fatalf("incomplete counters promoted: %+v %v", rows, err)
	}
}

type speculativeTransport func(*http.Request) (*http.Response, error)

func (f speculativeTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

type trailerErrorBody struct {
	io.Reader
	err error
}

func (b trailerErrorBody) Read(p []byte) (int, error) {
	n, e := b.Reader.Read(p)
	if e == io.EOF {
		return n, b.err
	}
	return n, e
}
func (b trailerErrorBody) Close() error { return nil }

// fak-test:runtime fast est=1s lane=default
func TestTrailerDrainFailureIsNotSuccessful(t *testing.T) {
	sentinel := errors.New("trailer read failed")
	client := &http.Client{Transport: speculativeTransport(func(*http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: 200, Header: http.Header{}, Body: trailerErrorBody{Reader: strings.NewReader("data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\ndata: [DONE]\n"), err: sentinel}}, nil
	})}
	var row Row
	_, err := measureOnce(context.Background(), Config{Client: client}, Arm{BaseURL: "http://test"}, nil, 4, &row, nil)
	if !errors.Is(err, sentinel) || row.OK {
		t.Fatalf("drain failure accepted: row=%+v err=%v", row, err)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestDraftCounterPresenceRoundTrip(t *testing.T) {
	for _, input := range []string{`{}`, `{"draft_n":3}`, `{"draft_n":3,"draft_n_accepted":0}`, `{"draft_n":0,"draft_n_accepted":0}`} {
		var timing ServerTimings
		if err := json.Unmarshal([]byte(input), &timing); err != nil {
			t.Fatal(err)
		}
		data, err := json.Marshal(timing)
		if err != nil {
			t.Fatal(err)
		}
		var before, after map[string]json.RawMessage
		json.Unmarshal([]byte(input), &before)
		json.Unmarshal(data, &after)
		for _, key := range []string{"draft_n", "draft_n_accepted"} {
			if string(before[key]) != string(after[key]) {
				t.Fatalf("%s changed: input=%s output=%s", key, input, data)
			}
		}
		var roundTrip ServerTimings
		if err := json.Unmarshal(data, &roundTrip); err != nil {
			t.Fatal(err)
		}
		if roundTrip.draftPresent != timing.draftPresent || roundTrip.acceptedPresent != timing.acceptedPresent {
			t.Fatal("counter presence changed")
		}
	}
}
