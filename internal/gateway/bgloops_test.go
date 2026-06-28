package gateway

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/bgloop"
)

func TestHandleFakLoopsReturnsSnapshot(t *testing.T) {
	sup := bgloop.New()
	_ = sup.Register(bgloop.Loop{Name: "heartbeat", Interval: time.Second, Tick: func(context.Context) error { return nil }})
	s := &Server{loops: sup}

	rec := httptest.NewRecorder()
	s.handleFakLoops(rec, httptest.NewRequest(http.MethodGet, "/v1/fak/loops", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d want 200", rec.Code)
	}
	var resp BgloopsResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp.Schema != "fak.bgloops.v1" {
		t.Errorf("schema=%q want fak.bgloops.v1", resp.Schema)
	}
	if len(resp.Loops) != 1 || resp.Loops[0].Name != "heartbeat" {
		t.Fatalf("loops=%+v want one heartbeat", resp.Loops)
	}
}

func TestHandleFakLoopsRejectsNonGet(t *testing.T) {
	s := &Server{loops: bgloop.New()}
	rec := httptest.NewRecorder()
	s.handleFakLoops(rec, httptest.NewRequest(http.MethodPost, "/v1/fak/loops", nil))
	if rec.Code == http.StatusOK {
		t.Errorf("POST /v1/fak/loops should be rejected, got 200")
	}
}

func TestWriteBgloopMetricsFamily(t *testing.T) {
	sup := bgloop.New()
	_ = sup.Register(bgloop.Loop{Name: "heartbeat", Interval: time.Second, Tick: func(context.Context) error { return nil }})
	s := &Server{loops: sup}

	var b strings.Builder
	s.writeBgloopMetrics(&b)
	out := b.String()
	for _, want := range []string{
		"fak_bgloop_registered 1",
		`fak_bgloop_ticks_total{loop="heartbeat"}`,
		`fak_bgloop_up{loop="heartbeat"`,
	} {
		if !strings.Contains(out, want) {
			t.Errorf("bgloop metrics missing %q:\n%s", want, out)
		}
	}
}

func TestNewBgloopSupervisorRegistersHeartbeat(t *testing.T) {
	t.Setenv("FAK_BGLOOP_HEARTBEAT_S", "") // default interval -> heartbeat registered
	sup := newBgloopSupervisor(&Server{})
	if sup == nil {
		t.Fatal("newBgloopSupervisor returned nil")
	}
	if _, ok := sup.Get("heartbeat"); !ok {
		t.Error("expected a built-in heartbeat loop to be registered")
	}
}
