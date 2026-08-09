package main

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/fleetbus"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
	"github.com/anthony-chaudhary/fak/internal/session"
)

type fakeGwBusApplier struct {
	witness string
	changed bool
	err     error
	calls   int
}

func (f *fakeGwBusApplier) ReloadRoute() (string, bool, error) {
	f.calls++
	return f.witness, f.changed, f.err
}

func TestFleetBusGatewayApplier(t *testing.T) {
	for _, tc := range []struct {
		name       string
		gateway    gwBusApplier
		wantStatus fleetbus.AckStatus
		wantReason fleetbus.RefuseReason
		wantPrefix string
		wantCount  int
	}{
		{name: "changed", gateway: &fakeGwBusApplier{witness: "source=routes.json reloads=2", changed: true}, wantStatus: fleetbus.AckApplied, wantPrefix: "gateway-route-reloaded:", wantCount: 1},
		{name: "unchanged", gateway: &fakeGwBusApplier{witness: "source=routes.json reloads=2"}, wantStatus: fleetbus.AckApplied, wantPrefix: "gateway-route-unchanged:", wantCount: 0},
		{name: "not configured", wantStatus: fleetbus.AckRefused, wantReason: fleetbus.ApplyRefused},
		{name: "reload rejected", gateway: &fakeGwBusApplier{err: errors.New("last-good kept")}, wantStatus: fleetbus.AckRefused, wantReason: fleetbus.ApplyRefused},
		{name: "hollow witness", gateway: &fakeGwBusApplier{changed: true}, wantStatus: fleetbus.AckRefused, wantReason: fleetbus.ApplyRefused},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out := (&fleetBusApplier{gateway: tc.gateway}).Apply(fleetbus.Directive{Op: gwBusReloadRoute})
			if out.Status != tc.wantStatus || out.Reason != tc.wantReason || out.Affected != tc.wantCount {
				t.Fatalf("outcome = %#v, want status=%s reason=%s affected=%d", out, tc.wantStatus, tc.wantReason, tc.wantCount)
			}
			if tc.wantPrefix != "" && !strings.HasPrefix(out.Witness, tc.wantPrefix) {
				t.Fatalf("witness = %q, want prefix %q", out.Witness, tc.wantPrefix)
			}
		})
	}
}

func TestFleetBusGatewayApplierMutatesRealRouteWatcher(t *testing.T) {
	path := filepath.Join(t.TempDir(), "routes.json")
	write := func(model string) {
		t.Helper()
		manifest := modelroute.Manifest{
			Version: modelroute.Version,
			Default: modelroute.Plan{Members: []modelroute.Member{{Model: "default", Role: "primary"}}},
			Rules: []modelroute.Rule{{
				Name:  "route-x",
				Match: modelroute.Match{Aspect: modelroute.AspectToolCall, Tool: "x"},
				Plan:  modelroute.Plan{Members: []modelroute.Member{{Model: model}}},
			}},
		}
		if err := os.WriteFile(path, manifest.JSON(), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	write("alpha")
	loaded, err := modelroute.LoadManifest(path)
	if err != nil {
		t.Fatal(err)
	}
	live := modelroute.NewLive(&loaded)
	srv := &gateway.Server{}
	srv.SetRouteWatcher(modelroute.NewWatcher(path, live, 0, nil))
	write("beta")

	out := (&fleetBusApplier{gateway: serveGwBusApplier{srv: srv}}).Apply(fleetbus.Directive{Op: gwBusReloadRoute})
	if out.Status != fleetbus.AckApplied || out.Affected != 1 || !strings.Contains(out.Witness, "reloads=1") {
		t.Fatalf("outcome = %#v, want a witnessed real route swap", out)
	}
	decision := live.Route(modelroute.Subject{Aspect: modelroute.AspectToolCall, Tool: "x"})
	if len(decision.Plan.Members) != 1 || decision.Plan.Members[0].Model != "beta" {
		t.Fatalf("live route = %#v, want beta after fleet apply", decision)
	}
}

func TestFleetBusGatewayAndSessionVocabulariesAreDisjoint(t *testing.T) {
	gw := &fakeGwBusApplier{witness: "route", changed: true}
	tbl := session.NewTable()
	tbl.Restore("s1", session.State{TraceID: "s1", Run: session.Running})
	ap := &fleetBusApplier{tbl: tbl, gateway: gw}

	out := ap.Apply(fleetbus.Directive{Op: "pause", Selector: fleetbus.Selector{All: true}})
	if out.Status != fleetbus.AckApplied {
		t.Fatalf("session op refused: %#v", out)
	}
	if gw.calls != 0 {
		t.Fatalf("session op reached gateway applier %d time(s)", gw.calls)
	}
	if got := tbl.Get("s1").Run; got != session.Paused {
		t.Fatalf("session state = %v, want paused", got)
	}

	out = ap.Apply(fleetbus.Directive{Op: gwBusReloadRoute})
	if out.Status != fleetbus.AckApplied || gw.calls != 1 {
		t.Fatalf("gateway op outcome=%#v calls=%d", out, gw.calls)
	}
	if got := tbl.Get("s1").Run; got != session.Paused {
		t.Fatalf("gateway op mutated session state to %v", got)
	}
}

func TestFleetBusGatewayApplierDirBusFold(t *testing.T) {
	bus, err := fleetbus.OpenDir(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	now := time.Date(2026, 8, 6, 12, 0, 0, 0, time.UTC)
	instances := make([]fleetbus.Instance, 0, 2)
	appliers := make([]*fleetBusApplier, 0, 2)
	for i := 1; i <= 2; i++ {
		id := fmt.Sprintf("gateway-%d", i)
		inst, refusal := fleetbus.NewInstance(id, "test-host", "serve", i, "", fleetBusAdvertisedOps(), now)
		if refusal != nil {
			t.Fatalf("new instance %s: %v", id, refusal)
		}
		if err := bus.Announce(inst); err != nil {
			t.Fatalf("announce %s: %v", id, err)
		}
		instances = append(instances, inst)
		appliers = append(appliers, &fleetBusApplier{gateway: &fakeGwBusApplier{
			witness: fmt.Sprintf("instance=%s source=routes.json generation=2", id),
			changed: true,
		}})
	}

	d, refusal := fleetbus.NewDirective("test-issuer", gwBusReloadRoute, "", fleetbus.Selector{All: true}, time.Minute, "test route fanout", now)
	if refusal != nil {
		t.Fatal(refusal)
	}
	d.Targets = []string{instances[0].ID, instances[1].ID}
	if err := bus.Publish(d); err != nil {
		t.Fatal(err)
	}
	for i := range instances {
		rep, err := fleetbus.Drain(bus, instances[i], appliers[i], now.Add(time.Second))
		if err != nil {
			t.Fatalf("drain %s: %v", instances[i].ID, err)
		}
		if rep.Applied != 1 || len(rep.Acks) != 1 {
			t.Fatalf("drain %s = %#v, want one applied ack", instances[i].ID, rep)
		}
	}
	acks, err := bus.Acks(d.ID)
	if err != nil {
		t.Fatal(err)
	}
	report := fleetbus.Fold(d, instances, acks, now.Add(2*time.Second))
	if report.Applied != 2 || report.Outstanding != 0 || !report.Complete {
		t.Fatalf("fold = %#v, want applied=2 outstanding=0 complete", report)
	}
	for _, row := range report.Rows {
		if row.Status != fleetbus.RowApplied || !strings.HasPrefix(row.Witness, "gateway-route-reloaded:") || row.Affected != 1 {
			t.Fatalf("row = %#v, want witnessed gateway application", row)
		}
	}
}

func TestFleetBusGatewayApplierDirBusRefusal(t *testing.T) {
	bus, err := fleetbus.OpenDir(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	now := time.Date(2026, 8, 6, 12, 0, 0, 0, time.UTC)
	inst, refusal := fleetbus.NewInstance("gateway-refuses", "test-host", "serve", 1, "", fleetBusAdvertisedOps(), now)
	if refusal != nil {
		t.Fatal(refusal)
	}
	if err := bus.Announce(inst); err != nil {
		t.Fatal(err)
	}
	d, refusal := fleetbus.NewDirective("test-issuer", gwBusReloadRoute, "", fleetbus.Selector{All: true}, time.Minute, "test refusal", now)
	if refusal != nil {
		t.Fatal(refusal)
	}
	d.Targets = []string{inst.ID}
	if err := bus.Publish(d); err != nil {
		t.Fatal(err)
	}
	if _, err := fleetbus.Drain(bus, inst, &fleetBusApplier{}, now.Add(time.Second)); err != nil {
		t.Fatal(err)
	}
	acks, err := bus.Acks(d.ID)
	if err != nil {
		t.Fatal(err)
	}
	report := fleetbus.Fold(d, []fleetbus.Instance{inst}, acks, now.Add(2*time.Second))
	if report.Refused != 1 || report.Outstanding != 0 || len(report.Rows) != 1 {
		t.Fatalf("fold = %#v, want refused=1 outstanding=0", report)
	}
	if row := report.Rows[0]; row.Status != fleetbus.RowRefused || row.Reason != fleetbus.ApplyRefused || !strings.Contains(row.Detail, "not configured") {
		t.Fatalf("refusal row = %#v, want closed visible refusal", row)
	}
}
