package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/fleetbus"
)

func TestFoldFleetControlRefusesUnreadableRoster(t *testing.T) {
	bus, err := fleetbus.OpenDir(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	d, refusal := fleetbus.NewDirective("issuer", "pause", "", fleetbus.Selector{All: true}, time.Minute, "test", time.Now())
	if refusal != nil {
		t.Fatal(refusal)
	}
	inst, refusal := fleetbus.NewInstance("serve-1", "host", "serve", 1, "", []fleetbus.Op{"pause"}, time.Now())
	if refusal != nil {
		t.Fatal(refusal)
	}
	if err := bus.Announce(inst); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(bus.Root, "instances", "serve-1.json")
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(path, 0o755); err != nil {
		t.Fatal(err)
	}
	_, err = foldFleetControl(bus, d, fleetbus.DefaultInstanceTTL)
	if err == nil || !strings.Contains(err.Error(), "read roster") {
		t.Fatalf("err=%v, want visible roster failure", err)
	}
}
func TestFoldFleetControlRefusesUnreadableAcknowledgements(t *testing.T) {
	bus, err := fleetbus.OpenDir(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	d, refusal := fleetbus.NewDirective("issuer", "pause", "", fleetbus.Selector{All: true}, time.Minute, "test", time.Now())
	if refusal != nil {
		t.Fatal(refusal)
	}
	path := filepath.Join(bus.Root, "acks", d.ID+".jsonl")
	if err := os.Mkdir(path, 0o755); err != nil {
		t.Fatal(err)
	}
	_, err = foldFleetControl(bus, d, fleetbus.DefaultInstanceTTL)
	if err == nil || !strings.Contains(err.Error(), "read acknowledgements") {
		t.Fatalf("err=%v, want visible ack failure", err)
	}
}
