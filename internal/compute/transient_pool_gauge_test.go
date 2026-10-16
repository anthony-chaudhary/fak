package compute

import "testing"

func TestTransientPoolGaugeTracksParkReuseDrain(t *testing.T) {
	var g transientPoolGauge
	g.parked(100)
	g.parked(300)
	g.reused(100)
	if n, b, p := g.TransientPoolStats(); n != 1 || b != 300 || p != 400 {
		t.Fatalf("after reuse = (%d,%d,%d), want (1,300,400)", n, b, p)
	}
	g.drained()
	if n, b, p := g.TransientPoolStats(); n != 0 || b != 0 || p != 400 {
		t.Fatalf("after drain = (%d,%d,%d), want (0,0,400)", n, b, p)
	}
}
