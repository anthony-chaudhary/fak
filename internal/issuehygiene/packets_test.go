package issuehygiene

import (
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

func packetFixture() scorecard.Payload {
	return scorecard.Payload{Schema: Schema, KPIs: []scorecard.KPI{
		{Key: "priority_coverage", Defects: []string{"#12 z", "#2 b"}},
		{Key: "class_coverage", Defects: []string{"#12 a", "#7 c", "#2 b"}},
	}}
}

func TestPlanPacketsDefaultIsStableDisjointAndComplete(t *testing.T) {
	first, err := PlanPackets(packetFixture(), PacketOptions{})
	if err != nil {
		t.Fatal(err)
	}
	second, err := PlanPackets(packetFixture(), PacketOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("planner is not deterministic:\nfirst=%#v\nsecond=%#v", first, second)
	}
	if first.PacketSize != DefaultPacketSize || len(first.Packets) != 3 {
		t.Fatalf("default bound changed: size=%d packets=%d", first.PacketSize, len(first.Packets))
	}
	wantIssues := [][]int{{2}, {7}, {12}}
	seenIssues := map[int]bool{}
	seenDefects := map[string]bool{}
	for i, packet := range first.Packets {
		if !reflect.DeepEqual(packet.Issues, wantIssues[i]) {
			t.Fatalf("packet %d issues=%v, want %v", i, packet.Issues, wantIssues[i])
		}
		for _, issue := range packet.Issues {
			if seenIssues[issue] {
				t.Fatalf("issue #%d appears in overlapping packets", issue)
			}
			seenIssues[issue] = true
		}
		for _, defect := range packet.Defects {
			if seenDefects[defect] {
				t.Fatalf("defect %q appears more than once", defect)
			}
			seenDefects[defect] = true
		}
	}
	if first.IssueCount != 3 || first.DefectCount != 4 || len(seenDefects) != 4 {
		t.Fatalf("coverage mismatch: plan=%+v seen=%v", first, seenDefects)
	}
}

func TestPlanPacketsOversizedFailsClosed(t *testing.T) {
	_, err := PlanPackets(packetFixture(), PacketOptions{PacketSize: 2})
	if err == nil || !strings.Contains(err.Error(), "--unsafe-oversized") {
		t.Fatalf("unsafe packet size was not rejected: %v", err)
	}
	_, err = PlanPackets(packetFixture(), PacketOptions{PacketSize: 2, UnsafeOversized: true, PriceRef: "dos-price"})
	if err == nil || !strings.Contains(err.Error(), "price and review") {
		t.Fatalf("missing review gate was not rejected: %v", err)
	}

	plan, err := PlanPackets(packetFixture(), PacketOptions{
		PacketSize: 2, UnsafeOversized: true,
		PriceRef: "dispatch-price:sha256:abc", ReviewRef: "operator-review:10356",
	})
	if err != nil {
		t.Fatal(err)
	}
	if !plan.UnsafeOverride || plan.PriceRef == "" || plan.ReviewRef == "" || len(plan.Packets) != 2 {
		t.Fatalf("oversized metadata missing: %+v", plan)
	}
}

func TestPlanPacketsRejectsNonIssueDefect(t *testing.T) {
	_, err := PlanPackets(scorecard.Payload{Schema: Schema, KPIs: []scorecard.KPI{{Defects: []string{"global defect"}}}}, PacketOptions{})
	if err == nil {
		t.Fatal("non-issue defect accepted")
	}
}
