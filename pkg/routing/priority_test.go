package routing

import (
	"reflect"
	"testing"
)

func TestParsePriorityClass(t *testing.T) {
	cases := []struct {
		in   string
		want PriorityClass
	}{
		{"premium", PriorityClassPremium},
		{" Premium ", PriorityClassPremium},
		{"PREMIUM", PriorityClassPremium},
		{"flex", PriorityClassFlex},
		{" Flex ", PriorityClassFlex},
		{"FLEX", PriorityClassFlex},
		{"", PriorityClassNone},
		{"   ", PriorityClassNone},
		{"gold", PriorityClassNone},
		{"background", PriorityClassNone},
		{"premiumx", PriorityClassNone},
		{"flexible", PriorityClassNone},
	}
	for _, c := range cases {
		if got := ParsePriorityClass(c.in); got != c.want {
			t.Errorf("ParsePriorityClass(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestPriorityClassesVocabulary(t *testing.T) {
	want := []string{"", "premium", "flex"}
	if got := PriorityClasses(); !reflect.DeepEqual(got, want) {
		t.Fatalf("PriorityClasses() = %v, want %v", got, want)
	}
	// The copy is private: mutating it must not move the vocabulary.
	got := PriorityClasses()
	got[0] = "mutated"
	if again := PriorityClasses(); !reflect.DeepEqual(again, want) {
		t.Fatalf("vocabulary mutated through a returned copy: %v", again)
	}
}

func TestOrderForPriorityPeakPremiumTrailsFlex(t *testing.T) {
	members := []PriorityMember{
		{Name: "hive", Class: PriorityClassNone},
		{Name: "flex-a", Class: PriorityClassFlex},
		{Name: "strix-1", Class: PriorityClassPremium},
		{Name: "flex-b", Class: PriorityClassFlex},
	}
	got, applied := OrderForPriority(members, PriorityClassPremium, PeakSignal(true))
	want := []string{"hive", "strix-1", "flex-a", "flex-b"}
	if !applied || !reflect.DeepEqual(got, want) {
		t.Fatalf("order = %v applied=%v, want %v applied=true", got, applied, want)
	}
}

func TestOrderForPriorityPeakFlexKeepsDeclared(t *testing.T) {
	members := []PriorityMember{
		{Name: "flex-a", Class: PriorityClassFlex},
		{Name: "strix-1", Class: PriorityClassPremium},
	}
	got, applied := OrderForPriority(members, PriorityClassFlex, PeakSignal(true))
	if applied || !reflect.DeepEqual(got, []string{"flex-a", "strix-1"}) {
		t.Fatalf("order = %v applied=%v, want declared, applied=false", got, applied)
	}
}

func TestOrderForPriorityNoPeakKeepsDeclared(t *testing.T) {
	members := []PriorityMember{
		{Name: "flex-a", Class: PriorityClassFlex},
		{Name: "strix-1", Class: PriorityClassPremium},
	}
	got, applied := OrderForPriority(members, PriorityClassPremium, PeakSignal(false))
	if applied || !reflect.DeepEqual(got, []string{"flex-a", "strix-1"}) {
		t.Fatalf("order = %v applied=%v, want declared, applied=false", got, applied)
	}
}

func TestOrderForPriorityNoFlexMembersNoEffect(t *testing.T) {
	members := []PriorityMember{
		{Name: "hive", Class: PriorityClassNone},
		{Name: "strix-1", Class: PriorityClassPremium},
	}
	got, applied := OrderForPriority(members, PriorityClassPremium, PeakSignal(true))
	if applied || !reflect.DeepEqual(got, []string{"hive", "strix-1"}) {
		t.Fatalf("order = %v applied=%v, want declared, applied=false", got, applied)
	}
}

func TestOrderForPriorityEmptyMembers(t *testing.T) {
	got, applied := OrderForPriority(nil, PriorityClassPremium, PeakSignal(true))
	if applied || len(got) != 0 {
		t.Fatalf("order = %v applied=%v, want empty, applied=false", got, applied)
	}
}

func TestOrderForPriorityReplay(t *testing.T) {
	members := []PriorityMember{
		{Name: "flex-a", Class: PriorityClassFlex},
		{Name: "hive", Class: PriorityClassNone},
		{Name: "strix-1", Class: PriorityClassPremium},
		{Name: "flex-b", Class: PriorityClassFlex},
	}
	first, applied := OrderForPriority(members, PriorityClassPremium, PeakSignal(true))
	if !applied {
		t.Fatalf("first replay did not take effect")
	}
	for i := 0; i < 100; i++ {
		again, applied := OrderForPriority(members, PriorityClassPremium, PeakSignal(true))
		if !applied || !reflect.DeepEqual(again, first) {
			t.Fatalf("replay %d = %v applied=%v, want %v applied=true", i, again, applied, first)
		}
	}
	// The input is never mutated: the declared order still reads back.
	declared := []string{"flex-a", "hive", "strix-1", "flex-b"}
	for i, m := range members {
		if m.Name != declared[i] {
			t.Fatalf("input mutated at %d: %v", i, members)
		}
	}
}
