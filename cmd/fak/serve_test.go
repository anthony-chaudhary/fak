package main

import (
	"reflect"
	"testing"
)

func TestRepeatedStringFlagAccumulatesTrimmedValues(t *testing.T) {
	var f repeatedStringFlag
	if err := f.Set(" http://127.0.0.1:8001/v1 "); err != nil {
		t.Fatalf("Set first: %v", err)
	}
	if err := f.Set("http://127.0.0.1:8002/v1"); err != nil {
		t.Fatalf("Set second: %v", err)
	}
	want := []string{"http://127.0.0.1:8001/v1", "http://127.0.0.1:8002/v1"}
	if got := f.Values(); !reflect.DeepEqual(got, want) {
		t.Fatalf("Values() = %v, want %v", got, want)
	}
	if got := f.String(); got != "http://127.0.0.1:8001/v1,http://127.0.0.1:8002/v1" {
		t.Fatalf("String() = %q", got)
	}
	got := f.Values()
	got[0] = "mutated"
	if again := f.Values(); !reflect.DeepEqual(again, want) {
		t.Fatalf("Values() returned internal storage: %v", again)
	}
}

func TestRepeatedStringFlagRejectsEmptyValue(t *testing.T) {
	var f repeatedStringFlag
	if err := f.Set(" \t "); err == nil {
		t.Fatal("Set blank value succeeded, want error")
	}
}
