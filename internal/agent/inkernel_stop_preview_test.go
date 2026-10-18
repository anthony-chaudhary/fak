package agent

import (
	"errors"
	"reflect"
	"testing"
)

// fak-test:runtime fast est=50ms lane=default
func TestDFlash2StopPreviewIsPure(t *testing.T) {
	scanner := newIncrementalStopScanner(4, []string{"abcd", "cd"})
	scanner.appendPiece("xa")
	before := cloneDFlash2StopScanner(scanner)
	pieces := map[int]string{1: "b", 2: "c", 3: "d", 4: "ignored"}
	got, err := previewDFlash2Stop([]int{1, 2, 3, 4}, nil, scanner, func(ids []int) (string, error) { return pieces[ids[0]], nil })
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(scanner, before) || cap(got.scanner.tail) != cap(scanner.tail) || cap(got.scanner.hist) != cap(scanner.hist) {
		t.Fatal("preview changed source or capacities")
	}
	if !got.stopped || got.trim != 4 || !reflect.DeepEqual(got.tokens, []int{1, 2, 3}) {
		t.Fatalf("preview=%+v", got)
	}
	got.scanner.pushTail([]byte("changed"))
	if !reflect.DeepEqual(scanner, before) {
		t.Fatal("scanner alias")
	}
	empty := newIncrementalStopScanner(9, []string{"123456789"})
	copy := cloneDFlash2StopScanner(empty)
	if cap(copy.tail) != 9 || cap(copy.hist) != 9 {
		t.Fatal("empty capacity lost")
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestDFlash2StopPreviewBoundaries(t *testing.T) {
	for _, tc := range []struct {
		name    string
		tokens  []int
		stops   map[int]bool
		want    []int
		stopped bool
	}{
		{"eos-first", []int{0, 1}, map[int]bool{0: true}, nil, true},
		{"eos-middle", []int{1, 0, 2}, map[int]bool{0: true}, []int{1}, true},
		{"length", []int{1, 2}, nil, []int{1, 2}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			got, err := previewDFlash2Stop(tc.tokens, tc.stops, newIncrementalStopScanner(0, nil), func(ids []int) (string, error) { calls++; return "", nil })
			if err != nil || !reflect.DeepEqual(got.tokens, tc.want) || got.stopped != tc.stopped || calls != len(tc.want) {
				t.Fatalf("got=%+v calls=%d err=%v", got, calls, err)
			}
		})
	}
	s := newIncrementalStopScanner(len("αβ"), []string{"αβ"})
	first, err := previewDFlash2Stop([]int{1}, nil, s, func([]int) (string, error) { return "α", nil })
	if err != nil || first.stopped {
		t.Fatal(first, err)
	}
	second, err := previewDFlash2Stop([]int{2, 3}, nil, first.scanner, func([]int) (string, error) { return "β", nil })
	if err != nil || !second.stopped || second.trim != len("αβ") || len(second.tokens) != 1 {
		t.Fatal(second, err)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestDFlash2StopPreviewRefusesDecodeFailure(t *testing.T) {
	s := newIncrementalStopScanner(3, []string{"end"})
	want := errors.New("decode")
	got, err := previewDFlash2Stop([]int{1, 2}, nil, s, func(ids []int) (string, error) {
		if ids[0] == 2 {
			return "", want
		}
		return "e", nil
	})
	if !errors.Is(err, want) || got.scanner != nil || len(s.tail) != 0 {
		t.Fatalf("got=%+v err=%v source=%+v", got, err, s)
	}
}
