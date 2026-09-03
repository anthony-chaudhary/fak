package resulttier

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

func TestDefaultTierAssignment(t *testing.T) {
	// Zero and negative limits default to 5.
	for _, lim := range []int{0, -1, -5, -100} {
		req := PaginationRequest{Limit: lim}
		resolvedLim, offset, cont, err := ResolvePagination(req, 20)
		if err != nil {
			t.Fatalf("unexpected error for limit %d: %v", lim, err)
		}
		if resolvedLim != 5 {
			t.Fatalf("expected limit %d to default to 5, got %d", lim, resolvedLim)
		}
		if offset != 0 {
			t.Fatalf("expected offset 0, got %d", offset)
		}
		if cont.Tier != 5 {
			t.Fatalf("expected continuation Tier 5, got %d", cont.Tier)
		}
	}

	// Verify Slice with 0 and negative limits defaults to 5 items.
	items := make([]int, 20)
	for i := range items {
		items[i] = i
	}
	for _, lim := range []int{0, -1, -50} {
		sliced, cont, err := Slice(items, PaginationRequest{Limit: lim})
		if err != nil {
			t.Fatalf("Slice failed for limit %d: %v", lim, err)
		}
		if len(sliced) != 5 {
			t.Fatalf("expected Slice to return 5 items for limit %d, got %d", lim, len(sliced))
		}
		if cont.Tier != 5 {
			t.Fatalf("expected Slice continuation Tier 5, got %d", cont.Tier)
		}
	}
}

func TestStandardTiers(t *testing.T) {
	if Tier1 != 1 {
		t.Fatalf("expected Tier1=1, got %d", Tier1)
	}
	if Tier3 != 3 {
		t.Fatalf("expected Tier3=3, got %d", Tier3)
	}
	if Tier5 != 5 {
		t.Fatalf("expected Tier5=5, got %d", Tier5)
	}
	if Tier10 != 10 {
		t.Fatalf("expected Tier10=10, got %d", Tier10)
	}
	if DefaultTier != Tier5 {
		t.Fatalf("expected DefaultTier=Tier5 (5), got %d", DefaultTier)
	}
	if MaxStandardTier != Tier10 {
		t.Fatalf("expected MaxStandardTier=Tier10 (10), got %d", MaxStandardTier)
	}
	if Tier5.Int() != 5 {
		t.Fatalf("expected Tier5.Int()=5, got %d", Tier5.Int())
	}

	// Verify 1, 3, 5, 10 pass without widening reason.
	items := make([]int, 20)
	for i := range items {
		items[i] = i
	}
	for _, lim := range []int{1, 3, 5, 10} {
		req := PaginationRequest{Limit: lim}
		resolvedLim, _, cont, err := ResolvePagination(req, 20)
		if err != nil {
			t.Fatalf("expected limit %d to pass without widening reason: %v", lim, err)
		}
		if resolvedLim != lim {
			t.Fatalf("expected limit %d, got %d", lim, resolvedLim)
		}
		if cont.Tier != lim {
			t.Fatalf("expected continuation tier %d, got %d", lim, cont.Tier)
		}

		sliced, sliceCont, err := Slice(items, req)
		if err != nil {
			t.Fatalf("Slice failed for limit %d: %v", lim, err)
		}
		if len(sliced) != lim {
			t.Fatalf("expected Slice to return %d items, got %d", lim, len(sliced))
		}
		if sliceCont.Tier != lim {
			t.Fatalf("expected Slice continuation tier %d, got %d", lim, sliceCont.Tier)
		}
	}
}

func TestWideningRequiresReason(t *testing.T) {
	const wantErr = "TIER_WIDENING_REFUSED: limits > 10 require an explicit widening_reason. Inspect the initial bounded view (default tier 5) and paginate with offset/cursor"
	items := make([]int, 50)
	for _, lim := range []int{11, 15, 20, 50, 100} {
		// Empty widening reason
		req := PaginationRequest{Limit: lim}
		_, _, _, err := ResolvePagination(req, 20)
		if err == nil {
			t.Fatalf("expected error for limit %d without reason", lim)
		}
		if !errors.Is(err, ErrTierWideningRefused) {
			t.Fatalf("expected ErrTierWideningRefused, got %v", err)
		}
		if !strings.Contains(err.Error(), "TIER_WIDENING_REFUSED") {
			t.Fatalf("expected error to contain TIER_WIDENING_REFUSED, got %q", err.Error())
		}
		if err.Error() != wantErr {
			t.Fatalf("error message mismatch: got %q, want %q", err.Error(), wantErr)
		}

		// Whitespace-only reason also refused
		reqWhitespace := PaginationRequest{Limit: lim, WideningReason: "   \t\n  "}
		_, _, _, err = ResolvePagination(reqWhitespace, 20)
		if err == nil || !errors.Is(err, ErrTierWideningRefused) {
			t.Fatalf("expected ErrTierWideningRefused for whitespace reason, got %v", err)
		}

		// Slice should also fail with ErrTierWideningRefused
		_, _, err = Slice(items, req)
		if err == nil || !errors.Is(err, ErrTierWideningRefused) {
			t.Fatalf("expected Slice to fail with ErrTierWideningRefused for limit %d, got %v", lim, err)
		}
	}
}

func TestWideningWithReason(t *testing.T) {
	items := make([]int, 150)
	for i := range items {
		items[i] = i
	}
	for _, lim := range []int{11, 15, 20, 50, 100} {
		req := PaginationRequest{
			Limit:          lim,
			WideningReason: "large batch audit export",
		}
		resolvedLim, _, cont, err := ResolvePagination(req, 200)
		if err != nil {
			t.Fatalf("unexpected error for limit %d with reason: %v", lim, err)
		}
		if resolvedLim != lim {
			t.Fatalf("expected resolved limit %d, got %d", lim, resolvedLim)
		}
		if cont.Tier != lim {
			t.Fatalf("expected continuation tier %d, got %d", lim, cont.Tier)
		}

		sliced, sliceCont, err := Slice(items, req)
		if err != nil {
			t.Fatalf("Slice failed for limit %d with reason: %v", lim, err)
		}
		if len(sliced) != lim {
			t.Fatalf("expected Slice to return %d items, got %d", lim, len(sliced))
		}
		if sliceCont.Tier != lim {
			t.Fatalf("expected continuation tier %d, got %d", lim, sliceCont.Tier)
		}
	}
}

func TestSliceGeneric(t *testing.T) {
	items := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}

	// Page 1: default tier 5
	page1, cont1, err := Slice(items, PaginationRequest{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(page1) != 5 {
		t.Fatalf("expected 5 items, got %d", len(page1))
	}
	for i, v := range page1 {
		if v != i {
			t.Fatalf("expected item %d, got %d", i, v)
		}
	}
	if !cont1.HasMore || cont1.NextCursor != "5" {
		t.Fatalf("expected HasMore=true, NextCursor='5', got %+v", cont1)
	}

	// Page 2: using NextCursor from page 1
	page2, cont2, err := Slice(items, PaginationRequest{Cursor: cont1.NextCursor})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(page2) != 5 {
		t.Fatalf("expected 5 items, got %d", len(page2))
	}
	for i, v := range page2 {
		if v != i+5 {
			t.Fatalf("expected item %d, got %d", i+5, v)
		}
	}
	if !cont2.HasMore || cont2.NextCursor != "10" {
		t.Fatalf("expected HasMore=true, NextCursor='10', got %+v", cont2)
	}

	// Page 3: using NextCursor from page 2 (last page: 2 remaining items)
	page3, cont3, err := Slice(items, PaginationRequest{Cursor: cont2.NextCursor})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(page3) != 2 {
		t.Fatalf("expected 2 items, got %d", len(page3))
	}
	if page3[0] != 10 || page3[1] != 11 {
		t.Fatalf("expected items 10, 11, got %v", page3)
	}
	if cont3.HasMore || cont3.NextCursor != "" {
		t.Fatalf("expected HasMore=false, empty NextCursor, got %+v", cont3)
	}

	// Pagination using explicit Offset
	offPage, offCont, err := Slice(items, PaginationRequest{Offset: 5, Limit: 5})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(offPage) != 5 {
		t.Fatalf("expected 5 items, got %d", len(offPage))
	}
	for i, v := range offPage {
		if v != i+5 {
			t.Fatalf("expected item %d, got %d", i+5, v)
		}
	}
	if offCont.Offset != 5 || offCont.NextCursor != "10" || !offCont.HasMore {
		t.Fatalf("unexpected continuation with explicit offset: %+v", offCont)
	}

	// String slice generic check
	strItems := []string{"alpha", "beta", "gamma", "delta"}
	strPage, strCont, err := Slice(strItems, PaginationRequest{Limit: 3})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(strPage) != 3 || strPage[2] != "gamma" {
		t.Fatalf("expected ['alpha', 'beta', 'gamma'], got %v", strPage)
	}
	if !strCont.HasMore || strCont.NextCursor != "3" {
		t.Fatalf("expected HasMore=true, NextCursor='3', got %+v", strCont)
	}

	// Empty and nil slices
	var nilSlice []string
	nilRes, _, err := Slice(nilSlice, PaginationRequest{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if nilRes != nil {
		t.Fatalf("expected nil result for nil slice, got %v", nilRes)
	}

	emptySlice := []string{}
	emptyRes, _, err := Slice(emptySlice, PaginationRequest{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(emptyRes) != 0 {
		t.Fatalf("expected empty result, got %v", emptyRes)
	}
}

func TestCursorFormat(t *testing.T) {
	// 1. Cursor encoding via ResolvePagination:
	req := PaginationRequest{Limit: 5, Offset: 0}
	_, _, cont, err := ResolvePagination(req, 10)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cont.NextCursor != "5" {
		t.Fatalf("expected NextCursor '5', got %q", cont.NextCursor)
	}

	reqLast := PaginationRequest{Limit: 5, Offset: 5}
	_, _, contLast, err := ResolvePagination(reqLast, 10)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if contLast.NextCursor != "" {
		t.Fatalf("expected empty NextCursor on last page, got %q", contLast.NextCursor)
	}

	// 2. Cursor decoding formats in parseCursor
	testCases := []struct {
		cursor  string
		wantOff int
		wantErr bool
	}{
		{"", 0, false},
		{"   ", 0, false},
		{"0", 0, false},
		{"5", 5, false},
		{"42", 42, false},
		{"offset:10", 10, false},
		{"OFFSET:25", 25, false},
		{"cursor:15", 15, false},
		{"CURSOR:30", 30, false},
		{base64.StdEncoding.EncodeToString([]byte("0")), 0, false},
		{base64.StdEncoding.EncodeToString([]byte("50")), 50, false},
		{"-1", 0, true},
		{"-10", 0, true},
		{"offset:-5", 0, true},
		{"cursor:-8", 0, true},
		{base64.StdEncoding.EncodeToString([]byte("-10")), 0, true},
		{"not-a-number", 0, true},
		{"offset:abc", 0, true},
		{"cursor:xyz", 0, true},
	}

	for _, tc := range testCases {
		got, err := parseCursor(tc.cursor)
		if tc.wantErr {
			if err == nil {
				t.Errorf("parseCursor(%q): expected error, got nil", tc.cursor)
			}
		} else {
			if err != nil {
				t.Errorf("parseCursor(%q): unexpected error: %v", tc.cursor, err)
			}
			if got != tc.wantOff {
				t.Errorf("parseCursor(%q): got offset %d, want %d", tc.cursor, got, tc.wantOff)
			}
		}
	}

	// 3. Decoding cursor through ResolvePagination requests
	for _, tc := range []struct {
		cursor  string
		wantOff int
	}{
		{"5", 5},
		{"offset:10", 10},
		{"cursor:15", 15},
		{base64.StdEncoding.EncodeToString([]byte("8")), 8},
	} {
		pReq := PaginationRequest{Limit: 5, Cursor: tc.cursor}
		_, off, _, err := ResolvePagination(pReq, 30)
		if err != nil {
			t.Fatalf("ResolvePagination failed for cursor %q: %v", tc.cursor, err)
		}
		if off != tc.wantOff {
			t.Fatalf("ResolvePagination for cursor %q: got offset %d, want %d", tc.cursor, off, tc.wantOff)
		}
	}

	badReq := PaginationRequest{Cursor: "invalid-cursor"}
	_, _, _, err = ResolvePagination(badReq, 30)
	if err == nil || !strings.Contains(err.Error(), "invalid cursor") {
		t.Fatalf("expected invalid cursor error from ResolvePagination, got %v", err)
	}
}

func TestResolvePaginationBoundariesAndContinuation(t *testing.T) {
	// Total 10, Limit 5, Offset 0 -> first page
	req := PaginationRequest{Limit: 5, Offset: 0}
	lim, off, cont, err := ResolvePagination(req, 10)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if lim != 5 || off != 0 {
		t.Fatalf("expected lim=5, off=0, got lim=%d, off=%d", lim, off)
	}
	if cont.TotalAvailable != 10 {
		t.Fatalf("expected TotalAvailable=10, got %d", cont.TotalAvailable)
	}
	if cont.ReturnedCount != 5 {
		t.Fatalf("expected ReturnedCount=5, got %d", cont.ReturnedCount)
	}
	if !cont.HasMore {
		t.Fatalf("expected HasMore=true")
	}
	if cont.NextCursor != "5" {
		t.Fatalf("expected NextCursor='5', got %q", cont.NextCursor)
	}
	if !cont.Truncated {
		t.Fatalf("expected Truncated=true")
	}

	// Total 10, Limit 5, Offset 5 -> second / last page
	req = PaginationRequest{Limit: 5, Offset: 5}
	lim, off, cont, err = ResolvePagination(req, 10)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if lim != 5 || off != 5 {
		t.Fatalf("expected lim=5, off=5, got lim=%d, off=%d", lim, off)
	}
	if cont.ReturnedCount != 5 {
		t.Fatalf("expected ReturnedCount=5, got %d", cont.ReturnedCount)
	}
	if cont.HasMore {
		t.Fatalf("expected HasMore=false on last page")
	}
	if cont.NextCursor != "" {
		t.Fatalf("expected empty NextCursor on last page, got %q", cont.NextCursor)
	}
	if !cont.Truncated {
		t.Fatalf("expected Truncated=true because total (10) > limit (5)")
	}

	// Small collection (total 3 <= default limit 5) -> not truncated, no next page
	req = PaginationRequest{}
	lim, off, cont, err = ResolvePagination(req, 3)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if lim != 5 || off != 0 {
		t.Fatalf("expected lim=5, off=0, got lim=%d, off=%d", lim, off)
	}
	if cont.TotalAvailable != 3 || cont.ReturnedCount != 3 {
		t.Fatalf("expected TotalAvailable=3, ReturnedCount=3, got total=%d, returned=%d", cont.TotalAvailable, cont.ReturnedCount)
	}
	if cont.HasMore {
		t.Fatalf("expected HasMore=false for small collection")
	}
	if cont.Truncated {
		t.Fatalf("expected Truncated=false for small collection")
	}
	if cont.NextCursor != "" {
		t.Fatalf("expected empty NextCursor, got %q", cont.NextCursor)
	}

	// Empty collection (total = 0)
	req = PaginationRequest{}
	_, _, cont, err = ResolvePagination(req, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cont.TotalAvailable != 0 || cont.ReturnedCount != 0 || cont.HasMore || cont.Truncated {
		t.Fatalf("expected all zeros/falses for empty collection, got %+v", cont)
	}

	// Negative total treated as 0
	_, _, cont, err = ResolvePagination(req, -5)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cont.TotalAvailable != 0 || cont.ReturnedCount != 0 {
		t.Fatalf("expected TotalAvailable=0 for negative total, got %+v", cont)
	}

	// Offset beyond total
	req = PaginationRequest{Limit: 5, Offset: 20}
	_, off, cont, err = ResolvePagination(req, 10)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if off != 20 {
		t.Fatalf("expected off=20, got %d", off)
	}
	if cont.ReturnedCount != 0 {
		t.Fatalf("expected ReturnedCount=0 when offset > total, got %d", cont.ReturnedCount)
	}
	if cont.HasMore {
		t.Fatalf("expected HasMore=false when offset > total")
	}

	// Negative offset clamped to 0
	req = PaginationRequest{Limit: 5, Offset: -3}
	_, off, cont, err = ResolvePagination(req, 10)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if off != 0 || cont.Offset != 0 {
		t.Fatalf("expected negative offset clamped to 0, got off=%d cont.Offset=%d", off, cont.Offset)
	}
}

func TestJSONEncoding(t *testing.T) {
	req := PaginationRequest{
		Limit:          5,
		Offset:         10,
		Cursor:         "10",
		WideningReason: "audit",
	}
	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}
	var decodedReq PaginationRequest
	if err := json.Unmarshal(data, &decodedReq); err != nil {
		t.Fatalf("json.Unmarshal failed: %v", err)
	}
	if decodedReq != req {
		t.Fatalf("decoded request mismatch: got %+v, want %+v", decodedReq, req)
	}

	resp := ContinuationResponse{
		TotalAvailable: 100,
		ReturnedCount:  5,
		Offset:         0,
		NextCursor:     "5",
		HasMore:        true,
		Tier:           5,
		Truncated:      true,
	}
	respData, err := json.Marshal(resp)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}
	var decodedResp ContinuationResponse
	if err := json.Unmarshal(respData, &decodedResp); err != nil {
		t.Fatalf("json.Unmarshal failed: %v", err)
	}
	if decodedResp != resp {
		t.Fatalf("decoded response mismatch: got %+v, want %+v", decodedResp, resp)
	}
}
