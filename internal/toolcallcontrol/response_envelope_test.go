package toolcallcontrol

import "testing"

func TestAccountResponseRecordsActualsAndBranchesByIndependentDimension(t *testing.T) {
	payload := []byte(`[{"id":1},{"id":2}]`)
	limits := []ResponseLimit{
		{Dimension: ResponseItems, Maximum: 2},
		{Dimension: ResponseBytes, Maximum: int64(len(payload))},
		{Dimension: ResponseTokensEstimated, Maximum: 5},
	}

	tests := []struct {
		name       string
		limits     []ResponseLimit
		want       ResponseDisposition
		wantExceed []ResponseDimension
	}{
		{name: "at every ceiling passes", limits: limits, want: ResponsePass},
		{name: "items", limits: replaceResponseLimit(limits, ResponseItems, 1), want: ResponseBranch, wantExceed: []ResponseDimension{ResponseItems}},
		{name: "bytes", limits: replaceResponseLimit(limits, ResponseBytes, int64(len(payload))-1), want: ResponseBranch, wantExceed: []ResponseDimension{ResponseBytes}},
		{name: "estimated tokens", limits: replaceResponseLimit(limits, ResponseTokensEstimated, 4), want: ResponseBranch, wantExceed: []ResponseDimension{ResponseTokensEstimated}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := AccountResponse(payload, 2, tc.limits)
			if err != nil {
				t.Fatal(err)
			}
			if got.Disposition != tc.want {
				t.Fatalf("disposition=%q want=%q receipt=%+v", got.Disposition, tc.want, got)
			}
			if len(got.Exceeded) != len(tc.wantExceed) {
				t.Fatalf("exceeded=%v want=%v", got.Exceeded, tc.wantExceed)
			}
			for i := range tc.wantExceed {
				if got.Exceeded[i] != tc.wantExceed[i] {
					t.Fatalf("exceeded=%v want=%v", got.Exceeded, tc.wantExceed)
				}
			}
			if got.Actual.Items != 2 || got.Actual.Bytes != 19 || got.Actual.TokensEstimated != 5 {
				t.Fatalf("actual=%+v", got.Actual)
			}
			if got.Actual.TokenEstimateBasis != ResponseTokenEstimateBytesDiv4Ceil {
				t.Fatalf("token estimate basis=%q", got.Actual.TokenEstimateBasis)
			}
		})
	}
}

func TestAccountResponseReportsEveryExceededDimensionInStableOrder(t *testing.T) {
	got, err := AccountResponse([]byte("12345"), 3, []ResponseLimit{
		{Dimension: ResponseTokensEstimated, Maximum: 1},
		{Dimension: ResponseBytes, Maximum: 4},
		{Dimension: ResponseItems, Maximum: 2},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []ResponseDimension{ResponseItems, ResponseBytes, ResponseTokensEstimated}
	if len(got.Exceeded) != len(want) {
		t.Fatalf("exceeded=%v want=%v", got.Exceeded, want)
	}
	for i := range want {
		if got.Exceeded[i] != want[i] {
			t.Fatalf("exceeded=%v want=%v", got.Exceeded, want)
		}
	}
}

func TestAccountResponseRejectsAmbiguousLimitsAndCounts(t *testing.T) {
	tests := []struct {
		name   string
		items  int64
		limits []ResponseLimit
	}{
		{name: "negative actual items", items: -1},
		{name: "negative maximum", limits: []ResponseLimit{{Dimension: ResponseBytes, Maximum: -1}}},
		{name: "unknown dimension", limits: []ResponseLimit{{Dimension: ResponseDimension("rows"), Maximum: 1}}},
		{name: "duplicate dimension", limits: []ResponseLimit{{Dimension: ResponseItems, Maximum: 1}, {Dimension: ResponseItems, Maximum: 2}}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := AccountResponse([]byte("x"), tc.items, tc.limits); err == nil {
				t.Fatal("expected error")
			}
		})
	}
}

func replaceResponseLimit(in []ResponseLimit, dimension ResponseDimension, maximum int64) []ResponseLimit {
	out := append([]ResponseLimit(nil), in...)
	for i := range out {
		if out[i].Dimension == dimension {
			out[i].Maximum = maximum
		}
	}
	return out
}
