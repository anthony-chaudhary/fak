package routing

import "testing"

func TestExhaustedWalkRetryable(t *testing.T) {
	t.Parallel()
	transient := Failure{Class: FailureUnavailable, Action: ActionFailoverNext}
	rateLimited := Failure{Class: FailureRateLimited, Action: ActionRetrySame}
	fatal := Failure{Class: FailureBadPayload, Action: ActionFatal}
	modelMissing := Failure{Class: FailureModelMissing, Action: ActionFailoverNext}
	creditWall := Failure{Class: FailureCreditExhausted, Action: ActionFailoverNext}
	cases := []struct {
		name     string
		failures []Failure
		starved  int
		want     bool
	}{
		{name: "empty walk", want: false},
		{name: "starved only", starved: 1, want: true},
		{name: "transient only", failures: []Failure{transient}, want: true},
		{name: "rate limited only", failures: []Failure{rateLimited}, want: true},
		{name: "transient then fatal", failures: []Failure{transient, fatal}, want: false},
		{name: "fatal then transient", failures: []Failure{fatal, transient}, want: false},
		{name: "starved with fatal", failures: []Failure{fatal}, starved: 2, want: false},
		{name: "non-transient failover only", failures: []Failure{modelMissing, creditWall}, want: false},
		{name: "transient then model unavailable", failures: []Failure{transient, transient, transient, modelMissing}, want: true},
		{name: "starved then model unavailable", failures: []Failure{modelMissing}, starved: 1, want: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			if got := ExhaustedWalkRetryable(tc.failures, tc.starved); got != tc.want {
				t.Fatalf("ExhaustedWalkRetryable(%v, %d) = %v, want %v", tc.failures, tc.starved, got, tc.want)
			}
		})
	}
}

func TestClassifyInvalidModelNameIsModelUnavailable(t *testing.T) {
	t.Parallel()
	got := Classify(400, `{"error":{"message":"Invalid Model Name"}}`)
	if got.Class != FailureModelMissing || got.Action != ActionFailoverNext {
		t.Fatalf("Classify(400, invalid model name) = (%s, %s), want (%s, %s)", got.Class, got.Action, FailureModelMissing, ActionFailoverNext)
	}
}
