package routing

import "testing"

// fak-test:runtime fast est=1s
func TestFailureClassTerminalIsClosedOnCreditExhaustion(t *testing.T) {
	t.Parallel()
	for _, class := range FailureClasses() {
		want := class == FailureCreditExhausted
		if got := class.Terminal(); got != want {
			t.Errorf("%s.Terminal() = %v, want %v", class, got, want)
		}
		if class.Terminal() && class.Transient() {
			t.Errorf("%s is both Terminal and Transient", class)
		}
	}
	if FailureClass("").Terminal() || FailureClass("NOT_A_CLASS").Terminal() {
		t.Error("an unknown class must never be Terminal")
	}
}

// fak-test:runtime fast est=1s
func TestCreditWallBodiesClassifyTerminal(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name   string
		status int
		body   string
	}{
		{"bare 402", 402, ""},
		{"nebius 402 budget", 402, `{"detail":"Payment Required: You have exhausted your budget"}`},
		{"spaced usage limit on 429", 429, `{"error":"usage limit reached"}`},
		{"snake usage limit on 429", 429, `{"error":{"code":"usage_limit_reached"}}`},
		{"title-case usage limit on 429", 429, `{"error":"Usage Limit Reached"}`},
		{"wrapped usage limit on 403", 403, "usage\n  limit   reached"},
		{"insufficient quota on 429", 429, `{"error":{"code":"insufficient_quota"}}`},
	}
	for _, tc := range cases {
		f := Classify(tc.status, tc.body)
		if f.Class != FailureCreditExhausted || !f.Class.Terminal() || f.Action != ActionFailoverNext {
			t.Errorf("%s: Classify(%d) = {%s %s}, want terminal CREDIT_EXHAUSTED FAILOVER_NEXT", tc.name, tc.status, f.Class, f.Action)
		}
	}
	for _, tc := range []struct {
		status int
		body   string
	}{
		{429, `{"error":"Rate limit reached for requests"}`},
		{503, ""},
		{401, ""},
	} {
		if f := Classify(tc.status, tc.body); f.Class.Terminal() {
			t.Errorf("Classify(%d, %q) = %s, want a non-terminal class", tc.status, tc.body, f.Class)
		}
	}
}
