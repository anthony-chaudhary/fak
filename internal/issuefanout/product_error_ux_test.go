package issuefanout

// product_error_ux_test.go is the #2518 product/error-ux follow-on: the
// refusal-message QUALITY pass. #2512 (failure_paths_test.go) already pinned
// that each refusal names *a* recovery; this pins the stronger product bar the
// spine asks for — every message Build emits under failure tells the caller the
// exact next step in one read. Concretely: each refusal sets its fix off with an
// em-dash action clause led by an imperative verb, so no failure states only the
// violation and buries (or omits) the fix. It fails before the two fix-less
// messages — "title and leaf are required" and "unknown area" — grew their
// recovery clause and passes after.

import (
	"strings"
	"testing"
)

// actionVerbs are the imperative openers a well-formed recovery clause leads
// with: the word right after the em-dash names what the caller should DO.
var actionVerbs = map[string]bool{
	"set": true, "raise": true, "widen": true,
	"ship": true, "fix": true, "pick": true, "drop": true, "use": true,
}

// TestEveryRefusalLeadsWithAnAction drives every refusal site and asserts the
// message carries a " — <imperative>" action clause — the one-read next step the
// #2518 spine requires. A refusal that states only the violation (no em-dash) or
// buries the fix behind a non-action word fails here.
func TestEveryRefusalLeadsWithAnAction(t *testing.T) {
	for _, tc := range refusalContract {
		t.Run(tc.site, func(t *testing.T) {
			var err error
			if tc.drive != nil {
				err = tc.drive(t)
			} else {
				in := spineInput()
				tc.mutate(&in)
				_, err = Build(in)
			}
			if err == nil {
				t.Fatalf("the %s input was accepted, want a refusal", tc.site)
			}
			msg := err.Error()
			idx := strings.Index(msg, "—")
			if idx < 0 {
				t.Fatalf("refusal names no next step: %q has no \"—\" action clause — append \" — <imperative> ...\" naming the fix", msg)
			}
			recovery := strings.TrimSpace(msg[idx+len("—"):])
			fields := strings.Fields(recovery)
			if len(fields) == 0 {
				t.Fatalf("refusal has an empty action clause: %q", msg)
			}
			if first := strings.ToLower(fields[0]); !actionVerbs[first] {
				t.Errorf("recovery clause buries the fix in prose: %q leads with %q, want an imperative action verb (one of %v)", recovery, first, sortedVerbs())
			}
		})
	}
}

// TestImprovedMessagesPinWording is the #2518 witness: it pins the exact next
// step of the two messages this pass rewrote, so a later edit that drops the
// recovery regresses the product bar loudly instead of silently.
func TestImprovedMessagesPinWording(t *testing.T) {
	cases := []struct {
		site   string
		mutate func(*Input)
		want   string
	}{
		{
			site:   "title and leaf required names the fix",
			mutate: func(in *Input) { in.Title = " " },
			want:   "— set title to the spine's human name and leaf to its owning lane, then re-run",
		},
		{
			site:   "unknown area names the fix",
			mutate: func(in *Input) { in.Areas = []string{"bogus"} },
			want:   "— set the area filter to one of those, or drop it for the full taxonomy",
		},
	}
	for _, tc := range cases {
		t.Run(tc.site, func(t *testing.T) {
			in := spineInput()
			tc.mutate(&in)
			_, err := Build(in)
			if err == nil {
				t.Fatalf("Build accepted the %s input, want a refusal", tc.site)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("message wording drifted: got %q, want substring %q", err, tc.want)
			}
		})
	}
}

// sortedVerbs renders actionVerbs in a stable order for failure messages.
func sortedVerbs() []string {
	out := make([]string, 0, len(actionVerbs))
	for v := range actionVerbs {
		out = append(out, v)
	}
	// insertion sort — tiny fixed set, keeps the failure message deterministic
	for i := 1; i < len(out); i++ {
		for j := i; j > 0 && out[j-1] > out[j]; j-- {
			out[j-1], out[j] = out[j], out[j-1]
		}
	}
	return out
}
