package modelengine

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

type nativePrefixReplacementProbe struct{ calls int }

func (p *nativePrefixReplacementProbe) MatchLen([]int) int { p.calls++; return 16 }

// Estimate only; this test has not been timed.
// fak-test:runtime medium est=4s lane=default
func TestNativeSchedulerPrefixProviderReplacementKeepsLeaseOwner(t *testing.T) {
	for _, mode := range []string{"disable-radix", "disable-generic", "typed-nil", "generic", "radix"} {
		t.Run(mode, func(t *testing.T) {
			m := nativeSchedulerPrefillModel(t)
			prompt := nativeSchedulerQwenPrompt(48)
			oldTree, nextTree := radixkv.New(0), radixkv.New(0)
			b, n := oldTree.Lookup(prompt)
			oldTree.Done(oldTree.Insert(b, prompt[n:], nil))
			b, n = nextTree.Lookup(prompt[:16])
			peerLease := nextTree.Insert(b, prompt[n:16], nil)
			defer nextTree.Done(peerLease)
			s := newNativeScheduler(m, nativeSchedulerPrefillPrepare(map[string][]int{"old": prompt, "new": prompt}))
			s.SetRadixKV(oldTree)
			if err := s.SetQwenPrefillMaxTokensPerIteration(nativeQwenPrefillMinChunkTokens); err != nil {
				t.Fatal(err)
			}
			nativeSchedulerBeginManualDrain(t, s)
			defer nativeSchedulerEndManualDrain(s)
			old := nativeSchedulerAdmitLane(t, s, "old")
			if oldTree.Stats().ProtectedTokens != len(prompt) {
				t.Fatal("old admission did not acquire real prefix lease")
			}
			probe := &nativePrefixReplacementProbe{}
			switch mode {
			case "disable-radix":
				s.SetRadixKV(nil)
			case "disable-generic":
				s.SetPrefixTree(nil)
			case "typed-nil":
				var nilTree *radixkv.Tree
				s.SetPrefixTree(nilTree)
			case "generic":
				s.SetPrefixTree(probe)
			case "radix":
				s.SetRadixKV(nextTree)
			}
			if mode == "radix" {
				if s.RadixKV() != nextTree || s.PrefixTree() != nextTree {
					t.Fatal("radix provider views diverged")
				}
			} else if mode == "generic" {
				if s.RadixKV() != nil || s.PrefixTree() != probe {
					t.Fatal("generic provider did not replace radix")
				}
			} else if s.RadixKV() != nil || s.PrefixTree() != nil {
				t.Fatal("disabled provider remained reachable")
			}
			old.Cancel()
			nativeSchedulerDriveIteration(t, s)
			if !old.terminal || oldTree.Stats().ProtectedTokens != 0 {
				t.Fatal("old admission retained its replaced-tree lease")
			}
			if nextTree.Stats().ProtectedTokens != 16 {
				t.Fatal("replacement released unrelated peer lease")
			}
			before := s.PrefixStats()
			fresh := nativeSchedulerAdmitLane(t, s, "new")
			after := s.PrefixStats()
			if mode == "generic" || mode == "radix" {
				if after.Hits != before.Hits+1 || after.MatchedTokens != before.MatchedTokens+16 {
					t.Fatal("new admission used stale provider")
				}
			} else if after != before {
				t.Fatal("disabled provider still received lookup")
			}
			if mode == "generic" && probe.calls != 1 {
				t.Fatalf("new provider calls = %d", probe.calls)
			}
			fresh.Cancel()
			nativeSchedulerDriveIteration(t, s)
			if !fresh.terminal || nextTree.Stats().ProtectedTokens != 16 {
				t.Fatal("fresh retirement disturbed unrelated replacement lease")
			}
			oldTree.SetRetention(0)
			if oldTree.Stats().Tokens != 0 {
				t.Fatal("replaced tree is still pinned")
			}
		})
	}
}

// Estimate only; this test has not been timed.
// fak-test:runtime fast est=100ms lane=default
func TestNativeSchedulerPrefixLookupCapturesOwnerBeforeRecord(t *testing.T) {
	m := nativeSchedulerPrefillModel(t)
	prompt := nativeSchedulerQwenPrompt(32)
	tree := radixkv.New(0)
	b, n := tree.Lookup(prompt)
	tree.Done(tree.Insert(b, prompt[n:], nil))
	s := NewNativeScheduler(m)
	defer s.Close()
	s.SetRadixKV(tree)
	sess := m.NewSession()
	defer sess.Close()
	matched, boundary, owner := s.lookupPrefix(prompt)
	if matched != len(prompt) || boundary == nil || owner != tree {
		t.Fatal("lookup omitted its actual lease owner")
	}
	// Exercise precisely the setter window between the two production calls.
	s.SetRadixKV(nil)
	s.recordPrefixLookup(sess, prompt, matched, boundary, owner)
	s.closeLaneSession(sess)
	tree.SetRetention(0)
	if tree.Stats().Tokens != 0 {
		t.Fatal("record attributed the lease to a later provider")
	}
	// No holder is recorded for a miss; lookup releases that lease itself.
	s.SetRadixKV(tree)
	matched, boundary, owner = s.lookupPrefix(prompt)
	if matched != 0 || boundary != nil || owner != nil {
		t.Fatal("miss handed out an unowned root lease")
	}
}
