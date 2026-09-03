package trajctl

import (
	"testing"
)

func TestApologyRepeat(t *testing.T) {
	t.Run("BlockedOnApologize", func(t *testing.T) {
		prevTool := "bash"
		prevArgs := `{"command": "cat foo.txt"}`
		prevFailed := true
		currText := "I apologize for the mistake, I will now fix it"
		currTool := "bash"
		currArgs := `{"command": "cat foo.txt"}`

		isRepeat, altMsg := DetectApologyRepeat(prevTool, prevArgs, prevFailed, currText, currTool, currArgs)
		if !isRepeat {
			t.Fatalf("expected isRepeat=true, got false")
		}
		if altMsg != BlockedAlternativeMessage {
			t.Fatalf("expected altMsg=%q, got %q", BlockedAlternativeMessage, altMsg)
		}
	})

	t.Run("BlockedOnMyMistakeWithJSONWhitespace", func(t *testing.T) {
		prevTool := "read"
		prevArgs := `{"file_path": "foo.go"}`
		prevFailed := true
		currText := "That was my mistake. Let me read it again."
		currTool := "read"
		currArgs := `{"file_path":"foo.go"}`

		isRepeat, altMsg := DetectApologyRepeat(prevTool, prevArgs, prevFailed, currText, currTool, currArgs)
		if !isRepeat {
			t.Fatalf("expected isRepeat=true, got false")
		}
		if altMsg != BlockedAlternativeMessage {
			t.Fatalf("expected altMsg=%q, got %q", BlockedAlternativeMessage, altMsg)
		}
	})

	t.Run("BlockedOnSorryWithKeyReordering", func(t *testing.T) {
		prevTool := "edit"
		prevArgs := `{"file_path": "a.txt", "old": "1", "new": "2"}`
		prevFailed := true
		currText := "Sorry about that error, re-trying edit."
		currTool := "edit"
		currArgs := `{"new": "2", "file_path": "a.txt", "old": "1"}`

		isRepeat, altMsg := DetectApologyRepeat(prevTool, prevArgs, prevFailed, currText, currTool, currArgs)
		if !isRepeat {
			t.Fatalf("expected isRepeat=true, got false")
		}
		if altMsg != BlockedAlternativeMessage {
			t.Fatalf("expected altMsg=%q, got %q", BlockedAlternativeMessage, altMsg)
		}
	})

	t.Run("BlockedOnExcuseTheError", func(t *testing.T) {
		prevTool := "grep"
		prevArgs := `{"pattern": "TODO"}`
		prevFailed := true
		currText := "Excuse the error, executing the search again."
		currTool := "grep"
		currArgs := `{"pattern": "TODO"}`

		isRepeat, altMsg := DetectApologyRepeat(prevTool, prevArgs, prevFailed, currText, currTool, currArgs)
		if !isRepeat {
			t.Fatalf("expected isRepeat=true, got false")
		}
		if altMsg != BlockedAlternativeMessage {
			t.Fatalf("expected altMsg=%q, got %q", BlockedAlternativeMessage, altMsg)
		}
	})

	t.Run("BlockedOnIWasWrong", func(t *testing.T) {
		prevTool := "bash"
		prevArgs := "make test"
		prevFailed := true
		currText := "I was wrong about that failure. Re-running."
		currTool := "bash"
		currArgs := "make test"

		isRepeat, altMsg := DetectApologyRepeat(prevTool, prevArgs, prevFailed, currText, currTool, currArgs)
		if !isRepeat {
			t.Fatalf("expected isRepeat=true, got false")
		}
		if altMsg != BlockedAlternativeMessage {
			t.Fatalf("expected altMsg=%q, got %q", BlockedAlternativeMessage, altMsg)
		}
	})

	t.Run("BlockedOnEmptyArgs", func(t *testing.T) {
		prevTool := "status"
		prevArgs := ""
		prevFailed := true
		currText := "My apologies, re-checking status."
		currTool := "status"
		currArgs := "{}"

		isRepeat, altMsg := DetectApologyRepeat(prevTool, prevArgs, prevFailed, currText, currTool, currArgs)
		if !isRepeat {
			t.Fatalf("expected isRepeat=true for empty args vs {}, got false")
		}
		if altMsg != BlockedAlternativeMessage {
			t.Fatalf("expected altMsg=%q, got %q", BlockedAlternativeMessage, altMsg)
		}
	})

	t.Run("AllowedWhenPreviousNotFailed", func(t *testing.T) {
		prevTool := "bash"
		prevArgs := `{"command": "cat foo.txt"}`
		prevFailed := false
		currText := "I apologize for any delay, running cat again."
		currTool := "bash"
		currArgs := `{"command": "cat foo.txt"}`

		isRepeat, altMsg := DetectApologyRepeat(prevTool, prevArgs, prevFailed, currText, currTool, currArgs)
		if isRepeat {
			t.Fatalf("expected isRepeat=false when prevFailed=false, got true")
		}
		if altMsg != "" {
			t.Fatalf("expected empty altMsg, got %q", altMsg)
		}
	})

	t.Run("AllowedWhenToolDiffers", func(t *testing.T) {
		prevTool := "bash"
		prevArgs := `{"command": "cat foo.txt"}`
		prevFailed := true
		currText := "I apologize for the mistake, I will use read instead."
		currTool := "read"
		currArgs := `{"file_path": "foo.txt"}`

		isRepeat, altMsg := DetectApologyRepeat(prevTool, prevArgs, prevFailed, currText, currTool, currArgs)
		if isRepeat {
			t.Fatalf("expected isRepeat=false when tool differs, got true")
		}
		if altMsg != "" {
			t.Fatalf("expected empty altMsg, got %q", altMsg)
		}
	})

	t.Run("AllowedWhenArgsDiffer", func(t *testing.T) {
		prevTool := "bash"
		prevArgs := `{"command": "go test ."}`
		prevFailed := true
		currText := "I apologize for the mistake, adding verbose flag now."
		currTool := "bash"
		currArgs := `{"command": "go test -v ."}`

		isRepeat, altMsg := DetectApologyRepeat(prevTool, prevArgs, prevFailed, currText, currTool, currArgs)
		if isRepeat {
			t.Fatalf("expected isRepeat=false when args differ, got true")
		}
		if altMsg != "" {
			t.Fatalf("expected empty altMsg, got %q", altMsg)
		}
	})

	t.Run("AllowedWhenNoApology", func(t *testing.T) {
		prevTool := "bash"
		prevArgs := `{"command": "make build"}`
		prevFailed := true
		currText := "Checking the compilation once more."
		currTool := "bash"
		currArgs := `{"command": "make build"}`

		isRepeat, altMsg := DetectApologyRepeat(prevTool, prevArgs, prevFailed, currText, currTool, currArgs)
		if isRepeat {
			t.Fatalf("expected isRepeat=false when no apology, got true")
		}
		if altMsg != "" {
			t.Fatalf("expected empty altMsg, got %q", altMsg)
		}
	})

	t.Run("AllowedWhenEmptyTools", func(t *testing.T) {
		isRepeat, _ := DetectApologyRepeat("", "", true, "sorry", "", "")
		if isRepeat {
			t.Fatalf("expected isRepeat=false for empty tools")
		}
	})

	t.Run("ApologyPhrasesCoverage", func(t *testing.T) {
		phrases := []string{
			"I apologize for the confusion",
			"My apologies",
			"I'm sorry for the error",
			"Sorry about that",
			"That was my mistake",
			"Excuse the error",
			"Excuse my error",
			"Excuse this error",
			"I was wrong",
			"My fault, fixing it",
			"My bad",
			"Pardon the mistake",
			"Pardon the error",
			"Pardon me",
			"I made a mistake in the path",
			"I made an error in the flag",
			"I stand corrected",
			"I stood corrected",
		}

		for _, p := range phrases {
			if !HasApology(p) {
				t.Errorf("expected HasApology(%q)=true, got false", p)
			}
		}

		nonApologies := []string{
			"I will read the file now",
			"Here are the test results",
			"Refactoring the authentication handler",
			"Running buildcheck with vet",
		}

		for _, p := range nonApologies {
			if HasApology(p) {
				t.Errorf("expected HasApology(%q)=false, got true", p)
			}
		}
	})

	t.Run("CanonicalizeAndHashArgs", func(t *testing.T) {
		raw1 := `{"a": 1, "b": "hello"}`
		raw2 := `{"b": "hello", "a": 1}`
		raw3 := `{"a": 1, "b": "world"}`

		if HashArgs(raw1) != HashArgs(raw2) {
			t.Fatalf("expected HashArgs for key-reordered JSON to match, got %s != %s", HashArgs(raw1), HashArgs(raw2))
		}
		if HashArgs(raw1) == HashArgs(raw3) {
			t.Fatalf("expected HashArgs for differing content to mismatch")
		}

		if HashArgs("") != HashArgs("{}") {
			t.Fatalf("expected HashArgs(\"\") == HashArgs(\"{}\")")
		}
		if HashArgs("  ") != HashArgs("{}") {
			t.Fatalf("expected HashArgs(\"  \") == HashArgs(\"{}\")")
		}
	})

	t.Run("TrackerLifecycle", func(t *testing.T) {
		tracker := NewApologyRepeatTracker()

		// Initially clean
		isRepeat, _ := tracker.Check("I apologize", "bash", "make")
		if isRepeat {
			t.Fatalf("expected false on uninitialized tracker")
		}

		// Tool fails
		tracker.RecordOutcome("bash", "make", true)
		tool, args, failed := tracker.LastCall()
		if tool != "bash" || args != "make" || !failed {
			t.Fatalf("unexpected LastCall: %s %s %v", tool, args, failed)
		}

		// Model attempts identical call with apology -> blocked
		isRepeat, altMsg := tracker.Check("I apologize for the mistake", "bash", "make")
		if !isRepeat || altMsg != BlockedAlternativeMessage {
			t.Fatalf("expected repeat blocked with message, got repeat=%v msg=%q", isRepeat, altMsg)
		}

		// Model changes args -> allowed
		isRepeat, _ = tracker.Check("I apologize, adjusting target", "bash", "make test")
		if isRepeat {
			t.Fatalf("expected allowed on modified args")
		}

		// Adjusted call succeeds
		tracker.RecordOutcome("bash", "make test", false)

		// Model calls same tool again with apology -> allowed because previous succeeded
		isRepeat, _ = tracker.Check("Sorry, running again", "bash", "make test")
		if isRepeat {
			t.Fatalf("expected allowed because previous call succeeded")
		}

		// Reset clears
		tracker.Reset()
		tool, args, failed = tracker.LastCall()
		if tool != "" || args != "" || failed {
			t.Fatalf("expected empty LastCall after Reset")
		}
	})
}
