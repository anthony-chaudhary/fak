package grammar

import (
	"errors"
	"testing"
)

// fak-test:runtime fast est=1ms
func TestToolchoiceTerminalRequiredParallel(t *testing.T) {
	s, err := NewToolchoiceTerminal(ToolchoiceRequired, true)
	if err != nil {
		t.Fatal(err)
	}
	for _, event := range []ToolchoiceEvent{ToolchoiceResponseComplete, ToolchoiceToolsOpen, ToolchoiceCallComplete, ToolchoiceCallComplete} {
		if err := s.Accept(event); err != nil {
			t.Fatal(err)
		}
		if s.Terminal() {
			t.Fatal("required mode terminal before tools section closes")
		}
	}
	if err := s.Accept(ToolchoiceToolsClose); err != nil || !s.Terminal() {
		t.Fatalf("section close: %v", err)
	}
	if err := s.Accept(ToolchoiceCallComplete); !errors.Is(err, ErrToolchoiceTerminal) {
		t.Fatal("extra call admitted")
	}
	if err := s.Accept(ToolchoiceMessageClose); err != nil || !s.Terminal() {
		t.Fatalf("optional close: %v", err)
	}
	if err := s.Accept(ToolchoiceResponseComplete); !errors.Is(err, ErrToolchoiceTerminal) {
		t.Fatal("text after message close admitted")
	}
}

// fak-test:runtime fast est=1ms
func TestToolchoiceTerminalRefusals(t *testing.T) {
	s, _ := NewToolchoiceTerminal(ToolchoiceRequired, false)
	if err := s.Accept(ToolchoiceMessageClose); !errors.Is(err, ErrToolchoiceTerminal) {
		t.Fatal("premature close")
	}
	if err := s.Accept(ToolchoiceToolsOpen); err != nil {
		t.Fatal(err)
	}
	before := *s
	if err := s.Accept(ToolchoiceToolsClose); !errors.Is(err, ErrToolchoiceTerminal) || *s != before {
		t.Fatal("empty section not refused atomically")
	}
	if err := s.Accept(ToolchoiceCallComplete); err != nil {
		t.Fatal(err)
	}
	before = *s
	if err := s.Accept(ToolchoiceCallComplete); !errors.Is(err, ErrToolchoiceTerminal) || *s != before {
		t.Fatal("single-call limit not atomic")
	}
	if err := s.Accept(ToolchoiceToolsClose); err != nil || !s.Terminal() {
		t.Fatal("single-call section did not terminate")
	}
	if _, err := NewToolchoiceTerminal(ToolchoiceMode(255), false); !errors.Is(err, ErrToolchoiceTerminal) {
		t.Fatal("bad mode admitted")
	}
	var zero ToolchoiceTerminal
	if zero.Terminal() || !errors.Is(zero.Accept(ToolchoiceResponseComplete), ErrToolchoiceTerminal) {
		t.Fatal("zero value admits")
	}
}

// fak-test:runtime fast est=1ms
func TestToolchoiceTerminalNone(t *testing.T) {
	s, _ := NewToolchoiceTerminal(ToolchoiceNone, true)
	if s.Terminal() {
		t.Fatal("empty none response terminal")
	}
	if err := s.Accept(ToolchoiceToolsOpen); !errors.Is(err, ErrToolchoiceTerminal) {
		t.Fatal("none admits tools")
	}
	if err := s.Accept(ToolchoiceResponseComplete); err != nil || !s.Terminal() {
		t.Fatal("nonempty response not terminal")
	}
	if err := s.Accept(ToolchoiceMessageClose); err != nil {
		t.Fatal(err)
	}
}
