package vdso

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// IsWriteShaped is the SAME predicate the runtime destructive() override uses, so a
// regression in the needle list would change both the fast-path veto and the
// linter's prediction together. Pin the contract.
func TestIsWriteShapedMatchesNeedles(t *testing.T) {
	writeShaped := []string{"book_flight", "delete_account", "send_email", "update_row", "edit_doc", "run_job", "patch_file", "exec_cmd", "cancel_order", "write_blob"}
	for _, name := range writeShaped {
		if !IsWriteShaped(name) {
			t.Fatalf("IsWriteShaped(%q) = false, want true", name)
		}
	}
	readShaped := []string{"get_user_details", "search_direct_flight", "calculate", "convert_currency", "fetch_policy", "list_all_airports"}
	for _, name := range readShaped {
		if IsWriteShaped(name) {
			t.Fatalf("IsWriteShaped(%q) = true, want false", name)
		}
	}
	if len(WriteShapeNeedles()) == 0 {
		t.Fatal("WriteShapeNeedles() returned an empty list")
	}
}

// destructive() and IsWriteShaped() must agree on the tool-name path (the explicit
// "destructive" meta flag is the only other input to destructive()).
func TestDestructiveUsesIsWriteShaped(t *testing.T) {
	for _, name := range []string{"book_flight", "get_user", "delete_account", "search"} {
		c := &abi.ToolCall{Tool: name}
		if destructive(c) != IsWriteShaped(name) {
			t.Fatalf("destructive(%q)=%v but IsWriteShaped=%v; they must agree on the name path",
				name, destructive(c), IsWriteShaped(name))
		}
	}
}

// ClassifyNamespace distinguishes a single class, an unknown name, and a multi-class
// collision (the silent full-flush degrade the linter flags).
func TestClassifyNamespace(t *testing.T) {
	if ns, multi := ClassifyNamespace("search_direct_flight"); ns != "flights" || multi {
		t.Fatalf("single-class: got (%q,%v), want (flights,false)", ns, multi)
	}
	if ns, multi := ClassifyNamespace("convert_currency"); ns != "fx" || multi {
		t.Fatalf("single-class: got (%q,%v), want (fx,false)", ns, multi)
	}
	if ns, multi := ClassifyNamespace("totally_unknown_tool"); ns != "" || multi {
		t.Fatalf("unknown: got (%q,%v), want (\"\",false)", ns, multi)
	}
	if ns, multi := ClassifyNamespace("price_flight_in_currency"); ns != "" || !multi {
		t.Fatalf("multi-class: got (%q,%v), want (\"\",true)", ns, multi)
	}
}
