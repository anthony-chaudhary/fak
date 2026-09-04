package macromailbox

import (
	"testing"
)

// Invariant: Macro mailboxes must enforce cryptographic authentication and idempotent delivery semantics.
// Guard: Enqueue refuses unauthenticated messages and deduplicates repeated message IDs.

func TestMacroMailboxLifecycle(t *testing.T) {
	t.Parallel()

	id := Identity{"macro:tester", "local://tester", []byte("secret-key")}
	box := New(id)

	msg := Message{ID: "msg-lifecycle-1", To: id.Address, Body: []byte("payload")}
	msg.Auth = Sign(id, msg)

	receipt, err := box.Enqueue(msg)
	if err != nil || !receipt.Applied {
		t.Fatalf("failed enqueuing authenticated message: %v (receipt: %+v)", err, receipt)
	}

	delivered, dReceipt, err := box.Deliver("msg-lifecycle-1")
	if err != nil || !dReceipt.Applied || string(delivered.Body) != "payload" {
		t.Fatalf("failed delivering message: %v (receipt: %+v)", err, dReceipt)
	}
}
