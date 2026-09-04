package macromailbox

import (
	"fmt"
	"testing"
)

// BenchmarkMacroMailbox exercises mailbox enqueue, dequeue, and delivery in a loop.
func BenchmarkMacroMailbox(b *testing.B) {
	id := Identity{
		AgentID: "macro:benchmark",
		Address: "local://benchmark",
		Secret:  []byte("benchmark-secret-key"),
	}
	m := New(id)

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		msgID := fmt.Sprintf("bench-msg-%d", i)
		msg := Message{
			ID:   msgID,
			To:   id.Address,
			Body: []byte("benchmark-payload-data"),
		}
		msg.Auth = Sign(id, msg)

		receipt, err := m.Enqueue(msg)
		if err != nil || !receipt.Applied {
			b.Fatalf("enqueue failed at iteration %d: r=%+v err=%v", i, receipt, err)
		}

		delivered, delReceipt, err := m.Deliver(msgID)
		if err != nil || !delReceipt.Applied || len(delivered.Body) == 0 {
			b.Fatalf("deliver failed at iteration %d: r=%+v err=%v", i, delReceipt, err)
		}
	}
}
