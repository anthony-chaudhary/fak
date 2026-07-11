package dispatchtick

import (
	"reflect"
	"testing"
)

func TestReconcileDiscoveryReadmitsSoftEvictedWorker(t *testing.T) {
	auth := []string{"a", "b"}
	evicted := map[string]bool{"b": true}
	before := ReconcileDiscovery(auth, evicted, false)
	if !reflect.DeepEqual(before.Routable, []string{"a"}) {
		t.Fatalf("before=%+v", before)
	}
	after := ReconcileDiscovery(auth, evicted, true)
	if !reflect.DeepEqual(after.Routable, []string{"a", "b"}) || !reflect.DeepEqual(after.Readmitted, []string{"b"}) {
		t.Fatalf("after=%+v", after)
	}
}
