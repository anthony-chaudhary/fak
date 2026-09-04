package deployment

import (
	"path/filepath"
	"testing"
)

func BenchmarkDeployment(b *testing.B) {
	desired := representativeDesired()
	derivID, err := DerivationID(desired)
	if err != nil {
		b.Fatal(err)
	}
	files := map[string][]byte{
		"bin/agent":    []byte("benchmark-agent-payload"),
		"context.json": []byte(`{"active":"benchmark","version":1}`),
	}
	root := b.TempDir()
	store := Store{Root: filepath.Join(root, "store")}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r, err := NewRealization(derivID, desired.Target, files)
		if err != nil {
			b.Fatal(err)
		}
		if _, err := store.Materialize(r); err != nil {
			b.Fatal(err)
		}
	}
}
