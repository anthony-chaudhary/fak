package devindex

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/testenv/repotest"
)

func repoRootForSurface(t *testing.T) string {
	t.Helper()
	return repotest.Root(t)
}
