package strmatch

import (
	"reflect"
	"testing"
)

func TestTOMLTextHelpers(t *testing.T) {
	if got := StripUnquotedComment(`a = "#" # comment`, '#'); got != `a = "#" ` {
		t.Fatalf("StripUnquotedComment = %q", got)
	}
	want := []string{`"a,b"`, ` "c"`}
	if got := SplitQuoted(`"a,b", "c"`, ','); !reflect.DeepEqual(got, want) {
		t.Fatalf("SplitQuoted = %#v", got)
	}
}
