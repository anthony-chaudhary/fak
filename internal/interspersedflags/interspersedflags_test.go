package interspersedflags

import (
	"flag"
	"io"
	"reflect"
	"testing"
)

func TestParse(t *testing.T) {
	fs := flag.NewFlagSet("test", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	value := fs.String("value", "", "")
	got, err := Parse(fs, []string{"first", "--value", "set", "second"})
	if err != nil || *value != "set" || !reflect.DeepEqual(got, []string{"first", "second"}) {
		t.Fatalf("Parse = %#v value=%q err=%v", got, *value, err)
	}
}
