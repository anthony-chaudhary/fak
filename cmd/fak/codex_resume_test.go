package main

import (
 "bytes"
 "strings"
 "testing"
)
func TestCodexResumeUsage(t *testing.T){var out,err bytes.Buffer;if c:=runCodexResume(&out,&err,nil);c!=2{t.Fatalf("code=%d",c)};if !strings.Contains(err.String(),"usage: fak codex-resume"){t.Fatalf("stderr=%q",err.String())}}
