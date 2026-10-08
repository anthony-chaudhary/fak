// cron_progress_helper is the real child process for the cron silence/deadline test.
package main

import (
	"os"
	"time"
)

func main() {
	if len(os.Args) != 2 {
		os.Exit(2)
	}
	var count int
	switch os.Args[1] {
	case "progress":
		count = 7
	case "stalled":
		time.Sleep(3 * time.Second)
		return
	case "hard-ceiling":
		count = 50
	default:
		os.Exit(2)
	}
	for i := 0; i < count; i++ {
		if _, err := os.Stdout.WriteString("progress\n"); err != nil {
			os.Exit(1)
		}
		time.Sleep(100 * time.Millisecond)
	}
}
