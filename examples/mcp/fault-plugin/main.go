package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"runtime"
	"time"
)

type frame struct {
	Type    string `json:"type"`
	Payload string `json:"payload,omitempty"`
	PID     int    `json:"pid,omitempty"`
}

func main() {
	mode := flag.String("mode", "healthy", "healthy, crash, startup-hang, hang, leak, or malformed")
	flag.Parse()
	if *mode == "crash" {
		os.Exit(23)
	}
	if *mode == "startup-hang" {
		for {
			time.Sleep(time.Hour)
		}
	}
	if *mode == "leak" {
		for i := 0; i < 8; i++ {
			go func() {
				b := make([]byte, 1<<20)
				for {
					b[0]++
					runtime.Gosched()
				}
			}()
		}
	}
	ready, _ := json.Marshal(frame{Type: "ready", PID: os.Getpid()})
	fmt.Println(string(ready))
	if *mode == "malformed" {
		fmt.Println("not-json")
		select {}
	}
	scanner := bufio.NewScanner(os.Stdin)
	for scanner.Scan() {
		if *mode == "hang" || *mode == "leak" {
			for {
				time.Sleep(time.Hour)
			}
		}
		var in frame
		if json.Unmarshal(scanner.Bytes(), &in) != nil {
			continue
		}
		out, _ := json.Marshal(frame{Type: "result", Payload: "ok:" + in.Payload})
		fmt.Println(string(out))
	}
}
