package main

import (
	"context"
	"os"

	"github.com/anthony-chaudhary/fak/internal/harnessweb"
)

func main() { os.Exit(harnessweb.Run(context.Background(), os.Stdout, os.Stderr, os.Args[1:])) }
