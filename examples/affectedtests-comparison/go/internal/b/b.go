package b

import "example.com/diamond/internal/a"

func Value() int { return a.Value() + 1 }
