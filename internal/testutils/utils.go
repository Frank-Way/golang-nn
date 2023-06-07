// Package testutils provides functionality for testing other packages
package testutils

import "math/rand"

// Base represents common losstestutils struct
type Base struct {
	Name string
	Err  error
}

// RandomArray return slice of <size> floats in [0; 1)
func RandomArray(size int) []float64 {
	values := make([]float64, 0)
	for i := 0; i < size; i++ {
		values = append(values, rand.Float64())
	}
	return values
}
