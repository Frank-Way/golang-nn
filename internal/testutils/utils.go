package testutils

import "math/rand"

type Base struct {
	Name string
	Err  error
}

func Arange(start, stop float64) []float64 {
	var values []float64
	for i := start; i <= stop; i += 1.0 {
		values = append(values, i)
	}
	return values
}

func ArangeExclude(start, stop float64) []float64 {
	return Arange(start, stop-1.0)
}

func RandomArray(size int) []float64 {
	values := make([]float64, 0)
	for i := 0; i < size; i++ {
		values = append(values, rand.Float64())
	}
	return values
}
