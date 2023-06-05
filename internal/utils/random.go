package utils

import (
	"math/rand"
	"time"
)

var rng = rand.New(rand.NewSource(time.Now().UnixNano()))

func RandNormArray(size int, loc float64, scale float64) []float64 {
	arr := make([]float64, size)
	for i := 0; i < size; i++ {
		arr[i] = loc + rng.NormFloat64()*scale
	}

	return arr
}
