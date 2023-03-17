package utils

import "math"

const epsilon = 0.000001

func EqualApprox(a, b float64) bool {
	return math.Abs(a-b) < epsilon
}
