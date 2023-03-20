package expression

import (
	"fmt"
	"math"
)

type operation struct {
	inputsCount int
	impl        func(x []float64) float64
}

func newOperation(inputsCount int, impl func(x []float64) float64) *operation {
	return &operation{inputsCount: inputsCount, impl: impl}
}

func (o *operation) exec(x []float64) (float64, error) {
	if o.inputsCount > 0 {
		if o.inputsCount != len(x) {
			return 0, fmt.Errorf("invalid number of arguments, required %d, provided %v", o.inputsCount, x)
		}
	} else if o.inputsCount == 0 {
		if len(x) < 1 {
			return 0, fmt.Errorf("no arguments provided, required at least one")
		}
	}
	return o.impl(x), nil
}

func (o *operation) checkInputsCount(count int) bool {
	if o.inputsCount == 0 {
		return count > 0
	} else if o.inputsCount < 0 {
		return true
	} else {
		return count == o.inputsCount
	}
}

// getOperation return operation for given token. Allowed tokens are: `+`, `-`, `*`, `/`, ``, ``, ``, ``, ``, ``, ``
func getOperation(token string) (*operation, error) {
	switch token {
	case "+":
		return newOperation(2, func(x []float64) float64 { return x[0] + x[1] }), nil
	case "-":
		return newOperation(2, func(x []float64) float64 { return x[0] - x[1] }), nil
	case "*":
		return newOperation(2, func(x []float64) float64 { return x[0] * x[1] }), nil
	case "/":
		return newOperation(2, func(x []float64) float64 { return x[0] / x[1] }), nil
	case "sin":
		return newOperation(1, func(x []float64) float64 { return math.Sin(x[0]) }), nil
	case "cos":
		return newOperation(1, func(x []float64) float64 { return math.Cos(x[0]) }), nil
	case "tan":
		return newOperation(1, func(x []float64) float64 { return math.Tan(x[0]) }), nil
	case "pow":
		return newOperation(2, func(x []float64) float64 { return math.Pow(x[0], x[1]) }), nil
	case "sum":
		return newOperation(-1, func(x []float64) float64 {
			s := 0.0
			for _, v := range x {
				s += v
			}
			return s
		}), nil
	case "max":
		return newOperation(0, func(x []float64) float64 {
			// 0 inputsCount prevents zero-length inputs, so it is safe to pick first element (as long as it is not nil)
			max := x[0]
			for _, v := range x[1:] {
				if max < v {
					max = v
				}
			}
			return max
		}), nil
	case "min":
		return newOperation(0, func(x []float64) float64 {
			// 0 inputsCount prevents zero-length inputs, so it is safe to pick first element (as long as it is not nil)
			min := x[0]
			for _, v := range x[1:] {
				if min > v {
					min = v
				}
			}
			return min
		}), nil
	default:
		return nil, fmt.Errorf("unknown operation token: %s", token)
	}
}
