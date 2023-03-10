package datagen

import (
	"fmt"
	"nn/pkg/mmath/vector"
	"nn/pkg/wraperr"
)

type InputRange struct {
	Left  float64
	Right float64
	Count int
}

func NewInputRange(left float64, right float64, count int) (*InputRange, error) {
	res, err := func(left float64, right float64, count int) (*InputRange, error) {
		if left >= right {
			return nil, fmt.Errorf("left >= right: %f, %f", left, right)
		} else if count < 2 {
			return nil, fmt.Errorf("inputs must contain at least 2 points, required %d", count)
		}
		return &InputRange{Left: left, Right: right, Count: count}, nil
	}(left, right, count)

	if err != nil {
		return nil, wraperr.NewWrapErr(ErrCreate, err)
	}

	return res, nil
}

func (r *InputRange) inputs() (*vector.Vector, error) {
	return vector.LinSpace(r.Left, r.Right, r.Count)
}
