package vector

import (
	"fmt"
	"math"
	"nn/pkg/wraperr"
)

type UnaryOperation func(a float64) float64
type BinaryOperation func(a, b float64) float64

func Add(a, b float64) float64 {
	return a + b
}

func Sub(a, b float64) float64 {
	return a - b
}

func Mul(a, b float64) float64 {
	return a * b
}

func Div(a, b float64) float64 {
	return a / b
}

func (v *Vector) MulScalar(vector *Vector) (float64, error) {
	res, err := func(vector *Vector) (float64, error) {
		if vector == nil {
			return 0, fmt.Errorf("no second vector provided: %v", vector)
		} else if v.size != vector.size {
			return 0, fmt.Errorf("vectors sizes for scalar multiplication does no match: %d != %d", v.size, vector.size)
		}

		mul, err := v.Mul(vector)
		if err != nil {
			return 0, err
		}

		return mul.Sum(), nil
	}(vector)

	if err != nil {
		return 0, wraperr.NewWrapErr(ErrOperationExec, err)
	}

	return res, nil
}

func (v *Vector) ApplyFunc(operation UnaryOperation) *Vector {
	values := make([]float64, v.size)
	for i, value := range v.values {
		values[i] = operation(value)
	}

	vector, err := NewVector(values)
	if err != nil {
		panic(err)
	}

	return vector
}

func (v *Vector) ApplyFuncVec(vector *Vector, operation BinaryOperation) (*Vector, error) {
	res, err := func(vector *Vector, operation BinaryOperation) (*Vector, error) {
		if vector == nil {
			return nil, fmt.Errorf("no second vector provided: %v", vector)
		} else if vector.size != v.size {
			return nil, fmt.Errorf("can not operate vectors sized %d and %d", v.size, vector.size)
		}
		n := vector.size

		values := make([]float64, n)
		for i := 0; i < n; i++ {
			values[i] = operation(v.values[i], vector.values[i])
		}

		result, err := NewVector(values)
		if err != nil {
			return nil, err
		}

		return result, nil
	}(vector, operation)

	if err != nil {
		return nil, wraperr.NewWrapErr(ErrOperationExec, err)
	}

	return res, nil
}

func (v *Vector) ApplyFuncNum(number float64, operation BinaryOperation) *Vector {
	values := make([]float64, v.size)
	for i := 0; i < v.size; i++ {
		values[i] = operation(v.values[i], number)
	}

	vector, err := NewVector(values)
	if err != nil {
		panic(err)
	}

	return vector
}

func (v *Vector) Reduce(operation BinaryOperation) float64 {
	res := v.values[0]

	for i := 1; i < v.size; i++ {
		res = operation(res, v.values[i])
	}

	return res
}

func (v *Vector) Abs() *Vector {
	return v.ApplyFunc(math.Abs)
}

func (v *Vector) Exp() *Vector {
	return v.ApplyFunc(math.Exp)
}

func (v *Vector) Pow(scale float64) *Vector {
	return v.ApplyFunc(func(number float64) float64 {
		return math.Pow(number, scale)
	})
}

func (v *Vector) Sqr() *Vector {
	return v.Pow(2)
}

func (v *Vector) Sqrt() *Vector {
	return v.Pow(1.0 / 2.0)
}

func (v *Vector) Tanh() *Vector {
	return v.ApplyFunc(math.Tanh)
}

func (v *Vector) Add(vector *Vector) (*Vector, error) {
	return v.ApplyFuncVec(vector, Add)
}

func (v *Vector) Mul(vector *Vector) (*Vector, error) {
	return v.ApplyFuncVec(vector, Mul)
}

func (v *Vector) Sub(vector *Vector) (*Vector, error) {
	return v.ApplyFuncVec(vector, Sub)
}

func (v *Vector) Div(vector *Vector) (*Vector, error) {
	return v.ApplyFuncVec(vector, Div)
}

func (v *Vector) AddNum(number float64) *Vector {
	return v.ApplyFuncNum(number, Add)
}

func (v *Vector) MulNum(number float64) *Vector {
	return v.ApplyFuncNum(number, Mul)

}

func (v *Vector) SubNum(number float64) *Vector {
	return v.ApplyFuncNum(number, Sub)
}

func (v *Vector) DivNum(number float64) *Vector {
	return v.ApplyFuncNum(number, Div)
}

func (v *Vector) Sum() (sum float64) {
	return v.Reduce(Add)
}

func (v *Vector) Max() float64 {
	return v.Reduce(math.Max)
}

func (v *Vector) Min() float64 {
	return v.Reduce(math.Min)
}

func (v *Vector) Avg() float64 {
	return v.Sum() / float64(v.size)
}

func (v *Vector) Extend(scale int) (*Vector, error) {
	res, err := func(scale int) (*Vector, error) {
		if scale < 1 {
			return nil, fmt.Errorf("wrong scale for vector extending: %d", scale)
		}

		values := make([]float64, v.size*scale)
		for i := 0; i < v.size; i++ {
			for j := 0; j < scale; j++ {
				values[i*scale+j] = v.values[i]
			}
		}

		return NewVector(values)
	}(scale)

	if err != nil {
		return nil, wraperr.NewWrapErr(ErrOperationExec, err)
	}

	return res, nil
}

func (v *Vector) Stack(count int) (*Vector, error) {
	res, err := func(count int) (*Vector, error) {
		if count < 1 {
			return nil, fmt.Errorf("wrong count for vector stacking: %d", count)
		}

		values := make([]float64, v.size*count)
		for i := 0; i < v.size; i++ {
			for j := 0; j < count; j++ {
				values[i+j*v.size] = v.values[i]
			}
		}

		return NewVector(values)
	}(count)

	if err != nil {
		return nil, wraperr.NewWrapErr(ErrOperationExec, err)
	}

	return res, nil
}

func (v *Vector) Concatenate(vector *Vector) (*Vector, error) {
	res, err := func(vector *Vector) (*Vector, error) {
		if vector == nil {
			return nil, fmt.Errorf("no second vector provided: %v", vector)
		}

		values := make([]float64, v.size+vector.size)

		for i, value := range v.values {
			values[i] = value
		}
		for j, value := range vector.values {
			values[v.size+j] = value
		}

		return NewVector(values)

	}(vector)

	if err != nil {
		return nil, wraperr.NewWrapErr(ErrOperationExec, err)
	}

	return res, nil
}

func (v *Vector) Reverse() *Vector {
	values := make([]float64, v.size)

	for i, value := range v.values {
		values[v.size-i-1] = value
	}

	vector, err := NewVector(values)
	if err != nil {
		panic(err)
	}

	return vector
}

func (v *Vector) Slice(start, stop, step int) (*Vector, error) {
	res, err := func(start, stop, step int) (*Vector, error) {
		l := stop - start
		resLen := int(math.Ceil(float64(l) / float64(step)))
		if l < 1 {
			return nil, fmt.Errorf("wrong start and stop indicies for slice: %d >= %d", start, stop)
		} else if start < 0 {
			return nil, fmt.Errorf("negative start index for slice: %d", start)
		} else if stop < 1 {
			return nil, fmt.Errorf("negative or zero stop index for slice: %d", stop)
		} else if step < 1 {
			return nil, fmt.Errorf("negative or zero step for slice: %d", step)
		} else if v.size < stop {
			return nil, fmt.Errorf("stop index for slice out of vector size: %d > %d", stop, v.size)
		} else if resLen < 1 {
			return nil, fmt.Errorf("zero resulting slice length with start %d stop %d step %d size %d", start, stop, step, v.size)
		}

		values := make([]float64, resLen)
		for i, cnt := start, 0; i < stop && cnt < resLen; i, cnt = i+step, cnt+1 {
			values[cnt] = v.values[i]
		}

		vector, err := NewVector(values)
		if err != nil {
			return nil, err
		}

		return vector, nil
	}(start, stop, step)

	if err != nil {
		return nil, wraperr.NewWrapErr(ErrOperationExec, err)
	}

	return res, nil
}

func (v *Vector) Split(partSize int) ([]*Vector, error) {
	res, err := func(partSize int) ([]*Vector, error) {
		if partSize < 1 {
			return nil, fmt.Errorf("negative or zero part size in splitting vecttor: %d", partSize)
		} else if v.size < partSize {
			return nil, fmt.Errorf("splitting part size too big for vector: %d > %d", partSize, v.size)
		} else if v.size%partSize != 0 {
			return nil, fmt.Errorf("can not split vector sized %d into parts sized %d", v.size, partSize)
		}

		cnt := v.size / partSize
		vectors := make([]*Vector, cnt)

		for i := 0; i < cnt; i++ {
			slice, err := v.Slice(i*partSize, (i+1)*partSize, 1)
			if err != nil {
				return nil, err
			}
			vectors[i] = slice
		}

		return vectors, nil
	}(partSize)

	if err != nil {
		return nil, wraperr.NewWrapErr(ErrOperationExec, err)
	}

	return res, nil
}

func Join(vectors ...*Vector) (*Vector, error) {
	res, err := func(vectors ...*Vector) (*Vector, error) {
		if vectors == nil || len(vectors) < 1 {
			return nil, fmt.Errorf("no vectors provided to join: %v", vectors)
		}

		result := vectors[0].Copy()
		var err error
		for i := 1; i < len(vectors); i++ {
			result, err = result.Concatenate(vectors[i])
			if err != nil {
				return nil, err
			}
		}

		return result, nil
	}(vectors...)

	if err != nil {
		return nil, wraperr.NewWrapErr(ErrOperationExec, err)
	}

	return res, nil
}

func (v *Vector) Equal(vector *Vector) bool {
	if vector == nil {
		return false
	} else if v.size != vector.size {
		return false
	}

	for i := 0; i < v.size; i++ {
		if v.values[i] != vector.values[i] {
			return false
		}
	}

	return true
}

const Epsilon = 0.000001

func (v *Vector) EqualApprox(vector *Vector) bool {
	if vector == nil {
		return false
	} else if v.size != vector.size {
		return false
	}

	for i := 0; i < v.size; i++ {
		if math.Abs(v.values[i]-vector.values[i]) > Epsilon {
			return false
		}
	}

	return true
}

func LinSpace(start, stop float64, count int) (*Vector, error) {
	res, err := func(start, stop float64, count int) (*Vector, error) {
		if start >= stop {
			return nil, fmt.Errorf("start greater or equal stop value for linspace: %v >= %v", start, stop)
		}

		length := stop - start
		step := length / float64(count-1)

		values := make([]float64, count)
		for i, value := 0, start; i < count-1; i, value = i+1, value+step {
			values[i] = value
		}
		values[count-1] = stop

		return NewVector(values)
	}(start, stop, count)

	if err != nil {
		return nil, wraperr.NewWrapErr(ErrOperationExec, err)
	}

	return res, nil
}
