package vector

import (
	"fmt"
	"math"
	"nn/pkg/wraperr"
)

// UnaryOperation is function to apply to one Vector
type UnaryOperation func(a float64) float64

// BinaryOperation is function to apply to two Vector
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

// MulScalar makes scalar multiplication of vectors. Both vectors must have same size.
//
// Throws ErrExec
//
// Example:
//     | 1 |.MulScalar(| 4 |) = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
//     | 2 |           | 5 |
//     | 3 |           | 6 |
func (v *Vector) MulScalar(vector *Vector) (res float64, err error) {
	defer wraperr.WrapError(ErrExec, &err)

	if v == nil {
		return 0, ErrNil
	} else if vector == nil {
		return 0, fmt.Errorf("no second vector provided: %v", vector)
	} else if v.size != vector.size {
		return 0, fmt.Errorf("vectors sizes for scalar multiplication does no match: %d != %d", v.size, vector.size)
	}

	mul, err := v.Mul(vector)
	if err != nil {
		return 0, err
	}

	return mul.Sum(), nil
}

// ApplyFunc applies given UnaryOperation to Vector.
//
// Example:
//     | -1 |.ApplyFunc(math.Abs) = | 1 |
//     |  2 |                       | 2 |
//     |  3 |                       | 3 |
func (v *Vector) ApplyFunc(operation UnaryOperation) *Vector {
	if v == nil {
		return nil
	}
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

// ApplyFunc applies given BinaryOperation to this and given Vector. Vectors must be same size.
//
// Throws ErrExec error
//
// Example:
//     | 1 |.ApplyFuncVec(| 4 |, Add) = | 5 |
//     | 2 |              | 5 |         | 7 |
//     | 3 |              | 6 |         | 9 |
func (v *Vector) ApplyFuncVec(vector *Vector, operation BinaryOperation) (vec *Vector, err error) {
	defer wraperr.WrapError(ErrExec, &err)

	if v == nil {
		return nil, ErrNil
	} else if vector == nil {
		return nil, fmt.Errorf("no second vector provided: %v", vector)
	} else if vector.size != v.size {
		return nil, fmt.Errorf("can not operate vectors sized %d and %d", v.size, vector.size)
	}
	n := vector.size

	values := make([]float64, n)
	for i := 0; i < n; i++ {
		values[i] = operation(v.values[i], vector.values[i])
	}

	vec, err = NewVector(values)
	if err != nil {
		return nil, err
	}

	return vec, nil
}

// ApplyFuncNum applies given BinaryOperation to this and given float.
//
// Example:
//     | 1 |.ApplyFuncNum(4, Add) = | 5 |
//     | 2 |                        | 6 |
//     | 3 |                        | 7 |
func (v *Vector) ApplyFuncNum(number float64, operation BinaryOperation) *Vector {
	if v == nil {
		return nil
	}
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

// Reduce sequentially applies BinaryOperation to all values accumulating result in first value
//
// Example
//     | 1 |.Reduce(Mul) = 1 * 2 * 3 = 6
//     | 2 |
//     | 3 |
//
//     | 1 |.Reduce(math.Max) = math.Max(math.Max(1, 2), 3) = math.Max(2, 3) = 3
//     | 2 |
//     | 3 |
func (v *Vector) Reduce(operation BinaryOperation) float64 {
	if v == nil {
		return 0
	}
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

// Extend extends Vector by given scale
//
// Throws ErrExec error
//
// Example:
//     | 1 |.Extend(2) = | 1 |
//     | 2 |             | 1 |
//     | 3 |             | 2 |
//                       | 2 |
//                       | 3 |
//                       | 3 |
func (v *Vector) Extend(scale int) (vec *Vector, err error) {
	defer wraperr.WrapError(ErrExec, &err)

	if v == nil {
		return nil, ErrNil
	} else if scale < 1 {
		return nil, fmt.Errorf("wrong scale for vector extending: %d", scale)
	}

	values := make([]float64, v.size*scale)
	for i := 0; i < v.size; i++ {
		for j := 0; j < scale; j++ {
			values[i*scale+j] = v.values[i]
		}
	}

	return NewVector(values)
}

// Stack stacks Vector for given count
//
// Throws ErrExec error
//
// Example:
//     | 1 |.Stack(2) = | 1 |
//     | 2 |            | 2 |
//     | 3 |            | 3 |
//                      | 1 |
//                      | 2 |
//                      | 3 |
func (v *Vector) Stack(count int) (vec *Vector, err error) {
	defer wraperr.WrapError(ErrExec, &err)

	if v == nil {
		return nil, ErrNil
	} else if count < 1 {
		return nil, fmt.Errorf("wrong count for vector stacking: %d", count)
	}

	values := make([]float64, v.size*count)
	for i := 0; i < v.size; i++ {
		for j := 0; j < count; j++ {
			values[i+j*v.size] = v.values[i]
		}
	}

	return NewVector(values)
}

// Concatenate combines vectors writing values from given Vector to this Vector
//
// Throws ErrExec error
//
// Example:
//     | 1 |.Concatenate(| 4 |)   | 1 |
//     | 2 |             | 5 |  = | 2 |
//     | 3 |                      | 3 |
//                                | 4 |
//                                | 5 |
func (v *Vector) Concatenate(vector *Vector) (vec *Vector, err error) {
	defer wraperr.WrapError(ErrExec, &err)

	if v == nil {
		return nil, ErrNil
	} else if vector == nil {
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
}

// Reverse return reversed Vector
//
// Example:
//     | 1 |.Reverse() = | 3 |
//     | 2 |             | 2 |
//     | 3 |             | 1 |
func (v *Vector) Reverse() *Vector {
	if v == nil {
		return nil
	}
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

// Slice returns slice of Vector for given start (include) and stop (exclude) indices with given start
//
// Throws ErrExec error
//
// Example:
//     | 1 |.Slice(1, 5, 2) = | 2 |
//     | 2 |                  | 4 |
//     | 3 |
//     | 4 |
//     | 5 |
func (v *Vector) Slice(start, stop, step int) (vec *Vector, err error) {
	defer wraperr.WrapError(ErrExec, &err)

	if v == nil {
		return nil, ErrNil
	}
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

	return NewVector(values)
}

// Split splits Vector in parts sized by given value.
//
// Throws ErrExec
//
// Example:
//     | 1 |.Split(2) = [| 1 |, | 3 |, | 5 |]
//     | 2 |             | 2 |  | 4 |  | 6 |
//     | 3 |
//     | 4 |
//     | 5 |
//     | 6 |
//
//     | 1 |.Split(3) = [| 1 |, | 4 |]
//     | 2 |             | 2 |  | 5 |
//     | 3 |             | 3 |  | 6 |
//     | 4 |
//     | 5 |
//     | 6 |
func (v *Vector) Split(partSize int) (vecs []*Vector, err error) {
	defer wraperr.WrapError(ErrExec, &err)

	if v == nil {
		return nil, ErrNil
	} else if partSize < 1 {
		return nil, fmt.Errorf("negative or zero part size in splitting vecttor: %d", partSize)
	} else if v.size < partSize {
		return nil, fmt.Errorf("splitting part size too big for vector: %d > %d", partSize, v.size)
	} else if v.size%partSize != 0 {
		return nil, fmt.Errorf("can not split vector sized %d into parts sized %d", v.size, partSize)
	}

	cnt := v.size / partSize
	vecs = make([]*Vector, cnt)

	for i := 0; i < cnt; i++ {
		slice, err := v.Slice(i*partSize, (i+1)*partSize, 1)
		if err != nil {
			return nil, err
		}
		vecs[i] = slice
	}

	return vecs, nil
}

// Join sequentially calls Vector.Concatenate() for all given vectors
//
// Throws ErrExec
func Join(vectors ...*Vector) (vec *Vector, err error) {
	defer wraperr.WrapError(ErrExec, &err)

	if len(vectors) < 1 {
		return nil, fmt.Errorf("no vectors provided to join: %v", vectors)
	}

	vec = vectors[0].Copy()
	for i := 1; i < len(vectors); i++ {
		vec, err = vec.Concatenate(vectors[i])
		if err != nil {
			return nil, err
		}
	}

	return vec, nil
}

func (v *Vector) Equal(vector *Vector) bool {
	if v == nil || vector == nil {
		if (v != nil && vector == nil) || (v == nil && vector != nil) {
			return false // non-nil != nil and nil != non-nil
		} else {
			return true // nil == nil
		}
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

// Epsilon is max absolute error for checking floats equality.
//
// 1 ~= 1 +- Epsilon/2
//
// 1 != 1 +- Epsilon*2
const Epsilon = 0.000001

func (v *Vector) EqualApprox(vector *Vector) bool {
	if v == nil || vector == nil {
		if (v != nil && vector == nil) || (v == nil && vector != nil) {
			return false // non-nil != nil and nil != non-nil
		} else {
			return true // nil == nil
		}
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

// LinSpace create Vector with values [start; stop] sized `count`. Count must be greater than 1.
//
// Throws ErrExec
//
// Example:
//     LinSpace(1, 2, 6) = |   1 |
//                         | 1.2 |
//                         | 1.4 |
//                         | 1.6 |
//                         | 1.8 |
//                         |   2 |
func LinSpace(start, stop float64, count int) (vec *Vector, err error) {
	defer wraperr.WrapError(ErrExec, &err)

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
}
