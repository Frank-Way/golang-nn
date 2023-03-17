// Package vector provides functionality for Vector (wrap on slice of float)
package vector

import (
	"fmt"
	"nn/pkg/wraperr"
	"strings"
)

// Vector holds slice of float64. It provides some useful methods. It can be treated as immutable by using Copy().
//
// Example:
//
// Vector of 5 floats [1, 2, 3, 4, 5] is treated as column:
//     | 1 |
//     | 2 |
//     | 3 |
//     | 4 |
//     | 5 |
type Vector struct {
	values []float64
	size   int
}

// NewVector creates Vector from given slice of floats. There must be at least one value in slice.
//
// Throws ErrCreate error.
func NewVector(values []float64) (*Vector, error) {
	if len(values) < 1 {
		return nil, wraperr.NewWrapErr(ErrCreate,
			fmt.Errorf("no values provided: %v", values))
	}

	n := len(values)
	newValues := make([]float64, n)
	copy(newValues, values)

	return &Vector{
		values: newValues,
		size:   n,
	}, nil
}

// NewVectorOf creates Vector with given size, all elements equal to `value`. Size must be positive.
//
// Throws ErrCreate
func NewVectorOf(value float64, size int) (*Vector, error) {
	var values []float64
	if size > 0 {
		values = make([]float64, size)
	}
	for i := 0; i < size; i++ {
		values[i] = value // there is no mistake. this code won't be reached when values is nil (size < 1)
	}

	return NewVector(values)
}

// Zeros returns Vector of zeros. See NewVectorOf.
//
// Example:
//     z := Zeros(2) // create vector [0, 0]
func Zeros(size int) (*Vector, error) {
	return NewVectorOf(0, size)
}

// Size returns count of Vector's elements
func (v *Vector) Size() int {
	return v.size
}

// Get returns value by given index.
//
// Throws ErrNotFound error
func (v *Vector) Get(index int) (res float64, err error) {
	defer wraperr.WrapError(ErrNotFound, &err)

	if v == nil {
		return 0, ErrNil
	}
	if index < 0 || index >= v.size {
		return 0, fmt.Errorf("wrong index %d for vector sized %d", index, v.size)
	}

	return v.values[index], nil
}

// Example:
//     vec := NewVector([]float64{1, 2, 3})
//     fmt.Println(vec.String())
//     // [1 2 3]
func (v *Vector) String() string {
	if v == nil {
		return "<nil>"
	}
	return fmt.Sprintf("%v", v.values)
}

// Example:
//     vec := NewVector([]float64{1, 2, 3})
//     fmt.Println(vec.PrettyString())
//     // | 1 |
//     // | 2 |
//     // | 3 |
func (v *Vector) PrettyString() string {
	if v == nil {
		return "<nil>"
	}
	l, r := "|", "|"

	strValues := make([]string, v.size)
	maxLen := 0
	for i, value := range v.values {
		strValues[i] = fmt.Sprintf("%v", value)
		if maxLen < len(strValues[i]) {
			maxLen = len(strValues[i])
		}
	}

	for i, str := range strValues {
		pad := ""
		for j := 0; j < maxLen-len(str); j++ {
			pad += " "
		}
		strValues[i] = pad + str
	}

	format := "%s %s %s"
	for i, str := range strValues {
		strValues[i] = fmt.Sprintf(format, l, str, r)
	}

	return strings.Join(strValues, "\n")
}

// Example:
//     vec := NewVector([]float64{1, 2, 3})
//     fmt.Println(vec.PrettyString())
//     // vector 3x1
func (v *Vector) ShortString() string {
	if v == nil {
		return "<nil>"
	}
	return fmt.Sprintf("vector %dx%d", v.size, 1)
}

// Raw returns copy of Vector's inner slice
func (v *Vector) Raw() []float64 {
	values := make([]float64, v.size)
	copy(values, v.values)
	return values
}

// Copy return deep copy of Vector
func (v *Vector) Copy() *Vector {
	vector, _ := NewVector(v.Raw())
	return vector
}
