package vector

import (
	"fmt"
	"nn/pkg/wraperr"
	"strings"
)

type Vector struct {
	values []float64
	size   int
}

func NewVector(values []float64) (*Vector, error) {
	if values == nil || len(values) < 1 {
		return nil, wraperr.NewWrapErr(ErrCreate,
			fmt.Errorf("no values provided: %v", values))
	}

	n := len(values)
	newValues := make([]float64, n)
	for i, value := range values {
		newValues[i] = value
	}

	return &Vector{
		values: newValues,
		size:   n,
	}, nil
}

func NewVectorOf(value float64, size int) (*Vector, error) {
	var values []float64
	for i := 0; i < size; i++ {
		values = append(values, value)
	}

	return NewVector(values)
}

func Zeros(size int) (*Vector, error) {
	return NewVectorOf(0, size)
}

func (v *Vector) Size() int {
	return v.size
}

func (v *Vector) Get(index int) (float64, error) {
	if index < 0 || index >= v.size {
		return 0, wraperr.NewWrapErr(ErrNotFound,
			fmt.Errorf("wrong index %d for vector sized %d", index, v.size))
	}

	return v.values[index], nil
}

func (v *Vector) String() string {
	return fmt.Sprintf("%v", v.values)
}

func (v *Vector) PrettyString() string {
	lu, ld, l, lo, ru, rd, r, ro := "⸢", "⸤", "│", "[", "⸣", "⸥", "│", "]"

	if v.size == 1 {
		return fmt.Sprintf("%s %v %s", lo, v.values[0], ro)
	}

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

	for i, str := range strValues {
		format := "%s %s %s"
		if i == 0 {
			strValues[i] = fmt.Sprintf(format, lu, str, ru)
		} else if i == v.size-1 {
			strValues[i] = fmt.Sprintf(format, ld, str, rd)
		} else {
			strValues[i] = fmt.Sprintf(format, l, str, r)
		}
	}

	return strings.Join(strValues, "\n")
}

func (v *Vector) Raw() []float64 {
	values := make([]float64, v.size)
	for i, value := range v.values {
		values[i] = value
	}
	return values
}

func (v *Vector) Copy() *Vector {
	vector, _ := NewVector(v.Raw())
	return vector
}
