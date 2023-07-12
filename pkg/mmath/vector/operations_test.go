package vector

import (
	"github.com/stretchr/testify/require"
	"math"
	"testing"
)

type binOpStruct struct {
	name     string
	inA      []float64
	inB      []float64
	expected []float64
	err      error
}

func testBinaryOperation(t *testing.T, tests []binOpStruct, op string) {
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			a, err := NewVector(test.inA)
			require.NoError(t, err)
			b, err := NewVector(test.inB)
			require.NoError(t, err)
			var actual *Vector = nil
			switch op {
			case "a":
				actual, err = a.Add(b)
			case "m":
				actual, err = a.Mul(b)
			case "s":
				actual, err = a.Sub(b)
			case "d":
				actual, err = a.Div(b)
			}
			if test.err == nil {
				require.NoError(t, err)
				require.Equal(t, len(test.expected), actual.size)
				for i, value := range test.expected {
					require.Equal(t, value, actual.values[i])
				}
			} else {
				require.Error(t, err)
				require.Error(t, err, test.err)
			}

		})
	}
}

func TestVector_Add(t *testing.T) {
	tests := []binOpStruct{
		{name: "correct addition", inA: []float64{1, 2, 3}, inB: []float64{2, 3, 4}, expected: []float64{3, 5, 7}},
		{name: "wrong sizes", inA: []float64{1, 2, 3}, inB: []float64{2, 3, 4, 5}, err: ErrExec},
	}
	testBinaryOperation(t, tests, "a")
}

func TestVector_Mul(t *testing.T) {
	tests := []binOpStruct{
		{name: "correct multiplication", inA: []float64{1, 2, 3}, inB: []float64{2, 3, 4}, expected: []float64{2, 6, 12}},
		{name: "wrong sizes", inA: []float64{1, 2, 3}, inB: []float64{2, 3, 4, 5}, err: ErrExec},
	}
	testBinaryOperation(t, tests, "m")
}

func TestVector_Sub(t *testing.T) {
	tests := []binOpStruct{
		{name: "correct subtraction", inA: []float64{1, 2, 3}, inB: []float64{2, 3, 4}, expected: []float64{-1, -1, -1}},
		{name: "wrong sizes", inA: []float64{1, 2, 3}, inB: []float64{2, 3, 4, 5}, err: ErrExec},
	}
	testBinaryOperation(t, tests, "s")
}

func TestVector_Div(t *testing.T) {
	tests := []binOpStruct{
		{name: "correct division", inA: []float64{1, 2, 3}, inB: []float64{2, 3, 4}, expected: []float64{1.0 / 2.0, 2.0 / 3.0, 3.0 / 4.0}},
		{name: "wrong sizes", inA: []float64{1, 2, 3}, inB: []float64{2, 3, 4, 5}, err: ErrExec},
	}
	testBinaryOperation(t, tests, "d")
}

func TestVector_Extend(t *testing.T) {
	tests := []struct {
		name     string
		in       []float64
		scale    int
		expected []float64
		err      error
	}{
		{name: "vector of size 3, scale by 3, no error", in: []float64{1, 2, 3}, scale: 3, expected: []float64{1, 1, 1, 2, 2, 2, 3, 3, 3}},
		{name: "vector of size 3, scale by 1, no error", in: []float64{1, 2, 3}, scale: 1, expected: []float64{1, 2, 3}},
		{name: "vector of size 3, scale by 0, error", in: []float64{1, 2, 3}, scale: 0, err: ErrExec},
		{name: "vector of size 3, scale by -3, error", in: []float64{1, 2, 3}, scale: -3, err: ErrExec},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			vector, err := NewVector(test.in)
			require.NoError(t, err)

			extended, err := vector.Extend(test.scale)
			if test.err == nil {
				require.NoError(t, err)
				require.Equal(t, len(test.expected), extended.Size())
				for i, val := range test.expected {
					require.Equal(t, val, extended.values[i])
				}
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, test.err)
			}
		})
	}
}

func TestVector_Stack(t *testing.T) {
	tests := []struct {
		name     string
		in       []float64
		count    int
		expected []float64
		err      error
	}{
		{name: "vector of size 3, stack by 3, no error", in: []float64{1, 2, 3}, count: 3, expected: []float64{1, 2, 3, 1, 2, 3, 1, 2, 3}},
		{name: "vector of size 3, stack by 1, no error", in: []float64{1, 2, 3}, count: 1, expected: []float64{1, 2, 3}},
		{name: "vector of size 3, stack by 0, error", in: []float64{1, 2, 3}, count: 0, err: ErrExec},
		{name: "vector of size 3, stack by -3, error", in: []float64{1, 2, 3}, count: -3, err: ErrExec},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			vector, err := NewVector(test.in)
			require.NoError(t, err)

			extended, err := vector.Stack(test.count)
			if test.err == nil {
				require.NoError(t, err)
				require.Equal(t, len(test.expected), extended.Size())
				for i, val := range test.expected {
					require.Equal(t, val, extended.values[i])
				}
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, test.err)
			}
		})
	}
}

func TestVector_Concatenate(t *testing.T) {
	tests := []struct {
		name     string
		inA      []float64
		inB      []float64
		expected []float64
		nilCheck bool
		err      error
	}{
		{name: "vector of size 2 concat vector of size 3", inA: []float64{1, 2}, inB: []float64{3, 4, 5}, expected: []float64{1, 2, 3, 4, 5}},
		{name: "vector of size 1 concat vector of size 1", inA: []float64{1}, inB: []float64{3}, expected: []float64{1, 3}},
		{name: "vector of size 2 concat vector of size 1", inA: []float64{1, 2}, inB: []float64{3}, expected: []float64{1, 2, 3}},
		{name: "vector of size 2 concat nil vector", inA: []float64{1, 2}, inB: []float64{3, 4, 5}, nilCheck: true, err: ErrExec},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			a, err := NewVector(test.inA)
			require.NoError(t, err)
			b, err := NewVector(test.inB)
			require.NoError(t, err)
			if test.nilCheck {
				b = nil
			}

			actual, err := a.Concatenate(b)
			if test.err == nil {
				require.NoError(t, err)
				require.Equal(t, len(test.expected), actual.Size())
				for i, value := range test.expected {
					require.Equal(t, value, actual.values[i])
				}
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, test.err)
			}
		})
	}
}

func TestVector_Sum(t *testing.T) {
	vector, err := NewVector([]float64{1, 2, 3})
	require.NoError(t, err)

	actual := vector.Sum()
	expected := 1.0 + 2.0 + 3.0
	require.Equal(t, expected, actual)
}

func TestVector_MulScalar(t *testing.T) {
	tests := []struct {
		name     string
		inA      []float64
		inB      []float64
		expected float64
		err      error
		nilCheck bool
	}{
		{name: "Mul scalar vectors of size 3, no error", inA: []float64{1, 2, 3}, inB: []float64{4, 5, 6}, expected: 1*4 + 2*5 + 3*6},
		{name: "Mul scalar vectors of size 1 and 2, error", inA: []float64{1}, inB: []float64{4, 5}, err: ErrExec},
		{name: "Mul scalar of vector and nil, error", inA: []float64{1, 2, 3}, inB: []float64{4, 5, 6}, nilCheck: true, err: ErrExec},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			a, err := NewVector(test.inA)
			require.NoError(t, err)
			b, err := NewVector(test.inB)
			require.NoError(t, err)
			if test.nilCheck {
				b = nil
			}

			actual, err := a.MulScalar(b)
			if test.err == nil {
				require.NoError(t, err)
				require.Equal(t, test.expected, actual)
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, test.err)
			}
		})
	}
}

type binNumOpStruct struct {
	name     string
	in       []float64
	number   float64
	expected []float64
}

func testBinaryNumOperation(t *testing.T, tests []binNumOpStruct, op string) {
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			vector, err := NewVector(test.in)
			require.NoError(t, err)
			var actual *Vector
			switch op {
			case "a":
				actual = vector.AddNum(test.number)
			case "m":
				actual = vector.MulNum(test.number)
			case "s":
				actual = vector.SubNum(test.number)
			case "d":
				actual = vector.DivNum(test.number)
			}

			require.NotNil(t, actual)
			require.Equal(t, len(test.expected), actual.size)
			for i, value := range test.expected {
				require.Equal(t, value, actual.values[i])
			}

		})
	}
}

func TestVector_AddNum(t *testing.T) {
	tests := []binNumOpStruct{
		{name: "vector of size 3 + 1", in: []float64{1, 2, 3}, number: 1, expected: []float64{2, 3, 4}},
		{name: "vector of size 1 + -1", in: []float64{1}, number: -1, expected: []float64{0}},
	}

	testBinaryNumOperation(t, tests, "a")
}

func TestVector_MulNum(t *testing.T) {
	tests := []binNumOpStruct{
		{name: "vector of size 3 * 2", in: []float64{1, 2, 3}, number: 2, expected: []float64{2, 4, 6}},
		{name: "vector of size 1 * -1", in: []float64{1}, number: -1, expected: []float64{-1}},
	}

	testBinaryNumOperation(t, tests, "m")
}

func TestVector_SubNum(t *testing.T) {
	tests := []binNumOpStruct{
		{name: "vector of size 3 - 1", in: []float64{1, 2, 3}, number: 1, expected: []float64{0, 1, 2}},
		{name: "vector of size 1 - -1", in: []float64{1}, number: -1, expected: []float64{2}},
	}

	testBinaryNumOperation(t, tests, "s")
}

func TestVector_DivNum(t *testing.T) {
	tests := []binNumOpStruct{
		{name: "vector of size 3 / 2", in: []float64{1, 2, 3}, number: 2, expected: []float64{1.0 / 2.0, 2.0 / 2.0, 3.0 / 2.0}},
		{name: "vector of size 1 / -1", in: []float64{1}, number: -1, expected: []float64{-1}},
	}

	testBinaryNumOperation(t, tests, "d")
}

func TestVector_Abs(t *testing.T) {
	vector, err := NewVector([]float64{-1, 2, -3})
	require.NoError(t, err)

	abs := vector.Abs()
	require.NotNil(t, abs)
	require.Equal(t, 3, abs.Size())
	require.Equal(t, []float64{1, 2, 3}, abs.values)
}

func TestVector_Max(t *testing.T) {
	vector, err := NewVector([]float64{-1, 2, -3})
	require.NoError(t, err)

	max := vector.Max()
	require.Equal(t, 2.0, max)
}

func TestVector_Min(t *testing.T) {
	vector, err := NewVector([]float64{-1, 2, -3})
	require.NoError(t, err)

	min := vector.Min()
	require.Equal(t, -3.0, min)
}

func TestVector_Avg(t *testing.T) {
	vector, err := NewVector([]float64{-1, 2, -3})
	require.NoError(t, err)

	avg := vector.Avg()
	require.Equal(t, -2.0/3.0, avg)
}

func TestVector_Exp(t *testing.T) {
	vector, err := NewVector([]float64{-1, 2, -3})
	require.NoError(t, err)

	exp := vector.Exp()
	require.NotNil(t, exp)
	require.Equal(t, 3, exp.Size())
	require.Equal(t, []float64{math.Exp(-1), math.Exp(2), math.Exp(-3)}, exp.values)
}

func TestVector_Pow(t *testing.T) {
	vector, err := NewVector([]float64{-1, 2, -3})
	require.NoError(t, err)

	pow := vector.Pow(3)
	require.NotNil(t, pow)
	require.Equal(t, 3, pow.Size())
	require.Equal(t, []float64{math.Pow(-1, 3), math.Pow(2, 3), math.Pow(-3, 3)}, pow.values)
}

func TestVector_Sqr(t *testing.T) {
	vector, err := NewVector([]float64{-1, 2, -3})
	require.NoError(t, err)

	sqr := vector.Sqr()
	require.NotNil(t, sqr)
	require.Equal(t, 3, sqr.Size())
	require.Equal(t, []float64{math.Pow(-1, 2), math.Pow(2, 2), math.Pow(-3, 2)}, sqr.values)
}

func TestVector_Sqrt(t *testing.T) {
	vector, err := NewVector([]float64{1, 2, 3})
	require.NoError(t, err)

	sqrt := vector.Sqrt()
	require.NotNil(t, sqrt)
	require.Equal(t, 3, sqrt.Size())
	require.Equal(t, []float64{math.Sqrt(1), math.Sqrt(2), math.Sqrt(3)}, sqrt.values)
}

func TestVector_Tanh(t *testing.T) {
	vector, err := NewVector([]float64{1, 2, 3})
	require.NoError(t, err)

	tanh := vector.Tanh()
	require.NotNil(t, tanh)
	require.Equal(t, 3, tanh.Size())
	require.Equal(t, []float64{math.Tanh(1), math.Tanh(2), math.Tanh(3)}, tanh.values)
}

func TestVector_Reverse(t *testing.T) {
	vector, err := NewVector([]float64{1, 2, 3})
	require.NoError(t, err)

	reverse := vector.Reverse()
	require.NotNil(t, reverse)
	require.Equal(t, 3, reverse.Size())
	require.Equal(t, []float64{3, 2, 1}, reverse.values)
}

func TestVector_Slice(t *testing.T) {
	tests := []struct {
		name     string
		in       []float64
		start    int
		stop     int
		step     int
		expected []float64
		err      error
	}{
		{name: "{1, 2, 3, 4}[0:4:1], no error", in: []float64{1, 2, 3, 4}, start: 0, stop: 4, step: 1, expected: []float64{1, 2, 3, 4}},
		{name: "{1, 2, 3, 4}[1:4:1], no error", in: []float64{1, 2, 3, 4}, start: 1, stop: 4, step: 1, expected: []float64{2, 3, 4}},
		{name: "{1, 2, 3, 4}[2:4:1], no error", in: []float64{1, 2, 3, 4}, start: 2, stop: 4, step: 1, expected: []float64{3, 4}},
		{name: "{1, 2, 3, 4}[3:4:1], no error", in: []float64{1, 2, 3, 4}, start: 3, stop: 4, step: 1, expected: []float64{4}},
		{name: "{1, 2, 3, 4}[4:4:1], error", in: []float64{1, 2, 3, 4}, start: 4, stop: 4, step: 1, err: ErrExec},
		{name: "{1, 2, 3, 4}[0:3:1], no error", in: []float64{1, 2, 3, 4}, start: 0, stop: 3, step: 1, expected: []float64{1, 2, 3}},
		{name: "{1, 2, 3, 4}[0:2:1], no error", in: []float64{1, 2, 3, 4}, start: 0, stop: 2, step: 1, expected: []float64{1, 2}},
		{name: "{1, 2, 3, 4}[0:1:1], no error", in: []float64{1, 2, 3, 4}, start: 0, stop: 1, step: 1, expected: []float64{1}},
		{name: "{1, 2, 3, 4}[0:0:1], error", in: []float64{1, 2, 3, 4}, start: 0, stop: 0, step: 1, err: ErrExec},
		{name: "{1, 2, 3, 4}[0:4:2], no error", in: []float64{1, 2, 3, 4}, start: 0, stop: 4, step: 2, expected: []float64{1, 3}},
		{name: "{1, 2, 3, 4}[1:4:2], no error", in: []float64{1, 2, 3, 4}, start: 1, stop: 4, step: 2, expected: []float64{2, 4}},
		{name: "{1, 2, 3, 4}[1:3:2], no error", in: []float64{1, 2, 3, 4}, start: 1, stop: 3, step: 2, expected: []float64{2}},
		{name: "{1, 2, 3, 4}[-1:4:1], error", in: []float64{1, 2, 3, 4}, start: -1, stop: 4, step: 1, err: ErrExec},
		{name: "{1, 2, 3, 4}[0:5:1], error", in: []float64{1, 2, 3, 4}, start: 0, stop: 5, step: 1, err: ErrExec},
		{name: "{1, 2, 3, 4}[0:4:-1], error", in: []float64{1, 2, 3, 4}, start: 0, stop: 4, step: -1, err: ErrExec},
		{name: "{1, 2, 3, 4}[3:0:1], error", in: []float64{1, 2, 3, 4}, start: 3, stop: 0, step: 1, err: ErrExec},
		{name: "{1, 2, 3, 4}[4:6:1], error", in: []float64{1, 2, 3, 4}, start: 4, stop: 6, step: 1, err: ErrExec},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			vector, err := NewVector(test.in)
			require.NoError(t, err)

			sliced, err := vector.Slice(test.start, test.stop, test.step)
			if test.err == nil {
				require.NoError(t, err)
				require.Equal(t, len(test.expected), sliced.Size())
				for i, value := range test.expected {
					require.Equal(t, value, sliced.values[i])
				}
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, test.err)
			}
		})
	}
}

func TestVector_Split(t *testing.T) {
	tests := []struct {
		name     string
		in       []float64
		partSize int
		expected [][]float64
		err      error
	}{
		{name: "{1, 2, 3, 4, 5, 6} split to 1, no error", in: []float64{1, 2, 3, 4, 5, 6}, partSize: 1, expected: [][]float64{{1}, {2}, {3}, {4}, {5}, {6}}},
		{name: "{1, 2, 3, 4, 5, 6} split to 2, no error", in: []float64{1, 2, 3, 4, 5, 6}, partSize: 2, expected: [][]float64{{1, 2}, {3, 4}, {5, 6}}},
		{name: "{1, 2, 3, 4, 5, 6} split to 3, no error", in: []float64{1, 2, 3, 4, 5, 6}, partSize: 3, expected: [][]float64{{1, 2, 3}, {4, 5, 6}}},
		{name: "{1, 2, 3, 4, 5, 6} split to 6, no error", in: []float64{1, 2, 3, 4, 5, 6}, partSize: 6, expected: [][]float64{{1, 2, 3, 4, 5, 6}}},
		{name: "{1, 2, 3, 4, 5, 6} split to 7, error", in: []float64{1, 2, 3, 4, 5, 6}, partSize: 7, err: ErrExec},
		{name: "{1, 2, 3, 4, 5, 6} split to 4, error", in: []float64{1, 2, 3, 4, 5, 6}, partSize: 4, err: ErrExec},
		{name: "{1, 2, 3, 4, 5, 6} split to -1, error", in: []float64{1, 2, 3, 4, 5, 6}, partSize: -1, err: ErrExec},
		{name: "{1, 2, 3, 4, 5, 6} split to 0, error", in: []float64{1, 2, 3, 4, 5, 6}, partSize: 0, err: ErrExec},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			base, err := NewVector(test.in)
			require.NoError(t, err)

			split, err := base.Split(test.partSize)
			if test.err == nil {
				require.NoError(t, err)
				require.Equal(t, len(test.expected), len(split))
				for i, row := range test.expected {
					require.Equal(t, len(row), split[i].size)
					for j, value := range row {
						require.Equal(t, value, split[i].values[j])
					}
				}
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, test.err)
			}
		})
	}
}

func TestJoin(t *testing.T) {
	tests := []struct {
		name     string
		in       [][]float64
		expected []float64
		nilCheck bool
		err      error
	}{
		{name: "{1, 2} + {3} + {4, 5, 6}, no error", in: [][]float64{{1, 2}, {3}, {4, 5, 6}}, expected: []float64{1, 2, 3, 4, 5, 6}, err: nil},
		{name: "{1, 2} + {3} + {4, 5, 6} + nil, no error", in: [][]float64{{1, 2}, {3}, {4, 5, 6}}, nilCheck: true, err: ErrExec},
		{name: "no inputs, error", err: ErrExec},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var vectors []*Vector
			for _, arr := range test.in {
				vector, err := NewVector(arr)
				require.NoError(t, err)
				vectors = append(vectors, vector)
			}
			if test.nilCheck {
				vectors = append(vectors, nil)
			}

			vector, err := Join(vectors...)
			if test.err == nil {
				require.NoError(t, err)
				require.Equal(t, len(test.expected), vector.size)
				for i, value := range test.expected {
					require.Equal(t, value, vector.values[i])
				}
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, test.err)
			}
		})
	}
}

func TestVector_Equal(t *testing.T) {
	tests := []struct {
		name     string
		inA      []float64
		inB      []float64
		nilCheck bool
		expected bool
	}{
		{name: "{1,2}=={1,2}", inA: []float64{1, 2}, inB: []float64{1, 2}, expected: true},
		{name: "{1,2}!={2,1}", inA: []float64{1, 2}, inB: []float64{2, 1}, expected: false},
		{name: "{1,2}!={1,2,3}", inA: []float64{1, 2}, inB: []float64{1, 2, 3}, expected: false},
		{name: "{1,2}!={1.00001,2}", inA: []float64{1, 2}, inB: []float64{1.00001, 2}, expected: false},
		{name: "{1,2}!={1.0000001,2}", inA: []float64{1, 2}, inB: []float64{1.0000001, 2}, expected: false},
		{name: "{1,2}!=mil", inA: []float64{1, 2}, inB: []float64{1, 2}, nilCheck: true, expected: false},
	}

	for _, test := range tests {
		a, err := NewVector(test.inA)
		require.NoError(t, err)
		b, err := NewVector(test.inB)
		require.NoError(t, err)
		if test.nilCheck {
			b = nil
		}

		actual := a.Equal(b)
		require.Equal(t, test.expected, actual)
	}
}

func TestVector_EqualApprox(t *testing.T) {
	tests := []struct {
		name     string
		inA      []float64
		inB      []float64
		nilCheck bool
		expected bool
	}{
		{name: "{1,2}=={1,2}", inA: []float64{1, 2}, inB: []float64{1, 2}, expected: true},
		{name: "{1,2}!={2,1}", inA: []float64{1, 2}, inB: []float64{2, 1}, expected: false},
		{name: "{1,2}!={1,2,3}", inA: []float64{1, 2}, inB: []float64{1, 2, 3}, expected: false},
		{name: "{1,2}=={1.0000001,2}", inA: []float64{1, 2}, inB: []float64{1.0000001, 2}, expected: true},
		{name: "{1,2}!={1.00001,2}", inA: []float64{1, 2}, inB: []float64{1.00001, 2}, expected: false},
		{name: "{1,2}!=nil", inA: []float64{1, 2}, inB: []float64{1, 2}, nilCheck: true, expected: false},
	}

	for _, test := range tests {
		a, err := NewVector(test.inA)
		require.NoError(t, err)
		b, err := NewVector(test.inB)
		require.NoError(t, err)
		if test.nilCheck {
			b = nil
		}

		actual := a.EqualApprox(b)
		require.Equal(t, test.expected, actual)
	}
}

func TestLinSpace(t *testing.T) {
	tests := []struct {
		name     string
		start    float64
		stop     float64
		count    int
		expected []float64
		err      error
	}{
		{name: "4 values in [1;2]", start: 1, stop: 2, count: 4, expected: []float64{1, 1 + 1.0/3.0, 1 + 2.0/3.0, 2}},
		{name: "2 values in [1;2]", start: 1, stop: 2, count: 2, expected: []float64{1, 2}},
		{name: "2 values in [2;1], error", start: 2, stop: 1, count: 2, err: ErrExec},
		{name: "11 values in [-0.1;1.1]", start: -0.1, stop: 1.1, count: 11},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actual, err := LinSpace(test.start, test.stop, test.count)
			if test.err == nil {
				require.NoError(t, err)
				for i, value := range test.expected {
					require.True(t, math.Abs(value-actual.values[i]) < 0.000001)
				}
				t.Logf("%s", actual.String())
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, test.err)
			}
		})
	}
}
