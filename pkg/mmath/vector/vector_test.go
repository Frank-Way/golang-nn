package vector

import (
	"github.com/stretchr/testify/require"
	"testing"
)

func TestNewVector(t *testing.T) {
	tests := []struct {
		name string
		in   []float64
		err  error
	}{
		{name: "create vector, no error", in: []float64{1, 2, 3}},
		{name: "empty vector, error", in: []float64{}, err: ErrCreate},
		{name: "nil vector, error", err: ErrCreate},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := NewVector(test.in)
			if test.err == nil {
				require.NoError(t, err)
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, test.err)
			}
		})
	}
}

func TestNewVectorOf(t *testing.T) {
	tests := []struct {
		name     string
		value    float64
		size     int
		expected []float64
		err      error
	}{
		{name: "3 values of 0, no error", value: 0, size: 3, expected: []float64{0, 0, 0}},
		{name: "-3 values of 0, error", value: 0, size: -3, err: ErrCreate},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			vector, err := NewVectorOf(test.value, test.size)
			if test.err == nil {
				require.NoError(t, err)
				require.Equal(t, test.size, vector.size)
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

func TestZeros(t *testing.T) {
	tests := []struct {
		name string
		size int
		err  error
	}{
		{name: "size 3, no error", size: 3},
		{name: "size 0, error", err: ErrCreate},
		{name: "size -1, error", size: -1, err: ErrCreate},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			vector, err := Zeros(test.size)
			if test.err == nil {
				require.NoError(t, err)
				require.Equal(t, test.size, vector.size)
				for i := 0; i < vector.size; i++ {
					require.Equal(t, 0.0, vector.values[i])
				}
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, test.err)
			}
		})
	}
}

func TestVector_Size(t *testing.T) {
	vector, err := NewVector([]float64{1, 2, 3})
	require.NoError(t, err)

	actual := vector.Size()
	require.Equal(t, 3, actual)
}

func TestVector_Get(t *testing.T) {
	tests := []struct {
		name  string
		in    []float64
		index int
		err   error
	}{
		{name: "3 inputs, index 0, no error", in: []float64{1, 2, 3}, index: 0},
		{name: "3 inputs, index 1, no error", in: []float64{1, 2, 3}, index: 1},
		{name: "3 inputs, index 2, no error", in: []float64{1, 2, 3}, index: 2},
		{name: "3 inputs, index 3, error", in: []float64{1, 2, 3}, index: 3, err: ErrNotFound},
		{name: "3 inputs, index -1, error", in: []float64{1, 2, 3}, index: 3, err: ErrNotFound},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			vector, err := NewVector(test.in)
			require.NoError(t, err)

			actual, err := vector.Get(test.index)
			if test.err == nil {
				require.NoError(t, err)
				require.Equal(t, test.in[test.index], actual)
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, test.err)
			}
		})
	}
}

func TestVector_String(t *testing.T) {
	tests := []struct {
		name     string
		in       []float64
		expected string
	}{
		{name: "3 int inputs", in: []float64{1, 2, 3}, expected: "[1 2 3]"},
		{name: "3 float inputs", in: []float64{1.0 / 2, 1.0 / 3, 1.0 / 4}, expected: "[0.5 0.3333333333333333 0.25]"},
		{name: "1 int input", in: []float64{1}, expected: "[1]"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			vector, err := NewVector(test.in)
			require.NoError(t, err)

			actual := vector.String()
			require.Equal(t, test.expected, actual)
		})
	}
}

func TestVector_PrettyString(t *testing.T) {
	tests := []struct {
		name     string
		in       []float64
		expected string
	}{
		{name: "3 int inputs", in: []float64{1, 2, 3}, expected: "| 1 |\n| 2 |\n| 3 |"},
		{name: "3 float inputs", in: []float64{1.0 / 2, 1.0 / 3, 1.0 / 4}, expected: "|                0.5 |\n| 0.3333333333333333 |\n|               0.25 |"},
		{name: "1 int input", in: []float64{1}, expected: "| 1 |"},
		{name: "2 int inputs", in: []float64{1, -20}, expected: "|   1 |\n| -20 |"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			vector, err := NewVector(test.in)
			require.NoError(t, err)

			actual := vector.PrettyString()
			require.Equal(t, test.expected, actual)
		})
	}
}

func TestVector_Raw(t *testing.T) {
	values := []float64{1, 2, 3}

	vector, err := NewVector(values)
	require.NoError(t, err)

	raw := vector.Raw()
	require.Equal(t, values, raw)

	// check: values and raw are different arrays
	values[1] = 5
	require.NotEqual(t, 5.0, vector.values[1])
	require.NotEqual(t, values, raw)
}

func TestVector_Copy(t *testing.T) {
	values := []float64{1, 2, 3}

	vector, err := NewVector(values)
	require.NoError(t, err)

	cp := vector.Copy()
	require.Equal(t, values, cp.values)
	require.Equal(t, vector.values, cp.values)

	// check: vector and cp are different vectors
	values[1] = 5
	require.Equal(t, vector.values, cp.values)

	vector = vector.AddNum(1)
	require.NotEqual(t, vector.values, cp.values)
}
