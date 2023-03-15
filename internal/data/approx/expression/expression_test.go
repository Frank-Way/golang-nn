package expression

import (
	"github.com/stretchr/testify/require"
	"math"
	"testing"
)

func TestNewExpression(t *testing.T) {
	tests := []struct {
		name string
		in   string
		err  error
	}{
		{name: "correct inputs", in: "(+ (sin (+ (* 2 x0) 1)) (* x0 (sin (sum 1 2 3 4))))"},
		{name: "imbalanced inputs", in: "((+ (sin (+ (* 2 x0) 1)) (* x0 (sin (sum 1 2 3 4))))", err: ErrParse},
		{name: "unknown token", in: "(++ (sin (+ (* 2 x0) 1)) (* x0 (sin (sum 1 2 3 4))))", err: ErrParse},
		{name: "wrong arguments count", in: "(+ (sin (+ (* 2 x0 3) 1)) (* x0 (sin (sum 1 2 3 4))))", err: ErrParse},
		{name: "invalid variable", in: "(+ (sin (+ (* 2 X0) 1)) (* x0 (sin (sum 1 2 3 4))))", err: ErrParse},
		{name: "nan 2.1.", in: "(+ (sin (+ (* 2.1. x0) 1)) (* x0 (sin (sum 1 2 3 4))))", err: ErrParse},
		{name: "nan a", in: "(+ (sin (+ (* a x0) 1)) (* x0 (sin (sum 1 2 3 4))))", err: ErrParse},
		{name: "illegal character _", in: "(+ _ (sin (+ (* 2 x0) 1)) (* x0 (sin (sum 1 2 3 4))))", err: ErrParse},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := NewExpression(test.in)
			if test.err == nil {
				require.NoError(t, err)
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, test.err)
			}
		})
	}
}

func TestExpression_Exec(t *testing.T) {
	tests := []struct {
		name     string
		in       string
		args     []float64
		expected float64
		err      error
	}{
		{
			name:     "correct inputs",
			in:       "(+ (sin (+ (* 2 x0) 1)) (* x0 (sin (sum 1 2 3 4))))",
			args:     []float64{1},
			expected: math.Sin(2*1+1) + 1*math.Sin(1+2+3+4),
		},
		{
			name:     "many arguments, no error",
			in:       "(+ (sin (+ (* 2 x0) 1)) (* x0 (sin (sum 1 2 3 4))))",
			args:     []float64{1, 2},
			expected: math.Sin(2*1+1) + 1*math.Sin(1+2+3+4),
		},
		{
			name: "too little arguments count",
			in:   "(+ (sin (+ (* 2 x0) 1)) (* x0 (sin (sum 1 2 3 4))))",
			args: []float64{},
			err:  ErrExec,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			expression, err := NewExpression(test.in)
			require.NoError(t, err)
			exec, err := expression.Exec(test.args)
			if test.err == nil {
				require.NoError(t, err)
				require.Equal(t, test.expected, exec)
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, test.err)
			}
		})
	}
}

func TestExpression_PrettyString(t *testing.T) {
	expression, err := NewExpression("(+ (sin (+ (* 2 x0) 1)) (* x0 (sin (sum 1 2 3 4))))")
	require.NoError(t, err)

	s := expression.PrettyString()
	expected := "(+ (sin (+ (* 2 x0) 1)) (* x0 (sin (sum 1 2 3 4))))\n├── +\n├── (sin (+ (* 2 x0) 1))\n│   ├── sin\n│   └── (+ (* 2 x0) 1)\n│       ├── +\n│       ├── (* 2 x0)\n│       │   ├── *\n│       │   ├── 2\n│       │   └── x0\n│       └── 1\n└── (* x0 (sin (sum 1 2 3 4)))\n    ├── *\n    ├── x0\n    └── (sin (sum 1 2 3 4))\n        ├── sin\n        └── (sum 1 2 3 4)\n            ├── sum\n            ├── 1\n            ├── 2\n            ├── 3\n            └── 4\n"
	require.Equal(t, expected, s)
}

func TestExpression_ShortString(t *testing.T) {
	expression, err := NewExpression("(+ (sin (+ (* 2 x0) 1)) (* x0 (sin (sum 1 2 3 4))))")
	require.NoError(t, err)

	s := expression.ShortString()
	expected := "(+ (sin (+ (* 2 x0) 1)) (* x0 (sin (sum 1 2 3 4))))"
	require.Equal(t, expected, s)
}

func TestExpression_String(t *testing.T) {
	expression, err := NewExpression("(+ (sin (+ (* 2 x0) 1)) (* x0 (sin (sum 1 2 3 4))))")
	require.NoError(t, err)

	s := expression.String()
	expected := "(+ (sin (+ (* 2 x0) 1)) (* x0 (sin (sum 1 2 3 4))))"
	require.Equal(t, expected, s)
}

func TestExpression_Copy(t *testing.T) {
	expression, err := NewExpression("(+ (sin (+ (* 2 x0) 1)) (* x0 (sin (sum 1 2 3 4))))")
	require.NoError(t, err)

	copy := expression.Copy()
	require.True(t, copy != expression)
	require.True(t, copy.Equal(expression))
}
