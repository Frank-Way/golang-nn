package dataset

import (
	"github.com/stretchr/testify/require"
	"nn/internal/testutils"
	"nn/internal/testutils/testfactories"
	"testing"
)

type DataParameters struct {
	X testfactories.MatrixParameters
	Y testfactories.MatrixParameters
}

func newData(t *testing.T, parameters DataParameters) *Data {
	x := testfactories.NewMatrix(t, parameters.X)
	y := testfactories.NewMatrix(t, parameters.Y)
	data, err := NewData(x, y)
	require.NoError(t, err)

	return data
}

func TestNewData(t *testing.T) {
	tests := []struct {
		testutils.Base
		x         testfactories.MatrixParameters
		y         testfactories.MatrixParameters
		nilCheckX bool
		nilCheckY bool
	}{
		{
			Base: testutils.Base{Name: "2x2 and 2x1 data"},
			x:    testfactories.MatrixParameters{Rows: 2, Cols: 2, Values: []float64{1, 2, 3, 4}},
			y:    testfactories.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{5, 6}},
		},
		{
			Base: testutils.Base{Name: "2x2 and 3x1 data, error", Err: ErrCreate},
			x:    testfactories.MatrixParameters{Rows: 2, Cols: 2, Values: []float64{1, 2, 3, 4}},
			y:    testfactories.MatrixParameters{Rows: 3, Cols: 1, Values: []float64{5, 6, 7}},
		},
		{
			Base:      testutils.Base{Name: "2x2 and 2x1 data, x nil, y nil", Err: ErrCreate},
			x:         testfactories.MatrixParameters{Rows: 2, Cols: 2, Values: []float64{1, 2, 3, 4}},
			y:         testfactories.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{5, 6}},
			nilCheckX: true,
			nilCheckY: true,
		},
		{
			Base:      testutils.Base{Name: "2x2 and 2x1 data, x nil", Err: ErrCreate},
			x:         testfactories.MatrixParameters{Rows: 2, Cols: 2, Values: []float64{1, 2, 3, 4}},
			y:         testfactories.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{5, 6}},
			nilCheckX: true,
		},
		{
			Base:      testutils.Base{Name: "2x2 and 2x1 data, y nil", Err: ErrCreate},
			x:         testfactories.MatrixParameters{Rows: 2, Cols: 2, Values: []float64{1, 2, 3, 4}},
			y:         testfactories.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{5, 6}},
			nilCheckY: true,
		},
	}

	for _, test := range tests {
		t.Run(test.Name, func(t *testing.T) {
			x := testfactories.NewMatrix(t, test.x)
			y := testfactories.NewMatrix(t, test.y)
			if test.nilCheckX {
				x = nil
			}
			if test.nilCheckY {
				y = nil
			}

			data, err := NewData(x, y)
			if test.Err == nil {
				require.NoError(t, err)
				require.Equal(t, x.Rows(), data.X.Rows())
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, test.Err)
			}
		})
	}
}

func TestData_Copy(t *testing.T) {
	data := newData(t, DataParameters{
		X: testfactories.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}},
		Y: testfactories.MatrixParameters{Rows: 2, Cols: 5, Values: []float64{7, 8, 9, 10, 11, 12, 13, 14, 15, 16}},
	})

	cp := data.Copy()
	require.True(t, data != cp)
	require.True(t, data.X.Equal(cp.X))
	require.True(t, data.Y.Equal(cp.Y))
}

func TestData_Equal(t *testing.T) {
	data := newData(t, DataParameters{
		X: testfactories.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}},
		Y: testfactories.MatrixParameters{Rows: 2, Cols: 5, Values: []float64{7, 8, 9, 10, 11, 12, 13, 14, 15, 16}},
	})

	cp := data.Copy()
	require.True(t, data != cp)
	require.True(t, data.Equal(cp))
	require.True(t, cp.Equal(data))
	require.True(t, data.Equal(data))
}

func TestData_Shuffle(t *testing.T) {
	data := newData(t, DataParameters{
		X: testfactories.MatrixParameters{Rows: 5, Cols: 1, Values: []float64{1, 2, 3, 4, 5}},
		Y: testfactories.MatrixParameters{Rows: 5, Cols: 1, Values: []float64{6, 7, 8, 9, 10}},
	})

	shuffled, indices := data.Shuffle()
	for i := 0; i < data.X.Rows(); i++ {
		actualX, err := shuffled.X.GetRow(i)
		require.NoError(t, err)

		expectedX, err := data.X.GetRow(indices[i])
		require.NoError(t, err)

		require.True(t, actualX.Equal(expectedX))

		actualY, err := shuffled.Y.GetRow(i)
		require.NoError(t, err)

		expectedY, err := data.Y.GetRow(indices[i])
		require.NoError(t, err)

		require.True(t, actualY.Equal(expectedY))
	}
}

func TestData_Split(t *testing.T) {
	tests := []struct {
		testutils.Base
		source    DataParameters
		pivot     int
		expectedA DataParameters
		expectedB DataParameters
	}{
		{
			Base: testutils.Base{Name: "5x1 and 5x3, pivot 1"},
			source: DataParameters{
				X: testfactories.MatrixParameters{Rows: 5, Cols: 1, Values: []float64{1, 2, 3, 4, 5}},
				Y: testfactories.MatrixParameters{Rows: 5, Cols: 3, Values: []float64{6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}},
			},
			pivot: 1,
			expectedA: DataParameters{
				X: testfactories.MatrixParameters{Rows: 1, Cols: 1, Values: []float64{1}},
				Y: testfactories.MatrixParameters{Rows: 1, Cols: 3, Values: []float64{6, 7, 8}},
			},
			expectedB: DataParameters{
				X: testfactories.MatrixParameters{Rows: 4, Cols: 1, Values: []float64{2, 3, 4, 5}},
				Y: testfactories.MatrixParameters{Rows: 4, Cols: 3, Values: []float64{9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}},
			},
		},
		{
			Base: testutils.Base{Name: "5x1 and 5x3, pivot 2"},
			source: DataParameters{
				X: testfactories.MatrixParameters{Rows: 5, Cols: 1, Values: []float64{1, 2, 3, 4, 5}},
				Y: testfactories.MatrixParameters{Rows: 5, Cols: 3, Values: []float64{6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}},
			},
			pivot: 2,
			expectedA: DataParameters{
				X: testfactories.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{1, 2}},
				Y: testfactories.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{6, 7, 8, 9, 10, 11}},
			},
			expectedB: DataParameters{
				X: testfactories.MatrixParameters{Rows: 3, Cols: 1, Values: []float64{3, 4, 5}},
				Y: testfactories.MatrixParameters{Rows: 3, Cols: 3, Values: []float64{12, 13, 14, 15, 16, 17, 18, 19, 20}},
			},
		},
		{
			Base: testutils.Base{Name: "5x1 and 5x3, pivot 4"},
			source: DataParameters{
				X: testfactories.MatrixParameters{Rows: 5, Cols: 1, Values: []float64{1, 2, 3, 4, 5}},
				Y: testfactories.MatrixParameters{Rows: 5, Cols: 3, Values: []float64{6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}},
			},
			pivot: 4,
			expectedA: DataParameters{
				X: testfactories.MatrixParameters{Rows: 4, Cols: 1, Values: []float64{1, 2, 3, 4}},
				Y: testfactories.MatrixParameters{Rows: 4, Cols: 3, Values: []float64{6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}},
			},
			expectedB: DataParameters{
				X: testfactories.MatrixParameters{Rows: 1, Cols: 1, Values: []float64{5}},
				Y: testfactories.MatrixParameters{Rows: 1, Cols: 3, Values: []float64{18, 19, 20}},
			},
		},
		{
			Base: testutils.Base{Name: "5x1 and 5x3, pivot 0, error", Err: ErrSplit},
			source: DataParameters{
				X: testfactories.MatrixParameters{Rows: 5, Cols: 1, Values: []float64{1, 2, 3, 4, 5}},
				Y: testfactories.MatrixParameters{Rows: 5, Cols: 3, Values: []float64{6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}},
			},
			pivot: 0,
		},
		{
			Base: testutils.Base{Name: "5x1 and 5x3, pivot -1, error", Err: ErrSplit},
			source: DataParameters{
				X: testfactories.MatrixParameters{Rows: 5, Cols: 1, Values: []float64{1, 2, 3, 4, 5}},
				Y: testfactories.MatrixParameters{Rows: 5, Cols: 3, Values: []float64{6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}},
			},
			pivot: -1,
		},
		{
			Base: testutils.Base{Name: "5x1 and 5x3, pivot 5, error", Err: ErrSplit},
			source: DataParameters{
				X: testfactories.MatrixParameters{Rows: 5, Cols: 1, Values: []float64{1, 2, 3, 4, 5}},
				Y: testfactories.MatrixParameters{Rows: 5, Cols: 3, Values: []float64{6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}},
			},
			pivot: 5,
		},
		{
			Base: testutils.Base{Name: "5x1 and 5x3, pivot 10, error", Err: ErrSplit},
			source: DataParameters{
				X: testfactories.MatrixParameters{Rows: 5, Cols: 1, Values: []float64{1, 2, 3, 4, 5}},
				Y: testfactories.MatrixParameters{Rows: 5, Cols: 3, Values: []float64{6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}},
			},
			pivot: 10,
		},
	}

	for _, test := range tests {
		t.Run(test.Name, func(t *testing.T) {
			source := newData(t, test.source)

			a, b, err := source.Split(test.pivot)
			if test.Err == nil {
				require.NoError(t, err)
				expectedA := newData(t, test.expectedA)
				expectedB := newData(t, test.expectedB)
				require.True(t, a.Equal(expectedA))
				require.True(t, b.Equal(expectedB))
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, test.Err)
			}
		})
	}
}

func TestData_Strings(t *testing.T) {
	data := newData(t, DataParameters{
		X: testfactories.MatrixParameters{Rows: 5, Cols: 3},
		Y: testfactories.MatrixParameters{Rows: 5, Cols: 2},
	})

	t.Log("ShortString():\n" + data.ShortString())
	t.Log("String():\n" + data.String())
	t.Log("PrettyString():\n" + data.PrettyString())
}

func TestData_Batches(t *testing.T) {
	tests := []struct {
		testutils.Base
		in       *Data
		batch    int
		expected []*Data
	}{
		{
			Base: testutils.Base{Name: "6 rows, 2 batch"},
			in: newData(t, DataParameters{
				X: testfactories.MatrixParameters{Rows: 6, Cols: 1, Values: []float64{1, 2, 3, 4, 5, 6}},
				Y: testfactories.MatrixParameters{Rows: 6, Cols: 1, Values: []float64{7, 8, 9, 10, 11, 12}},
			}),
			batch: 2,
			expected: []*Data{
				newData(t, DataParameters{
					X: testfactories.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{1, 2}},
					Y: testfactories.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{7, 8}},
				}),
				newData(t, DataParameters{
					X: testfactories.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{3, 4}},
					Y: testfactories.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{9, 10}},
				}),
				newData(t, DataParameters{
					X: testfactories.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{5, 6}},
					Y: testfactories.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{11, 12}},
				}),
			},
		},
		{
			Base: testutils.Base{Name: "6 rows, 4 batch"},
			in: newData(t, DataParameters{
				X: testfactories.MatrixParameters{Rows: 6, Cols: 1, Values: []float64{1, 2, 3, 4, 5, 6}},
				Y: testfactories.MatrixParameters{Rows: 6, Cols: 1, Values: []float64{7, 8, 9, 10, 11, 12}},
			}),
			batch: 4,
			expected: []*Data{
				newData(t, DataParameters{
					X: testfactories.MatrixParameters{Rows: 4, Cols: 1, Values: []float64{1, 2, 3, 4}},
					Y: testfactories.MatrixParameters{Rows: 4, Cols: 1, Values: []float64{7, 8, 9, 10}},
				}),
				newData(t, DataParameters{
					X: testfactories.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{5, 6}},
					Y: testfactories.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{11, 12}},
				}),
			},
		},
		{
			Base: testutils.Base{Name: "6 rows, 8 batch"},
			in: newData(t, DataParameters{
				X: testfactories.MatrixParameters{Rows: 6, Cols: 1, Values: []float64{1, 2, 3, 4, 5, 6}},
				Y: testfactories.MatrixParameters{Rows: 6, Cols: 1, Values: []float64{7, 8, 9, 10, 11, 12}},
			}),
			batch: 8,
			expected: []*Data{
				newData(t, DataParameters{
					X: testfactories.MatrixParameters{Rows: 6, Cols: 1, Values: []float64{1, 2, 3, 4, 5, 6}},
					Y: testfactories.MatrixParameters{Rows: 6, Cols: 1, Values: []float64{7, 8, 9, 10, 11, 12}},
				}),
			},
		},
		{
			Base: testutils.Base{Name: "6 rows, -2 batch", Err: ErrSplit},
			in: newData(t, DataParameters{
				X: testfactories.MatrixParameters{Rows: 6, Cols: 1, Values: []float64{1, 2, 3, 4, 5, 6}},
				Y: testfactories.MatrixParameters{Rows: 6, Cols: 1, Values: []float64{7, 8, 9, 10, 11, 12}},
			}),
			batch: -2,
		},
	}

	for _, test := range tests {
		t.Run(test.Name, func(t *testing.T) {
			gen, count, err := test.in.Batches(test.batch)
			if test.Err == nil {
				require.NoError(t, err)
				require.Equal(t, len(test.expected), count)
				for i := 0; i < count; i++ {
					data1, err := gen(i)
					require.NoError(t, err)
					require.True(t, data1.Equal(test.expected[i]))

					data2, err := gen(i)
					require.NoError(t, err)
					require.True(t, data2.Equal(test.expected[i]))

					require.True(t, data1 == data2)
				}
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, test.Err)
			}
		})
	}
}
