package test

import (
	"github.com/stretchr/testify/require"
	"nn/internal/data/dataset"
	"nn/internal/test/utils"
	"nn/internal/test/utils/fabrics"
	"testing"
)

func TestNewData(t *testing.T) {
	tests := []struct {
		utils.Base
		x         fabrics.MatrixParameters
		y         fabrics.MatrixParameters
		nilCheckX bool
		nilCheckY bool
	}{
		{
			Base: utils.Base{Name: "2x2 and 2x1 data"},
			x:    fabrics.MatrixParameters{Rows: 2, Cols: 2, Values: []float64{1, 2, 3, 4}},
			y:    fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{5, 6}},
		},
		{
			Base: utils.Base{Name: "2x2 and 3x1 data, error", Err: dataset.ErrCreate},
			x:    fabrics.MatrixParameters{Rows: 2, Cols: 2, Values: []float64{1, 2, 3, 4}},
			y:    fabrics.MatrixParameters{Rows: 3, Cols: 1, Values: []float64{5, 6, 7}},
		},
		{
			Base:      utils.Base{Name: "2x2 and 2x1 data, x nil, y nil", Err: dataset.ErrCreate},
			x:         fabrics.MatrixParameters{Rows: 2, Cols: 2, Values: []float64{1, 2, 3, 4}},
			y:         fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{5, 6}},
			nilCheckX: true,
			nilCheckY: true,
		},
		{
			Base:      utils.Base{Name: "2x2 and 2x1 data, x nil", Err: dataset.ErrCreate},
			x:         fabrics.MatrixParameters{Rows: 2, Cols: 2, Values: []float64{1, 2, 3, 4}},
			y:         fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{5, 6}},
			nilCheckX: true,
		},
		{
			Base:      utils.Base{Name: "2x2 and 2x1 data, y nil", Err: dataset.ErrCreate},
			x:         fabrics.MatrixParameters{Rows: 2, Cols: 2, Values: []float64{1, 2, 3, 4}},
			y:         fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{5, 6}},
			nilCheckY: true,
		},
	}

	for i := range tests {
		t.Run(tests[i].Name, func(t *testing.T) {
			x := fabrics.NewMatrix(t, tests[i].x)
			y := fabrics.NewMatrix(t, tests[i].y)
			if tests[i].nilCheckX {
				x = nil
			}
			if tests[i].nilCheckY {
				y = nil
			}

			data, err := dataset.NewData(x, y)
			if tests[i].Err == nil {
				require.NoError(t, err)
				require.Equal(t, x.Rows(), data.X.Rows())
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, tests[i].Err)
			}
		})
	}
}

func TestData_Copy(t *testing.T) {
	data := fabrics.NewData(t, fabrics.DataParameters{
		X: fabrics.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}},
		Y: fabrics.MatrixParameters{Rows: 2, Cols: 5, Values: []float64{7, 8, 9, 10, 11, 12, 13, 14, 15, 16}},
	})

	cp := data.Copy()
	require.True(t, data != cp)
	require.True(t, data.X.Equal(cp.X))
	require.True(t, data.Y.Equal(cp.Y))
}

func TestData_Equal(t *testing.T) {
	data := fabrics.NewData(t, fabrics.DataParameters{
		X: fabrics.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{1, 2, 3, 4, 5, 6}},
		Y: fabrics.MatrixParameters{Rows: 2, Cols: 5, Values: []float64{7, 8, 9, 10, 11, 12, 13, 14, 15, 16}},
	})

	cp := data.Copy()
	require.True(t, data != cp)
	require.True(t, data.Equal(cp))
	require.True(t, cp.Equal(data))
	require.True(t, data.Equal(data))
}

func TestData_Shuffle(t *testing.T) {
	data := fabrics.NewData(t, fabrics.DataParameters{
		X: fabrics.MatrixParameters{Rows: 5, Cols: 1, Values: []float64{1, 2, 3, 4, 5}},
		Y: fabrics.MatrixParameters{Rows: 5, Cols: 1, Values: []float64{6, 7, 8, 9, 10}},
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
		utils.Base
		source    fabrics.DataParameters
		pivot     int
		expectedA fabrics.DataParameters
		expectedB fabrics.DataParameters
	}{
		{
			Base: utils.Base{Name: "5x1 and 5x3, pivot 1"},
			source: fabrics.DataParameters{
				X: fabrics.MatrixParameters{Rows: 5, Cols: 1, Values: []float64{1, 2, 3, 4, 5}},
				Y: fabrics.MatrixParameters{Rows: 5, Cols: 3, Values: []float64{6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}},
			},
			pivot: 1,
			expectedA: fabrics.DataParameters{
				X: fabrics.MatrixParameters{Rows: 1, Cols: 1, Values: []float64{1}},
				Y: fabrics.MatrixParameters{Rows: 1, Cols: 3, Values: []float64{6, 7, 8}},
			},
			expectedB: fabrics.DataParameters{
				X: fabrics.MatrixParameters{Rows: 4, Cols: 1, Values: []float64{2, 3, 4, 5}},
				Y: fabrics.MatrixParameters{Rows: 4, Cols: 3, Values: []float64{9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}},
			},
		},
		{
			Base: utils.Base{Name: "5x1 and 5x3, pivot 2"},
			source: fabrics.DataParameters{
				X: fabrics.MatrixParameters{Rows: 5, Cols: 1, Values: []float64{1, 2, 3, 4, 5}},
				Y: fabrics.MatrixParameters{Rows: 5, Cols: 3, Values: []float64{6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}},
			},
			pivot: 2,
			expectedA: fabrics.DataParameters{
				X: fabrics.MatrixParameters{Rows: 2, Cols: 1, Values: []float64{1, 2}},
				Y: fabrics.MatrixParameters{Rows: 2, Cols: 3, Values: []float64{6, 7, 8, 9, 10, 11}},
			},
			expectedB: fabrics.DataParameters{
				X: fabrics.MatrixParameters{Rows: 3, Cols: 1, Values: []float64{3, 4, 5}},
				Y: fabrics.MatrixParameters{Rows: 3, Cols: 3, Values: []float64{12, 13, 14, 15, 16, 17, 18, 19, 20}},
			},
		},
		{
			Base: utils.Base{Name: "5x1 and 5x3, pivot 4"},
			source: fabrics.DataParameters{
				X: fabrics.MatrixParameters{Rows: 5, Cols: 1, Values: []float64{1, 2, 3, 4, 5}},
				Y: fabrics.MatrixParameters{Rows: 5, Cols: 3, Values: []float64{6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}},
			},
			pivot: 4,
			expectedA: fabrics.DataParameters{
				X: fabrics.MatrixParameters{Rows: 4, Cols: 1, Values: []float64{1, 2, 3, 4}},
				Y: fabrics.MatrixParameters{Rows: 4, Cols: 3, Values: []float64{6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}},
			},
			expectedB: fabrics.DataParameters{
				X: fabrics.MatrixParameters{Rows: 1, Cols: 1, Values: []float64{5}},
				Y: fabrics.MatrixParameters{Rows: 1, Cols: 3, Values: []float64{18, 19, 20}},
			},
		},
		{
			Base: utils.Base{Name: "5x1 and 5x3, pivot 0, error", Err: dataset.ErrSplit},
			source: fabrics.DataParameters{
				X: fabrics.MatrixParameters{Rows: 5, Cols: 1, Values: []float64{1, 2, 3, 4, 5}},
				Y: fabrics.MatrixParameters{Rows: 5, Cols: 3, Values: []float64{6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}},
			},
			pivot: 0,
		},
		{
			Base: utils.Base{Name: "5x1 and 5x3, pivot -1, error", Err: dataset.ErrSplit},
			source: fabrics.DataParameters{
				X: fabrics.MatrixParameters{Rows: 5, Cols: 1, Values: []float64{1, 2, 3, 4, 5}},
				Y: fabrics.MatrixParameters{Rows: 5, Cols: 3, Values: []float64{6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}},
			},
			pivot: -1,
		},
		{
			Base: utils.Base{Name: "5x1 and 5x3, pivot 5, error", Err: dataset.ErrSplit},
			source: fabrics.DataParameters{
				X: fabrics.MatrixParameters{Rows: 5, Cols: 1, Values: []float64{1, 2, 3, 4, 5}},
				Y: fabrics.MatrixParameters{Rows: 5, Cols: 3, Values: []float64{6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}},
			},
			pivot: 5,
		},
		{
			Base: utils.Base{Name: "5x1 and 5x3, pivot 10, error", Err: dataset.ErrSplit},
			source: fabrics.DataParameters{
				X: fabrics.MatrixParameters{Rows: 5, Cols: 1, Values: []float64{1, 2, 3, 4, 5}},
				Y: fabrics.MatrixParameters{Rows: 5, Cols: 3, Values: []float64{6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}},
			},
			pivot: 10,
		},
	}

	for i := range tests {
		t.Run(tests[i].Name, func(t *testing.T) {
			source := fabrics.NewData(t, tests[i].source)

			a, b, err := source.Split(tests[i].pivot)
			if tests[i].Err == nil {
				require.NoError(t, err)
				expectedA := fabrics.NewData(t, tests[i].expectedA)
				expectedB := fabrics.NewData(t, tests[i].expectedB)
				require.True(t, a.Equal(expectedA))
				require.True(t, b.Equal(expectedB))
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, tests[i].Err)
			}
		})
	}
}

func TestData_Strings(t *testing.T) {
	data := fabrics.NewData(t, fabrics.DataParameters{
		X: fabrics.MatrixParameters{Rows: 5, Cols: 3},
		Y: fabrics.MatrixParameters{Rows: 5, Cols: 2},
	})

	t.Log("ShortString():\n" + data.ShortString())
	t.Log("String():\n" + data.String())
	t.Log("PrettyString():\n" + data.PrettyString())
}
