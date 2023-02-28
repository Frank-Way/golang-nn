package fabrics

import (
	"github.com/stretchr/testify/require"
	"nn/internal/data/dataset"
	"testing"
)

type DatasetParameters struct {
	Train  DataParameters
	Tests  DataParameters
	Valid  DataParameters
	Single *DataParameters
}

func NewDataset(t *testing.T, parameters DatasetParameters) *dataset.Dataset {
	if parameters.Single != nil {
		data := NewData(t, *parameters.Single)
		ds, err := dataset.NewDatasetSplit(data, dataset.DefaultDataSplitParameters)
		require.NoError(t, err)

		return ds
	}
	train := NewData(t, parameters.Train)
	tests := NewData(t, parameters.Tests)
	valid := NewData(t, parameters.Valid)

	ds, err := dataset.NewDataset(train, tests, valid)
	require.NoError(t, err)

	return ds
}
