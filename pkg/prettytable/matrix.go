package prettytable

import (
	"fmt"
	"nn/pkg/mmath/matrix"
	"nn/pkg/mmath/vector"
)

func MatrixToGroup(header string, matrix *matrix.Matrix) *Group {
	if matrix == nil {
		return nil
	}
	g := &Group{Columns: make([]*Column, matrix.Cols())}

	t := matrix.T()
	for i := 0; i < t.Rows(); i++ {
		col, err := t.GetRow(i)
		if err != nil {
			panic(err)
		}
		h := ""
		if i == 0 {
			h = header
		}
		g.Columns[i] = VectorToColumn(h, col)
	}

	return g
}

func VectorToColumn(header string, vector *vector.Vector) *Column {
	if vector == nil {
		return nil
	}
	c := &Column{
		Header: header,
		Values: make([]string, vector.Size()),
	}
	for i, value := range vector.Raw() {
		c.Values[i] = fmt.Sprintf("%f", value)
	}
	return c
}
