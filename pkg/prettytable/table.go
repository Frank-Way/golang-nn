package prettytable

import (
	"fmt"
	"strings"
	"sync"
)

type Column struct {
	Header string
	Values []string
}

type Group struct {
	Columns []Column
}

func Build(groups []Group, fillGaps bool) (string, error) {
	if len(groups) < 1 {
		return "", fmt.Errorf("no table column groups provided: %v", groups)
	}

	length := 0
	for _, group := range groups {
		if len(group.Columns) < 1 {
			return "", fmt.Errorf("empty group")
		}
		for _, column := range group.Columns {
			l := column.length()
			if l > length {
				length = l
			}
		}
	}
	for i, group := range groups {
		for j, column := range group.Columns {
			l := column.length()
			if l != length {
				if fillGaps {
					group.Columns[j].Values = append(column.Values, make([]string, length-l)...)
				} else {
					return "", fmt.Errorf("%d'th column's of %d'th group size mismatches max column size: %d != %d",
						j, i, l, length)
				}
			}
		}
	}

	padded := make([][][]string, 2*len(groups)-1)
	for i, group := range groups {
		padded[2*i] = group.pad()
		if i != len(groups)-1 {
			padded[2*i+1] = groupsSep(group.length())
		}
	}

	builders := make([]strings.Builder, length+2)
	for i := 0; i < length+2; i++ {
		builders[i] = getSB()
	}
	defer func() {
		for _, builder := range builders {
			putSB(builder)
		}
	}()

	for _, paddedGroup := range padded {
		for _, paddedColumn := range paddedGroup {
			for k, paddedValue := range paddedColumn {
				builders[k].WriteString(paddedValue)
			}
		}
	}

	sb := getSB()
	defer putSB(sb)
	sb.WriteString(builders[0].String())
	for i := 1; i < length+2; i++ {
		sb.WriteString(crlf)
		sb.WriteString(builders[i].String())
	}

	return sb.String(), nil
}

var (
	crlf                = "\n"
	cross               = "+"
	crossSpaced         = verticalSep + cross + verticalSep
	horizontalSep       = "|"
	horizontalSepSpaced = sep + horizontalSep + sep
	verticalSep         = "-"
	verticalSepSpaced   = verticalSep + verticalSep + verticalSep
	sep                 = " "
	sepSpaced           = sep + sep + sep
	padder              = " "
)

var pool = sync.Pool{New: func() interface{} {
	return strings.Builder{}
}}

func getSB() strings.Builder {
	return pool.Get().(strings.Builder)
}

func putSB(sb strings.Builder) {
	sb.Reset()
	pool.Put(sb)
}

func pad(value string, width int) (string, error) {
	length := len(value)
	if length > width {
		return "", fmt.Errorf("value %q bigger than width %d", value, width)
	}

	sb := getSB()
	defer putSB(sb)

	for delta := width - length; delta > 0; delta-- {
		sb.WriteString(padder)
	}
	sb.WriteString(value)

	return sb.String(), nil
}

func columnSep(length int) []string {
	result := make([]string, length+2)
	result[0] = sepSpaced
	result[1] = verticalSepSpaced
	for i := 2; i < length+2; i++ {
		result[i] = sepSpaced
	}

	return result
}

func groupsSep(length int) [][]string {
	result := make([]string, length+2)
	result[0] = horizontalSepSpaced
	result[1] = crossSpaced
	for i := 2; i < length+2; i++ {
		result[i] = horizontalSepSpaced
	}

	return [][]string{result}
}

func (c Column) width() int {
	width := len(c.Header)
	if len(c.Values) == 0 {
		return width
	}

	for _, value := range c.Values {
		l := len(value)
		if l > width {
			width = l
		}
	}

	return width
}

func (c Column) length() int {
	return len(c.Values)
}

func (c Column) pad() []string {
	result := make([]string, c.length()+2)
	w := c.width()
	var err error

	result[0], err = pad(c.Header, w)
	if err != nil {
		panic(err)
	}

	result[1] = strings.Repeat(verticalSep, w)

	for i, value := range c.Values {
		result[i+2], err = pad(value, w)
		if err != nil {
			panic(err)
		}
	}

	return result
}

func (g Group) length() int {
	max := 0
	for _, column := range g.Columns {
		length := column.length()
		if length > max {
			max = length
		}
	}
	return max
}

func (g Group) pad() [][]string {
	result := make([][]string, 2*len(g.Columns)-1)
	for i, column := range g.Columns {
		result[2*i] = column.pad()
		if i != len(g.Columns)-1 {
			result[2*i+1] = columnSep(column.length())
		}
	}

	return result
}
