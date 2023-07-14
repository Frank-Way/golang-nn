// Package utils provides some useful methods used by other packages
package utils

import (
	"encoding/json"
	"fmt"
	"strings"
)

func MarshalJSON(typeName string, value interface{}) string {
	if value == nil {
		return "<nil>"
	}
	b, err := json.Marshal(value)
	if err != nil {
		return "invalid value"
	}
	return fmt.Sprintf("%s(%s)", typeName, string(b))
}

type SPStringer interface {
	ShortString() string
	PrettyString() string
	String() string
}

type Format string

var (
	BaseFormat   Format = "%s: %s, "
	ShortFormat  Format = "%s: %s, "
	PrettyFormat Format = "%s\n%s\n"
)

// FormatObject return map formatted by given format.
//
// Example:
//     m := map[string]string{
//         `attr1`: `val1`,
//         `attr2`: `val2`,
//     }
//     fmt.Println(m, BaseFormat)
//     // {attr1: val1, attr2: val2}
//
//     fmt.Println(m, PrettyFormat)
//     // attr1
//     //   val1
//     // attr2
//     //   val2
func FormatObject(object map[string]string, format Format) string {
	var sb strings.Builder
	for key, value := range object {
		if value == "<nil>" {
			continue
		}
		if format == PrettyFormat {
			sb.WriteString(fmt.Sprintf(string(format), key, AddIndent(value, 2)))
		} else {
			sb.WriteString(fmt.Sprintf(string(format), key, value))
		}
	}
	switch format {
	case BaseFormat:
		return fmt.Sprintf("{%s}", sb.String()[:sb.Len()-2])
	case ShortFormat:
		return fmt.Sprintf("{%s}", sb.String()[:sb.Len()-2])
	case PrettyFormat:
		return sb.String()[:sb.Len()-1]
	}

	panic(fmt.Sprintf("illegal format: %v", format))
}

// AddIndent split source by `\n`, add <count> number of spaces at begin of each line, return lines joined by `\n`
//
// Example:
//     fmt.Println(AddIndent(`1\n 2\n3`, 3))
//     //    1
//     //     2
//     //    3
func AddIndent(source string, count int) string {
	split := strings.Split(source, "\n")
	indent := strings.Repeat(" ", count)
	for i, str := range split {
		split[i] = indent + str
	}

	return strings.Join(split, "\n")
}

// PrettyStrings call PrettyString() on slice of values and combine results in yaml-like list format
func PrettyStrings(values []SPStringer) string {
	var sb strings.Builder
	for i, value := range values {
		if i > 0 {
			sb.WriteString("\n")
		}
		sb.WriteString("-")
		sb.WriteString(AddIndent(PrettyString(value), 2)[1:])
	}
	return sb.String()
}

// Strings call String() on slice of values and join results by comma
func Strings(values []SPStringer) string {
	strArr := make([]string, len(values))
	for i, value := range values {
		strArr[i] = String(value)
	}
	return fmt.Sprintf("[%s]", strings.Join(strArr, ", "))
}

// ShortStrings call ShortString() on slice of values and join results by comma
func ShortStrings(values []SPStringer) string {
	strArr := make([]string, len(values))
	for i, value := range values {
		strArr[i] = ShortString(value)
	}
	return fmt.Sprintf("[%s]", strings.Join(strArr, ", "))
}

func String(value SPStringer) string {
	if value == nil {
		return "<nil>"
	} else {
		return value.String()
	}
}

func ShortString(value SPStringer) string {
	if value == nil {
		return "<nil>"
	} else {
		return value.ShortString()
	}
}

func PrettyString(value SPStringer) string {
	if value == nil {
		return "<nil>"
	} else {
		return value.PrettyString()
	}
}
