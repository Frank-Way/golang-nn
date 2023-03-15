package utils

import (
	"fmt"
	"strings"
)

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

func FormatObject(object map[string]string, format Format) string {
	var sb strings.Builder
	for key, value := range object {
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

func AddIndent(source string, count int) string {
	split := strings.Split(source, "\n")
	indent := Repeat(" ", count)
	for i, str := range split {
		split[i] = indent + str
	}

	return strings.Join(split, "\n")
}

func Repeat(symbol string, count int) string {
	var sb strings.Builder
	for i := 0; i < count; i++ {
		sb.WriteString(symbol)
	}

	return sb.String()
}

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

func Strings(values []SPStringer) string {
	strArr := make([]string, len(values))
	for i, value := range values {
		strArr[i] = String(value)
	}
	return fmt.Sprintf("[%s]", strings.Join(strArr, ", "))
}

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
