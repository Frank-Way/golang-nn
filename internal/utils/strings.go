package utils

import "strings"

type PrettyStringer interface {
	PrettyString() string
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

func PrettyString(name string, value PrettyStringer) string {
	var sb strings.Builder
	sb.WriteString(name + "\n")
	sb.WriteString(AddIndent(value.PrettyString(), 2))
	return sb.String()
}