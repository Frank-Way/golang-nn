package expression

import (
	"fmt"
	"strings"
)

type sTree struct {
	root     string
	children []*sTree
}

func (t *sTree) hasChildren() bool {
	return t.children != nil && len(t.children) > 0
}

func (t *sTree) string() string {
	var sb strings.Builder
	t.print(&sb, "", "")
	return sb.String()
}

func (t *sTree) print(sb *strings.Builder, prefix, childrenPrefix string) {
	sb.WriteString(prefix)
	sb.WriteString(t.root)
	sb.WriteString("\n")
	for i, child := range t.children {
		if i < len(t.children)-1 {
			child.print(sb, childrenPrefix+"├── ", childrenPrefix+"│   ")
		} else {
			child.print(sb, childrenPrefix+"└── ", childrenPrefix+"    ")
		}
	}
}

func splitBalanced(input string, sep rune) ([]string, error) {
	if input == "" {
		return []string{}, nil
	}
	brackets := []rune("()")
	var openBracket, closeBracket = brackets[0], brackets[1]
	var start, balance int
	var resultRaw [][]rune
	chars := []rune(input)
	for i := 0; i < len(chars); i++ {
		char := chars[i]
		if char == openBracket { // открывающая скобка - увеличиваем баланс
			balance += 1
		} else if char == closeBracket { // закрывающая скобка - уменьшаем баланс
			balance -= 1
		}
		if balance == 0 { // в состоянии баланса смотрим
			if char == sep { // на очередном разделителе - вырезаем, запоминаем индекс и идем дальше
				resultRaw = append(resultRaw, chars[start:i])
				start = i + 1
			}
			if i == len(chars)-1 { // в конце строки - дорезаем до конца
				resultRaw = append(resultRaw, chars[start:i+1])
			}
		}
	}

	if balance != 0 {
		return nil, fmt.Errorf("string %q not balanced: %d", input, balance)
	}

	result := make([]string, len(resultRaw))
	for i, raw := range resultRaw {
		result[i] = runesToString(raw)
	}

	return result, nil
}

func runesToString(runes []rune) string {
	var sb strings.Builder
	for _, v := range runes {
		sb.WriteRune(v)
	}
	return sb.String()
}

func splitRecursively(expression string) (*sTree, error) {
	root := expression
	var children []*sTree
	if strings.HasPrefix(expression, "(") && strings.HasSuffix(expression, ")") {
		split, err := splitBalanced(expression[1:len(expression)-1], []rune(" ")[0])
		if err != nil {
			return nil, err
		}
		children = make([]*sTree, len(split))
		for i, str := range split {
			children[i], err = splitRecursively(str)
			if err != nil {
				return nil, err
			}
		}
	} else {
		children = make([]*sTree, 0)
	}
	return &sTree{
		root:     root,
		children: children,
	}, nil
}

func (t *sTree) equal(tree *sTree) bool {
	if t.root != tree.root {
		return false
	}

	if t.children == nil && tree.children == nil {
		return true
	}

	if t.children == nil || tree.children == nil {
		return false
	}

	if len(t.children) != len(tree.children) {
		return false
	}

	for i := range tree.children {
		if !t.children[i].equal(tree.children[i]) {
			return false
		}
	}

	return true
}

func (t *sTree) copy() *sTree {
	cp := &sTree{
		root: t.root,
	}
	if t.children != nil {
		cp.children = make([]*sTree, len(t.children))
	}
	for i, child := range t.children {
		cp.children[i] = child.copy()
	}

	return cp
}
