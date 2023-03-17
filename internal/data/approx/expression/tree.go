package expression

import (
	"fmt"
	"strings"
)

// sTree is tree with string as root and slice of sTree as children
type sTree struct {
	root     string
	children []*sTree
}

func (t *sTree) hasChildren() bool {
	return len(t.children) > 0
}

// string builds tree using recursion.
//
// Example:
//     t := &sTree{
//         root: "n0", children: []*sTree{
//             {root: "n1", children: []*sTree{}},
//             {root: "n2", children: []*sTree{
//                 {root: "n3", children: []*sTree{}}
//             }},
//         },
//     }
//     fmt.Println(t.string())
//     // n0
//     // ├── n1
//     // └── n2
//     //     └── n3
func (t *sTree) string() string {
	var sb strings.Builder
	t.print(&sb, "", "")
	return sb.String()
}

// print prints root and all the children to given strings.Builder
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

// splitBalanced split input by provided separator according to bracket's balance
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
		if char == openBracket {
			balance += 1
		} else if char == closeBracket {
			balance -= 1
		}
		if balance == 0 {
			if char == sep { // on sep - cut balanced substring and move start index to next position
				resultRaw = append(resultRaw, chars[start:i])
				start = i + 1
			}
			if i == len(chars)-1 { // if last symbol is closing bracket
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

// splitRecursively builds sTree for given expression recursively using splitBalanced
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
	if t == nil || tree == nil {
		if (t != nil && tree == nil) || (t == nil && tree != nil) {
			return false // non-nil != nil and nil != non-nil
		} else {
			return true // nil == nil
		}
	}

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
