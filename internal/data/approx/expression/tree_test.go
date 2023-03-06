package expression

import (
	"github.com/stretchr/testify/require"
	"testing"
)

func TestSTree_string(t *testing.T) {
	tree := &sTree{
		root: "0-a",
		children: []*sTree{
			{root: "1-a", children: []*sTree{
				{root: "2-a"},
				{root: "2-b", children: []*sTree{
					{root: "3-a"},
				}},
			}},
			{root: "1-b", children: nil},
			{root: "1-c", children: []*sTree{
				{root: "2-c", children: nil},
				{root: "2-d", children: []*sTree{
					{root: "3-b", children: []*sTree{
						{root: "4-a", children: []*sTree{
							{root: "5-a"},
						}},
					}},
					{root: "3-c"},
				}},
				{root: "2-e"},
			}},
		},
	}

	t.Log("\n" + tree.string())
}

func TestSplitBalanced(t *testing.T) {
	tests := []struct {
		name     string
		in       string
		expected []string
		err      bool
	}{
		{
			name:     "correct input",
			in:       "1-a 1-b 1-c (2-a 2-b 2-c (3-a)) (2-d 2-e) 1-d (2-f)",
			expected: []string{"1-a", "1-b", "1-c", "(2-a 2-b 2-c (3-a))", "(2-d 2-e)", "1-d", "(2-f)"},
		},
		{
			name: "imbalanced input",
			in:   "1-a (1-b 1-c (2-a 2-b 2-c (3-a)) (2-d 2-e) 1-d (2-f)",
			err:  true,
		},
		{
			name:     "empty input",
			expected: []string{},
		},
	}

	for i := range tests {
		t.Run(tests[i].name, func(t *testing.T) {
			balanced, err := splitBalanced(tests[i].in, []rune(" ")[0])
			if tests[i].err {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
				require.Equal(t, len(tests[i].expected), len(balanced))
				for j, str := range tests[i].expected {
					require.Equal(t, str, balanced[j])
				}
			}
		})
	}
}

func TestSplitRecursively(t *testing.T) {
	tests := []struct {
		name     string
		in       string
		expected *sTree
		err      bool
	}{
		{
			name: "correct input",
			in:   "(0-a (1-a (2-a 2-b (3-a)) 1-b 1-c (2-c 2-d (3-b (4-a (5-a)) 3-c) 2-e)))",
			expected: &sTree{
				root: "(0-a (1-a (2-a 2-b (3-a)) 1-b 1-c (2-c 2-d (3-b (4-a (5-a)) 3-c) 2-e)))", children: []*sTree{
					{root: "0-a", children: []*sTree{}},
					{root: "(1-a (2-a 2-b (3-a)) 1-b 1-c (2-c 2-d (3-b (4-a (5-a)) 3-c) 2-e))", children: []*sTree{
						{root: "1-a", children: []*sTree{}},
						{root: "(2-a 2-b (3-a))", children: []*sTree{
							{root: "2-a", children: []*sTree{}},
							{root: "2-b", children: []*sTree{}},
							{root: "(3-a)", children: []*sTree{
								{root: "3-a", children: []*sTree{}},
							}},
						}},
						{root: "1-b", children: []*sTree{}},
						{root: "1-c", children: []*sTree{}},
						{root: "(2-c 2-d (3-b (4-a (5-a)) 3-c) 2-e)", children: []*sTree{
							{root: "2-c", children: []*sTree{}},
							{root: "2-d", children: []*sTree{}},
							{root: "(3-b (4-a (5-a)) 3-c)", children: []*sTree{
								{root: "3-b", children: []*sTree{}},
								{root: "(4-a (5-a))", children: []*sTree{
									{root: "4-a", children: []*sTree{}},
									{root: "(5-a)", children: []*sTree{
										{root: "5-a", children: []*sTree{}},
									}},
								}},
								{root: "3-c", children: []*sTree{}},
							}},
							{root: "2-e", children: []*sTree{}},
						}},
					}},
				}},
		},
		{
			name: "imbalanced input",
			in:   "((0-a (1-a (2-a 2-b (3-a)) 1-b 1-c (2-c 2-d (3-b (4-a (5-a)) 3-c) 2-e)))",
			err:  true,
		},
		{
			name: "correct simple input",
			in:   "(0-a 0-b (1-a) 0-c (1-b 1-c))",
			expected: &sTree{
				root: "(0-a 0-b (1-a) 0-c (1-b 1-c))", children: []*sTree{
					{root: "0-a", children: []*sTree{}},
					{root: "0-b", children: []*sTree{}},
					{root: "(1-a)", children: []*sTree{
						{root: "1-a", children: []*sTree{}},
					}},
					{root: "0-c", children: []*sTree{}},
					{root: "(1-b 1-c)", children: []*sTree{
						{root: "1-b", children: []*sTree{}},
						{root: "1-c", children: []*sTree{}},
					}},
				},
			},
		},
	}

	for i := range tests {
		t.Run(tests[i].name, func(t *testing.T) {
			tree, err := splitRecursively(tests[i].in)
			if tests[i].err {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
				require.True(t, tests[i].expected.equal(tree))
			}
		})
	}
}

func TestSTree_copy(t *testing.T) {
	tree := &sTree{
		root: "0-a",
		children: []*sTree{
			{root: "1-a", children: []*sTree{
				{root: "2-a"},
				{root: "2-b", children: []*sTree{
					{root: "3-a"},
				}},
			}},
			{root: "1-b", children: nil},
			{root: "1-c", children: []*sTree{
				{root: "2-c", children: nil},
				{root: "2-d", children: []*sTree{
					{root: "3-b", children: []*sTree{
						{root: "4-a", children: []*sTree{
							{root: "5-a"},
						}},
					}},
					{root: "3-c"},
				}},
				{root: "2-e"},
			}},
		},
	}

	cp := tree.copy()

	require.True(t, tree != cp)
	require.True(t, tree.equal(cp))
}
