The graphs are saved as json files 
as:
{"nodes": ["i_j", "k_l", "m_n"],
"edges": [["i_j", "k_l"], ["k_l", "m_n"]],
}

one json per patient

nodes are: "i_j" means: the node whose label is i in layer (aka timepoint) j.
edges are pairs of nodes. edges are directed forward (in the example: j<l, l<n)

** note: labels are >= 1. layers are >=0 and <N.
nodes with label 0 are placeholders for drawing purposes.

The node labels are specific to pre-labeled tumor file (namely, re-labling the tumor file does not guarantee json files staying correct without re-matching).
