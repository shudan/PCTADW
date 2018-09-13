======
PCTADW
======

Embeddings of directed network with text-associated nodes.

Installation
---------
python setup.py install

How to use
---------

**Example**
"$pctadw --input_text example.text --input_edges example.edges --model_name PCTADW-2 --output output.embedding "

**--input_text**:  *input_text_filename*
   The text in nth line is the text associated with node n.

**--input__edges**: *input_edges_filename*
   Each line is a directed edge pair, e.g.
   0 1
   1 2
   3 4
   ...
**--model_name**: *model_name*

   1. PCTADW-1

   2. PCTADW-2

**--output__**: *output_filename*
   The nth line is the representation vector for node n.

**Full Command List**
   The full list of command line options can be checked with "$pctadw -h"



