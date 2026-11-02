# Blind listening test (paper, Sec. V "Listening test")

Open `index.html` in a browser (or host this folder, e.g. with GitHub Pages), enter a listener number
(1, 2, 3, ... — every listener gets a different but balanced assignment), answer the three parts
(~15 minutes, headphones), press **Download results** and send back `results_<n>.json`.

Score all returned files and write the paper's table:

    python research/build_human_eval.py --tally results_*.json --tex paper/figures/human_eval.tex

Rebuild the page from the renders: `python research/build_human_eval.py --out docs/listening_test`.
