#!/bin/bash
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode arxiv.tex > /dev/null; bibtex arxiv > bib.log 2>&1; pdflatex -interaction=nonstopmode arxiv.tex > /dev/null; pdflatex -interaction=nonstopmode arxiv.tex > build.log 2>&1
echo "pages $(pdfinfo arxiv.pdf | awk '/Pages/{print $2}')"
echo "overfull $(grep -c 'Overfull \\hbox' arxiv.log)"
grep -E "undefined|Undefined" arxiv.log | sort -u | head -20
grep -E "^! " arxiv.log | head
grep -i "warning" bib.log | head
