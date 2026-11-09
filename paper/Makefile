LATEX = pdflatex -interaction=nonstopmode -halt-on-error

.PHONY: all clean arxiv

all: main.pdf

main.pdf: main.tex sections/*.tex references.bib
	$(LATEX) main
	bibtex main
	$(LATEX) main
	$(LATEX) main

# Build a self-contained tarball for arXiv submission.
# arXiv compiles from a flat directory, so figures are copied in alongside the
# sources and the bbl is shipped (arXiv does not run bibtex).
arxiv: main.pdf
	rm -rf arxiv-build arxiv-submission.tar.gz
	mkdir -p arxiv-build
	cp main.tex tmlr.sty tmlr.bst algorithm.sty algorithmic.sty fancyhdr.sty references.bib main.bbl arxiv-build/ 2>/dev/null || true
	cp -r sections arxiv-build/
	mkdir -p arxiv-build/images
	cp -r ../images/. arxiv-build/images/
	cd arxiv-build && tar -czf ../arxiv-submission.tar.gz .
	@echo "Created arxiv-submission.tar.gz"

clean:
	rm -f *.aux *.log *.out *.bbl *.blg *.toc main.pdf
	rm -rf arxiv-build arxiv-submission.tar.gz
