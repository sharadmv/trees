default: compile

compile:
	pdflatex paper.tex
	bibtex paper
	pdflatex paper.tex
	pdflatex paper.tex

clean:
	rm *.aux *.log *.pdf *.out *.bbl *.blg
