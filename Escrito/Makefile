TEX_FILE = tesis

all: tesis.pdf

other:
	pdflatex  -shell-escape $(TEX_FILE)
	bibtex $(TEX_FILE)
	pdflatex  -shell-escape $(TEX_FILE)
	pdflatex  -shell-escape $(TEX_FILE)

$(TEX_FILE).pdf: $(TEX_FILE).tex ./img/*.png
	arara $(TEX_FILE).tex

.PHONY: tesis
