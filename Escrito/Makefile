TEX_FILE = tesis

all: tesis.pdf

$(TEX_FILE).pdf: $(TEX_FILE).tex ./Imagenes/*
	pdflatex  -shell-escape $(TEX_FILE)
	bibtex $(TEX_FILE)
	pdflatex  -shell-escape $(TEX_FILE)
	pdflatex  -shell-escape $(TEX_FILE)

.PHONY: tesis
