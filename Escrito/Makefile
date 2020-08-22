TEX_FILE = tesis
CODE_DIR = ./code

all: tesis.pdf

other: $(TEX_FILE).tex ./img/*.png $(CODE_DIR)/*.py
	pdflatex  -shell-escape $(TEX_FILE)
	bibtex $(TEX_FILE)
	pdflatex  -shell-escape $(TEX_FILE)
	pdflatex  -shell-escape $(TEX_FILE)

$(TEX_FILE).pdf: $(TEX_FILE).tex ./img/*.png $(CODE_DIR)/*.py
	arara $(TEX_FILE).tex

.PHONY: tesis
