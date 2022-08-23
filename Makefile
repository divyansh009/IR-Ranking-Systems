PYTHON = python3

# Defining an array variable
FILES = input output

# This target is executed whenever we just type `make`
.DEFAULT_GOAL = help

# The @ makes sure that the command itself isn't echoed in the terminal
help:
	@echo "---------------HELP-----------------"
	@echo "To setup the project type make setup --optional"
	@echo "To run the boolean retrieval model type make boolean"
	@echo "To run the TF-IDF model type make tfidf"
	@echo "To run the BM25 model type make bm25"
	@echo "------------------------------------"

# This generates the desired project file structure
# A very important thing to note is that macros (or makefile variables) are referenced in the target's code with a single dollar sign ${}, but all script variables are referenced with two dollar signs $${}
setup:	
	@echo "Checking if project files are generated..."
	@[ -d data ] || (echo "No directory found, generating..." && mkdir data)

# The ${} notation is specific to the make syntax and is very similar to bash's $() 	
run:
	@unzip -o 21111027-ir-systems.zip
	@unzip -o 21111027-qrel.zip
	@rm -rf Output_files
	@mkdir Output_files
ifdef filename
	@${PYTHON} All/final.py ${filename}
else
	@echo 'filename not given. Using default filename'
	@${PYTHON} All/final.py  query.txt
endif

