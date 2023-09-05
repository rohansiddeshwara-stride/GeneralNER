#!/bin/bash

# Create and activate a Python virtual environment
pip install virtualenv

python -m venv env
source env/bin/activate

# Install Flair
pip install flair

# Download Stanford CoreNLP
wget https://nlp.stanford.edu/software/stanford-corenlp-4.5.4.zip

# Unzip Stanford CoreNLP
unzip stanford-corenlp-4.5.4.zip -d stanford-corenlp-4.5.4

# Change directory to Stanford CoreNLP folder
cd stanford-corenlp-4.5.4

# Start Stanford CoreNLP server
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -annotators "tokenize,ssplit,pos,lemma,ner" -ner.useSUTime false &

# Return to the previous directory
cd ..

# Install Stanza
pip install stanza

# Upgrade Spacy and download the large English model
pip install -U spacy
python -m spacy download en_core_web_lg

# Clean up the downloaded zip file
rm stanford-corenlp-4.5.4.zip


