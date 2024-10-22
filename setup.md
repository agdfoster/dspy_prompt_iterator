# Setup instructions

## Activate your virtual environment (assuming it's already created)

source path/to/your/venv/bin/activate

## Install dependencies

pip install -r requirements.txt

## Install NLTK

pip install nltk

## Download NLTK data packages directly from the command line

python -m nltk.downloader punkt stopwords

## WEAT

For the WEAT analysis, you'll need to provide a path to pre-trained word vectors. You can download GloVe vectors from Stanford NLP or use any other suitable word embedding model.
Download GloVe vectors for WEAT analysis. We'll use the 50-dimensional vectors for this example:

```bash
mkdir -p data/word_vectors
cd data/word_vectors
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
cd ../..
```

Convert the GloVe vectors to word2vec format (which is required by gensim):

```bash
python -c "from gensim.scripts.glove2word2vec import glove2word2vec; glove2word2vec('data/word_vectors/glove.6B.50d.txt', 'data/word_vectors/glove.6B.50d.w2v.txt')"
```

Adjust the word sets in @metrics/weat_analysis.py to your needs.

```python
# The default word set won't be that useful for you, so adjust it to your needs.
male_terms = ['he', 'him', 'man', 'boy', 'father']
female_terms = ['she', 'her', 'woman', 'girl', 'mother']
career_terms = ['career', 'corporate', 'salary', 'office']
family_terms = ['home', 'family', 'children', 'parents']
```

## Config

- there's some congig in @extract_responses.py like NUM_ROWS_TO_PROCESS, USE_LARGE_DATASET, etc.

## REPORT GENERATION

To generate the report, simply run the main.py script as before. The report and figures will be saved in the analysis_output folder.
If you want to convert the Markdown report to PDF, you can use a tool like pandoc or a Python library like mdpdf. Here's how you can do it with pandoc:
1. Install pandoc and wkhtmltopdf (required for PDF conversion)
2. Run the following command:

```bash
pandoc analysis_output/llm_analysis_report.md -o analysis_output/llm_analysis_report.pdf --pdf-engine=wkhtmltopdf
```

This will create a PDF version of the report alongside the Markdown version.

## LLM ANALYSIS

1. the prompts for the LLM analysis are in @llm_analysis.py
2. you might want to adjust the prompts, they're first pass.
