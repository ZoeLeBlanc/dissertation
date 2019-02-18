# This repository is for all python scripts required to gather and clean the datasets used in my dissertation.

### Description of Files

1. get_clean_hathitrust.py
This script gets Hathi Trust extracted features using either downloaded metadata or scraped ids that correspond to the Hathi Trust Volume Id. This script gets all the tokenlists of a volume and separates them out by page. 

1. hathi_trust_scraper.py
This script scrapes Hathi Trust Record page if the collection creator is not working. It creates a list of volume identifiers that can be used by the `get_clean_hathitrust.py` script.

1. get_clean_imagelucida.py

1. process_hathitrust.py and process_imagelucida.py
This script runs the hathi trust tokens or Image Lucida data through both nltk and spacy to remove OCR errors, and allows for removing of stopwords and aggregating by volume.

1. ner scripts

