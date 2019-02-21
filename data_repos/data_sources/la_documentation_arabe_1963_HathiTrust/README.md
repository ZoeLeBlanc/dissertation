# Publication Documentation
La Documentation Arabe

### Place of Publication
Cairo, Egypt

### Language
French

### Publication House
Anbāʾ al-Sharq al-Awsaṭ.  National Publications House

### Frequency Published
Weekly

### Publication Run
 1962-1963

### Dates/Volumes Contained
- 1963 from Hathi Trust

### Link to Digitized Copy
[https://catalog.hathitrust.org/Record/007579658](https://catalog.hathitrust.org/Record/007579658)

### Date Last Compiled
2019-02-01

### Code Snippets
`process_hathitrust.py`
```
# in get_hathi()

title = hathi_vol.split('._')[0]
dates = hathi_vol.split('_')[-1]
vols = hathi_vol.split('._')[1].split('_')[0]
              
```