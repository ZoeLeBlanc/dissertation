# Publication Documentation
al-Kitāb al-sanawī

### Place of Publication
Cairo, Egypt

### Language
Arabic

### Publication House
UAR Maṣlaḥat al-Istiʻlāmāt

### Frequency Published
Annually

### Publication Run
 1959-1966

### Dates/Volumes Contained
- 1962-1966 from Hathi Trust

### Link to Digitized Copy
[https://catalog.hathitrust.org/Record/003914125](https://catalog.hathitrust.org/Record/003914125)

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