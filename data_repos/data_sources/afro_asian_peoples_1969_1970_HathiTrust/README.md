# Publication Documentation
Afro-Asian People's

### Place of Publication
Cairo, Egypt

### Language
English

### Publication House
The Permanent Secretariat of Afro-Asian People's Solidarity Organization

### Frequency Published
Monthly

### Publication Run
1969-1972

### Dates/Volumes Contained
- 1969-1970 from Hathi Trust

### Link to Digitized Copy
[https://catalog.hathitrust.org/Record/006064527](https://catalog.hathitrust.org/Record/006064527)

### Date Last Compiled
2019-02-01

### Code Snippets
`process_hathitrust.py`
```
# in get_hathi()

title = hathi_vol.split('._')[0]
dates = hathi_vol.split('_')[-1]
vols = ('_').join(hathi_vol.split('._')[1].split('_')[:-1])
              
```