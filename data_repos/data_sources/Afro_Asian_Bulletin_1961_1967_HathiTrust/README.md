# Publication Documentation
Afro-Asian Bulletin

### Place of Publication
Cairo, Egypt

### Language
English

### Publication House
The Permanent Secretariat of Afro-Asian People's Solidarity Organization

### Frequency Published
Monthly

### Publication Run
1958-1968

### Dates/Volumes Contained
- 1961-1967 from Hathi Trust

### Link to Digitized Copy
[https://catalog.hathitrust.org/Record/000055797](https://catalog.hathitrust.org/Record/000055797)

### Date Last Compiled
2019-02-01

### Code Snippets
`process_hathitrust.py`
```
# in get_hathi()

title = hathi_vol.split('_:')[0]
dates = hathi_vol.split('_')[-1]
vols = ('_').join(hathi_vol.split('._')[1].split('_')[:-1])
```