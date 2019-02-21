# Publication Documentation
Afro-Asian and World Affairs

### Place of Publication
New Dehli, India

### Language
English

### Publication House
Institute of Afro-Asian and World Affairs

### Frequency Published


### Publication Run
1964-1968

### Dates/Volumes Contained
- 1964-1968 from Hathi Trust

### Link to Digitized Copy
[https://catalog.hathitrust.org/Record/000495647](https://catalog.hathitrust.org/Record/000495647)


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