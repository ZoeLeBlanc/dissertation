# Publication Documentation
Middle East News Economic Weekly

### Place of Publication
Cairo, Egypt

### Language
English

### Publication House
Middle East News Agency. Anbāʼ al-Sharq al-Awsaṭ.

### Frequency Published
Weekly

### Publication Run
1961-1979

### Dates/Volumes Contained
- 1964-1972 from Hathi Trust

### Link to Digitized Copy
[https://catalog.hathitrust.org/Record/008564927](https://catalog.hathitrust.org/Record/008564927)

### Date Last Compiled
2019-02-01

### Code Snippets
`process_hathitrust.py`
```
# in get_hathi()

title = hathi_vol.split('._')[0]
dates = ('_').join(hathi_vol.split('_')[-2:])
vols = ('_').join(hathi_vol.split('._')[-1].split('_')[0:2])           
```