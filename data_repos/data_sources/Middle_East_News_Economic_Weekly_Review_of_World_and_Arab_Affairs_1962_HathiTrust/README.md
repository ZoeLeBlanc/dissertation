# Publication Documentation
Middle East News Economic Weekly of World and Arab Affairs

### Place of Publication
Cairo, Egypt

### Language
English

### Publication House
Middle East News Agency. Anbāʼ al-Sharq al-Awsaṭ.

### Frequency Published
Weekly

### Publication Run
1962

### Dates/Volumes Contained
- 1962 from Hathi Trust

### Link to Digitized Copy
[https://catalog.hathitrust.org/Record/008564924](https://catalog.hathitrust.org/Record/008564924)

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