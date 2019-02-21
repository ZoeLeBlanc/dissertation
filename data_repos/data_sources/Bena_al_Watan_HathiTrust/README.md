# Publication Documentation
Bina al-Watan

### Place of Publication
Cairo, Egypt

### Language
Arabic

### Publication House
Egyptian National Publications House. al-Muʼassasah al-Miṣrīyah al-ʻĀmmah lil-Ibnāʼ wa-al-Nashr wa-al-Tawzīʻ wa-al-Ṭibāʻah

### Frequency Published
Monthly

### Publication Run
1958-1966 (Though 1958 is only referenced in Humum al Sahafiyah by Awatif so not sure if the real publication date)

### Dates/Volumes Contained
- 1962-1964 from Hathi Trust

### Link to Digitized Copy
[https://catalog.hathitrust.org/Record/000047730](https://catalog.hathitrust.org/Record/000047730)

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