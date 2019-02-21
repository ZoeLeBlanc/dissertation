# Publication Documentation
Nahḍat Ifrīqīyah, African Renaissance

### Place of Publication
Cairo, Egypt

### Language
Arabic

### Publication House
African Affairs Bureau and Cairo University. 	Editor: Muḥammad ʻAbd al-ʻAzīz Isḥaq

### Frequency Published
Monthly

### Publication Run
November 1957 - February 1964

### Dates/Volumes Contained
- 1962-1964 from Hathi Trust

### Link to Digitized Copy
[https://catalog.hathitrust.org/Record/000063924](https://catalog.hathitrust.org/Record/000063924)

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