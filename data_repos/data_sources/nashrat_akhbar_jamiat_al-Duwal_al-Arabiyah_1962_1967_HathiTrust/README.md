# Publication Documentation
Nashrat akhbār Jāmiʻat al-Duwal al-ʻArabīyah

### Place of Publication
Cairo, Egypt

### Language
Arabic

### Publication House
Idārat al-Istiʻlām wa-al-Nashr bi-al-Amānah al-ʻĀmmah li-Jāmiʻat al-Duwal al-ʻArabīyah, League of Arab States

### Frequency Published
Monthly

### Publication Run
1962-1969. Continued as Nashrah ikhbārīyah from 1969-1974

### Dates/Volumes Contained
- 1962-1967 from Hathi Trust

### Link to Digitized Copy
[https://catalog.hathitrust.org/Record/003839852](https://catalog.hathitrust.org/Record/003839852)

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