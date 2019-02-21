# Publication Documentation
al-Nashrah al-Ifriqiyah al-Asyawiyah

### Place of Publication
Cairo, Egypt

### Language
Arabic

### Publication House
The Permanent Secretariat of Afro-Asian People's Solidarity Organization
(al-Sikritārīyah al-Dāʼimah li-Tiḍāmun al-Shuʻūb al-Ifrīqīyah al-Āsyawīyah)
Dār al-Kutub al-Miṣrīyah. Fihrist al-Dawrīyāt
### Frequency Published
Monthly

### Publication Run
1958-1967

### Dates/Volumes Contained
- 1961-1967 from Hathi Trust

### Link to Digitized Copy
[https://catalog.hathitrust.org/Record/008564849](https://catalog.hathitrust.org/Record/008564849)

### Date Last Compiled
2019-02-01

### Code Snippets
`process_hathitrust.py`
```
# in get_hathi()

title = hathi_vol.split('._')[0]
dates = hathi_vol.split('_')[-1]
v = ('_').join(hathi_vol.split('._')[1:])
vols = ('_').join(v.split('_')[:-1])
              
```