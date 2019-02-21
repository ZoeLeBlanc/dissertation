# Publication Documentation
African Recorder

### Related (Earlier, Later) Titles
African Chronicle

### Place of Publication
New Delhi, India

### Language
English

### Publication House
Need to research more about origins. Worldcat lists M. H. Samuel as the publisher [http://uva.worldcat.org/oclc/1461403](http://uva.worldcat.org/oclc/1461403)

### Frequency Published
BiWeekly

### Publication Run
1962-1999

### Dates/Volumes Contained
- 1967-1973 from Hathi Trust

### Link to Digitized Copy
[https://catalog.hathitrust.org/Record/000494970](https://catalog.hathitrust.org/Record/000494970)

### Date Last Compiled
2019-02-01

### Code Snippets
`process_hathitrust.py`
```
# in get_hathi()

df_hathi = pd.read_csv(directory + file)
hathi_vol = file.split('/')[-1].split('_grouped')[0]
print(hathi_vol)
title = hathi_vol.split('_v')[0]
dates = hathi_vol.split('_')[-1]
vols = hathi_vol.split('_')[-2]
```