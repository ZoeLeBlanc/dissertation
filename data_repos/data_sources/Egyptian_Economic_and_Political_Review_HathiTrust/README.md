# Publication Documentation
Egyptian Economic and Political Science Review

### Place of Publication
Cairo, Egypt

### Language
English

### Publication House
I think this comes out of Cairo University, but need to double check. Worldcat lists Adel Mahmoud Sabit as the publisher [http://uva.worldcat.org/oclc/2260691](http://uva.worldcat.org/oclc/2260691)

### Frequency Published
Monthly

### Publication Run
1954-1962

### Dates/Volumes Contained
- 1954-1962 from Hathi Trust

### Link to Digitized Copy
[https://catalog.hathitrust.org/Record/000542442](https://catalog.hathitrust.org/Record/000542442)

### Date Last Compiled
2019-02-01

### Code Snippets
`process_hathitrust.py`
```
# in get_hathi()

hathi_vol = file.split('/')[-1].split('_grouped')[0]
title = hathi_vol.split('review')[0]
title = title + 'review'
dates = hathi_vol.split('review')[1].split('_')[-2:]
dates = ('-').join(dates)
vols = hathi_vol.split('review')[1].split('_')[:-2]
vols = ('_').join(vols)
```