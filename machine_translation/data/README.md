# Datasets

## Europarl

This directory should contain [Europarl datasets](https://www.statmt.org/europarl)
for the languages to be used.

For example you can obtain the Romanian-English dataset via:
```bash
$ curl https://www.statmt.org/europarl/v7/ro-en.tgz | tar -xzv
```

## Riksrevisjonen

We also support (Bokmål) Norvegian-English bilingual corpuses produced with data from the [Riksrevisjonen](https://data.europa.eu/data/datasets/elrc_1061) and [Responsible Business Norway](https://live.european-language-grid.eu/catalogue/corpus/3006).

Obtain and prepare the *Riksrevisjonen* dataset via:
```bash
$ curl https://elrc-share.eu/repository/download/a5d2470201e311e9b7d400155d0267060fffdc9258a741659ce9e52ef15a7c26/ | funzip | python3 tmxconvert.py
```

Obtain and prepare the *Responsible Business Norway* dataset via:
```bash
$ curl https://elrc-share.eu/repository/download/b80b2426e70811e7b7d400155d026706978b3385b0154ef18b2c6a2c3fb739ec/ | funzip | python3 tmxconvert.py
```
