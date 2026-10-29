# Sources and indicator notes

## Primary data source

**World Bank Indicators API** — official public API documentation:  
https://datahelpdesk.worldbank.org/knowledgebase/articles/889392-about-the-indicators-api-documentation

API pattern used by the app:

`https://api.worldbank.org/v2/country/{ISO3}/indicator/{INDICATOR_CODE}?date={START}:{END}&format=json&per_page=20000`

The app requests the current year minus 70 through the current year minus 1, inclusive. It reports actual non-null observation coverage and count because the API may have missing years. Data retrieval date is shown in the app and exports. Public World Bank API access does not require a credential.

## Country coverage

Mexico (MEX), Brazil (BRA), Argentina (ARG), Colombia (COL), Chile (CHL), Peru (PER), Venezuela (VEN), Ecuador (EC), and Uruguay (URY). Brazil is a Latin American comparison country; Portuguese is its principal language.

## Indicators

Each entry below lists the exact World Bank indicator code used. The linked indicator pages are official World Bank Data pages.

| Indicator | Code | Units shown by app | Definition / caution |
|---|---|---|---|
| Life expectancy at birth | `SP.DYN.LE00.IN` | years | Average years a newborn would live if current mortality patterns continued. [World Bank](https://data.worldbank.org/indicator/SP.DYN.LE00.IN) |
| Unemployment, national estimate | `SL.UEM.TOTL.NE.ZS` | % of labor force | National-estimate unemployment rate. [World Bank](https://data.worldbank.org/indicator/SL.UEM.TOTL.NE.ZS) |
| GDP per capita | `NY.GDP.PCAP.CD` | current US dollars per person | GDP divided by midyear population; not average wealth or household income. [World Bank](https://data.worldbank.org/indicator/NY.GDP.PCAP.CD) |
| Crude birth rate | `SP.DYN.CBRT.IN` | births per 1,000 people per year | Annual live births per 1,000 population; not a birth count. [World Bank](https://data.worldbank.org/indicator/SP.DYN.CBRT.IN) |
| Net migration | `SM.POP.NETM` | people, net total | Immigrants minus emigrants over the period; not an emigration count. [World Bank](https://data.worldbank.org/indicator/SM.POP.NETM) |
| Intentional homicide rate | `VC.IHR.PSRC.P5` | per 100,000 people | Intentional homicide victims per 100,000 population. [World Bank](https://data.worldbank.org/indicator/VC.IHR.PSRC.P5) |
| Consumer price inflation | `FP.CPI.TOTL.ZG` | % annual change | Annual change in the consumer price index. [World Bank](https://data.worldbank.org/indicator/FP.CPI.TOTL.ZG) |
| Central government debt | `GC.DOD.TOTL.GD.ZS` | % of GDP | Central government debt as a share of GDP. [World Bank](https://data.worldbank.org/indicator/GC.DOD.TOTL.GD.ZS) |
| Total reserves including gold | `FI.RES.TOTL.CD` | current US dollars | Total reserves including gold. [World Bank](https://data.worldbank.org/indicator/FI.RES.TOTL.CD) |
| External balance on goods and services | `NE.RSB.GNFS.ZS` | % of GDP | External balance on goods and services as a share of GDP. [World Bank](https://data.worldbank.org/indicator/NE.RSB.GNFS.ZS) |

## Methodology and limitations

- Null observations are omitted from the modeling table and never replaced with random or fabricated data.
- The app uses a polynomial regression of degree 3–8 with centered time `t = year − base year`.
- R² and MAE are labeled in-sample. A chronological holdout MAE, where available, is shown separately.
- A polynomial fit is descriptive and does not establish causation. Extrapolation is especially uncertain and may be implausible.
- The app does not treat ±MAE as a 95% prediction interval.
- The editable table is a user-modified scenario and must be distinguished from official source observations.
