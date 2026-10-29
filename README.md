# Hispanic Heritage Historical Data Explorer

A Streamlit classroom app for exploring World Bank historical indicators for nine Latin American countries, fitting polynomial trends, editing a transparent scenario table, calculating values, comparing countries, and exploring functions/calculus.

## Research question and team choices

Customize the research question in the app's Research setup panel and in your presentation. A suggested question is:

> How does the historical trend in a selected social, demographic, or economic indicator differ between two Latin American countries, and how reliable is a polynomial model beyond the observed years?

Choose a featured country, indicator, comparison country, and mathematics pathway (Honors Algebra 2, Precalculus, or Calculus). Brazil is included as a Latin American comparison country; Portuguese is Brazil's principal language, so it is not a Spanish-speaking country.

## Files

- `app.py` — complete Streamlit application.
- `requirements.txt` — minimal unpinned dependencies.
- `requirements` — same dependency list without an extension, if your workflow explicitly requires that filename.
- `sources.md` — indicator definitions and official source references.

## Run locally

Use Python 3.10–3.12 for broad compatibility with Streamlit and the scientific Python stack.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python -m streamlit run app.py
```

The app retrieves data from the public World Bank API at runtime. An internet connection is required for fresh API retrieval. Responses are cached for one hour. No API key or credential is required.

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, `README.md`, and `sources.md` to the repository root.
3. Open Streamlit Community Cloud and connect the GitHub repository.
4. Select the repository, branch, and `app.py` as the main file.
5. Deploy. Check the app's logs if installation or startup fails.

## Deploy to Render

Create a **Python Web Service** connected to the repository.

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `python -m streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT`

If Render's shell does not expand `$PORT` in your selected environment, inspect the deploy logs and use Render's documented environment-variable behavior. The command above is the requested single-line start command.

## Verification checklist

1. **Source accuracy:** select several countries/indicators; compare the API endpoint, indicator code, units, and coverage with the World Bank indicator page. Confirm null years are not filled with fabricated numbers.
2. **Editing/reset:** change a value and observe the user-modified scenario warning; press Reset to source and confirm the source values return.
3. **Validation:** try duplicate years, fractional years, blank/nonnumeric values, and an invalid indicator-specific value. Modeling should be blocked until corrected.
4. **Insufficient observations:** select a high polynomial degree or edit the table to fewer than degree + 1 distinct years. The app should explain the minimum-data requirement.
5. **Equation and metrics:** check `t = year - base year`, degree, in-sample R², in-sample MAE, fitted values, and residuals. R² is not proof of causation.
6. **Single/batch agreement:** calculate one year with Calculate Value, then request the same year in a batch. Values should agree up to floating-point display precision.
7. **Invalid batch input:** enter `2020, 20xx, 2050` or a year outside 1950–2100. The app should show a validation message.
8. **Controls:** change country, indicator, degree, sample increment, forecast horizon, comparison country, course, context, function, graph toggles, interval, current x, and simulation type. Confirm each affects its intended output.
9. **Exports:** download and open CSV, `.xlsx`, and PDF. Verify that the workbook has summary, working data, residuals when available, and calculated-year sheets. Verify the PDF opens and includes metadata, equation/metrics when fitted, chart, and previews.
10. **API failure:** test with internet disabled or a temporary network block. The app should show a timeout/request/no-data message rather than substituting random values.
11. **Deployment:** open the deployed app in a fresh browser, check logs, and repeat the core checks.

## Likely errors

- **World Bank request timeout / 5xx:** retry later or choose a different indicator/country. The app does not fabricate replacement data.
- **No observations:** some indicators have country/year gaps; inspect the API endpoint and indicator page. Missing values are excluded, not treated as zero.
- **Polynomial degree error:** degree `d` requires at least `d+1` distinct valid years.
- **Data editor validation error:** ensure years are unique integers and values are numeric/finite and satisfy the app's indicator checks.
- **Dependency installation error:** inspect deployment logs and use a supported Python version (3.10–3.12 recommended); avoid pinning random versions without diagnosing the conflict.
- **PDF export issue:** inspect ReportLab stack traces in logs and check that data are not malformed.
- **Function parsing error:** use supported notation such as `x**2`, `1/(x-2)`, `exp(-x)`, `log(x)`, `sqrt(x)`, `sin(x)`, or `cos(x)`. This is not a general Python execution box.
- **Graph gaps:** undefined rational/logarithmic values are masked as NaN; graph sampling may still only approximate where a discontinuity lies.

## How to explain the app

### Functions and data flow
`cached_wb_fetch` retrieves JSON from the official World Bank API and caches it. The source table is retained separately from the editable working table. Validation runs before fitting. `fit_polynomial` centers time with `t = year - base_year`, then fits a polynomial regression. Calculated years use the same centered transformation. Exports are generated from the current selection and working scenario.

### Mathematics and assumptions
A degree-`d` polynomial is a descriptive curve fit, not a causal model. Centering time makes coefficients easier to work with numerically but does not remove overfitting risk. In-sample R² and MAE describe fit on the same observations used to train the model. A chronological holdout MAE is shown separately when enough data exist. Polynomial extrapolation can grow rapidly and become implausible; no prediction interval is claimed.

In the function lab, SymPy differentiates supported symbolic expressions. The app numerically checks derivative signs around candidate critical points and concavity changes around candidate inflection points. Numerical sampling can miss roots, domain gaps, or narrow features; reported classifications are approximations where noted. Absolute-extrema candidates are only as reliable as the detected critical points and domain coverage.

### Limitations
- API coverage varies by country and indicator; the requested date range does not guarantee 70 observations.
- Edited values are scenarios and must not be represented as original World Bank data.
- Indicator definitions and methods can change over time.
- Historical trends do not establish why a change occurred.
- Polynomial forecasts are not official forecasts.
- The context visuals are scaled classroom illustrations, not physical or medical simulations.
- The app is for education, not policy, investment, diagnosis, or treatment decisions.
