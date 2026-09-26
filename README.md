# MCP-ECO

Data and analysis code for a study of the Model Context Protocol (MCP) server ecosystem:
capability, disclosure and provenance signals, and their relationship to adoption.

The manuscript is currently under review. Results, tables and figures are **not** included
in this repository; they are reproduced by running the code on the data provided here.

## Repository layout

```
MCP-ECO/
├── README.md
├── LICENSE                  (code)
├── LICENSE-DATA             (data)
├── checksums.csv
├── code/
│   ├── config.py
│   ├── utils.py
│   ├── 01_prepare.py
│   ├── 02_validation.py
│   ├── 03_reclassify.py
│   ├── 04_features.py
│   ├── 05_rq1_lca.py
│   ├── 06_rq2_adoption.py
│   ├── 07_rq3_disclosure.py
│   ├── 08_rq4_language.py
│   ├── 09_panel_robustness.py
│   ├── 10_report.py
│   ├── run_all.py
│   ├── prepare_release.py
│   └── requirements.txt
└── data/
    ├── servers.csv.gz
    ├── features_server.csv.gz
    ├── tools.csv.gz
    ├── downloads_weekly.csv.gz
    ├── packages.csv.gz
    ├── packages_enriched.csv.gz
    ├── repos.csv.gz
    ├── probe_log.csv.gz
    ├── tool_versions.csv.gz
    ├── server_sources.csv.gz
    ├── sample_flow.csv.gz
    ├── validation_sample_filled.csv.gz
    ├── vendor_validation_sample_filled.csv.gz
    └── dictionaries/
```

## Data

| File | Unit of observation | Contents |
|---|---|---|
| `servers.csv` | server | Registry census: namespace, verification method, publisher, repository and package links, sample indicators |
| `features_server.csv` | server | Server-level capability, disclosure and category features |
| `tools.csv` | tool | Tool inventories: name, description, input schema, behavioural annotations, rule-based flags, inventory source |
| `downloads_weekly.csv` | package × ISO week | Weekly download counts with spike and partial-week flags |
| `packages.csv`, `packages_enriched.csv` | package | Package metadata, release history, provenance attestation, download window |
| `repos.csv` | repository | Owner type, stars, licence, continuous integration, security policy, releases |
| `probe_log.csv` | probe attempt | Every capability-probe attempt and its outcome, including failures |
| `tool_versions.csv` | package × version | Tool inventories for earlier releases (within-package panel) |
| `server_sources.csv` | server × source | Which registry or directory listed each server |
| `sample_flow.csv` | stage | Collection and deduplication record with counts |
| `validation_sample_filled.csv` | tool | 300 human-coded tools used to validate the tool-level measures |
| `vendor_validation_sample_filled.csv` | server | 100 human-coded servers used to validate connected service and publisher-vendor identity |
| `dictionaries/` | — | Versioned keyword dictionaries and coding rules |

Files are gzipped; the code reads `.csv.gz` directly, so no manual decompression is needed.
`checksums.csv` lists the SHA-256 hash of every data file.

### Collection summary

The census snapshot was taken in September 2026 from public application programming
interfaces. Download histories cover the 365 days (npm) and 181 days (Python Package Index)
preceding the snapshot. Tool inventories were obtained metadata-only: only the protocol
handshake and the tool-listing call were issued. **No tool was ever invoked and no
credentials were supplied.** Local probes ran in disposable containers with installation
scripts disabled and the network disabled before start-up. Every probe attempt, including
every failure, is recorded in `probe_log.csv`.

### Responsible release

Descriptions matching injection-style patterns (for example, instructing a model to conceal
actions from the user or to ignore prior instructions) are withheld: 169 tool records carry
`description_redacted = TRUE`, with the description and title replaced by a redaction marker.
The aggregate measures used in the analysis are unaffected, so all reported results reproduce
from the released files. Matching expressions are in `data/dictionaries/`.

Human coder identifiers are pseudonymous (`ID_01`, `ID_02`, …) and carry no personal data.
All other content originates from public registries, package indexes and repositories.

## Reproducing the analysis

```bash
git clone https://github.com/arad1367/MCP-ECO.git
cd MCP-ECO/code
pip install -r requirements.txt
python run_all.py
```

Run time is roughly 15 to 25 minutes, most of it in `05_rq1_lca.py` (latent class models and
bootstrapped likelihood-ratio tests). Results are written to `output/tables`,
`output/figures` and `output/logs`, none of which are tracked in this repository.

Scripts may also be run individually in numbered order:

| Script | Purpose |
|---|---|
| `01_prepare.py` | Loads sources, builds server- and tool-level base files, derives signals, controls, exposure windows and the version panel |
| `02_validation.py` | Validates rule-based measures against the human coding; design-weighted prevalence |
| `03_reclassify.py` | Cross-validated supervised re-estimation of weak measures; final tool flags |
| `04_features.py` | Server-level features from final flags; analysis samples; selection balance |
| `05_rq1_lca.py` | Descriptives, co-occurrence and latent class analysis |
| `06_rq2_adoption.py` | PPML and negative binomial adoption models with clustered and bootstrap inference |
| `07_rq3_disclosure.py` | Declared versus measured agreement, misdeclaration rates, logistic models |
| `08_rq4_language.py` | Model-directed language prevalence, correlates and adoption association |
| `09_panel_robustness.py` | Within-package version panel and robustness battery |
| `10_report.py` | Key-results summary and figures |

`prepare_release.py` is the script that produced this repository's `data/` folder from the
raw collection output, including the redaction step.

Everything is deterministic: the random seed is fixed in `config.py`, and repeated runs
reproduce identical estimates.

## Environment

Tested with Python 3.12 and pandas 3.0.2, numpy 2.4.4, scipy 1.17.1, statsmodels 0.15.0,
scikit-learn 1.8.0 and matplotlib 3.10.8. The minimums pinned in `requirements.txt` are
lower; recent versions should work.

## Licence

Code is released under the MIT Licence. Data are released under CC BY 4.0. The underlying
registry, package-index and repository content remains subject to the terms of its
respective sources.

## Citation

Citation details will be added once the manuscript is published. Until then, please cite
this repository by its URL and commit hash.

## Contact
Pejman Ebrahimi
- Department of Information Systems & Computer Science, University of Liechtenstein, Liechtenstein
- email: pejman.ebrahimi@uni.li and pejman.ebrahimi77@gmail.com , ORCID: https://orcid.org/0000-0003-0125-3707 

