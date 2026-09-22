# 3D ICE JOSS pre-submission audit

Audit date: 17 August 2026

Sources: the current 3D ICE repository, the public GitHub repository, and the
current [JOSS review checklist](https://joss.readthedocs.io/en/latest/review_checklist.html),
[review criteria](https://joss.readthedocs.io/en/latest/review_criteria.html),
[paper format](https://joss.readthedocs.io/en/latest/paper.html), and
[scope guidance](https://joss.theoj.org/about).

## Readiness decision

**Current recommendation: do not submit yet.** 3D ICE has strong foundations
for a JOSS submission—an OSI-approved license, tagged releases, automated tests,
continuous integration, data provenance metadata, contributor guidance, and a
working JOSS build—but it still has three material pre-review risks:

1. The public GitHub repository was created on 21 March 2026. JOSS asks for at
   least six months of public development history, preferably with releases,
   public issues or pull requests, and external engagement. The earliest
   six-month date is 21 September 2026.
2. 3D ICE is a web-based research tool. JOSS expects such submissions to expose
   a core library or demonstrate unusually strong domain modelling, modularity,
   testing, and local verifiability. The binary/metadata data contract and test
   suite are good evidence, but the architectural case needs to be explicit and
   the browser core should continue moving out of the large HTML entry point.
3. The project is effectively single-author and GitHub currently reports no
   public issues or pull requests. The paper does not yet contain evidence of
   external use, community feedback, presentations, or research outputs enabled
   by the software.

Submission after 21 September is not automatically safe: the intervening public
history should show normal project work, a substantive release, and genuine
feedback or use where available.

## Repository evidence snapshot

| Evidence | Observation on 17 August 2026 |
| --- | --- |
| Public repository | `yuwang115/3d-ice`, created 21 March 2026 |
| License | Plain-text MIT `LICENSE` file |
| Development history | 82 commits in the current history; activity in February, March, April, and August 2026 |
| Contributors | One effective code contributor represented by two author identities |
| Releases | `v0.1.0`, `v0.1.1`, and `v0.1.2`, all published 21 March 2026 |
| Public collaboration | GitHub currently reports 0 issues and 0 pull requests |
| Quality assurance | Python, JavaScript, metadata, bundle-smoke, and browser E2E jobs pass in CI |
| JavaScript coverage | 99.11% line coverage for the three tested polar-feature modules |
| Paper build | JOSS draft workflow produces the `joss-paper` artifact |
| Packaging | Python dependencies declared in `pyproject.toml`; tagged compatibility bundle workflow present |
| Community pathways | `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, issue reporting guidance, and a support email are present |
| Citation metadata | `CITATION.cff` and `codemeta.json` are validated in CI; archival DOI is not yet available |

## JOSS checklist assessment

Status meanings: **Pass** = evidence is already sufficient; **Partial** = likely
reviewable but should be strengthened; **Blocker** = material desk-screening or
acceptance risk.

| JOSS area | Status | Evidence and gap | Required action |
| --- | --- | --- | --- |
| Open repository | Pass | Source is publicly browsable on GitHub and users can open issues or propose changes. | Keep the repository public throughout review. |
| OSI-approved license | Pass | A complete MIT license is present. | No action. |
| Author contribution | Pass | Yu Wang is the dominant contributor in the commit history. | Confirm whether anyone made substantial non-code contributions that warrant authorship; funding or general supervision alone is insufficient. |
| Research purpose | Pass | The software integrates research-grade cryosphere datasets for communication, teaching, and cross-domain exploration. | Keep the research purpose prominent in the paper and documentation. |
| Web-software scope | Partial / high risk | The offline pipeline, explicit binary/metadata contract, worker boundary, modules, and tests demonstrate domain modelling. The main browser application is still concentrated in a large HTML entry point. | Add an architecture document and continue extracting testable domain logic into modules. Frame the data contract and polar rendering decisions clearly in the paper. |
| Development timeline | Blocker until at least 21 September 2026 | The public repository is less than six months old and activity is concentrated in a few bursts. | Continue visible, substantive development and defer submission until the public-history threshold has been crossed. |
| Open development | Blocker / high risk | Three releases exist, but all were published on the repository's first day; there are no retained public issues or pull requests. | Use issues and pull requests for genuine future changes, publish a later release, and document decisions publicly. Do not manufacture engagement. |
| Collaborative effort | Blocker / high risk | The project is single-author with no documented external engagement. JOSS accepts single-author projects when community use or influence is evidenced. | Gather verifiable examples of use, feedback, teaching, presentations, or requests. If none exist, obtain and document real domain-user feedback before submission. |
| Installation | Pass / verify once more | Static preview and editable Python installation are documented; dependencies are declared. | Perform the clean reviewer walkthrough in a fresh checkout before submission. |
| Functionality | Partial | CI exercises core data functions and browser paths, but not every scientific layer can be regenerated without large upstream source files. | Document one small, reproducible end-to-end scientific example and expected outputs. |
| Performance claims | Pass with caution | The paper makes qualitative responsiveness and deployment claims, not numerical speed claims. | Avoid unbenchmarked performance comparisons. If numerical claims are added, publish the benchmark method and results. |
| Installation documentation | Pass | README and contributing guide cover the browser runtime, Python pipeline, and test extras. | Update the documented Node.js minimum: the coverage command requires Node.js 22.5 or newer; CI uses Node.js 24. |
| Example usage | Partial | Commands are present, but there is no compact worked research scenario with inputs, outputs, and interpretation. | Add a reviewer-sized example using bundled or openly downloadable data. |
| Functionality/API documentation | Partial | Repository layout and commands are described, but preparation-script parameters and the binary/metadata contract lack a dedicated reference. | Add architecture and data-contract documentation outside the JOSS paper. |
| Automated tests | Pass | Five CI jobs cover Python, JavaScript, metadata, bundle compatibility, and browser E2E behaviour. | Preserve green CI and consider publishing Python coverage. |
| Community guidelines | Pass | Contribution, bug reporting, enhancement, support, and conduct pathways are present. | Add issue templates when genuine issue traffic begins. |
| Paper summary | Pass after current revision | A non-specialist description of purpose and major functionality is present. | Keep jargon and implementation detail out of the opening paragraph. |
| Statement of need | Pass after current revision | Problem, audience, and research context are explicit. | Keep detailed tool comparisons in the dedicated state-of-the-field section. |
| State of the field | Pass after current revision | The revised draft compares Quantarctica/QGIS, NASA Worldview, and CesiumJS and gives a build-versus-contribute rationale. | Ask a cryosphere/GIS colleague to check whether an important competing tool is missing. |
| Software design | Pass after current revision | The revised draft explains the offline/runtime split, data contract, quantization trade-off, worker boundary, and static deployment choice. | Support these claims with the planned architecture/data-contract documentation. |
| Research impact statement | Partial / high risk | The revised draft accurately presents reproducible materials and community-readiness signals, but there is no documented external adoption or enabled publication. | Replace or augment near-term-significance evidence with verified use, feedback, presentations, integrations, or publications before submission. |
| AI usage disclosure | Pass subject to author confirmation | The revised draft discloses known OpenAI Codex/GPT-5 assistance and human verification. | Confirm whether any additional AI tools or model versions were used in code, documentation, or paper work and add them before submission. |
| References | Partial | Dataset and related-software references are present. The tagged GitHub release is cited, but it is not a permanent archive. | After review, archive the accepted release with Zenodo or Figshare, add the DOI, and replace the provisional release citation. |
| Paper length and build | Pass | The target range is 750–1750 words and the automated draft workflow is available. | Recompile after every substantive paper change and inspect the PDF visually. |

## State-of-the-field evidence

| Tool | Primary strength | Difference from 3D ICE | Position in the paper |
| --- | --- | --- | --- |
| Quantarctica/QGIS | Comprehensive Antarctic data package, desktop GIS analysis environment, and visualization platform | Requires a local GIS workflow and is Antarctic-centred; 3D ICE prioritizes zero-install, curated 3D comparison of Antarctica and Greenland | Complementary: Quantarctica supports analysis; 3D ICE supports immediate contextual exploration and communication. |
| NASA Worldview | Rapid web browsing, comparison, animation, and download of global satellite imagery, including polar views | Optimized for image layers and time-sensitive Earth observation rather than 3D ice geometry, subsurface fields, and model-derived process layers | Complementary: Worldview supplies broad imagery access; 3D ICE supplies a domain-specific 3D view. |
| CesiumJS | General-purpose, high-precision WGS84 3D globe and map framework | A developer library rather than a cryosphere application; using or extending it would not provide the polar data preparation, provenance, curation, or scientific layer semantics | Build-versus-contribute justification: the scholarly contribution is the domain pipeline and data model, not a replacement rendering engine. |
| Three.js | General-purpose WebGL rendering library used by 3D ICE | Supplies graphics primitives but no geospatial or cryosphere data model | Reused directly so 3D ICE can focus on domain-specific software rather than reimplementing graphics infrastructure. |

## Evidence that requires author input

The following should not be inferred or invented. Supply only verifiable items:

- courses, lectures, workshops, conference demonstrations, or outreach events
  in which 3D ICE has been used;
- research groups, educators, students, or external users who have tried it;
- concrete feedback that changed a feature, dataset, interface, or workflow;
- publications, proposals, project reports, or research workflows that link to
  or use 3D ICE;
- usage metrics with a defined source and time window;
- domain collaborators who made substantial intellectual, project-direction,
  documentation, data-integration, or software contributions;
- any generative-AI tools or models used beyond the OpenAI Codex/GPT-5 work
  currently disclosed in the paper.

## Recommended submission gate

Submit only after all of the following are true:

- [ ] Public repository history exceeds six months.
- [ ] A substantive post-March release is published.
- [ ] Normal public issues/pull requests document ongoing work and decisions.
- [ ] At least one verifiable external-use or community-feedback example is in
      the research impact statement.
- [ ] Web-software scope is supported by architecture and data-contract docs.
- [ ] A compact worked scientific example can be reproduced from a clean clone.
- [ ] All tests and the JOSS draft build pass from the release candidate.
- [ ] The full author list and affiliation details are confirmed.
- [ ] The AI disclosure lists every tool/model and verification procedure.
- [ ] The accepted release is archived and its DOI is added to the paper and
      citation metadata.

