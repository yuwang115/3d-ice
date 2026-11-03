# Research-edition panel redesign

**Status:** agreed with the author on 2026-11-01. Phases 1 and 2 are built on `feat/research-panel` (see [Phase 1](#phase-1-structure) and [Phase 2](#phase-2-behaviour)); phase 3 is not.

**Scope:** the left panel of the research edition, which has two hand-written locale shells:

- `static/tools/3D-interactive-cryosphere-explorer.html`
- `static/zh/tools/3D-interactive-cryosphere-explorer.html`

The public edition (`/explore/`) is out of scope and must not change.

**Baseline:** `origin/main` at 2f51bfe (v0.3.1 plus CPOM, sea ice and ice-core sites), measured in a 1440 × 900 window.

## Why

The panel grew one feature at a time. It now holds all of this in a 340 px column:

- 30 checkboxes, 8 sliders, 7 selects and 6 buttons
- 11 legend blocks
- an 11-part Data Snapshot

| Measure (1440 × 900) | Today |
| --- | --- |
| Panel height, nothing switched on | 2,135 px (2.5 screens) |
| Panel height, ISMIP6 projection on | 3,849 px (4.4 screens) |
| First control from the top of the panel | 373 px |
| Top of Interactive Scenarios | 1,476 px |
| Distance from a layer toggle to its legend | 800–1,750 px |
| Projection year slider | about 190 px for 2015–2300 (1.5 years per pixel) |

The first 373 px, 43 % of the first screen, are logo, intro, tour link and language.

What goes wrong:

1. **View Controls mixes six kinds of things.** It holds data selection, display parameters, place search, twelve scientific layers, render options and viewer actions under one heading.
2. **Exclusivity is hidden.** Layers that paint the same surface switch each other off silently, and nothing comes back afterwards. Switching on the projection unticks up to eleven layers with no message. The full set of rules is in [Behaviour rules](#behaviour-rules).
3. **Legends are far from their toggles.** They are stacked at the bottom in a fixed order.
4. **Unavailable layers are handled two ways in Greenland.**
   - Hidden: basal melt, thermal driving, CPOM and the projection.
   - Greyed out without a reason: effective pressure and channels.

   The terrain options are named inconsistently too: Antarctica's name the product ("BedMachine v4 — Balanced") and Greenland's do not ("Balanced"). The dataset select truncates at 340 px.
5. **Time features are cramped.** The projection readout wraps to three lines, and the thickness-change legend ticks wrap to two rows.
6. **Labels are long.** 22 of them start with "Show", and the source and period parentheticals wrap.
7. **There are duplicates and misplacements.**
   - There are two fullscreen buttons.
   - Reset Camera sits in the panel.
   - Switching flowlines on inserts a 220–330 px "Pick a flowline" placeholder.
   - On phones the drawer opens on the logo.
8. **The Data Snapshot ignores the view.** It lists every dataset, on screen or not.

## Decisions

Agreed with the author on 2026-11-01:

| Topic | Decision |
| --- | --- |
| Structure | One scrolling panel of collapsible, themed groups. Tabs and a catalogue-plus-active-list were considered and not chosen. |
| Exclusive layers | Shown as one-of choices. A time feature that takes over the scene says so in place, and puts back what it switched off when it ends. |
| Legends | Inline, under the control that draws them. |
| Time features | The play button, slider, readout and chart move to a timeline dock along the bottom of the 3D view. The panel keeps the choice and its options. |
| One at a time | The four time features exclude each other, so CPOM and sea ice can no longer be combined. |
| Header | A one-row text wordmark with language and an ⓘ menu, until the logo refresh is adopted. |
| JOSS | Build now, and re-render Figure 1(a) when the new panel ships (on `docs/joss-paper-revision`). |

Defaults proposed in the same discussion, which the author did not object to:

- **The public edition stays as it is.** New styles and behaviour are gated to `data-edition="research"`.
- **Merge order.** The redesign lands on main first. `feat/glads-hydrology` and the ISMIP7 branch are rebased onto it afterwards.
- **Restore after a time feature.**
  - When the author turns the feature off, every toggle it changed returns to its earlier state.
  - When it ends because a conflicting layer was switched on, that new choice stands and nothing else is restored.
- **Data and sources** lists the terrain and the layers on screen, with a "Show all datasets" link.
- **On phones** the dock is one row: play, slider and value. The readout and chart stay in the drawer.
- **Other modes.** The dock is hidden in showcase and embed modes, and it hides with the capture panel's H key.

## Prototype

A clickable prototype of this plan was built before any product code. It is kept outside the repository.

How it is built:

- **Same ids and tokens.** It keeps the explorer's control ids and copies its colour tokens.
- **Real legends.** Ramps and tick positions are sampled from the live explorer.
- **Real series.** The ISMIP6, NSIDC and CPOM series come from the data packages.
- **A stand-in runtime.** A small script applies the exclusivity, one-at-a-time and restore rules below.
- **A still 3D view.** The view is a picture, so sliders move the readouts and charts but not the scene.

Measured in it (2026-11-01):

| Measure | Today | Prototype | Target |
| --- | --- | --- | --- |
| Panel height, nothing on (1440 × 900) | 2,135 px | 990 px | ≤ 1,000 px |
| Panel height, ISMIP6 on | 3,849 px | 1,295 px | ≤ 1,300 px |
| First layer toggle from the top | about 1,000 px | 530 px | in the first screen |
| Speed toggle to its legend | about 1,770 px | 60 px | same screen |
| Projection slider, 1440 × 900 / 1280 × 720 | about 190 px | 824 / 664 px | ≥ 800 px at 1440 |

The readouts match the live explorer to the digit at the same settings:

- September 2026 sea ice: 17.77 million km², −1.20 (−6%), 3rd lowest of 48.
- CPOM, end of the record: −1,890 ± 320 Gt and +5.2 mm.
- Projection, 2300: +1.46 m, models −0.08 to +3.99 m.

Changes the prototype made to the plan:

- **The bed-elevation legend sits under the Bed chip in Scene.** Bed elevation is the default colouring, so its legend belongs with the switch that shows the bed. With it moved, Beneath the ice can start collapsed. The default open groups are Scene, Ice flow and Change over time.
- **Units move from the end ticks into the caption.** For example, "Ice thickness change since 2015, m". No tick row then wraps at 340 px, which today's thickness-change legend does.
- **Legend notes move into each time feature's "About" disclosure.** These are scale explanations such as "symmetric log scale, linear within ±10 m".
- **The projection's own flowline toggle (`projectionFlowline`) goes from the research panel.** Ice flow → Flowlines is the one switch. The public edition keeps its copy.
- **The terrain size hint shows only for the HD grid.** It warns that the grid is large and that a desktop is recommended. The light grid's size is in its tooltip.
- **A time feature the terrain lacks gives its reason on a second line.** That stops its label wrapping.
- **Dock readouts are one line at 1440 px and wrap at narrower widths.** They do not truncate, because a hidden number is worse than a second line.
- **On phones the status chip is hidden**, as it already is on the live explorer. Otherwise it collides with the toolbar.

## Information architecture

Antarctica, nothing switched on:

```
3D ICE  Research                                   EN | 中文   ⓘ
[ Antarctica | Greenland ]
Search places, stations, ice cores…
Terrain  BedMachine v4 ▾                  Grid  10 km | 4 km

▾ Scene
    Ice surface ☑   Ice base ☑   Bed ☑   Sea level ☐
    Bed elevation, m   ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬   −7000 … 0 … +3500
    Ice opacity             ───────o──   0.65
    Vertical exaggeration   ────o─────   55×
▾ Ice flow
    ☐ Surface speed
    ☐ Flowlines  ·  click one for its profile
▸ Beneath the ice
    Colour the bed by   (•) Elevation  ( ) Basal friction  ( ) Effective pressure
    ☐ Subglacial channels
▸ Ocean and ice shelves
    ☐ Ocean streamlines   [Surface] [Upper] [Mid] [Lower]
    Ice-shelf base      (•) Off  ( ) Basal melt  ( ) Thermal driving
▾ Change over time                                       one at a time
    (•) Present day
    ( ) Observed elevation change      1995–2026
    ( ) Observed sea ice               1978–2026
    ( ) Future projection              2015–2300
    ( ) Ice-free rebound
▸ Places and labels      stations · names · ice cores · basins
▸ Display                animate flow · wireframe
▸ Data and sources
```

- **Greenland** offers Terrain "BedMachine v6 | QRF 2025" and Grid "3 km | 1 km".
- **Phones** use the same groups. The drawer's first screen shows the header row, Where and Scene.
- **Collapsing.** A group that holds a control that is on opens automatically. Otherwise each group remembers whether it was open, in localStorage, for the research edition only.

### Where every control goes

Every id is kept; controls move, they are not renamed.

| Group | Controls | Inline legend | Notes |
| --- | --- | --- | --- |
| Header | `panelLocaleSwitcher` | | The wordmark and ⓘ menu are new. `panelSubtitle` and the tour link move into the menu. `panelCloseButton` stays for the phone drawer. |
| Where | `regionPreset`, `polarSearchInput`, `polarSearchResults`, `polarSearchStatus`, `polarFeatureDetails`, `resolutionPreset` | | The region becomes a two-way segmented control. The dataset select splits into Terrain and Grid. The feature card sits under the search box, so a collapsed group never hides it. |
| Scene | `showIce`, `showIceBottom`, `showBed`, `showSea`, `showGroundingLine`, `iceOpacity`, `exaggeration` | `bedLegendSection` | The bed's default colouring is elevation, so its legend sits under the Bed chip. The grounding line (added 2026-11-02) is a chip beside Sea level, so the group keeps its height. |
| Ice flow | `showVelocity`, `showFlowline`, `flowlineProfileCardMount` | `velocityLegendSection`, also with flowlines alone | The placeholder card becomes a one-line hint. A picked flowline's profile still opens here. |
| Beneath the ice | Colour the bed by: Elevation / `showBasalFriction` / `showEffectivePressure`, then `showSubglacialChannels` | `basalFrictionLegendSection`, `effectivePressureLegendSection`, `channelLegendSection` | The GlaDS branch adds `showSheetThickness` as a fourth bed option. Collapsed by default. |
| Ocean and ice shelves | `showOceanCurrents` and `oceanCurrentLayerControls`; Ice-shelf base: Off / `showBasalMelt` / `showThermalDriving` | `oceanLegendSection`, `basalMeltLegendSection`, `thermalDrivingLegendSection` | The four depth bands show only while streamlines are on. |
| Change over time | Present day / `showSurfaceChange` / `showSeaIce` / `showIceProjection` / `showIsostaticRebound`, each with its options | `surfaceChangeLegendSection`, `seaIceLegendSection`, `projectionLegendSection`, `reboundLegendNote` | Sliders, play buttons, readouts and charts move to the dock (table below). |
| Places and labels | `showResearchStations`, `showGeographicNames`, `showIceCores`, `showRefinedBasins` | | |
| Display | `animateFlow`, `wireframe` | | `resetView` moves to the viewer toolbar. The panel's `fullscreenToggle` goes; `viewerFullscreenToggle` remains. |
| Data and sources | `metaList` | | Filtered to what is on screen. |

### What stays in the panel and what moves to the dock

| Time feature | In the panel | In the dock |
| --- | --- | --- |
| Observed elevation change (CPOM) | `surfaceChangeView`, `surfaceChangeFade`, `surfaceChangeNote`, legend | `surfaceChangePlay`, `surfaceChangeTime`, `surfaceChangeTimeValue`, `surfaceChangeReadout`, `surfaceChangeBasin`, `surfaceChangeChart`, `surfaceChangeChartNote` |
| Observed sea ice (NSIDC) | `seaIceMedianEdge`, `seaIceNote`, legend | `seaIcePlay`, `seaIceYear`, `seaIceYearValue`, `seaIceMonth`, `seaIceMonthValue`, `seaIceReadout`, `seaIceChart`, `seaIceChartNote` |
| Future projection (ISMIP6) | `projectionScenario`, `projectionColorMode`, `projectionNote`, legend | `projectionPlay`, `projectionYear`, `projectionYearValue`, `projectionReadout`, `projectionChart` |
| Ice-free rebound | `reboundModel`, `reboundModelNote`, `reboundSeaLevel`, `reboundSeaLevelNote`, `highlightEmergentLand`, `reboundLegendNote` | `reboundProgress`, `reboundProgressValue`, `reboundProgressNote` |

ISMIP7 joins as a projection source select in the projection's options. It stays research-only while the data are unpublished.

### Labels

"Show" goes from every label. Sources move to the ⓘ text or the Data and sources list.

| Today | New |
| --- | --- |
| Show bed topography / Show ice surface / Show ice bottom surface | Bed / Ice surface / Ice base |
| Show sea level (0 m) | Sea level (0 m) |
| Show ice surface velocity / Show ice flowlines | Surface speed / Flowlines |
| Show basal friction / Show effective pressure / Show subglacial channels | Basal friction / Effective pressure / Subglacial channels |
| Show ocean streamlines | Ocean streamlines |
| Show ice-shelf basal melt / Show ice-shelf thermal driving | Basal melt / Thermal driving, under "Ice-shelf base" |
| Show observed elevation change (CPOM, 1995–2026) | Observed elevation change · 1995–2026 |
| Show observed sea ice (NSIDC, 1978–2026) | Observed sea ice · 1978–2026 |
| Show future projection (ISMIP6, to 2300) | Future projection · 2015–2300 |
| Show ice-free isostatic rebound | Ice-free rebound |
| Show research stations / geographic names / ice-core sites / refined basins | Research stations / Place names / Ice-core sites / Refined basins (Greenland: Basins) |
| Animate ice & ocean flow / Wireframe mode | Animate flow / Wireframe |
| Bed Elevation Color, Ice Surface Velocity Color, … (legend headings) | A one-line caption under the control, such as "Bed elevation, m" |

- **Labels written by the runtime change too.** For example, `basinToggleLabel` in the region registry, and its `zh` strings.
- **The Chinese labels** change in step, for the author to review.

## Behaviour rules

### Today's exclusivity, from the runtime

The rules come from `ICE_PROJECTION_EXCLUSIVE_TOGGLES`, `hideBedStressOverlaysExcept`, `hideIceSurfaceForRiseOverlay`, `leaveLayersForSurfaceChange` and `leaveLayersForSeaIce` in `static/tools/js/explorer-app.js`.

| Switching on | Effect today |
| --- | --- |
| ISMIP6 projection | Switches off velocity colour, basal friction, effective pressure, channels, basal melt, thermal driving, ocean streamlines, refined basins, rebound, CPOM and sea ice. Switches on flowlines and sea level. Ending the projection brings none of them back. |
| CPOM elevation change | Switches off velocity colour and rebound. |
| Sea ice | Switches off the projection and rebound. Switches on sea level, and switches it off again if it was left so. |
| Basal friction or effective pressure | Switches off the other one and rebound. |
| Basal melt or thermal driving | Switches off the other one, and unticks Ice surface and Ice base. |
| Velocity colour | Switches off CPOM. |
| Any layer in the projection's list | Switches off the projection. |

The redesign adds one rule: CPOM and sea ice exclude each other.

### One-of choices

Two surfaces take one colouring at a time:

- **The bed:** elevation (the default), basal friction or effective pressure, and later water-sheet thickness. The elevation legend stays visible under an overlay's legend, because elevation still colours the sea floor and ice-free land.
- **The ice-shelf base:** off, basal melt or thermal driving. Choosing melt or thermal driving hides the ice surface and the ice base, so the colouring is seen unobstructed. Setting it back to Off shows each again, if this choice was what hid it.

How to build it:

- **Keep the existing checkboxes and restyle them as a segmented control,** using CSS `:has(input:checked)`. The runtime writes `.checked` directly in dozens of places without firing events. A separate widget that mirrors the checkboxes would drift out of step; styling the inputs themselves cannot.
- **Elevation and Off are not checkboxes.** Each is a button that unticks its group and dispatches `change`. Its pressed state is the group matching `:not(:has(input:checked))`.

### Time features

- **One at a time.** They are chosen from a radio list, with Present day as the off state.
- **Entering one** records the state of every toggle it will change. It then applies its rules and marks the affected rows "Off while the projection is shown" (or the name of whichever feature it is).
- **Returning to Present day** restores the recorded state.
- **Switching straight to another time feature** restores first, then applies the new feature's rules.
- **Switching on a conflicting layer** ends the time feature, and only that layer changes (see the default above).

### Unavailable layers

There is one state everywhere: the row stays, disabled, with a short reason taken from the registry. Examples:

- "Antarctica only"
- "BedMachine terrain only"
- "BedMachine 10 km grid only"

Rows are no longer hidden per region.

### Header

- **The row:** the wordmark "3D ICE" in Space Grotesk, a "Research" tag, EN | 中文, and ⓘ.
- **The ⓘ menu** holds the region intro (today's subtitle) and "Take the guided tour" (`../explore/?tour=1`).
  - It has no link to the home page. The compatibility bundle serves the explorer pages and not the home page, so on a host such as yuwang.blog a link home would point outside the bundle (`tests/test_site_embed.py`).
  - The analytics script counts any link into the other edition, wherever it sits, so the tour link needs no new wiring.
- **The logo images are no longer fetched on this page.** That saves 244 KB (light) or 336 KB (dark). The other writer of the logo's `src`, `applyRuntimeTheme()`, must tolerate the element being gone.

### Timeline dock

- **Place:** inside `viewerShell`, along the bottom edge, as wide as the viewer. It appears while a time feature is active.
- **Contents:** as in the table above.
- **Charts need a width parameter.** They are drawn into fixed 260 × 96 boxes today: `ICE_PROJECTION_CHART`, `SEA_ICE_CHART` and the CPOM chart's box.
- **Phones:** one row above the drawer. The readout and chart stay in the drawer's time group.
- **Other modes:** hidden in `mode=showcase` and `mode=preview-embed`, and hides with the capture panel (H).

### Data and sources

- **By default** it lists the terrain, the layers on screen and their sources.
- **"Show all datasets"** expands the full list.
- **`metaSectionOpenState`** keeps remembering which parts are open.

## Compatibility constraints

### Ids and markup

- **Every control keeps its id.** Presets (`applyExplorerPresetControls`), the public edition's stand-ins (`getStandInSpec`) and the tests all address controls by id.
- **The exceptions are** the panel's `fullscreenToggle`, which the runtime must then treat as optional, and the logo `<img>`.
- **The two locale shells are hand-written copies.** Change them together; a new test should assert that they carry the same ids in the same groups.
- **The public page loads the same `explorer.css`.** Scope new rules under `html[data-edition="research"]`, or use new class names.

### Tests

Two unit tests pin today's nesting and must be rewritten for the new groups:

- `tests/test_polar_features_integration.py` (Places inside View Controls)
- `tests/test_isostatic_rebound_integration.py` (rebound controls inside Interactive Scenarios)

Playwright cannot click a control inside a closed `<details>`. The e2e tests get an `open_group()` helper, and groups that hold a ticked control open automatically.

### Layout code

- **Tick legends are laid out by measuring their width** (`scheduleTickLegendLayout`). Re-run it when a group opens.
- **The phone drawer computes its bounds from its content** (`syncMobilePanelBounds`). Re-run it when a group opens or closes.

### Recordings and unmerged work

- **Recorders:**
  - The LinkedIn demo recorder drives controls by id from a release checkout. Check its scripts before recording on the new release.
  - The film recorders use frozen, detached checkouts and are unaffected.
- **Unmerged work.** Each branch adds one panel control and is rebased after the redesign.
  - `feat/glads-hydrology`: 81 commits behind `origin/main`, about 24 files in conflict.
  - `claude/ismip7-antarctica-3d-viz-aeaa9c`: 79 behind, unpushed, data git-ignored.

### JOSS

Figure 1(a) shows today's panel. Re-render it once the panel's look is final, after phase 2. The dock appears only with a time feature on, and panel (a) shows none.

1. Re-render it on `docs/joss-paper-revision`.
2. Post `@editorialbot generate pdf` on openjournals/joss-reviews#11368.

## Phases

### Phase 1: structure

Markup and CSS, with small runtime changes.

Work:

- **Header row and ⓘ menu.**
- **Where group:** the segmented region control, the search box with its feature card, and Terrain and Grid.
- **Collapsible groups,** which open automatically and remember their state.
- **Short labels** in both locales.
- **Inline legends.**
- **One disabled state** with a reason, for unavailable layers.
- **Viewer actions:** Reset view in the viewer toolbar, one fullscreen button, and the one-line flowline hint.

Done when, at 1440 × 900:

- The panel is at most 1,000 px tall with nothing on (2,135 today).
- Every label and legend tick fits on one line at 340 px.
- Both locales carry every id, in the agreed groups.
- The public edition's pages and tests are unchanged.

#### Built

On `feat/research-panel`, cut from `origin/main` at 2f51bfe. Measured at 1440 × 900:

| Measure | Before | Phase 1 | Target |
| --- | --- | --- | --- |
| Panel height, nothing on (English) | 2,135 px | 959 px | ≤ 1,000 px |
| Panel height, nothing on (Chinese) | | 981 px | ≤ 1,000 px |

No label, and no row of legend ticks, wraps at 340 px with every group open.

How it is built:

- **Markup.** Both shells carry the same groups, ids and order. `regionPreset` and `resolutionPreset` stay as hidden selects, the runtime's source of truth; the region buttons and the Terrain and Grid controls set them and dispatch `change`.
- **Styles.** `static/tools/css/research-panel.css`, which only the research pages load, after `explorer.css`. Its class names start with `rp-`.
- **Behaviour.** `static/tools/js/research-panel.js` opens, closes and counts the groups, mirrors the selects and runs the ⓘ menu. It watches the layer toggles (`input[data-layer]`) itself, so no caller has to remember to tell it:
  - a `change` listener catches the reader's clicks;
  - a wrapper on each toggle's `checked` setter catches the writes that fire no event, from presets, the place search and the time features.

  An inline script after the panel applies the stored open and closed state before the first paint, so a return visit does not open on the defaults and snap shut. The public page fetches neither module.

  `static/tools/js/panel-state.js` holds the decisions with no DOM, under Node's test runner:
  - which terrain and grid a dataset is, from the registry's new `terrain`, `terrainLabel`, `grid`, `gridLabel` and `desktopRecommended` fields;
  - where a layer the terrain lacks can be found, which the runtime words as the reason;
  - which groups start open. Only groups the reader has opened or closed are stored, under `3d-ice-research-panel-groups`, so a group the page opened by itself, or a later default, is never pinned.
- **Edition flags.** `legendUnits` and `unavailableLayers` in `editions.js` keep the public edition as it was: units on the first tick, and unavailable rows hidden.
- **Runtime.** Tick legends are laid out again when their section is shown or their group opens. The ocean depth bands now show only while the streamlines are on.

Tests:

- `tests/test_research_panel_structure.py` holds both shells to the agreed groups, order and ids. Against `origin/main` 36 of its 39 checks fail.
- `tests/js/panel-state.test.mjs` covers `panel-state.js` (100 % of lines).
- `tests/e2e/test_research_panel.py` checks the height budget, the groups (including a layer the place search switches on, and a group the reader closed), the Where controls and reasons, one-line labels and ticks, the ⓘ menu, Reset view, the Chinese page and the phone drawer.
- The two nesting tests are rewritten for the groups, and the other research e2e tests open a group with `tests/e2e/panel.py` before they click inside it.

### Phase 2: behaviour

Work:

- **One-of controls** for the bed and the ice-shelf base.
- **Change over time as one choice,** with in-place notes on the rows a time feature switched off, and the restore rules.
- **Data and sources** filtered to what is on screen.

Done when:

- An e2e test covers every row of the exclusivity table, including restore.
- A unit test fails any preset that would switch on two layers from one one-of choice.
- The panel is at most 1,300 px tall with the ISMIP6 projection on (3,849 today).

Then release v0.4.0 and re-render JOSS Figure 1(a).

#### Built

On `feat/research-panel`, after phase 1. Measured in headless Chromium at 1440 × 900:

| Measure | Before | Phase 1 | Phase 2 | Target |
| --- | --- | --- | --- | --- |
| Panel height, nothing on (English) | 2,135 px | 959 px | 945 px | ≤ 1,000 px |
| Panel height, nothing on (Chinese) | | 981 px | 962 px | ≤ 1,000 px |
| Panel height, ISMIP6 projection on (English) | 3,849 px | 1,726 px | 1,515 px | ≤ 1,300 px |

**The 1,300 px target needs the dock, so it moves to phase 3.** With the projection on, its year slider, readout and chart take 268 px of the panel, and they leave it only for the timeline dock. Phase 2 saved 211 px: the About boxes, and group headings 4 px shorter on a mouse-driven screen (touch keeps the taller ones). The prototype's 1,295 px was measured with its dock.

How it is built:

- **One-of choices.** Each surface is a `[data-choice]` field. The layers keep their checkboxes, drawn as round markers inside pills; the first pill (`[data-choice-none]`: Elevation, Off) is a button that switches them all off. Its pressed look comes from CSS (`:not(:has(input:checked))`), and the panel keeps its `aria-pressed` in step. A choice gives its reasons and the time feature's notes on one line under the pills: a single reason when every option shares it ("Antarctica only"), otherwise one line per option ("Effective pressure: Antarctica only").
- **Ice surface and ice base.** Off brings each back if basal melt or thermal driving hid it, and so does a time feature that clears the ice-shelf base, whose ice would otherwise stay hidden.
- **Change over time.** `static/tools/js/research-time.js`. The four feature toggles stay checkboxes, drawn with round markers under a *Present day* option. They keep checkbox semantics, so a click on the feature that is on also returns to the present day. The panel stops the reader's click on a feature and makes the change itself, in order, each step as the toggle's own change event:
  1. the feature that was on goes, so the runtime lowers what it raised;
  2. the layers it switched off come back;
  3. the next feature starts.
- **What a feature changes.** `planTimeChange()` in `panel-state.js` keeps two lists per feature: the layers it switched off that were on, and the layers it switched on that were off. Moving straight to another feature carries over what that feature changes the same way, so nothing flickers. The lists come from the runtime's own rules (`ICE_PROJECTION_EXCLUSIVE_TOGGLES`, the new `REBOUND_EXCLUSIVE_TOGGLES`, `leaveLayersForSurfaceChange()`, `leaveLayersForSeaIce()`), and a layer the reader sets by hand drops out of them.
- **Conflicting layers.** Switching on a layer the feature replaced ends it, and nothing else comes back. The runtime already did this for the projection, for velocity against CPOM and for friction and effective pressure against the rebound. The panel now does it for every layer the rebound replaces, on the research page only.
- **About boxes.** The legend note and the longer note of the observed elevation change, the observed sea ice and the projection sit in a closed *About* box. The rebound's notes stay beside the control each one explains.
- **Data and sources.** The terrain and the layers on screen, with their sections and sources; *Show all N datasets* lists everything the terrain offers.
- **Presets.** `SHOWCASE_PRESETS` and `EXPLORER_PRESETS` move to `static/tools/js/view-presets.js`, so a node test can hold every view, and every tour stop, to `findChoiceConflicts()`.

Tests:

- `tests/e2e/test_panel_exclusivity.py`: one case per row of the exclusivity table, restore included, on the real packages except the ocean streamlines.
- `tests/js/panel-state.test.mjs` covers `planTimeChange()` and `findChoiceConflicts()`; `tests/js/view-presets.test.mjs` fails any view or tour stop that would switch on two layers of one choice.
- `tests/test_research_panel_structure.py` pins the choices, the time list, the About boxes and the data switch in both shells.

### Phase 3: timeline dock

Work: the dock, wide charts, the phone row, and the mode handling.

Done when:

- The panel is at most 1,300 px tall with the ISMIP6 projection on (moved from phase 2).
- The projection slider is at least 800 px long at 1440 × 900.
- Readouts stay on one line.
- Chart labels are legible at 1280 × 720.

## Test plan

- **Height budget (e2e),** at 1440 × 900 and 1280 × 720, with nothing on and with the projection on.
- **Shell parity (unit):** the same ids, in the same groups, in `static/tools/` and `static/zh/tools/`.
- **Exclusivity and restore (e2e):** one case per row of the exclusivity table.
- **Existing e2e tests** are updated with `open_group()`. The public-edition e2e tests are untouched and still pass.
- **Visual checks:** 1440 × 900, 1280 × 720 and 375 × 812, in light and dark themes.

## Open questions

1. Group names and order.
2. Whether the ice-free rebound belongs under Change over time. It is a what-if, not a record, though it fits the dock as "deglaciation 0–100 %".
3. Restore when a time feature is ended by a conflicting layer. The default keeps everything else off; the alternative restores everything and lets the new choice win any conflict.
4. Whether to keep the "Research" tag in the header.
5. The Chinese labels, for the author to review.
6. When ISMIP7 can leave research-only.

## Follow-up: the public edition and the edition switch

On 2026-11-02 the author asked for three things: the site's icon in front of the wordmark, a public panel redesigned to match this one, and a switch between the two editions. That request lifts the rule above that the public edition must not change. It also settles open question 4: the switch names the edition, so the *Research* tag goes.

The author's choices:

- **Edition switch.** It sits on a second masthead row. The first row holds the icon, the wordmark and the ⓘ button. The second holds *Public | Research* (*科普版 | 专业版*) beside the language switch, in the same style. The page's own edition is marked, and the other one is a link to its page in the same language. That link carries the region on screen (`?region=`), so the reader lands on the same ice sheet.
- **Public panel.** It takes this panel's look, with every group open and none folding.
  - The logo banner goes, and the intro and what the explorer is built from move into the ⓘ menu. The tour card stays on top.
  - Antarctica and Greenland become two buttons, and each legend sits under the layer that draws it.
  - *Reset view* joins the viewer's buttons.
  - The copy and the explainers are the reviewed ones, moved.

Built on `feat/public-panel`. Measured in headless Chromium at 1440 × 900, with nothing on:

| Panel | 0.4.0 | Now |
| --- | --- | --- |
| Public, English / Chinese | 1,877 / 1,796 px | 979 / 992 px |
| Research, English | 945 px | 981 px |

The estimate offered with the choice was about 850 px, short enough to fit a 1440 × 900 window. It left out the masthead's second row and undercounted the tour card, so the last group sits about 110 px below the fold.

How it is built:

- **Styles.** `css/research-panel.css` is now `css/panel.css`, and every explorer page loads it after `explorer.css`. The research edition's own rules find nothing on the public page. The public pages load `explore.css` after it.
- **Script.** `js/panel-masthead.js` holds the ⓘ menu, the region buttons and the switch's link for both editions. `js/research-panel.js` keeps the groups, the terrain and grid controls, the one-of choices and Change over time, and is still fetched by the research page only.
- **Tour.** While the tour plays, the public panel holds the card and the legends of the stop, as before. The groups give up their headings and rows for that time.
- **Wordmark.** The author found the outlined *3D* weak and chose, of four options rendered in the panel, a solid two-tone name: *3D* in the accent colour (`--rp-wordmark-3d`, lighter on the dark theme) and *ICE* in ink, with the icon at 28 px. The other options were a gradient *3D*, the name over the line *Interactive Cryosphere Explorer*, and one colour in two weights.
- **Logo.** No explorer page draws the logo banner now, so its styles, the runtime's theme swap of its `src` and the two logo addresses in each page's theme script are gone. The logo images stay: the home page and the README show them.
