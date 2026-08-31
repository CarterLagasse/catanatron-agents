# Catanatron -> Colonist.io UI Overhaul Plan (Simple Catan Art, No Emoji) — v3 Audit + Fixes

> Goal: colonist-like tactile board with simple flat Catan art, no emoji, no photoreal. Engine stays Python `catanatron`.

## 0. Executive Summary

After `6a672fe UI Improvements` the board moved from SVG textures → filled hexes, but two regressions slipped in the last sand-gap pass: (1) board sand `#e6d2a0` was ΔE≈4 from desert `#e8d68a` → desert blended into background; (2) ports were re-parented to a `Board` overlay at `1.18*size` outside the clipped hex → ships sat on sand, bridges floated over gaps, rotation/z wrong, blue border clipped them. This doc audits **all** changes `git diff HEAD` (23 files, +923/-358), diagnoses the breakage, and records the fix pass (`GAP 0.975`, sand `#c9b99a` vs desert `#f0d060`, ships back inside hex + wood bridge).

## 1. What Changed — Full Audit

### Stack / Infra
- `ui/package.json:1` `vite@6 + react@19 + react-router@7 + @mui/material@7 + sass + framer-motion 13` (added `framer-motion`), `ui/vite.config.ts:11` `port 4000 host:true`
- `docker-compose.yml:40` `3000:3000 → 4000:4000`, `ui/Dockerfile` npm start, anon `/app/node_modules` stale-volume footnote
- `ui/vite.config.ts` `envPrefix CTRON_`, `configuration.ts:1` `API_URL http://localhost:5001`

### Board Rendering — before → after
- **Base**: DOM hex grid `Board.tsx:1` `absolute div.tile/edge/node` + `utils/coordinates.ts:6` `SQRT3`, `ZoomableBoard.tsx:1` `react-zoom-pan-pinch 3.7` `TransformWrapper minScale 0.55 maxScale 2.4 centerOnInit` `+/-⟲` `Board.scss:135-145`, `Board.tsx:30` `TOOLBAR 132 HEADER 48 DRAWER 280` sync `computeDefaultSize() *0.985`
- **Tiles**: `Tile.tsx:78` `TERRAIN_COLORS` WOOD `#1e5a2e` BRICK `#b94a2e` SHEEP `#7dc67e` (was `#9ac17a` mint → closer green per request) WHEAT `#e7c65a` ORE `#6e7681` DESERT `#f0d060` (was `#d9c8a3` → `#e8d68a` → now vivid yellow to contrast sand) WATER `#4a8fd3`; `Tile.tsx:152` `GAP 0.975` `displayW/H = w/h * GAP` `left x - displayW/2` → sand shows between hexes/outer rim; `Board.scss:70` `.tile clip-path polygon(50% 0 100% 25% …)` `filter drop-shadow 0 2px 2px` `::after` soft gloss `0.14/0.06`, `hover scale 1.015` (was `scale 1.02` + `border 1.5px` + double shadows). `Tile.scss:160` `tile--wood/brick…` now only `saturate` (was `+ drop-shadow`) to avoid double-shadow.
- **Motifs**: `Tile.tsx:216` `tile-motif` `motif--wood/brick/sheep/wheat/ore` `Tile.scss:114` inline SVG data URI at `bottom 10%` (`tree pair 28px`, `terraces 30px`, `sheep 30px`, `wheat 26px`, `mountain 30px`)
- **Numbers**: `Tile.tsx:26` `NumberToken Paper` + `Tile.scss:3` centered beige disc `4.2rem #fdf6e3 #c9b896` flex-centered via `.tile {display:flex justify-center align-center}`, red 6/8, pips `•`, pulse gold

### Ports / Ships / Bridges — evolution
- **v1**: `Tile.tsx:PortBoat 0.58*size` inside water hex at `Tile.scss:89` `.port-boat absolute translate -50% -50%` hull `#5d4037` deck `#d7ccc8` sail `#fff8e1` + `port-badge 2:1`
- **v2 (broken)**: `Board.tsx:64` `portOverlays` moved boats to `1.18*size` outside clipped hex over sand + `port-bridge` at `0.90*size` `size*0.48 x 0.11` `rotate atan2+90` `z 2/3`. Ships landed on sand (`#e6d2a0`), bridge wood `#c9a86a` low contrast on sand, clipped/buried under blue outline, 8-direction offsets uneven.
- **v3 (fixed)**: `Tile.tsx:172` ports back **inside** blue hex: `bridgeOff 0.32*size` + `boatOff 0.62*size` `bridgeAngle = atan2+90` `bridge 0.10*size x 0.38*size` at `calc(50% + x)` over blue, `PortBoat 0.62*size` over blue, both `overflow:hidden` safe (`0.62 < apothem 0.866`). `Board.scss:103` `.port-bridge` kept as inside-tile wood `linear-gradient #c9a86a→#a6844b` `border #7a5c2e` `repeating planks`. Removed `Board.tsx portOverlays` + `port-boat-overlay` (was outside). Water hex fill remains `#4a8fd3` distinct from sand.

### Pieces
- `Node.tsx:11` `Building` now SVG house/city (was `div 80% rect + ::after gloss`): `FILL RED #fe0400 BLUE #2278ff ORANGE #ffa500 WHITE #f2f2f2` `stroke rgba(0,0,0,0.72)` `drop-shadow 1.5px`; settlement `viewBox 0 0 24 20` `M12 2.8 L3.2 10.2 V17.2 H9.2 V11.8 …` door slit `rgba(0,0,0,0.22)` `svg.settlement 92%`; city `viewBox 0 0 26 18` `M3 17.5 V7 H6.2 V4.2 …` crenellations `108%` two windows `rgba(0,0,0,0.18)`. `Board.scss:210` `.node` `svg.settlement 92% svg.city 108%` `hover scale 1.07` no gloss.
- `Edge.tsx:1` road `14px pill 999px` `border 1.5px rgba(0,0,0,0.55)` `inset highlight + drop-shadow` `hover brightness 1.08`
- `Robber.tsx:30` minimalist pawn `viewBox 0 0 24 32` `M12 3.5 A5.5 … L7.6 24.4` `fill #3a3f45 stroke #1a1d22 1.1` `drop-shadow 2px` (was `32x40` `#6b7280` belt/highlight/base ring/shadow ellipse). `Board.scss:57` `.robber translate -50% -50% 420ms`
- `ResourceIcon.tsx:1` vector WOOD tree `#2e7d32` BRICK wall `#b71c1c` SHEEP fluffy `#e8f5e9` WHEAT stalk `#fdd835` ORE mountain `#546e7a`

### chrome / UX
- `CostStrip.tsx:1` `ResourceIcon + MUI Home/LocationCity/Signpost/Style` (was emoji `🏠🏙️`), `PhaseBanner.tsx`, `DiceRoll.tsx` `framer-motion` `38px #fdf6e3` pips, `PlayerStateBox.scss` `is-current` gold glow, `ResourceCards.tsx` `AnimatedCard spring 420/22` `Wo/Br…` labels, `LeftDrawer/RightDrawer` unified `md down`, `HomePage.tsx:36` `mini-hex 18x20 clip-path` lobby, `GameScreen.scss:99` unified `padding 280`.

## 2. What Went Wrong (Diagnosis Required)

### Background == Desert (regression)
- `Board.scss:124` was `radial dark #0d2137` → `#4a8fd3` (to unify with water) → `#e6d2a0` sand in gap pass. `#e6d2a0` (230,210,160) vs `TERRAIN_COLORS.DESERT #e8d68a` (232,214,138) ΔL≈2 ΔC≈12 → desert hex blended into board sand, tile edge invisible, map read as `desert == gap`. User flagged "background is now the desert color".

### Ports All Messed Up
- Moving ships to `Board` overlay at `1.18*size` put them **outside** the blue water hex onto sand `#e6d2a0` → ships looked beached on desert. Bridge at `0.90*size` straddled the sand gap where hex `clip-path` had already shrunk (`GAP 0.962`) → bridge was half-clipped by `overflow:hidden`, half floating over sand with `z 2` under `water` but over `tile` due to `Board` order, rotation `atan2+90` mis-aligned for `WEST/EAST` (flat sides) vs `NORTH*` diagonals (used `0.78/0.45` factor but overlay used full `1.18` dist → diagonal boats overshot). Blue border `1.5px #2a5a8a` previously also overrode ships (ship `z 2` under border).

### Hexes Don't Align Perfectly
- `GAP 0.962` (3.8% shrink) + `computeDefaultSize *0.96` (4% board shrink) = effective `0.923` → gap `~4.5px` at size 60, settlements (`size*0.5` at true vertex) appeared floating `~2px` off hex corner over sand. Combined with `transform scale 1.02` + double `drop-shadow` from `Tile.scss` variants made edges look heavy/uneven.

## 3. What Was Fixed in This Pass

- **Distinct desert vs sand**: `Board.scss:124` sand gap now `#c9b99a` (201,185,154) warm taupe; `Tile.tsx:84` desert now `#f0d060` (240,208,96) vivid yellow ΔE≈28 distinct. Water `#4a8fd3` stays blue, so 3-way contrast: `desert yellow / sand taupe / water blue`.
- **Sand gap tuning**: `Tile.tsx:154` `GAP 0.975` (2.5% not 3.8%) + `Board.tsx:30` `*0.985` (1.5% rim not 4%) → total `0.960` gap `~1.5px` per side, intersections "chill" but settlements still attached. `Board.scss:70` `filter drop-shadow 0 2px 2px` single, `Tile.scss:161` variants now only `saturate` (removed duplicate shadows).
- **Ports back inside**: `Tile.tsx:170-191` removed `Board` overlay, restored `PortBoat` + new `port-bridge` **inside** `tile` at `0.32*size` (bridge) + `0.62*size` (boat) both `< apothem 0.866` → over blue, not sand, not clipped, `z 1` under badge `z 2`. Bridge `width 0.10*size height 0.38*size` wood `#c9a86a→#a6844b` planks visible on blue contrast. `Board.scss:103` `.port-bridge` simplified, `Board.tsx` overlay deleted, `Tile.scss:89` `.port-boat` remains inside-tile absolute.
- **Alignment**: nodes/edges still use true geometry `tilePixelVector` + `getNodeDelta(size*0.865)`, visual hex `0.975` gap leaves sand at vertices where SVG houses (`Node.tsx`) sit centered over gap — colonist-like.

## 4. What Still Needs Improvement

1. **Port direction fidelity** — 8-direction `N/NW/NE…` uses `0.78/0.45` approx for pointy-top hex; for flat `WEST/EAST` the apothem axis is slightly off vs `size*0.866` — diagonal ports sit ~2px inset vs cardinal. Replace with exact hex vector `angle = dirToAngle(direction)` + `r = apothem * factor`.
2. **Bridge length vs gap** — bridge inside hex reads as dock, not span to shore. True colonist bridge spans sand gap from land vertex to water edge (requires knowing adjacent land hex). Needs `Board` to lookup neighboring land coordinate and draw `line` across gap, not inside single hex.
3. **Settlement/city scale** — SVG houses `92%/108%` at `node size*0.5` are slightly small on `sm` (`size~32` → house `15px`). Tune `Node.tsx:45` to `size*0.55` + `sm` media `0.62`.
4. **Desert motif** — desert currently solid `#f0d060` no motif; colonist desert has dunes/cactus speck. Add subtle `tile-motif motif--desert` at 12% opacity.
5. **Water outer ring completeness** — outer ring currently 18 water/port hexes but board `computeDefaultSize *0.985` leaves thin sand rim uneven on `md` vs `sm`. Add `board-padding 8px` CSS instead of JS scale for consistent rim.
6. **A11y / colorblind** — sheep `#7dc67e` vs wood `#1e5a2e` luminance delta 45 is ok, but wheat `#e7c65a` vs desert `#f0d060` delta 8 close — add stronger `tile-motif` shape contrast or `stroke` outline on desert.
7. **Performance** — `framer-motion` `789kB` bundle, `tile_*.svg 67-337kB` still shipped though `TERRAIN_COLORS` filled; tree-shake or remove `brickTile` imports if unused.

## 5. Design Tokens (Current)

`TERRAIN_COLORS` `Tile.tsx:78` WOOD `#1e5a2e` BRICK `#b94a2e` SHEEP `#7dc67e` WHEAT `#e7c65a` ORE `#6e7681` DESERT `#f0d060` WATER `#4a8fd3` `BOARD_SAND #c9b99a` `NUMBER_DISC #fdf6e3 #c9b896` `BRIDGE #c9a86a→#a6844b #7a5c2e` `BOAT hull #5d4037 deck #d7ccc8 sail #fff8e1`

## 6. Verification

- `npm run build` `vite` `790kB` `css 32.8kB` (was `32.11` → `32.81` + gap/bridge), `docker compose restart react-ui` `Vite 574ms` `localhost:4000`, `grep -r "🌲|🧱|🐑" ui/src` → 0, manual `HUMAN+3 bots BASE ROLL->BUILD->ROB` flow, `sm 600 md 960` drawer, `TransformWrapper 0.55-2.4`.

## 7. References

`ui/src/pages/Tile.tsx:78,152,172` `Tile.scss:89,114` `Board.scss:70,103,124,210` `Board.tsx:30` `Node.tsx:11` `Robber.tsx:30` `ResourceIcon.tsx:1` `docker-compose.yml:40`
