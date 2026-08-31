# Minimal Fixes — Black Screen After Spec Selection

> Scope: **only** crash fix for `GameScreen` turning black after `HomePage -> Start`. No overhaul, no color/layout redesign, no new features. Based on `git diff HEAD` audit (plan mode).

## Context

After `Create Game` (`POST /api/games` → `navigate("/games/:id")`) the app mounts `GameScreen -> ZoomableBoard -> Board -> Tile` with 9 outer `PORT` hexes. `main {background: #030000}` fills viewport; any render throw unmounts the tree and appears as solid black (loader `GridLoader #000000` invisible on black).

## Root Cause (Confirmed)

**1. `ReferenceError: RESOURCES is not defined` — `ui/src/pages/Tile.tsx:95`**

```ts
// before (broken, after PORT_RESOURCES rename)
const PORT_RESOURCES: ResourceCard[] = [...] // line 73
// ...
fill={resource in RESOURCES ? TERRAIN_COLORS[resource] : "#fff"} // line 95 — RESOURCES deleted
```

`git diff HEAD -- Tile.tsx` shows `const RESOURCES` removed but use-site left. Every `PORT` renders `PortBoat` → throw → React unmount → black.

**2. Wrong port angles — `ui/src/utils/coordinates.ts:43-49` (secondary, no crash but mis-places harbors)**

```ts
EAST: 90,  // duplicates SOUTH (=90) — should be 0
WEST: 270, // duplicates NORTH (-90) — should be 180
NORTHWEST: 210 // actually down-left (210°), should be -150 (up-left)
```

Caused diagonal harbors to point south.

No evidence that deleted `tile_*.svg` (`D 7 files`) cause crash — imports were removed together with `RESOURCES`. `Board.scss` sand `#c9b99a` vs desert `#f0d060` is intentional contrast, not part of black-screen fix.

## Minimal Fixes (3 lines, 2 files)

### Fix 1 — `ui/src/pages/Tile.tsx:95` (P0, blocks all games)

**Before:**
```ts
<path d="M12 11 L12 6 L18 8.5 Z" fill={resource in RESOURCES ? TERRAIN_COLORS[resource as string] : "#fff"} />
```

**After:**
```ts
<path d="M12 11 L12 6 L18 8.5 Z" fill={PORT_RESOURCES.includes(resource) ? TERRAIN_COLORS[resource as string] : "#fff"} />
```

*Risk: none (type-safe, same branch). Fallback `"#fff"` for `GENERIC 3:1` unchanged.*

Optional tidy (no behavior change): delete stale comment line 75 `// kept for board overlay`.

### Fix 2 — `ui/src/utils/coordinates.ts:43-48` (P2, harbor orientation)

**Before:**
```ts
EAST: 90, WEST: 270, NORTHWEST: 210
```

**After:**
```ts
EAST: 0, WEST: 180, NORTHWEST: -150,
```

Keeps 6-direction pointy-top math (`apothem = size * SQRT3/2`, `dist = apothem * distFactor`, `angle -> cos/sin`). No caller change; `Tile.tsx:143-144` `bridgeOff 1.0 / boatOff 0.7` continue to work.

## Out of Scope (Explicitly NOT in this doc)

- Sand vs desert tuning (`Board.scss:142 #c9b99a`, `Tile.tsx:69 #f0d060`), `GAP 0.975`, `Board.tsx *0.985`, `Tile.scss` drop-shadow — already fixed, not touched.
- New `ErrorBoundary`, `GameScreen`/`HomePage` try/catch, `GridLoader` color, bundle size, motifs, `framer-motion`.
- Any port-bridge spanning sand gap to land vertex (requires neighbor lookup) — deferred.

## Verification (Must pass before merge)

1. `rg "RESOURCES" ui/src` → only `PORT_RESOURCES`
2. `npm run build` in `ui/` → `vite` `~790kB` success (no `ReferenceError`)
3. `docker compose restart react-ui` → `Vite ready` `localhost:4000`
4. Manual: `Home -> BASE/MINI/TOURNAMENT -> Start` → board renders (not black), `PORT` boats show `2:1/3:1` + wood bridge inside blue hex, no console error
5. `SOUTH/EAST/WEST/NORTHWEST` harbors point correctly (spot-check 2 ports)

## Rollback

Revert commit: `git checkout HEAD -- ui/src/pages/Tile.tsx ui/src/utils/coordinates.ts`.
