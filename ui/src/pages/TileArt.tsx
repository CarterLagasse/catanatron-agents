import { memo } from "react";

// Flat terrain illustrations, drawn on a hex of radius 100 centred on the origin.
// The middle of each tile stays clear for its number token.

export type Terrain = "WOOD" | "BRICK" | "SHEEP" | "WHEAT" | "ORE" | "DESERT";

export const TERRAIN_BASE: Record<Terrain, { fill: string; edge: string }> = {
  WOOD: { fill: "#3b8a48", edge: "#23582f" },
  BRICK: { fill: "#d0713f", edge: "#8f4522" },
  SHEEP: { fill: "#a3cb58", edge: "#6d9233" },
  WHEAT: { fill: "#eec14a", edge: "#b08723" },
  ORE: { fill: "#98a3ab", edge: "#606b74" },
  DESERT: { fill: "#ead3a0", edge: "#b39761" },
};

type At = [x: number, y: number, scale?: number];

const place = ([x, y, scale = 1]: At) => `translate(${x} ${y}) scale(${scale})`;

function Pine({ at }: { at: At }) {
  return (
    <g transform={place(at)}>
      <rect x="-2.6" y="9" width="5.2" height="9" fill="#5f3d20" />
      <path d="M0-28-11-9h4.5L-15 11h30L6.5-9H11Z" fill="#1f6136" />
      <path d="M0-28 11-9H6.5L15 11H0Z" fill="#2c7c45" />
    </g>
  );
}

function Forest() {
  const trees: At[] = [
    [-22, -74, 0.85], [24, -72, 0.9], [-58, -40, 1], [60, -38, 1.05], [-4, -48, 1.05],
    [-68, 6, 0.95], [68, 8, 0.95], [-56, 48, 1.05], [56, 50, 1.05], [-20, 74, 0.9], [22, 76, 0.9],
  ];
  return (
    <>
      <path d="M-90-20Q-40-46 10-22T90-30V40Q40 62-10 42T-90 50Z" fill="#348040" />
      {trees.map((at) => (
        <Pine key={`${at[0]},${at[1]}`} at={at} />
      ))}
    </>
  );
}

function BrickPile({ at }: { at: At }) {
  const rows: [number, number][] = [
    [-15, 8], [1, 8],
    [-23, 0], [-7, 0], [9, 0],
    [-15, -8],[1, -8],
  ];
  return (
    <g transform={place(at)}>
      <ellipse cx="0" cy="15" rx="30" ry="5" fill="#a9552b" />
      {rows.map(([x, y]) => (
        <g key={`${x},${y}`}>
          <rect x={x} y={y} width="14.5" height="7.5" rx="1" fill="#a33f1b" />
          <rect x={x} y={y} width="14.5" height="2.4" rx="1" fill="#d9754a" />
        </g>
      ))}
    </g>
  );
}

function Hills() {
  const piles: At[] = [[-44, -58, 0.95], [38, -60, 0.9], [-58, 32, 0.9], [54, 34, 0.95], [-2, 70, 0.9]];
  return (
    <>
      <path d="M-100-34Q-50-78 0-40T100-44V-100H-100Z" fill="#bd5e30" />
      <path d="M-100 62Q-46 22 0 58T100 52V100H-100Z" fill="#bd5e30" />
      <path d="M-90 10Q-66-8-40 8T-6 6M20 6Q48-14 92 4" fill="none" stroke="#dd8656" strokeWidth="5" strokeLinecap="round" />
      {piles.map((at) => (
        <BrickPile key={`${at[0]},${at[1]}`} at={at} />
      ))}
    </>
  );
}

function Sheep({ at, flip }: { at: At; flip?: boolean }) {
  return (
    <g transform={`${place(at)}${flip ? " scale(-1 1)" : ""}`}>
      <ellipse cx="0" cy="13" rx="15" ry="3.4" fill="#7da33e" />
      <path d="M-8 6v7M-3 6v7M4 6v7M9 6v7" stroke="#3b3532" strokeWidth="2.4" strokeLinecap="round" />
      <g fill="#fffdf6">
        <circle cx="-9" cy="-1" r="6.5" />
        <circle cx="-2" cy="-5" r="7" />
        <circle cx="6" cy="-3" r="6.5" />
        <circle cx="-4" cy="3" r="6.5" />
        <circle cx="5" cy="3" r="6.5" />
      </g>
      <ellipse cx="13" cy="-4" rx="4.6" ry="5.8" fill="#3b3532" />
      <ellipse cx="9" cy="-8" rx="3" ry="1.7" fill="#3b3532" />
      <circle cx="14.6" cy="-5.4" r="0.9" fill="#fffdf6" />
    </g>
  );
}

function Pasture() {
  const flowers: [number, number][] = [[-30, -30], [40, -24], [-72, 2], [74, 18], [-28, 42], [34, 66], [-52, -68], [8, -82]];
  const tufts: [number, number][] = [[-68, -34], [-6, -46], [64, -46], [-46, 16], [48, 2], [8, 44], [-46, 72], [58, 64]];
  return (
    <>
      <path d="M-100-20Q-50-52 0-24T100-30V40Q50 66 0 44T-100 50Z" fill="#b1d566" />
      {tufts.map(([x, y]) => (
        <path key={`${x},${y}`} d={`M${x - 5} ${y}q2-6 5 0q3-6 5 0`} fill="none" stroke="#7da33e" strokeWidth="2.2" strokeLinecap="round" />
      ))}
      {flowers.map(([x, y]) => (
        <circle key={`${x},${y}`} cx={x} cy={y} r="2.2" fill="#fff6c9" />
      ))}
      <Sheep at={[-42, -56, 1.05]} />
      <Sheep at={[42, -58, 1]} flip />
      <Sheep at={[-56, 32, 1]} />
      <Sheep at={[56, 34, 1.05]} flip />
      <Sheep at={[0, 70, 1]} />
    </>
  );
}

function Wheat({ at }: { at: At }) {
  const grains: [number, number, number][] = [
    [-3.4, -8, -22], [3.4, -8, 22], [-3.4, -14, -22], [3.4, -14, 22], [-3.4, -20, -22], [3.4, -20, 22],
  ];
  return (
    <g transform={place(at)}>
      <path d="M0 16V-24" stroke="#a9771a" strokeWidth="2.2" strokeLinecap="round" />
      {grains.map(([x, y, r]) => (
        <ellipse key={`${x},${y}`} cx={x} cy={y} rx="2.8" ry="4.6" transform={`rotate(${r} ${x} ${y})`} fill="#fbe48a" stroke="#b3811d" strokeWidth="0.9" />
      ))}
      <ellipse cx="0" cy="-27" rx="2.6" ry="4.6" fill="#fbe48a" stroke="#b3811d" strokeWidth="0.9" />
    </g>
  );
}

function Fields() {
  const stalks: At[] = [
    [-34, -66], [-18, -72], [18, -74], [34, -66],
    [-68, -22], [-58, -18], [62, -20], [72, -24],
    [-68, 34], [-54, 38], [56, 36], [70, 32],
    [-22, 74], [-6, 80], [10, 80], [26, 74],
  ];
  return (
    <>
      {[-72, -38, 40, 76].map((y) => (
        <path key={y} d={`M-100 ${y}Q0 ${y - 22} 100 ${y}`} fill="none" stroke="#dcaa2f" strokeWidth="9" />
      ))}
      {stalks.map((at) => (
        <Wheat key={`${at[0]},${at[1]}`} at={at} />
      ))}
    </>
  );
}

function Peak({ at }: { at: At }) {
  return (
    <g transform={place(at)}>
      <path d="M0-30-30 20H30Z" fill="#6c7782" />
      <path d="M0-30 30 20H6L-5-4Z" fill="#b2bcc3" />
      <path d="M0-30-10-13-4-16 1-10 4-15 10-13Z" fill="#fbfcfd" />
    </g>
  );
}

function Mountains() {
  const peaks: At[] = [[-34, -58, 1.05], [28, -64, 1.2], [-60, 18, 1], [60, 16, 1.05], [-22, 66, 0.95], [28, 68, 1.05]];
  const stones: [number, number][] = [[-8, -26], [-44, -18], [46, -18], [-30, 44], [44, 48], [-74, 44], [76, 44]];
  return (
    <>
      <path d="M-100-6Q-50-30 0-8T100-14V40Q50 22 0 36T-100 30Z" fill="#88939c" />
      {stones.map(([x, y]) => (
        <path key={`${x},${y}`} d={`M${x - 6} ${y}l4-5h5l3 5Z`} fill="#58626c" />
      ))}
      {peaks.map((at) => (
        <Peak key={`${at[0]},${at[1]}`} at={at} />
      ))}
    </>
  );
}

function Cactus({ at }: { at: At }) {
  return (
    <g transform={place(at)} fill="none" strokeLinecap="round">
      <ellipse cx="0" cy="19" rx="11" ry="3" fill="#cfb076" />
      <path d="M0 18V-16" stroke="#4d9352" strokeWidth="7" />
      <path d="M0 6H-10V-6" stroke="#4d9352" strokeWidth="5" strokeLinejoin="round" />
      <path d="M0 0H10V-11" stroke="#4d9352" strokeWidth="5" strokeLinejoin="round" />
      <path d="M1.8 16V-15" stroke="#6bb06c" strokeWidth="1.6" />
    </g>
  );
}

function Desert() {
  return (
    <>
      <path d="M-100-26Q-50-58 0-30T100-36V-100H-100Z" fill="#f2e0b6" />
      <path d="M-100 38Q-40 4 12 34T100 26V100H-100Z" fill="#dcc083" />
      <path d="M-100 68Q-40 44 16 66T100 60V100H-100Z" fill="#cfb076" />
      <Cactus at={[-44, -34, 1.05]} />
      <Cactus at={[50, 26, 1.15]} />
      <Cactus at={[-22, 62, 0.85]} />
      <path d="M36-52l5-6h7l4 6ZM-68 22l4-5h6l3 5Z" fill="#b39761" />
    </>
  );
}

const ART: Record<Terrain, () => React.JSX.Element> = {
  WOOD: Forest,
  BRICK: Hills,
  SHEEP: Pasture,
  WHEAT: Fields,
  ORE: Mountains,
  DESERT: Desert,
};

/** Hexagons of one terrain differ by a mirror flip, so a board doesn't look stamped. */
export const TerrainArt = memo(function TerrainArt({ terrain, flip }: { terrain: Terrain; flip: boolean }) {
  const Art = ART[terrain];
  return (
    <g transform={flip ? "scale(-1 1)" : undefined}>
      <Art />
    </g>
  );
});
