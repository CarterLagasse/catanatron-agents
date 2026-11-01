import "./ResourceIcon.scss";

export type Resource = "WOOD" | "BRICK" | "SHEEP" | "WHEAT" | "ORE";

// A small terrain-coloured hexagon, so each resource reads on light and dark backgrounds alike.
const HEX = "M12 1.5 21.1 6.75V17.25L12 22.5 2.9 17.25V6.75Z";

const FILL: Record<Resource, string> = {
  WOOD: "#2f7d42",
  BRICK: "#c8602f",
  SHEEP: "#92c044",
  WHEAT: "#ecb62e",
  ORE: "#7d8993",
};

const GLYPHS: Record<Resource, React.JSX.Element> = {
  WOOD: (
    <>
      <rect x="11" y="16" width="2" height="3" fill="#f6ecd2" />
      <path d="M12 4.6 8.5 10h1.6L7 16h10l-3.1-6h1.6Z" fill="#f6ecd2" />
    </>
  ),
  BRICK: (
    <g fill="#f8e9d6">
      <rect x="6.2" y="7.4" width="5.4" height="2.9" rx=".5" />
      <rect x="12.4" y="7.4" width="5.4" height="2.9" rx=".5" />
      <rect x="9.2" y="11" width="5.6" height="2.9" rx=".5" />
      <rect x="6.2" y="14.6" width="5.4" height="2.9" rx=".5" />
      <rect x="12.4" y="14.6" width="5.4" height="2.9" rx=".5" />
    </g>
  ),
  SHEEP: (
    <>
      <path d="M8.7 15.2v2.3M11 15.6v2M13.4 15.6v2M15.5 15.2v2.3" stroke="#39332f" strokeWidth="1.1" strokeLinecap="round" />
      <g fill="#fffdf6">
        <circle cx="8.5" cy="12.2" r="2.6" />
        <circle cx="11.4" cy="10.6" r="2.9" />
        <circle cx="14.3" cy="12" r="2.6" />
        <circle cx="10.4" cy="14" r="2.5" />
        <circle cx="13.4" cy="14" r="2.5" />
      </g>
      <ellipse cx="17" cy="11" rx="1.9" ry="2.4" fill="#39332f" />
    </>
  ),
  WHEAT: (
    <>
      <path d="M12 19V8" stroke="#6b470c" strokeWidth="1.3" strokeLinecap="round" />
      <g fill="#6b470c">
        <ellipse cx="12" cy="6.2" rx="1.5" ry="2.3" />
        <ellipse cx="9.8" cy="9.4" rx="1.4" ry="2.1" transform="rotate(-24 9.8 9.4)" />
        <ellipse cx="14.2" cy="9.4" rx="1.4" ry="2.1" transform="rotate(24 14.2 9.4)" />
        <ellipse cx="9.8" cy="13" rx="1.4" ry="2.1" transform="rotate(-24 9.8 13)" />
        <ellipse cx="14.2" cy="13" rx="1.4" ry="2.1" transform="rotate(24 14.2 13)" />
      </g>
    </>
  ),
  ORE: (
    <>
      <path d="M12 5.4 5.8 17h12.4Z" fill="#46505a" />
      <path d="M12 5.4 18.2 17h-5l-2.3-5.2Z" fill="#dfe4e8" />
      <path d="M12 5.4 9.7 9.8 11.3 9l1.3 1.3 1.7-.5Z" fill="#fff" />
    </>
  ),
};

export default function ResourceIcon({ resource, size = 14 }: { resource: Resource; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" className={`ri ri-${resource.toLowerCase()}`} aria-hidden>
      <path d={HEX} fill={FILL[resource]} stroke={FILL[resource]} strokeWidth="1.5" strokeLinejoin="round" />
      {GLYPHS[resource]}
    </svg>
  );
}
