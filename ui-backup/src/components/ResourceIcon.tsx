import "./ResourceIcon.scss";

export type Resource = "WOOD" | "BRICK" | "SHEEP" | "WHEAT" | "ORE";

// Colors matched to goal.png reference image
const RESOURCE_COLORS: Record<Resource, { main: string; dark: string; light: string }> = {
  WOOD: { main: "#1e5d29", dark: "#004405", light: "#2e7d32" },
  BRICK: { main: "#e26427", dark: "#b84d1e", light: "#ef7b3a" },
  SHEEP: { main: "#8bb60e", dark: "#6a8a0b", light: "#a5d63a" },
  WHEAT: { main: "#f0b615", dark: "#c89211", light: "#f6a03f" },
  ORE: { main: "#9ba09c", dark: "#7d827f", light: "#b8bbb9" },
};

export default function ResourceIcon({ resource, size = 14 }: { resource: Resource; size?: number }) {
  const s = size;
  const c = RESOURCE_COLORS[resource];
  switch (resource) {
    case "WOOD":
      return (
        <svg width={s} height={s} viewBox="0 0 24 24" className="ri ri-wood" aria-hidden>
          <path d="M12 3 L6 14 H8 L5 19 H19 L16 14 H18 Z" fill={c.main} stroke={c.dark} strokeWidth="1.2" />
          <rect x="10.5" y="19" width="3" height="3" fill={c.dark} />
        </svg>
      );
    case "BRICK":
      return (
        <svg width={s} height={s} viewBox="0 0 24 24" className="ri ri-brick" aria-hidden>
          <rect x="3" y="5" width="18" height="14" rx="1.5" fill={c.main} stroke={c.dark} strokeWidth="1.2" />
          <path d="M3 10 H21 M3 15 H21 M9 5 V19 M15 5 V19" stroke="rgba(0,0,0,0.35)" strokeWidth="0.9" />
          <path d="M3 10 H21 M3 15 H21" stroke="rgba(255,255,255,0.18)" strokeWidth="0.6" />
        </svg>
      );
    case "SHEEP":
      return (
        <svg width={s} height={s} viewBox="0 0 24 24" className="ri ri-sheep" aria-hidden>
          <ellipse cx="12" cy="13" rx="7.5" ry="5.2" fill={c.light} stroke={c.dark} strokeWidth="1.1" />
          <circle cx="9" cy="11" r="1.1" fill="#1b1b1b" />
          <circle cx="15" cy="11" r="1.1" fill="#1b1b1b" />
          <ellipse cx="12" cy="8.5" rx="4.8" ry="3.2" fill={c.main} stroke={c.dark} strokeWidth="1" />
          <ellipse cx="7.2" cy="9.2" rx="1.6" ry="1.2" fill={c.main} stroke={c.dark} strokeWidth="0.9" />
          <ellipse cx="16.8" cy="9.2" rx="1.6" ry="1.2" fill={c.main} stroke={c.dark} strokeWidth="0.9" />
          <path d="M8 16.5 L7 19 H10 Z" fill={c.dark} />
          <path d="M16 16.5 L14 19 H17 Z" fill={c.dark} />
        </svg>
      );
    case "WHEAT":
      return (
        <svg width={s} height={s} viewBox="0 0 24 24" className="ri ri-wheat" aria-hidden>
          <path d="M12 19 V5" stroke={c.dark} strokeWidth="1.6" strokeLinecap="round" />
          <ellipse cx="12" cy="5" rx="2.2" ry="2.8" fill={c.main} stroke={c.dark} strokeWidth="1" />
          <ellipse cx="10" cy="7.5" rx="1.6" ry="1.2" fill={c.light} stroke={c.dark} strokeWidth="0.9" transform="rotate(-18 10 7.5)" />
          <ellipse cx="14" cy="7.5" rx="1.6" ry="1.2" fill={c.light} stroke={c.dark} strokeWidth="0.9" transform="rotate(18 14 7.5)" />
          <ellipse cx="10" cy="10.8" rx="1.6" ry="1.2" fill={c.light} stroke={c.dark} strokeWidth="0.9" transform="rotate(-18 10 10.8)" />
          <ellipse cx="14" cy="10.8" rx="1.6" ry="1.2" fill={c.light} stroke={c.dark} strokeWidth="0.9" transform="rotate(18 14 10.8)" />
          <path d="M9 16 Q12 15 15 16" stroke={c.dark} strokeWidth="0.9" fill="none" opacity="0.7" />
        </svg>
      );
    case "ORE":
      return (
        <svg width={s} height={s} viewBox="0 0 24 24" className="ri ri-ore" aria-hidden>
          <path d="M4 18 L12 4 L20 18 Z" fill={c.main} stroke={c.dark} strokeWidth="1.2" />
          <path d="M8.5 11 L12 4 L15.5 11 Z" fill={c.light} stroke={c.dark} strokeWidth="1" />
          <path d="M7 14 L10 9 L13 12 L16 9 L17.5 14 Z" fill="rgba(0,0,0,0.18)" />
        </svg>
      );
    default:
      return null;
  }
}
