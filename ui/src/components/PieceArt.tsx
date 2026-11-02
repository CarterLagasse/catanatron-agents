import type { DevelopmentCard } from "../utils/api.types";

// Player pieces, shared by the board and the buy buttons. Colours come from .piece-colors.

export function SettlementArt({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 28 28" className={className} aria-hidden>
      <path className="piece-outline" d="M3.5 25V14L9.5 6.5h9L24.5 14v11Z" />
      <path className="piece-side" d="M15.5 14h9v11h-9Z" />
      <path className="piece-face" d="M3.5 14 9.5 6.5 15.5 14v11h-12Z" />
      <path className="piece-roof" d="M9.5 6.5h9l6 7.5H15.5Z" />
    </svg>
  );
}

export function CityArt({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 36 32" className={className} aria-hidden>
      <path className="piece-outline" d="M2.5 29V11.5L8.5 3h5l6 8.5V13H29l4.5 6v10Z" />
      <path className="piece-face" d="M19.5 19h14v10h-14Z" />
      <path className="piece-roof" d="M19.5 19V13H29l4.5 6Z" />
      <path className="piece-side" d="M14.5 11.5h5V29h-5Z" />
      <path className="piece-face" d="M2.5 11.5 8.5 3l6 8.5V29h-12Z" />
      <path className="piece-roof" d="M8.5 3h5l6 8.5h-5Z" />
    </svg>
  );
}

export function RoadArt() {
  return (
    <svg viewBox="0 0 28 28" aria-hidden>
      <g transform="rotate(-38 14 14)">
        <path className="piece-outline" d="M1.5 14 4.5 9.5h19l3 4.5-3 4.5h-19Z" />
        <path className="piece-face" d="M1.5 14 4.5 9.5h19l3 4.5-3 4.5h-19Z" />
        <path className="piece-roof" d="M3.2 11.5 4.5 9.5h19l1.3 2Z" />
        <path className="piece-side" d="M3.2 16.5h21.6l-1.3 2h-19Z" />
      </g>
    </svg>
  );
}

export function DevCardArt() {
  return (
    <svg viewBox="0 0 28 28" aria-hidden>
      <rect x="8" y="3.5" width="15" height="21" rx="2.4" transform="rotate(9 15.5 14)" fill="#3f2a7d" stroke="#1a232c" strokeWidth="1.3" />
      <rect x="5" y="3.5" width="15" height="21" rx="2.4" fill="#6f50c8" stroke="#1a232c" strokeWidth="1.3" />
      <rect x="7" y="5.5" width="11" height="17" rx="1.2" fill="none" stroke="#b9a3ea" strokeWidth="0.9" />
      <path d="m12.5 9 1.5 3.1 3.4.5-2.5 2.4.6 3.4-3-1.6-3 1.6.6-3.4-2.5-2.4 3.4-.5Z" fill="#f2b632" />
    </svg>
  );
}

// A filled shape with a darker side, outlined last so the shading can't cover the outline.
function Shaded({ d, shade, fill, dark }: { d: string; shade: string; fill: string; dark: string }) {
  return (
    <>
      <path d={d} fill={fill} stroke="none" />
      <path d={shade} fill={dark} stroke="none" />
      <path d={d} fill="none" />
    </>
  );
}

const SWORD = (
  <>
    <path d="M16 .5 17.4 4v18h-2.8V4Z" fill="#e6ebef" />
    <rect x="11.5" y="21.6" width="9" height="2.2" rx="1.1" fill="#f2b632" />
    <rect x="15" y="23.8" width="2" height="3.7" fill="#7a4a24" />
    <circle cx="16" cy="28.6" r="1.4" fill="#f2b632" />
  </>
);

const DEV_CARD_LOGOS: Record<DevelopmentCard | "VICTORY_POINT", React.JSX.Element> = {
  KNIGHT: (
    <>
      <g transform="rotate(45 16 16)">{SWORD}</g>
      <g transform="rotate(-45 16 16)">{SWORD}</g>
      <Shaded
        d="M16 9 22.5 11.2V17c0 3.9-2.8 6.3-6.5 7.8-3.7-1.5-6.5-3.9-6.5-7.8v-5.8Z"
        shade="M16 9 22.5 11.2V17c0 3.9-2.8 6.3-6.5 7.8Z"
        fill="#d0432f"
        dark="#a52f22"
      />
      <path d="M16 12.2v9M12.8 15.8h6.4" stroke="#f6d27a" strokeWidth="1.9" strokeLinecap="round" />
    </>
  ),
  YEAR_OF_PLENTY: (
    <>
      <circle cx="21.5" cy="5.6" r="3" fill="#ecb62e" />
      <circle cx="26.8" cy="8" r="3" fill="#d0432f" />
      <circle cx="28" cy="13.6" r="2.9" fill="#92c044" />
      <Shaded
        d="M2.6 25.6c-.6-1.6.4-3 1.9-3.4C9 20.8 13 15 16.6 6.8L27.5 18c-5 7-15 10.7-21.5 9.5-1.5-.3-2.9-.7-3.4-1.9Z"
        shade="M6 27.5c6.5 1.2 16.5-2.5 21.5-9.5l-2.7-2.8c-5 6-12.8 10-20.3 9.8-.8.1-1.1 1.9 1.5 2.5Z"
        fill="#ecb03a"
        dark="#b67f1f"
      />
      <ellipse cx="22" cy="12.4" rx="3.2" ry="8" transform="rotate(-44 22 12.4)" fill="#5b3a12" />
      <circle cx="20.8" cy="10.8" r="2.7" fill="#2f9a52" />
      <circle cx="24.3" cy="13.5" r="2.3" fill="#f2c94c" />
    </>
  ),
  MONOPOLY: (
    <>
      <Shaded
        d="M6 23.5 3.5 11l7.2 5L16 6.5 21.3 16l7.2-5L26 23.5Z"
        shade="M16 6.5 21.3 16l7.2-5L26 23.5H16Z"
        fill="#f2b632"
        dark="#d9951c"
      />
      <rect x="5.5" y="23.5" width="21" height="4" rx="1" fill="#b9791a" />
      <circle cx="3.5" cy="10.5" r="1.9" fill="#fff3c9" />
      <circle cx="16" cy="5.2" r="2.1" fill="#fff3c9" />
      <circle cx="28.5" cy="10.5" r="1.9" fill="#fff3c9" />
      <circle cx="16" cy="25.5" r="1.3" fill="#e0533c" />
      <circle cx="10.5" cy="25.5" r="1" fill="#6fd3e8" />
      <circle cx="21.5" cy="25.5" r="1" fill="#6fd3e8" />
    </>
  ),
  ROAD_BUILDING: (
    <>
      <path d="M9.5 8h5.5L21 28.5H2Z" fill="#59636d" />
      <g transform="rotate(38 23 12)">
        <rect x="21.5" y="9" width="2.5" height="16" rx="1" fill="#a06a35" />
        <path d="M18 5h8.5c.8 0 1.5.7 1.5 1.5V9c0 .8-.7 1.5-1.5 1.5H18Z" fill="#dfe4e8" />
      </g>
    </>
  ),
  VICTORY_POINT: (
    <>
      <path d="M9.5 7H5c0 5 2.2 8.3 6.5 8.8M22.5 7H27c0 5-2.2 8.3-6.5 8.8" fill="none" stroke="#f2b632" strokeWidth="2.2" />
      <Shaded
        d="M9 3.5h14V12c0 4.2-3 7.5-7 7.5S9 16.2 9 12Z"
        shade="M16 3.5h7V12c0 4.2-3 7.5-7 7.5Z"
        fill="#f2b632"
        dark="#d9951c"
      />
      <rect x="14.4" y="19.5" width="3.2" height="4" fill="#d9951c" />
      <path d="M10 28.5c0-3.3 2.2-5 6-5s6 1.7 6 5Z" fill="#b9791a" />
      <path d="m16 6.3 1.3 2.7 3 .4-2.2 2.1.5 3-2.6-1.4-2.6 1.4.5-3-2.2-2.1 3-.4Z" fill="#fff8df" stroke="none" />
    </>
  ),
};

export function DevCardLogo({ card }: { card: DevelopmentCard | "VICTORY_POINT" }) {
  return (
    <svg viewBox="0 0 32 32" stroke="#1a232c" strokeWidth="0.9" strokeLinejoin="round" aria-hidden>
      {DEV_CARD_LOGOS[card]}
    </svg>
  );
}

export function CardsArt() {
  return (
    <svg viewBox="0 0 28 28" aria-hidden>
      <g stroke="#1a232c" strokeWidth="1.3">
        <rect x="8.5" y="5" width="11" height="17" rx="2" transform="rotate(-24 14 24)" fill="#2f7d42" />
        <rect x="8.5" y="5" width="11" height="17" rx="2" fill="#c8602f" />
        <rect x="8.5" y="5" width="11" height="17" rx="2" transform="rotate(24 14 24)" fill="#fff6df" />
      </g>
      <path transform="rotate(24 14 24)" d="M14 9l3.5 2v4l-3.5 2-3.5-2v-4Z" fill="#ecb62e" />
    </svg>
  );
}
