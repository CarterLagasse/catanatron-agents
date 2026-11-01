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
