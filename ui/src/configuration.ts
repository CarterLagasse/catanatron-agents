// A production build without an address calls the server that served it.
export const API_URL: string =
  import.meta.env.CTRON_API_URL ||
  (import.meta.env.DEV ? "http://localhost:5001" : "");

// The classic UI is a separate site. Without an address in production, the switch to it is hidden.
export const CLASSIC_UI_URL: string | null =
  import.meta.env.CTRON_CLASSIC_UI_URL ||
  (import.meta.env.DEV ? `http://${window.location.hostname}:4001` : null);
