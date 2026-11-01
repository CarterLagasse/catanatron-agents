export const API_URL: string =
  import.meta.env.CTRON_API_URL || "http://localhost:5001";

// The new UI is a separate site. Without an address in production, the switch to it is hidden.
export const NEW_UI_URL: string | null =
  import.meta.env.CTRON_NEW_UI_URL ||
  (import.meta.env.DEV ? `http://${window.location.hostname}:4000` : null);
