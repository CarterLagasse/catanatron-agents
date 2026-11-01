import { useCallback, useContext, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import { useSnackbar } from "notistack";

import ACTIONS from "../actions";
import { store } from "../store";
import type { GameAction } from "../utils/api.types";
import { postAction } from "../utils/apiClient";

/** Sends the human's moves. Ignores clicks while one is on its way, and reports failures. */
export default function useGameActions() {
  const { gameId } = useParams();
  const { dispatch } = useContext(store);
  const { enqueueSnackbar } = useSnackbar();
  const [busy, setBusy] = useState(false);
  const inFlight = useRef(false);

  const act = useCallback(
    async (...actions: GameAction[]) => {
      if (!gameId || inFlight.current) return;
      inFlight.current = true;
      setBusy(true);
      try {
        let latest = null;
        for (const action of actions) {
          latest = await postAction(gameId, action);
        }
        if (latest) dispatch({ type: ACTIONS.SET_GAME_STATE, data: latest });
      } catch (error) {
        console.error("Move failed", error);
        enqueueSnackbar("That move didn't go through. Check the game server and try again.", {
          variant: "error",
          autoHideDuration: 5000,
        });
      } finally {
        inFlight.current = false;
        setBusy(false);
      }
    },
    [gameId, dispatch, enqueueSnackbar],
  );

  return { act, busy };
}
