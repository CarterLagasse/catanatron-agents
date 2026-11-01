import React, { createContext, useReducer } from "react";
import ACTIONS from "./actions";
import { type GameState } from "./utils/api.types";
import { rollGains, type RollGains } from "./utils/gameLogic";

export type BuildMode = "ROAD" | "SETTLEMENT" | "CITY" | null;

export type CatanState = {
  gameState: GameState | null;
  buildMode: BuildMode;
  // What each roll paid out, by action record index. Only known for rolls seen live.
  gainsByRecord: Record<number, RollGains>;
};
type ReducerAction = {
  type: keyof typeof ACTIONS;
  data?: any;
};

const initialState: CatanState = {
  gameState: null,
  buildMode: null,
  gainsByRecord: {},
};

const store = createContext<{
  state: CatanState;
  dispatch: React.ActionDispatch<[action: ReducerAction]>;
}>({ state: initialState, dispatch: () => {} });
const { Provider } = store;

const StateProvider = ({ children }: { children: React.ReactNode }) => {
  const [state, dispatch] = useReducer(
    (state: CatanState, action: ReducerAction): CatanState => {
      switch (action.type) {
        case ACTIONS.RESET_GAME:
          return initialState;
        case ACTIONS.SET_GAME_STATE: {
          const next: GameState = action.data;
          const gains = state.gameState && rollGains(state.gameState, next);
          return {
            ...state,
            gameState: next,
            buildMode: null,
            gainsByRecord: gains
              ? { ...state.gainsByRecord, [next.action_records.length - 1]: gains }
              : state.gainsByRecord,
          };
        }
        case ACTIONS.SET_BUILD_MODE:
          return { ...state, buildMode: action.data };
        default:
          throw new Error("Unknown Reducer Action: " + action.type);
      }
    },
    initialState
  );

  return <Provider value={{ state, dispatch }}>{children}</Provider>;
};

export { store, StateProvider };
