import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { SnackbarProvider } from "notistack";
import { createTheme, ThemeProvider } from "@mui/material/styles";
import Fade from "@mui/material/Fade";

import GameScreen from "./pages/GameScreen";
import HomePage from "./pages/HomePage";
import { StateProvider } from "./store";

import "./App.scss";
import ReplayScreen from "./pages/ReplayScreen";

const theme = createTheme({
  palette: {
    mode: "dark",
    primary: { main: "#f2b632", contrastText: "#241a00" },
    secondary: { main: "#6a4cbe" },
    background: { default: "#0e1a24", paper: "#1c2f3f" },
  },
  typography: {
    fontFamily: '"Nunito", "Segoe UI", sans-serif',
    button: { fontWeight: 800, textTransform: "none" },
  },
  shape: { borderRadius: 10 },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <StateProvider>
        <SnackbarProvider
          maxSnack={2}
          autoHideDuration={4000}
          TransitionComponent={Fade}
          TransitionProps={{ timeout: 100 }}
        >
          <Router>
            <Routes>
              <Route
                path="/games/:gameId/states/:stateIndex"
                element={<GameScreen replayMode={true} />}
              />
              <Route path="/replays/:gameId" element={<ReplayScreen />} />
              <Route
                path="/games/:gameId"
                element={<GameScreen replayMode={false} />}
              />
              <Route path="/" element={<HomePage />} />
            </Routes>
          </Router>
        </SnackbarProvider>
      </StateProvider>
    </ThemeProvider>
  );
}

export default App;
