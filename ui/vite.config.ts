import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { createServer, defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";

// Serves the classic UI from ../ui-backup alongside this one, so the switch on
// the home page has somewhere to go after a plain `npm start`.
function classicUi(): Plugin {
  const root = resolve(__dirname, "../ui-backup");
  return {
    name: "classic-ui",
    apply: "serve",
    configureServer(server) {
      const { logger } = server.config;
      if (process.env.VITEST) return;
      if (!existsSync(resolve(root, "node_modules"))) {
        logger.warn("Classic UI not started: run `npm ci` in ui-backup to enable it.");
        return;
      }
      createServer({ root, configFile: resolve(root, "vite.config.ts"), logLevel: "warn" })
        .then(async (classic) => {
          server.httpServer?.once("close", () => classic.close());
          await classic.listen();
          logger.info(`  Classic UI: ${classic.resolvedUrls?.local[0]}`);
        })
        .catch((error) => logger.warn(`Classic UI not started: ${error.message}`));
    },
  };
}

export default defineConfig({
  plugins: [react(), classicUi()],
  envPrefix: "CTRON_",
  test: {
    environment: "jsdom",
    setupFiles: "vitest.setup.ts",
  },
  server: {
    port: 4000,
    host: true, // needed for the Docker Container port mapping to work
  },
});
