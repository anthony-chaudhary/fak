const { test, expect } = require("@playwright/test");

const crypto = require("node:crypto");
const fs = require("node:fs/promises");
const http = require("node:http");
const path = require("node:path");

const schema = "fak.prompt-vault-color-palette-grade/1";
const viewport = { width: 1440, height: 900 };
const artifactPath = process.env.PALETTE_ARTIFACT;
const artifactID = process.env.PALETTE_ARTIFACT_ID;
const reportPath = process.env.PALETTE_REPORT;
const screenshotPath = process.env.PALETTE_SCREENSHOT;

for (const [name, value] of Object.entries({
  PALETTE_ARTIFACT: artifactPath,
  PALETTE_ARTIFACT_ID: artifactID,
  PALETTE_REPORT: reportPath,
  PALETTE_SCREENSHOT: screenshotPath,
})) {
  if (!value) throw new Error(`${name} is required`);
}

test.use({
  channel: process.env.PALETTE_BROWSER_CHANNEL || "chrome",
  colorScheme: "light",
  locale: "en-US",
  reducedMotion: "no-preference",
  viewport,
});

let artifactBytes;
let server;
let origin;

test.beforeAll(async () => {
  artifactBytes = await fs.readFile(artifactPath);
  server = http.createServer((request, response) => {
    const requestPath = new URL(request.url, "http://127.0.0.1").pathname;
    if (requestPath !== "/" && requestPath !== "/index.html") {
      response.writeHead(404, { "content-type": "text/plain; charset=utf-8" });
      response.end("not found\n");
      return;
    }
    response.writeHead(200, {
      "cache-control": "no-store",
      "content-type": "text/html; charset=utf-8",
    });
    response.end(artifactBytes);
  });
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  origin = `http://127.0.0.1:${address.port}`;
});

test.afterAll(async () => {
  if (!server) return;
  await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
});

function colorsChanged(before, after) {
  return before.length === 5 && after.length === 5 && before.every((color, index) => color !== after[index]);
}

async function discover(page) {
  return page.evaluate(() => {
    const hexPattern = /^#[0-9A-F]{6}$/i;
    const visible = (element) => {
      const style = getComputedStyle(element);
      const rect = element.getBoundingClientRect();
      return rect.width > 0 && rect.height > 0 && style.display !== "none" &&
        style.visibility !== "hidden" && Number.parseFloat(style.opacity || "1") > 0.05;
    };
    const rgba = (value) => {
      const match = value.match(/rgba?\(([^)]+)\)/i);
      if (!match) return null;
      const parts = match[1].split(/[ ,/]+/).filter(Boolean).map(Number);
      if (parts.length < 3 || parts.slice(0, 3).some(Number.isNaN)) return null;
      return { r: parts[0], g: parts[1], b: parts[2], a: parts[3] ?? 1 };
    };
    const hexFromRGB = (color) => color ? `#${[color.r, color.g, color.b]
      .map((part) => Math.round(part).toString(16).padStart(2, "0"))
      .join("")}`.toUpperCase() : "";
    const relativeLuminance = (color) => {
      const channel = (value) => {
        const normalized = value / 255;
        return normalized <= 0.04045 ? normalized / 12.92 : ((normalized + 0.055) / 1.055) ** 2.4;
      };
      return 0.2126 * channel(color.r) + 0.7152 * channel(color.g) + 0.0722 * channel(color.b);
    };
    const contrast = (foreground, background) => {
      if (!foreground || !background) return 0;
      const first = relativeLuminance(foreground);
      const second = relativeLuminance(background);
      return (Math.max(first, second) + 0.05) / (Math.min(first, second) + 0.05);
    };
    const seconds = (value) => value.split(",").map((entry) => {
      const trimmed = entry.trim();
      return trimmed.endsWith("ms") ? Number.parseFloat(trimmed) / 1000 : Number.parseFloat(trimmed);
    });

    document.querySelectorAll("[data-fak-grade-swatch], [data-fak-grade-hex], [data-fak-grade-lock], [data-fak-grade-generate]")
      .forEach((element) => {
        element.removeAttribute("data-fak-grade-swatch");
        element.removeAttribute("data-fak-grade-hex");
        element.removeAttribute("data-fak-grade-lock");
        element.removeAttribute("data-fak-grade-generate");
      });

    const entries = [];
    const seen = new Set();
    const hexElements = [...document.querySelectorAll("body *")]
      .filter((element) => visible(element) && hexPattern.test(element.textContent.trim()));
    for (const hexElement of hexElements) {
      let swatch = hexElement;
      while (swatch && swatch !== document.body) {
        const rect = swatch.getBoundingClientRect();
        const background = rgba(getComputedStyle(swatch).backgroundColor);
        if (rect.width >= innerWidth * 0.12 && rect.height >= innerHeight * 0.45 && background?.a > 0.05) break;
        swatch = swatch.parentElement;
      }
      if (!swatch || swatch === document.body || seen.has(swatch)) continue;
      seen.add(swatch);
      entries.push({ swatch, hexElement });
    }
    entries.sort((left, right) => left.swatch.getBoundingClientRect().x - right.swatch.getBoundingClientRect().x);

    const swatches = entries.map(({ swatch, hexElement }, index) => {
      swatch.dataset.fakGradeSwatch = String(index);
      hexElement.dataset.fakGradeHex = String(index);
      const swatchRect = swatch.getBoundingClientRect();
      const hexRect = hexElement.getBoundingClientRect();
      const swatchStyle = getComputedStyle(swatch);
      const hexStyle = getComputedStyle(hexElement);
      const background = rgba(swatchStyle.backgroundColor);
      const foreground = rgba(hexStyle.color);
      const properties = swatchStyle.transitionProperty.split(",").map((value) => value.trim());
      const durations = seconds(swatchStyle.transitionDuration);
      const transitionSeconds = Math.max(0, ...properties.map((property, propertyIndex) =>
        property === "all" || property === "background" || property === "background-color"
          ? durations[propertyIndex % durations.length] || 0
          : 0));
      const lock = [...swatch.querySelectorAll("button, [role=button]")].find((button) => {
        const words = [button.textContent, button.getAttribute("aria-label"), button.getAttribute("title")]
          .filter(Boolean).join(" ");
        return !hexPattern.test(button.textContent.trim()) && /lock|unlock|🔒|🔓/i.test(words);
      });
      if (lock) lock.dataset.fakGradeLock = String(index);
      return {
        background: hexFromRGB(background),
        centered: Math.abs((hexRect.x + hexRect.width / 2) - (swatchRect.x + swatchRect.width / 2)) <= Math.max(40, swatchRect.width * 0.2) &&
          Math.abs((hexRect.y + hexRect.height / 2) - (swatchRect.y + swatchRect.height / 2)) <= swatchRect.height * 0.2,
        contrast: contrast(foreground, background),
        height: swatchRect.height,
        hex: hexElement.textContent.trim().toUpperCase(),
        lock: lock ? {
          label: lock.getAttribute("aria-label") || lock.getAttribute("title") || "",
          pressed: lock.getAttribute("aria-pressed"),
          text: lock.textContent.trim(),
        } : null,
        transition_seconds: transitionSeconds,
        width: swatchRect.width,
        x: swatchRect.x,
        y: swatchRect.y,
      };
    });

    const generate = [...document.querySelectorAll("button, [role=button]")].find((button) => {
      const words = [button.textContent, button.getAttribute("aria-label"), button.getAttribute("title")]
        .filter(Boolean).join(" ");
      return visible(button) && /\bgenerate\b/i.test(words);
    });
    if (generate) generate.dataset.fakGradeGenerate = "true";

    const externalResources = [...document.querySelectorAll("script[src], link[href], img[src], iframe[src]")]
      .map((element) => element.getAttribute("src") || element.getAttribute("href") || "")
      .filter((value) => /^(?:https?:)?\/\//i.test(value));
    const widths = swatches.map((swatch) => swatch.width);
    const heights = swatches.map((swatch) => swatch.height);
    const first = swatches[0];
    const last = swatches.at(-1);
    const equalGeometry = swatches.length === 5 &&
      Math.max(...widths) - Math.min(...widths) <= 2 &&
      Math.max(...heights) - Math.min(...heights) <= 2 &&
      Math.max(...swatches.map((swatch) => swatch.y)) - Math.min(...swatches.map((swatch) => swatch.y)) <= 2 &&
      first.x <= 2 && last.x + last.width >= innerWidth - 2 &&
      swatches.slice(1).every((swatch, index) => Math.abs(swatch.x - (swatches[index].x + swatches[index].width)) <= 2);

    return {
      equalGeometry,
      externalResources,
      generateFound: Boolean(generate),
      swatches,
    };
  });
}

async function load(page) {
  await page.goto(`${origin}/index.html`, { waitUntil: "load" });
  await page.waitForTimeout(450);
  return discover(page);
}

function palette(snapshot) {
  return snapshot.swatches.map((swatch) => swatch.hex);
}

test("grades the frozen Color Palette behavior and captures a desktop render", async ({ browserName, context, page }) => {
  const checks = [];
  const add = (id, passed, detail) => checks.push({ id, passed: Boolean(passed), detail });
  const attempt = async (id, action) => {
    try {
      const result = await action();
      add(id, result.passed, result.detail);
    } catch (error) {
      add(id, false, String(error.message || error).split("\n")[0]);
    }
  };

  await context.grantPermissions(["clipboard-read", "clipboard-write"], { origin });
  await page.addInitScript(() => {
    let state = 0x8323;
    Math.random = () => {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      return state / 0x100000000;
    };
  });

  const initial = await load(page);
  add(
    "single_file_vanilla",
    path.basename(artifactPath).toLowerCase() === "index.html" && initial.externalResources.length === 0,
    `index.html with ${initial.externalResources.length} external runtime resource(s)`,
  );
  add(
    "five_equal_desktop_swatches",
    initial.swatches.length === 5 && initial.equalGeometry,
    `${initial.swatches.length} swatch(es); equal full-width geometry=${initial.equalGeometry}`,
  );
  add(
    "centered_hex_labels",
    initial.swatches.length === 5 && initial.swatches.every((swatch) => swatch.hex === swatch.background && swatch.centered),
    `${initial.swatches.filter((swatch) => swatch.hex === swatch.background && swatch.centered).length}/5 labels centered and matched to the background`,
  );
  add(
    "contrast_aware_text",
    initial.swatches.length === 5 && initial.swatches.every((swatch) => swatch.contrast >= 4.5),
    `minimum HEX/background contrast ${initial.swatches.length ? Math.min(...initial.swatches.map((swatch) => swatch.contrast)).toFixed(2) : "0.00"}:1`,
  );
  add(
    "smooth_color_transition",
    initial.swatches.length === 5 && initial.swatches.every((swatch) => swatch.transition_seconds > 0 && swatch.transition_seconds <= 1),
    `${initial.swatches.filter((swatch) => swatch.transition_seconds > 0 && swatch.transition_seconds <= 1).length}/5 swatches transition background color within one second`,
  );

  await attempt("generate_button_replaces_unlocked", async () => {
    const before = palette(await load(page));
    if (await page.locator("[data-fak-grade-generate]").count() !== 1) return { passed: false, detail: "Generate control not found" };
    await page.locator("[data-fak-grade-generate]").click();
    const after = palette(await discover(page));
    return { passed: colorsChanged(before, after), detail: `${before.join(" ")} -> ${after.join(" ")}` };
  });

  let lockToggle = { passed: false, detail: "lock controls not found" };
  let lockPersistence = { passed: false, detail: "lock controls not found" };
  try {
    const before = await load(page);
    if (before.swatches.length === 5 && before.swatches.every((swatch) => swatch.lock) &&
        await page.locator('[data-fak-grade-lock="0"]').count() === 1) {
      const unlockedIndicator = before.swatches[0].lock;
      const unlockedColors = palette(before);
      await page.locator('[data-fak-grade-lock="0"]').click();
      const locked = await discover(page);
      const lockedIndicator = locked.swatches[0]?.lock;
      const lockedColors = palette(locked);
      await page.locator("[data-fak-grade-generate]").click();
      const regenerated = await discover(page);
      const regeneratedColors = palette(regenerated);
      lockPersistence = {
        passed: regeneratedColors.length === 5 && regeneratedColors[0] === lockedColors[0] &&
          regeneratedColors.slice(1).every((color, index) => color !== lockedColors[index + 1]),
        detail: `locked ${lockedColors[0]} -> ${regeneratedColors[0]}; unlocked changed ${regeneratedColors.slice(1).filter((color, index) => color !== lockedColors[index + 1]).length}/4`,
      };
      await page.locator('[data-fak-grade-lock="0"]').click();
      const unlockedAgain = await discover(page);
      const finalIndicator = unlockedAgain.swatches[0]?.lock;
      const changedOnLock = JSON.stringify(unlockedIndicator) !== JSON.stringify(lockedIndicator);
      const changedOnUnlock = JSON.stringify(lockedIndicator) !== JSON.stringify(finalIndicator);
      const returned = finalIndicator?.pressed === unlockedIndicator?.pressed || finalIndicator?.text === unlockedIndicator?.text;
      lockToggle = {
        passed: changedOnLock && changedOnUnlock && returned && unlockedColors[0] === lockedColors[0],
        detail: `${unlockedIndicator?.text || unlockedIndicator?.label} -> ${lockedIndicator?.text || lockedIndicator?.label} -> ${finalIndicator?.text || finalIndicator?.label}`,
      };
    }
  } catch (error) {
    lockToggle.detail = String(error.message || error).split("\n")[0];
    lockPersistence.detail = lockToggle.detail;
  }
  add("lock_toggles_both_states", lockToggle.passed, lockToggle.detail);
  add("locked_color_survives_regeneration", lockPersistence.passed, lockPersistence.detail);

  await attempt("spacebar_regenerates_unlocked", async () => {
    const before = palette(await load(page));
    await page.evaluate(() => {
      if (document.activeElement instanceof HTMLElement) document.activeElement.blur();
      document.body.tabIndex = -1;
      document.body.focus();
    });
    await page.keyboard.press("Space");
    const after = palette(await discover(page));
    return { passed: colorsChanged(before, after), detail: `${before.join(" ")} -> ${after.join(" ")}` };
  });

  let copiedValue = { passed: false, detail: "HEX control not found" };
  let copiedFeedback = { passed: false, detail: "HEX control not found" };
  try {
    const before = await load(page);
    const expectedColor = before.swatches[0]?.hex;
    if (expectedColor && await page.locator('[data-fak-grade-hex="0"]').count() === 1) {
      await page.locator('[data-fak-grade-hex="0"]').click();
      const clipboard = await page.evaluate(() => navigator.clipboard?.readText());
      let feedback = false;
      try {
        await page.waitForFunction(() => [...document.querySelectorAll("body *")].some((element) => {
          const style = getComputedStyle(element);
          const rect = element.getBoundingClientRect();
          return /\bcopied\b/i.test(element.textContent) && rect.width > 0 && rect.height > 0 &&
            style.display !== "none" && style.visibility !== "hidden" && Number.parseFloat(style.opacity || "1") > 0.1;
        }), null, { timeout: 1000 });
        feedback = true;
      } catch {
        // The check below records the missing visible confirmation.
      }
      copiedValue = { passed: clipboard === expectedColor, detail: `clipboard=${clipboard || "EMPTY"}; expected=${expectedColor}` };
      copiedFeedback = { passed: feedback, detail: `visible Copied confirmation=${feedback}` };
    }
  } catch (error) {
    copiedValue.detail = String(error.message || error).split("\n")[0];
    copiedFeedback.detail = copiedValue.detail;
  }
  add("hex_click_copies_value", copiedValue.passed, copiedValue.detail);
  add("copy_shows_confirmation", copiedFeedback.passed, copiedFeedback.detail);

  let renderSHA256 = "";
  await attempt("desktop_render_captured", async () => {
    await load(page);
    await fs.mkdir(path.dirname(screenshotPath), { recursive: true });
    await page.screenshot({ path: screenshotPath, fullPage: false });
    const bytes = await fs.readFile(screenshotPath);
    renderSHA256 = crypto.createHash("sha256").update(bytes).digest("hex");
    return { passed: bytes.length > 0, detail: `${viewport.width}x${viewport.height} PNG sha256:${renderSHA256}` };
  });

  const passed = checks.filter((check) => check.passed).length;
  const report = {
    schema,
    artifact_id: artifactID,
    artifact_sha256: crypto.createHash("sha256").update(artifactBytes).digest("hex"),
    observed_at: process.env.PALETTE_OBSERVED_AT || new Date().toISOString(),
    browser: {
      name: browserName,
      version: await page.context().browser().version(),
      channel: process.env.PALETTE_BROWSER_CHANNEL || "chrome",
    },
    viewport,
    acceptance_passed: passed,
    acceptance_total: checks.length,
    terminal_verdict: passed === checks.length ? "PASS" : "FAIL",
    render: {
      file: path.basename(screenshotPath),
      sha256: renderSHA256,
    },
    checks,
  };
  await fs.mkdir(path.dirname(reportPath), { recursive: true });
  await fs.writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`);

  const failures = checks.filter((check) => !check.passed).map((check) => check.id);
  expect(report.terminal_verdict, `failed checks: ${failures.join(", ")}`).toBe("PASS");
});
