/**
 * The scene draws heights 52000/3800 times taller than horizontal distances before the
 * exaggeration slider multiplies them. Version 0.3.0 labelled the slider value as if it
 * were the exaggeration, and the public copy said the default view was stretched "about
 * five times" when it was stretched about 66 times. These tests tie the labels and the copy
 * to the true factor.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { EXPLORE_CONTENT, TOUR_CHAPTERS } from "../../static/tools/js/explore-content.js";
import {
  BASE_HORIZONTAL_VERTICAL_SCALE_RATIO,
  formatVerticalExaggeration,
  trueVerticalExaggeration,
} from "../../static/tools/js/vertical-scale.js";

const PAGES = [
  "static/explore/index.html",
  "static/zh/explore/index.html",
  "static/tools/3D-interactive-cryosphere-explorer.html",
  "static/zh/tools/3D-interactive-cryosphere-explorer.html",
];
const NUMBER_WORDS_EN = { 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine" };
const NUMBER_WORDS_ZH = { 5: "五", 6: "六", 7: "七", 8: "八", 9: "九" };

function sliderAttributes(page) {
  const html = readFileSync(page, "utf8");
  const input = html.match(/<input[^>]*id="exaggeration"[^>]*>/);
  assert.ok(input, `${page} has an exaggeration slider`);
  const attr = (name) => Number(input[0].match(new RegExp(`${name}="([^"]+)"`))[1]);
  const label = html.match(/<span id="exaggerationValue"[^>]*>([^<]*)<\/span>/);
  assert.ok(label, `${page} has an exaggeration label`);
  return { min: attr("min"), value: attr("value"), label: label[1] };
}

test("the true exaggeration is the slider value times the base ratio", () => {
  assert.equal(BASE_HORIZONTAL_VERTICAL_SCALE_RATIO, 52000 / 3800);
  assert.ok(Math.abs(trueVerticalExaggeration(4.8) - 65.684) < 1e-3);
  assert.ok(Math.abs(trueVerticalExaggeration(1) - 13.684) < 1e-3);
});

test("the label shows the true factor, rounded to a whole number", () => {
  assert.equal(formatVerticalExaggeration(4.8), "66×");
  assert.equal(formatVerticalExaggeration(4), "55×");
  assert.equal(formatVerticalExaggeration(0.5), "7×");
  assert.equal(formatVerticalExaggeration("4.2"), "57×");
  assert.equal(formatVerticalExaggeration(Number.NaN), "");
});

test("every explorer page's static label already shows its slider's true factor", () => {
  for (const page of PAGES) {
    const { value, label } = sliderAttributes(page);
    assert.equal(label, formatVerticalExaggeration(value), page);
  }
});

test("the first tour stop states the true stretch of the view it shows", () => {
  const factor = Math.round(trueVerticalExaggeration(TOUR_CHAPTERS[0].view.sliders.exaggeration));
  const en = EXPLORE_CONTENT["en-US"].chapters["ice-continent"].body.join(" ");
  const zh = EXPLORE_CONTENT["zh-CN"].chapters["ice-continent"].body.join(" ");
  assert.match(en, new RegExp(`stretched about ${factor} times`));
  assert.match(zh, new RegExp(`拉伸了约 ${factor} 倍`));
});

test("the stretch explainer gives the true stretch at the slider's lowest setting", () => {
  const { min } = sliderAttributes("static/explore/index.html");
  const factor = Math.round(trueVerticalExaggeration(min));
  const en = EXPLORE_CONTENT["en-US"].info.exaggeration.body.join(" ");
  const zh = EXPLORE_CONTENT["zh-CN"].info.exaggeration.body.join(" ");
  assert.ok(NUMBER_WORDS_EN[factor], `add a word for ${factor}`);
  assert.match(en, new RegExp(`about ${NUMBER_WORDS_EN[factor]} times`));
  assert.match(zh, new RegExp(`约${NUMBER_WORDS_ZH[factor]}倍`));
  assert.doesNotMatch(en, /true proportions\b[^.]*\bat 1×/i);
  assert.doesNotMatch(en, /At 1× you see the true proportions/);
});
