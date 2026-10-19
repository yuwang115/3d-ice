/**
 * Source credits for the research edition's Data Snapshot.
 *
 * A source is plain text, a link `{ text, url }`, or an array of the two run together, so
 * that one line can credit a data set and the article that describes it.
 *
 * No DOM or scene dependencies: this module runs unchanged under Node's test runner.
 */

export function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

export function renderSourceValue(source) {
  if (!source) return "";
  if (Array.isArray(source)) return source.map(renderSourceValue).join("");
  if (typeof source === "string") return escapeHtml(source);
  const text = escapeHtml(source.text || "");
  const url = escapeHtml(source.url || "");
  if (!url) return text;
  return `<a class="meta-link" href="${url}" target="_blank" rel="noopener noreferrer">${text}</a>`;
}

export function renderSourceLine(label, source, suffix = "") {
  return `<li class="meta-compact"><strong>${escapeHtml(label)}:</strong> ${renderSourceValue(source)}${suffix}</li>`;
}
