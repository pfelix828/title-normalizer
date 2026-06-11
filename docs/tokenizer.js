/* Tokenizer mirroring src/dataset.py:tokenize exactly.
 *
 * Python reference:
 *   title = title.lower().strip()
 *   title = title.replace("\t", " ")
 *   title = re.sub(r"[&/–—]", " ", title)
 *   title = re.sub(r"[^\w\s.-]", "", title)   # \w is Unicode-aware in Python
 *   title = re.sub(r"\s+", " ", title)
 *   tokens = title.split()
 *
 * Python's \w matches Unicode alphanumerics (categories L* and N*) plus
 * underscore — and nothing else — so the JS keep-class is [\p{L}\p{N}_].
 */
(function (global) {
  "use strict";

  function tokenize(title) {
    let t = String(title).toLowerCase().trim();
    t = t.split("\t").join(" ");
    t = t.replace(/[&/–—]/g, " ");
    t = t.replace(/[^\p{L}\p{N}_\s.\-]/gu, "");
    t = t.replace(/\s+/g, " ").trim();
    return t.length ? t.split(" ") : [];
  }

  function encode(title, token2id, maxLength) {
    const unk = token2id["<UNK>"];
    const tokens = tokenize(title);
    const ids = tokens.map((tok) =>
      Object.prototype.hasOwnProperty.call(token2id, tok) ? token2id[tok] : unk
    );
    const out = ids.slice(0, maxLength);
    while (out.length < maxLength) out.push(0);
    return { tokens: tokens, ids: out };
  }

  const api = { tokenize: tokenize, encode: encode };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else global.TitleTokenizer = api;
})(typeof window !== "undefined" ? window : globalThis);
