/* Demo app: load vocab + ONNX model, classify titles as you type, and run a
 * parity self-test against recorded PyTorch outputs (fixtures.json). */
(function () {
  "use strict";

  var EXAMPLES = [
    "Sr. Dir. of Eng, EMEA",
    "VP Sales & Marketing",
    "principal pm – payments",
    "DIRECTOR, DATA ANALYTICS",
    "Chief People Officer",
    "Mkt Ops Lead",
  ];

  var PRETTY = {
    individual_contributor: "Individual Contributor", senior: "Senior", lead: "Lead",
    manager: "Manager", senior_manager: "Senior Manager", director: "Director",
    senior_director: "Senior Director", vp: "VP", svp: "SVP", c_suite: "C-suite",
    engineering: "Engineering", data: "Data", product: "Product", marketing: "Marketing",
    sales: "Sales", finance: "Finance", hr: "HR", operations: "Operations",
    design: "Design", legal: "Legal",
  };

  var $ = function (id) { return document.getElementById(id); };
  var session = null, vocab = null;

  function softmax(logits) {
    var m = Math.max.apply(null, logits);
    var exps = logits.map(function (v) { return Math.exp(v - m); });
    var s = exps.reduce(function (a, b) { return a + b; }, 0);
    return exps.map(function (v) { return v / s; });
  }

  function topK(probs, names, k) {
    return probs
      .map(function (p, i) { return { name: names[i], p: p }; })
      .sort(function (a, b) { return b.p - a.p; })
      .slice(0, k);
  }

  async function infer(ids) {
    var input = new ort.Tensor("int64", BigInt64Array.from(ids.map(function (x) { return BigInt(x); })), [1, vocab.max_length]);
    var out = await session.run({ tokens: input });
    return {
      sen: softmax(Array.from(out.seniority_logits.data)),
      func: softmax(Array.from(out.function_logits.data)),
    };
  }

  function renderPanel(prefix, probs, names) {
    var top = topK(probs, names, 4);
    $(prefix + "-pred").textContent = PRETTY[top[0].name] || top[0].name;
    $(prefix + "-bar").style.width = (top[0].p * 100).toFixed(1) + "%";
    $(prefix + "-conf").textContent = (top[0].p * 100).toFixed(1) + "% confident";
    $(prefix + "-alts").innerHTML = top.slice(1).map(function (a) {
      return '<div class="alt"><span>' + (PRETTY[a.name] || a.name) + "</span><span>" +
        (a.p * 100).toFixed(1) + "%</span></div>";
    }).join("");
  }

  async function classify(title) {
    if (!session || !title.trim()) { $("result-card").classList.add("hidden"); return; }
    var enc = TitleTokenizer.encode(title, vocab.token2id, vocab.max_length);
    if (!enc.tokens.length) {
      $("result-card").classList.add("hidden");
      $("status").textContent = "No recognizable words in that input — try a real job title.";
      return;
    }
    $("status").textContent = "Model loaded. Predictions run locally in your browser.";
    var r = await infer(enc.ids);
    renderPanel("sen", r.sen, vocab.seniority_names);
    renderPanel("func", r.func, vocab.function_names);
    $("tokens").innerHTML = enc.tokens.slice(0, vocab.max_length).map(function (tok) {
      var known = Object.prototype.hasOwnProperty.call(vocab.token2id, tok);
      return '<span class="tok' + (known ? "" : " unk") + '">' +
        tok.replace(/&/g, "&amp;").replace(/</g, "&lt;") + "</span>";
    }).join("");
    $("result-card").classList.remove("hidden");
  }

  /* Parity self-test: reproduce recorded PyTorch outputs through the JS
   * tokenizer + ONNX session. Argmax must match; confidence within 1e-3. */
  async function selfTest() {
    try {
      var res = await fetch("model/fixtures.json");
      var fixtures = await res.json();
      var pass = 0;
      for (var i = 0; i < fixtures.length; i++) {
        var f = fixtures[i];
        var enc = TitleTokenizer.encode(f.title, vocab.token2id, vocab.max_length);
        var r = await infer(enc.ids);
        var senTop = topK(r.sen, vocab.seniority_names, 1)[0];
        var funcTop = topK(r.func, vocab.function_names, 1)[0];
        if (senTop.name === f.seniority && funcTop.name === f.function &&
            Math.abs(senTop.p - f.seniority_confidence) < 1e-3 &&
            Math.abs(funcTop.p - f.function_confidence) < 1e-3) pass++;
      }
      var el = $("parity");
      el.textContent = "Browser parity check: " + pass + "/" + fixtures.length +
        " recorded PyTorch outputs reproduced exactly by this page (tokenizer + ONNX).";
      if (pass !== fixtures.length) el.classList.add("fail");
    } catch (e) { /* self-test is informative only */ }
  }

  async function main() {
    var examplesEl = $("examples");
    EXAMPLES.forEach(function (ex) {
      var b = document.createElement("button");
      b.className = "chip"; b.textContent = ex;
      b.addEventListener("click", function () { $("title-input").value = ex; classify(ex); });
      examplesEl.appendChild(b);
    });

    try {
      ort.env.wasm.numThreads = 1; // GitHub Pages is not cross-origin isolated
      var vres = await fetch("model/vocab.json");
      vocab = await vres.json();
      session = await ort.InferenceSession.create("model/model.onnx", { executionProviders: ["wasm"] });
      $("status").textContent = "Model loaded. Predictions run locally in your browser.";
    } catch (e) {
      $("status").textContent = "Model failed to load: " + e.message;
      return;
    }

    var t = null;
    $("title-input").addEventListener("input", function (ev) {
      clearTimeout(t);
      var v = ev.target.value;
      t = setTimeout(function () { classify(v); }, 150);
    });

    selfTest();
  }

  main();
})();
