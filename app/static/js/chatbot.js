/*
 * Chat page behaviour (chat.html).
 *
 *  - sends messages to POST /api/chat and appends the replies
 *  - shows a typing indicator while waiting, and a retry button if a request fails
 *  - turns the assistant's light markdown (**bold**, "- " lists) into safe HTML
 *  - switching the "Discussing" dropdown reloads the page for that conversation
 *
 * Security note: assistant text is HTML-escaped BEFORE any formatting is
 * applied, so the model's output can never inject markup into the page.
 */
(function () {
  "use strict";

  var app = document.getElementById("chat-app");
  if (!app) return;

  var messagesEl = document.getElementById("chat-messages");
  var emptyEl = document.getElementById("chat-empty");
  var inputEl = document.getElementById("chat-input");
  var sendBtn = document.getElementById("chat-send");
  var counterEl = document.getElementById("char-count");
  var clearBtn = document.getElementById("clear-btn");
  var contextSelect = document.getElementById("context-select");

  var predictionId = app.getAttribute("data-prediction-id") || "";
  var maxChars = parseInt(app.getAttribute("data-max-chars"), 10) || 1000;
  var REQUEST_TIMEOUT_MS = 70000;
  var busy = false;

  // ---------------------------------------------------------------- formatting

  function escapeHtml(text) {
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function inlineFormat(escaped) {
    return escaped
      .replace(/\*\*([^*]+?)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[\s(])\*([^*\s][^*]*?)\*(?=$|[\s).,;:!?])/g, "$1<em>$2</em>");
  }

  // Minimal markdown: paragraphs, "- " / "* " bullet lists, "1." numbered lists.
  function formatMessage(text) {
    var lines = escapeHtml(text.replace(/\r\n/g, "\n")).split("\n");
    var html = [];
    var paragraph = [];
    var listType = null; // "ul" | "ol"
    var listItems = [];

    function flushParagraph() {
      if (paragraph.length) {
        html.push("<p>" + inlineFormat(paragraph.join("<br>")) + "</p>");
        paragraph = [];
      }
    }
    function flushList() {
      if (listItems.length) {
        html.push(
          "<" + listType + ">" +
          listItems.map(function (item) { return "<li>" + inlineFormat(item) + "</li>"; }).join("") +
          "</" + listType + ">"
        );
        listItems = [];
        listType = null;
      }
    }

    lines.forEach(function (raw) {
      var line = raw.trim();
      var bullet = line.match(/^[-*\u2022]\s+(.*)$/);
      var numbered = line.match(/^\d+[.)]\s+(.*)$/);
      var heading = line.match(/^#{1,6}\s+(.*)$/);

      if (!line) {
        flushParagraph();
        flushList();
      } else if (bullet || numbered) {
        flushParagraph();
        var type = bullet ? "ul" : "ol";
        if (listType && listType !== type) flushList();
        listType = type;
        listItems.push((bullet || numbered)[1]);
      } else if (heading) {
        flushParagraph();
        flushList();
        html.push("<p><strong>" + inlineFormat(heading[1]) + "</strong></p>");
      } else {
        flushList();
        paragraph.push(line);
      }
    });
    flushParagraph();
    flushList();
    return html.join("");
  }

  // ------------------------------------------------------------------ DOM helpers

  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function hideEmptyState() {
    if (emptyEl) emptyEl.hidden = true;
    if (clearBtn) clearBtn.hidden = false;
  }

  function addMessage(role, text) {
    hideEmptyState();
    var row = document.createElement("div");
    row.className = "chat-msg " + role;
    var bubble = document.createElement("div");
    bubble.className = "chat-bubble";
    bubble.setAttribute("data-role", role);
    if (role === "assistant") {
      bubble.innerHTML = formatMessage(text);
      bubble.classList.add("is-formatted");
    } else {
      bubble.textContent = text;
    }
    row.appendChild(bubble);
    messagesEl.appendChild(row);
    scrollToBottom();
    return row;
  }

  function addTyping() {
    var row = document.createElement("div");
    row.className = "chat-msg assistant chat-typing";
    row.innerHTML =
      '<div class="chat-bubble" aria-label="The assistant is typing">' +
      '<span class="dot"></span><span class="dot"></span><span class="dot"></span></div>';
    messagesEl.appendChild(row);
    scrollToBottom();
    return row;
  }

  function addError(message, onRetry) {
    var row = document.createElement("div");
    row.className = "chat-msg chat-error";
    var box = document.createElement("div");
    box.className = "chat-error-box";
    var text = document.createElement("span");
    text.textContent = message;
    box.appendChild(text);
    if (onRetry) {
      var retry = document.createElement("button");
      retry.type = "button";
      retry.className = "chat-retry";
      retry.textContent = "Try again";
      retry.addEventListener("click", function () {
        row.remove();
        onRetry();
      });
      box.appendChild(retry);
    }
    row.appendChild(box);
    messagesEl.appendChild(row);
    scrollToBottom();
  }

  function setBusy(state) {
    busy = state;
    sendBtn.disabled = state;
    inputEl.disabled = state;
    if (!state) inputEl.focus();
  }

  // --------------------------------------------------------------------- sending

  function request(text) {
    setBusy(true);
    var typing = addTyping();

    var controller = new AbortController();
    var timer = setTimeout(function () { controller.abort(); }, REQUEST_TIMEOUT_MS);

    fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        prediction_id: predictionId ? parseInt(predictionId, 10) : null
      }),
      signal: controller.signal
    })
      .then(function (res) {
        return res.json().catch(function () { return {}; }).then(function (data) {
          return { status: res.status, ok: res.ok, data: data };
        });
      })
      .then(function (r) {
        typing.remove();
        if (r.status === 401) {
          window.location.href = "/login";
          return;
        }
        if (r.ok && r.data.reply) {
          addMessage("assistant", r.data.reply);
        } else {
          var retryable = r.status !== 400 && r.status !== 404;
          addError(
            r.data.error || "Something went wrong. Please try again.",
            retryable ? function () { request(text); } : null
          );
        }
      })
      .catch(function (err) {
        typing.remove();
        var message = err && err.name === "AbortError"
          ? "That took too long. Please try again."
          : "Couldn't reach the server. Check your connection and try again.";
        addError(message, function () { request(text); });
      })
      .then(function () {
        clearTimeout(timer);
        setBusy(false);
      });
  }

  function send(text) {
    text = (text || "").trim();
    if (!text || busy) return;
    if (text.length > maxChars) {
      addError("Please keep messages under " + maxChars + " characters.", null);
      return;
    }
    addMessage("user", text);
    inputEl.value = "";
    autoGrow();
    updateCounter();
    request(text);
  }

  // ------------------------------------------------------------------ input box

  function autoGrow() {
    inputEl.style.height = "auto";
    inputEl.style.height = Math.min(inputEl.scrollHeight, 140) + "px";
  }

  function updateCounter() {
    var length = inputEl.value.length;
    if (length >= maxChars * 0.8) {
      counterEl.hidden = false;
      counterEl.textContent = length + " / " + maxChars;
      counterEl.classList.toggle("is-near-limit", length >= maxChars * 0.95);
    } else {
      counterEl.hidden = true;
    }
  }

  inputEl.addEventListener("input", function () {
    autoGrow();
    updateCounter();
  });

  inputEl.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      send(inputEl.value);
    }
  });

  sendBtn.addEventListener("click", function () { send(inputEl.value); });

  document.querySelectorAll(".chat-suggestion").forEach(function (button) {
    button.addEventListener("click", function () {
      send(button.getAttribute("data-prompt"));
    });
  });

  // ------------------------------------------------------- toolbar: switch / clear

  if (contextSelect) {
    contextSelect.addEventListener("change", function () {
      window.location.href = "/chat" + (contextSelect.value ? "?prediction_id=" + contextSelect.value : "");
    });
  }

  if (clearBtn) {
    clearBtn.addEventListener("click", function () {
      if (!window.confirm("Clear this conversation? This can't be undone.")) return;
      fetch("/api/chat/clear", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prediction_id: predictionId ? parseInt(predictionId, 10) : null })
      })
        .then(function (res) {
          if (res.status === 401) { window.location.href = "/login"; return; }
          if (!res.ok) throw new Error("clear failed");
          window.location.reload();
        })
        .catch(function () {
          addError("Couldn't clear the conversation. Please try again.", null);
        });
    });
  }

  // ------------------------------------------------------------------- start up

  // Messages rendered by the server arrive as plain text; format the assistant's.
  document.querySelectorAll('.chat-bubble[data-role="assistant"]').forEach(function (bubble) {
    bubble.innerHTML = formatMessage(bubble.textContent);
    bubble.classList.add("is-formatted");
  });

  scrollToBottom();
  inputEl.focus();
})();
