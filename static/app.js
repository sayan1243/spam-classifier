const form = document.querySelector("#check-form");
const messageInput = document.querySelector("#message-input");
const characterCount = document.querySelector("#character-count");
const clearButton = document.querySelector("#clear-button");
const submitButton = document.querySelector("#submit-button");
const buttonLabel = submitButton.querySelector(".button-label");
const result = document.querySelector("#result");
const resultIcon = document.querySelector("#result-icon");
const resultKicker = document.querySelector("#result-kicker");
const resultTitle = document.querySelector("#result-title");
const resultDescription = document.querySelector("#result-description");
const dismissButton = document.querySelector("#dismiss-button");

function updateInputState() {
  characterCount.textContent = `${messageInput.value.length.toLocaleString()} / 5,000`;
  clearButton.disabled = messageInput.value.length === 0;
}

function showResult(type, kicker, title, description, icon) {
  result.className = `result ${type}`;
  resultIcon.textContent = icon;
  resultKicker.textContent = kicker;
  resultTitle.textContent = title;
  resultDescription.textContent = description;
  result.hidden = false;
}

function setLoading(isLoading) {
  submitButton.disabled = isLoading;
  buttonLabel.textContent = isLoading ? "Checking message…" : "Check message";
}

messageInput.addEventListener("input", updateInputState);

clearButton.addEventListener("click", () => {
  messageInput.value = "";
  updateInputState();
  result.hidden = true;
  messageInput.focus();
});

dismissButton.addEventListener("click", () => {
  result.hidden = true;
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = messageInput.value.trim();
  if (!message) {
    messageInput.focus();
    return;
  }

  result.hidden = true;
  setLoading(true);

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    const data = await response.json();

    if (!response.ok) {
      const detail = Array.isArray(data.detail)
        ? data.detail.map((issue) => issue.msg).join(" ")
        : data.detail;
      throw new Error(detail || "The message could not be checked. Please try again.");
    }

    if (data.is_spam) {
      showResult(
        "spam",
        "Classification",
        "This looks like spam",
        "Be cautious before replying, clicking links, or sharing personal information.",
        "!"
      );
    } else {
      showResult(
        "ham",
        "Classification",
        "This looks like a normal message",
        "No common spam pattern was detected. Still use your judgment with unexpected messages.",
        "✓"
      );
    }
  } catch (error) {
    showResult(
      "error",
      "Something went wrong",
      "We couldn’t check that message",
      error instanceof Error ? error.message : "Please try again in a moment.",
      "!"
    );
  } finally {
    setLoading(false);
  }
});

messageInput.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    form.requestSubmit();
  }
});
