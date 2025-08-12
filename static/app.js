
const log = document.getElementById("log");
const input = document.getElementById("msg");
const sendBtn = document.getElementById("send");

function append(kind, text) {
  const div = document.createElement("div");
  div.className = "row " + kind;
  div.textContent = text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

async function send() {
  const message = input.value.trim();
  if (!message) return;
  append("user", message);
  input.value = "";

  try {
    const r = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    if (!r.ok) {
      const t = await r.text();
      append("error", "Server error: " + r.status + " " + t);
      return;
    }
    const j = await r.json();
    append("bot", j.reply ?? "(no reply)");
  } catch (err) {
    append("error", "Network error: " + err.message);
  }
}

sendBtn.addEventListener("click", send);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});
