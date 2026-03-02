// ===============================
// Helpers
// ===============================

async function apiPost(url, data = {}) {
    const resp = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
    });

    if (!resp.ok) {
        let message = "Ошибка запроса";
        const contentType = (resp.headers.get("content-type") || "").toLowerCase();

        try {
            if (contentType.includes("application/json")) {
                const payload = await resp.json();
                message = payload.error || payload.message || JSON.stringify(payload);
            } else {
                const text = await resp.text();
                if (text) {
                    try {
                        const payload = JSON.parse(text);
                        message = payload.error || payload.message || text;
                    } catch {
                        message = text;
                    }
                }
            }
        } catch {
            message = "Ошибка запроса";
        }

        throw new Error(message);
    }

    return await resp.json();
}

function qs(sel) {
    return document.querySelector(sel);
}

function qsa(sel) {
    return document.querySelectorAll(sel);
}


// ===============================
// Token interaction (annotation)
// ===============================

document.addEventListener("click", async (e) => {

    const token = e.target.closest(".token");
    if (!token) return;

    const sentId = token.dataset.sentId;
    const tokenIdx = token.dataset.tokenIdx;
    const currentLabel = token.dataset.label || "";
    const confidence = token.dataset.confidence || "";

    const newLabel = prompt(
        `Изменить метку токена:\nТекущая: ${currentLabel}\nУверенность: ${confidence}`,
        currentLabel
    );

    if (newLabel === null) return;

    try {
        const res = await apiPost("/review/update_token", {
            sent_id: sentId,
            token_idx: tokenIdx,
            label: newLabel
        });

        token.dataset.label = newLabel;
        token.querySelector(".label").textContent = newLabel;

        token.classList.remove("conflict", "low-confidence");
        token.classList.add("confirmed");

    } catch (err) {
        alert("Ошибка обновления: " + err.message);
    }
});


// ===============================
// Review queue actions
// ===============================

document.addEventListener("click", async (e) => {

    const btn = e.target.closest("[data-queue-action]");
    if (!btn) return;

    const action = btn.dataset.queueAction;
    const itemId = btn.dataset.itemId;

    try {
        await apiPost("/review/queue_action", {
            action: action,
            item_id: itemId
        });

        const row = btn.closest(".review-item");
        if (row) row.remove();

    } catch (err) {
        alert("Ошибка действия: " + err.message);
    }
});


// ===============================
// Simple tab switching (review page)
// ===============================

document.addEventListener("click", (e) => {

    const tab = e.target.closest("[data-tab]");
    if (!tab) return;

    const name = tab.dataset.tab;

    qsa(".tab-content").forEach(el => {
        el.style.display = el.dataset.tab === name ? "block" : "none";
    });

    qsa("[data-tab]").forEach(t => t.classList.remove("active"));
    tab.classList.add("active");
});


// ===============================
// Toggle raw text blocks
// ===============================

document.addEventListener("click", (e) => {

    const btn = e.target.closest("[data-toggle-raw]");
    if (!btn) return;

    const targetId = btn.dataset.toggleRaw;
    const el = document.getElementById(targetId);

    if (!el) return;

    el.style.display = el.style.display === "none" ? "block" : "none";
});


// ===============================
// Utility: flash messages
// ===============================

function showFlash(message, type = "success") {
    const flash = document.createElement("div");
    flash.className = `flash ${type}`;
    flash.textContent = message;

    const container = qs(".container");
    if (container) container.prepend(flash);

    setTimeout(() => flash.remove(), 3000);
}
