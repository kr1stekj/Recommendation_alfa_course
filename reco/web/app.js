/* Диалог живёт в памяти вкладки. В localStorage хранится только случайный ID гостя. */
"use strict";
const $ = id => document.getElementById(id);
const kinds = {course: "Курс", article: "Статья", document: "Документ / шаблон", podcast: "Подкаст"};
let profileId, clarifications = [], skipQuestions = false, result, selected, busy = false, task = "";
let eventQueue = Promise.resolve();
let demoSession = null, sessionReady = false;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

async function api(path, data, method = "POST") {
  const response = await fetch(path, {method, credentials: "same-origin", headers: {"Content-Type": "application/json"}, body: data === undefined ? undefined : JSON.stringify(data)});
  if (response.status === 401) {
    demoSession = null; sessionReady = false; profileId = undefined;
    clearProfileContext(); renderAccount();
    const error = new Error("Демосессия завершилась. Войдите снова или продолжите как гость.");
    error.status = 401;
    $("summary").textContent = "Демовход завершён. Повторите поиск как гость или выберите тестовый профиль.";
    $("historyStatus").textContent = "Гостевой режим · повторите поиск";
    showError(error);
    throw error;
  }
  if (response.status === 403) throw new Error("Действие недоступно этому профилю. Обновите страницу и повторите вход.");
  if (response.status === 429) throw new Error("Сейчас слишком много демосессий. Попробуйте позже.");
  if (!response.ok) throw new Error(response.status === 400 ? "Проверьте запрос и выбранные ответы." : "Не удалось получить ответ. Проверьте подключение к сервису и повторите поиск.");
  return response.json();
}

async function ensureProfile() {
  if (!sessionReady) {
    const session = await api("/demo/session", undefined, "GET");
    demoSession = session.authenticated ? session : null;
    if (demoSession) $("smartMode").checked = true;
    sessionReady = true; renderAccount();
  }
  if (demoSession) { profileId = demoSession.profile_id; return; }
  try { profileId = localStorage.getItem("alfa-smart-profile"); } catch (_) { /* Работаем без сохранения между вкладками. */ }
  if (!/^anon:[A-Za-z0-9_-]{1,100}$/.test(profileId || "") || profileId.startsWith("anon:demo_")) {
    profileId = (await api("/profiles", {})).profile_id;
    try { localStorage.setItem("alfa-smart-profile", profileId); } catch (_) {}
  }
}

function renderAccount() {
  const account = demoSession?.account;
  $("loginButton").textContent = account ? account.title : "Войти";
  $("loginButton").setAttribute("aria-label", account ? `Сменить демопрофиль: ${account.title}` : "Войти в демопрофиль");
  $("logoutButton").hidden = !account;
  $("profileNote").textContent = account ? `Синтетический профиль · ${account.title}. Стартовых действий: ${account.seed_items}.` : "Гостевой профиль";
  $("resetHistory").textContent = account ? "Сбросить демопрофиль" : "Удалить мою историю";
  $("privacyNote").textContent = account ? "Тестовая история изолирована от реальных пользователей. При выходе эта копия удаляется." :
    "Без регистрации. Запрос не сохраняется в профиле; учитываются ваши действия с материалами.";
}

function clearProfileContext() {
  clarifications = []; skipQuestions = false; task = ""; selected = undefined; result = undefined;
  $("searchInput").value = ""; $("industry").value = ""; $("experience").value = "";
  $("questionPanel").hidden = true; $("llmStatus").hidden = true;
  $("results").replaceChildren();
  $("contextList").replaceChildren(element("p", "", "Контекст уточняется по необходимости"));
  $("historyStatus").textContent = "Загружаем контекст…";
  $("summary").textContent = "Готовим подборку…"; $("resultCount").textContent = "";
  $("emptyState").hidden = true;
  if ($("materialDialog").open) $("materialDialog").close();
}

async function changeAccount(path, body = {}) {
  if (busy) return;
  setBusy(true); $("loginError").hidden = true;
  try {
    // Дожидаемся событий прежнего профиля до смены cookie, не переносим их новому.
    await eventQueue.catch(() => {});
    const session = await api(path, body);
    demoSession = session.authenticated ? session : null; sessionReady = true;
    profileId = demoSession?.profile_id;
    clearProfileContext(); renderAccount();
    if (demoSession) $("smartMode").checked = true;
    await ensureProfile();
    if ($("loginDialog").open) $("loginDialog").close();
  } catch (error) {
    if ($("loginDialog").open) { $("loginError").textContent = error.message; $("loginError").hidden = false; }
    else showError(error);
    setBusy(false); return;
  }
  setBusy(false); await search();
}

$("loginButton").addEventListener("click", async () => {
  if (busy) return;
  setBusy(true);
  try {
    const data = await api("/demo/accounts", undefined, "GET");
    $("demoAccounts").replaceChildren(...data.accounts.map(account => {
      const button = element("button", "demo-account"); button.type = "button";
      button.append(element("strong", "", account.title), element("span", "", account.description),
        element("span", "small muted", account.seed_items ? `Синтетических действий: ${account.seed_items}` : "Без стартовой истории"));
      if (demoSession?.account.id === account.id) button.classList.add("current");
      button.setAttribute("aria-label", `Войти: ${account.title}`);
      button.addEventListener("click", () => changeAccount("/demo/login", {account_id: account.id}));
      return button;
    }));
    $("loginError").hidden = true; $("loginDialog").showModal();
  } catch (error) { showError(error); }
  finally { setBusy(false); }
});
$("logoutButton").addEventListener("click", () => changeAccount("/demo/logout"));

function setBusy(value) {
  busy = value;
  $("resultsSection").setAttribute("aria-busy", String(value));
  document.querySelectorAll("button, #smartMode").forEach(node => { node.disabled = value; });
  $("searchButton").textContent = value ? "Ищем…" : "Найти";
}

function showError(error) { $("error").textContent = error.message; $("error").hidden = false; }

async function search() {
  if (busy) return;
  setBusy(true); $("error").hidden = true;
  try {
    if (!profileId) await ensureProfile();
    result = await api("/search", {task, profile_id: profileId, clarifications, skip_questions: skipQuestions, search_mode: $("smartMode").checked ? "smart" : "normal", industry: $("industry").value.trim(), experience: $("experience").value.trim()});
    render();
    return true;
  } catch (error) {
    showError(error);
    if (!result) { $("results").replaceChildren(); $("summary").textContent = error.status === 401 ?
      "Демовход завершён. Повторите поиск как гость или выберите тестовый профиль." :
      "Каталог пока недоступен. После запуска сервиса повторите поиск."; }
    return false;
  } finally { setBusy(false); }
}

function render() {
  const smart = result.search_mode === "smart";
  const feed = Boolean(result.feed_mode);
  $("summary").textContent = result.summary;
  $("modeHint").textContent = smart ? "LLM уточнит запрос, если контекста недостаточно" : "Поиск по словам в каталоге";
  $("contextInputs").hidden = !smart;
  $("resultsLabel").textContent = !smart ? "ОБЫЧНЫЙ ПОИСК" : result.match_status === "broad" ? "УМНЫЙ ПОИСК" : result.match_status === "starter" ? "С ЧЕГО НАЧАТЬ" : result.match_status === "no_match" ? "АЛЬТЕРНАТИВЫ ИЗ КАТАЛОГА" : "ВАШ СЛЕДУЮЩИЙ ШАГ";
  $("resultsTitle").textContent = !smart ? "Результаты поиска" : result.match_status === "broad" ? "Предварительная подборка" : result.match_status === "no_match" ? "Попробуем посмотреть шире" : result.match_status === "starter" ? "С чего начать" : "Подборка для вас";
  $("resultCount").textContent = smart ? `${result.items.length} материалов в подборке` : `Найдено: ${result.total_matches}. Показано: ${result.items.length}`;
  $("historyStatus").textContent = !smart ? "История учитывается в умном режиме" : result.live_items || result.history_items ? `Учтено материалов: ${result.live_items + result.history_items}` : demoSession ? "Демопрофиль · без истории" : "Новый гость · без истории";
  if (feed) {
    $("resultsLabel").textContent = "РЕКОМЕНДАЦИИ";
    $("resultsTitle").textContent = result.live_items || result.history_items ? "Подборка для вас" : "С чего начать";
    $("resultCount").textContent = `${result.items.length} материалов в подборке`;
    $("historyStatus").textContent = result.live_items || result.history_items ?
      `Учтено материалов: ${result.live_items + result.history_items}` :
      demoSession ? "Демопрофиль · без истории" : "Новый гость · без истории";
  }
  $("refineButton").hidden = !smart || Boolean(result.question) || (!clarifications.length && !skipQuestions);
  $("refineButton").textContent = skipQuestions ? "Уточнить запрос" : "Сбросить уточнения";
  $("emptyState").hidden = result.items.length > 0;
  $("contextList").replaceChildren();
  for (const answer of result.dialogue || []) {
    const block = element("p"); block.append(element("span", "context-label", answer.question), element("strong", "", answer.answer)); $("contextList").append(block);
  }
  if (!(result.dialogue || []).length) $("contextList").append(element("p", "", feed ?
    "Темы ваших интересов и дополнительные материалы для знакомства" :
    smart ? "Контекст уточняется по необходимости" : "Поиск без персонализации и диалога"));
  const question = result.question;
  $("questionPanel").hidden = !question;
  $("questionOptions").replaceChildren();
  if (question) {
    $("questionStep").textContent = question.step === 1 ? "УТОЧНИМ ЗАДАЧУ" : "ЕЩЁ ОДНО УТОЧНЕНИЕ";
    $("questionTitle").textContent = question.title;
    question.options.forEach((option, index) => {
      const button = element("button", "option"); button.type = "button";
      button.append(element("span", "option-index", String(index + 1).padStart(2, "0")), element("span", "", option.label));
      button.addEventListener("click", async () => {
        const previous = clarifications;
        clarifications = [...previous, option.value];
        if (!await search()) clarifications = previous;
      });
      $("questionOptions").append(button);
    });
  }
  const status = result.assistant_status;
  $("llmStatus").hidden = !smart || Boolean(question) || ["idle", "skipped", "limit_reached", "not_requested", "not_needed", "already_answered"].includes(status);
  $("llmStatus").textContent = status === "ok" ? "Контекста достаточно — LLM не потребовался уточняющий вопрос." :
    status === "topic_mismatch" ? "Не удалось получить уточнение по теме запроса. Показываем подборку без дополнительного вопроса." :
    status === "disabled" ? "LLM не подключена. Уточняющие вопросы недоступны; показываем локальную подборку по смыслу." :
    "LLM сейчас недоступна для уточнения запроса. Показываем локальную подборку без дополнительных вопросов.";
  $("results").replaceChildren(...result.items.map(card));
}

function price(item) {
  if (item.access === "subscription") return "По подписке";
  if (item.access === "paid") return "Условия на платформе";
  if (item.price > 0) return `${item.price.toLocaleString("ru-RU")} ₽ в каталоге`;
  if (item.kind === "course" && item.price === 0 && item.access === "public") return "Бесплатно в каталоге";
  return "";
}

function metadata(item) {
  const source = {personal: "По вашим интересам", starter: "Для начала", popular: "Популярное", discovery: "Для знакомства"}[item.feed_source];
  return [source, kinds[item.kind] || "Материал", item.duration, item.lesson_quantity > 0 ? `${item.lesson_quantity} уроков` : ""].filter(Boolean).join(" · ");
}

function card(item, index) {
  const node = element("article", "card");
  const band = element("div", "card-band"); band.append(element("span", "pill", item.topics.join(" · ") || "Для вашего бизнеса"), element("span", "card-number", String(index + 1).padStart(2, "0")));
  const body = element("div", "card-body");
  if (item.continuation) body.append(element("span", "continuation", "↩ МОЖНО ПРОДОЛЖИТЬ"));
  body.append(element("div", "kind-meta", metadata(item)), element("h3", "", item.title));
  if (item.description) body.append(element("p", "card-description", item.description));
  const reason = element("p", "card-reason"); reason.append(element("strong", "", "Почему этот материал"), document.createTextNode(item.explanation)); body.append(reason);
  const footer = element("div", "card-footer"); const open = element("button", "open-card", item.continuation ? "Продолжить ↗" : "Посмотреть ↗");
  open.addEventListener("click", () => openMaterial(item)); footer.append(open, element("span", "card-price", price(item))); body.append(footer);
  if (item.next_material) {
    const next = element("div", "next-material"); const link = element("button", "", item.next_material.title);
    link.addEventListener("click", () => openMaterial(result.items.find(row => row.id === item.next_material.id)));
    next.append(element("span", "", "Далее →"), link); body.append(next);
  }
  node.append(band, body); return node;
}

async function recordEvent(item, type) {
  const eventId = globalThis.crypto?.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
  const event = {profile_id: profileId, item_id: item.id, event_type: type, event_id: eventId};
  // Поздний ответ click не должен затереть последующую отрицательную отметку.
  eventQueue = eventQueue.catch(() => {}).then(() => api("/events", event));
  return eventQueue;
}

function openMaterial(item) {
  if (!item || busy) return;
  selected = item;
  $("materialKind").textContent = kinds[item.kind]; $("materialTitle").textContent = item.title;
  $("materialDescription").textContent = item.description || "В каталоге нет аннотации. Подробности — на странице материала.";
  $("materialMeta").textContent = [metadata(item), price(item)].filter(Boolean).join(" · ");
  $("materialReason").textContent = item.explanation;
  $("materialTarget").textContent = item.target && item.target.length > 10 ? `Для кого: ${item.target}` : "";
  $("materialId").textContent = `ID: ${item.id}`;
  const safeUrl = item.material_url && item.material_url.startsWith("https://");
  $("materialLink").hidden = !safeUrl;
  if (safeUrl) $("materialLink").href = item.material_url; else $("materialLink").removeAttribute("href");
  $("eventStatus").textContent = "";
  if (!$("materialDialog").open) $("materialDialog").showModal();
  recordEvent(item, "click").then(() => { $("historyStatus").textContent = "История обновлена · учтём в следующем поиске"; }).catch(() => { $("eventStatus").textContent = "Материал открыт, но действие не сохранилось. Попробуйте ещё раз."; });
}

$("searchForm").addEventListener("submit", event => { event.preventDefault(); if (busy) return; task = $("searchInput").value.trim(); clarifications = []; skipQuestions = false; search(); });
document.querySelectorAll("[data-query]").forEach(button => button.addEventListener("click", () => { $("searchInput").value = button.dataset.query; $("searchForm").requestSubmit(); }));
$("skipQuestions").addEventListener("click", () => { skipQuestions = true; search(); });
$("refineButton").addEventListener("click", () => { clarifications = []; skipQuestions = false; search(); });
$("applyContext").addEventListener("click", () => { clarifications = []; skipQuestions = false; search(); });
$("smartMode").addEventListener("change", () => { task = $("searchInput").value.trim(); clarifications = []; skipQuestions = false; $("questionPanel").hidden = true; $("llmStatus").hidden = true; search(); });
$("aboutButton").addEventListener("click", () => $("aboutDialog").showModal());
document.querySelectorAll("[data-close]").forEach(button => button.addEventListener("click", () => $(button.dataset.close).close()));
document.querySelectorAll("dialog").forEach(dialog => dialog.addEventListener("click", event => { if (event.target === dialog) { const rect = dialog.getBoundingClientRect(); if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close(); } }));
document.querySelectorAll("[data-event]").forEach(button => button.addEventListener("click", async () => {
  if (!selected || busy) return;
  setBusy(true);
  try { await recordEvent(selected, button.dataset.event); $("materialDialog").close(); }
  catch (error) { $("eventStatus").textContent = "Не удалось сохранить отметку. Попробуйте ещё раз."; setBusy(false); return; }
  setBusy(false); await search();
}));
$("resetHistory").addEventListener("click", async () => {
  if (busy) return;
  if (demoSession) { await changeAccount("/demo/reset"); return; }
  setBusy(true);
  try { await eventQueue.catch(() => {}); if (profileId) await api(`/profiles/${encodeURIComponent(profileId)}`, undefined, "DELETE"); profileId = undefined; try { localStorage.removeItem("alfa-smart-profile"); } catch (_) {} clarifications = []; skipQuestions = false; await ensureProfile(); }
  catch (error) { showError(error); setBusy(false); return; }
  setBusy(false); await search();
});
$("results").append(element("div", "skeleton"), element("div", "skeleton"));
search();
