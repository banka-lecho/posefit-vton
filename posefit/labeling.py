"""Блок E: калибровочные наборы для ручной разметки.

Пороги фильтров (что считать плоской выкладкой, что — годным кадром из
отзыва) невозможно выставить из статистики: их надо сверить с глазами.
Модуль собирает стратифицированную выборку и запекает её в один HTML-файл
с миниатюрами внутри, чтобы разметка не зависела от путей на диске.

Файл остаётся локальным. В нём кадры реальных покупателей, и публиковать
его куда бы то ни было нельзя.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image

THUMB_PX = 420
JPEG_QUALITY = 78


@dataclass(frozen=True)
class Task:
    name: str
    title: str
    hint: str
    classes: list[tuple[str, str]]  # (код, подпись)
    paired: bool  # показывать ли эталонный кадр товара рядом


GARMENT = Task(
    name="garment",
    title="Кадр товара: плоская выкладка или съёмка на модели",
    hint="Калибрует блок D. «Другое» — инфографика, макро ткани, аксессуар.",
    classes=[("flat", "плоская выкладка"), ("on_model", "на модели"), ("other", "другое")],
    paired=False,
)

WILD = Task(
    name="wild",
    title="Кадр из отзыва: годен ли в обучающую пару",
    hint="Слева кадр из отзыва, справа эталон из карточки. Сверяй расцветку.",
    classes=[
        ("good", "годен"),
        ("no_person", "нет человека / вещь не надета"),
        ("crop", "торс обрезан или закрыт"),
        ("color_mismatch", "расцветка не та"),
    ],
    paired=True,
)

VERIFY = Task(
    name="verify",
    title="Проверка пары для тестового набора",
    hint="Слева человек, справа вещь из карточки. Отмечай только бесспорно верные пары.",
    classes=[
        ("ok", "пара верная"),
        ("wrong_colour", "расцветка не та"),
        ("not_visible", "вещь не видна или не надета"),
    ],
    paired=True,
)

TASKS = {t.name: t for t in (GARMENT, WILD, VERIFY)}


def sample(pool: pd.DataFrame, n: int, seed: int, strata: str = "category_group") -> pd.DataFrame:
    """Пропорциональная выборка по группам одежды, устойчивая к перезапуску."""
    shares = pool[strata].value_counts(normalize=True)
    parts = []
    for value, share in shares.items():
        take = max(1, round(n * share))
        subset = pool[pool[strata] == value]
        parts.append(subset.sample(min(take, len(subset)), random_state=seed))
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def thumb(path: Path) -> str:
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((THUMB_PX, THUMB_PX))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_items(rows: pd.DataFrame, raw_root: Path, task: Task) -> list[dict]:
    items = []
    for row in rows.itertuples():
        item = {
            "image_id": row.image_id,
            "rel_path": row.rel_path,
            "category": row.category_group,
            "thumb": thumb(raw_root / row.rel_path),
        }
        if task.paired:
            item["ref_path"] = row.ref_path
            item["ref_thumb"] = thumb(raw_root / row.ref_path)
        items.append(item)
    return items


def batch_id(items: list[dict]) -> str:
    """Отпечаток состава партии.

    Партии одной задачи обязаны различаться именем файла, ключом в localStorage
    и именем выгружаемого CSV. Иначе вторая партия открывается под теми же
    именами, что и первая, и выгрузка молча отдаёт чужие метки — так уже
    случилось и стоило часа ручной работы.
    """
    joined = "\n".join(sorted(item["image_id"] for item in items))
    return hashlib.blake2b(joined.encode("utf-8"), digest_size=4).hexdigest()


def render(task: Task, items: list[dict]) -> str:
    batch = batch_id(items)
    payload = json.dumps(
        {"task": task.name, "batch": batch, "classes": task.classes, "items": items},
        ensure_ascii=False,
    )
    keys = " ".join(f"<kbd>{i + 1}</kbd> {label}" for i, (_, label) in enumerate(task.classes))
    return _TEMPLATE.replace("__TITLE__", task.title).replace("__HINT__", task.hint) \
                    .replace("__KEYS__", keys).replace("__BATCH__", batch) \
                    .replace("__PAYLOAD__", payload)


_TEMPLATE = r"""<!doctype html>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  :root { color-scheme: light dark; --bg:#fff; --fg:#111; --mut:#666; --line:#ddd; --acc:#2563eb; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#15171a; --fg:#eceff2; --mut:#9aa3ad; --line:#2c3036; --acc:#5b8cff; }
  }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
  header { padding:12px 20px; border-bottom:1px solid var(--line); }
  h1 { font-size:16px; margin:0 0 4px; }
  .hint, .keys { color:var(--mut); font-size:13px; }
  .batch { color:var(--mut); font-weight:400; font-size:12px;
           font-family:ui-monospace,monospace; }
  kbd { border:1px solid var(--line); border-radius:4px; padding:1px 5px;
        font:12px ui-monospace,monospace; margin-left:8px; }
  main { display:flex; gap:24px; justify-content:center; align-items:flex-start;
         padding:20px; min-height:60vh; }
  figure { margin:0; text-align:center; }
  figure img { max-height:56vh; border-radius:6px; border:1px solid var(--line); }
  figcaption { color:var(--mut); font-size:12px; margin-top:6px; }
  .controls { display:flex; gap:8px; justify-content:center; flex-wrap:wrap; padding:0 20px 16px; }
  button { font:inherit; padding:8px 14px; border-radius:6px; border:1px solid var(--line);
           background:transparent; color:var(--fg); cursor:pointer; }
  button:hover { border-color:var(--acc); }
  button.done { background:var(--acc); color:#fff; border-color:var(--acc); }
  footer { border-top:1px solid var(--line); padding:12px 20px; display:flex;
           gap:16px; align-items:center; flex-wrap:wrap; }
  .bar { flex:1; height:6px; background:var(--line); border-radius:3px; overflow:hidden; min-width:160px; }
  .bar > i { display:block; height:100%; background:var(--acc); width:0; }
  .path { color:var(--mut); font-size:11px; font-family:ui-monospace,monospace;
          word-break:break-all; max-width:60ch; margin:0 auto; }
</style>
<header>
  <h1>__TITLE__ <span class="batch">партия __BATCH__</span></h1>
  <div class="hint">__HINT__</div>
  <div class="keys">__KEYS__ &nbsp;·&nbsp; <kbd>←</kbd> назад <kbd>→</kbd> вперёд</div>
</header>
<main id="stage"></main>
<div class="controls" id="controls"></div>
<div class="path" id="path"></div>
<footer>
  <span id="count"></span>
  <span class="bar"><i id="fill"></i></span>
  <button id="save">Скачать CSV</button>
</footer>
<script id="data" type="application/json">__PAYLOAD__</script>
<script>
const DATA = JSON.parse(document.getElementById("data").textContent);
// Ключ включает отпечаток партии: метки разных партий не смешиваются.
const KEY = "posefit-labels-" + DATA.task + "-" + DATA.batch;
let labels = {};
try { labels = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { labels = {}; }
let i = 0;

// Первый неразмеченный кадр: перезагрузка страницы не теряет прогресс.
while (i < DATA.items.length && labels[DATA.items[i].image_id]) i++;
if (i >= DATA.items.length) i = 0;

const stage = document.getElementById("stage");
const controls = document.getElementById("controls");

DATA.classes.forEach(([code, label], n) => {
  const b = document.createElement("button");
  b.textContent = (n + 1) + ". " + label;
  b.onclick = () => mark(code);
  controls.appendChild(b);
});

function persist() {
  try { localStorage.setItem(KEY, JSON.stringify(labels)); } catch (e) {}
}

function mark(code) {
  labels[DATA.items[i].image_id] = code;
  persist();
  if (i < DATA.items.length - 1) i++;
  render();
}

function render() {
  const it = DATA.items[i];
  stage.innerHTML = "";
  const add = (b64, cap) => {
    const fig = document.createElement("figure");
    const img = document.createElement("img");
    img.src = "data:image/jpeg;base64," + b64;
    const c = document.createElement("figcaption");
    c.textContent = cap;
    fig.append(img, c);
    stage.appendChild(fig);
  };
  add(it.thumb, "кадр для разметки");
  if (it.ref_thumb) add(it.ref_thumb, "эталон из карточки");

  document.getElementById("path").textContent = it.rel_path;
  const done = Object.keys(labels).length;
  document.getElementById("count").textContent =
    `${i + 1} / ${DATA.items.length} · размечено ${done} · ${it.category}`;
  document.getElementById("fill").style.width =
    (100 * done / DATA.items.length).toFixed(1) + "%";
  [...controls.children].forEach((b, n) => {
    b.classList.toggle("done", labels[it.image_id] === DATA.classes[n][0]);
  });
}

document.addEventListener("keydown", (e) => {
  if (e.key === "ArrowRight") { i = Math.min(i + 1, DATA.items.length - 1); render(); }
  else if (e.key === "ArrowLeft") { i = Math.max(i - 1, 0); render(); }
  else {
    const n = parseInt(e.key, 10);
    if (n >= 1 && n <= DATA.classes.length) mark(DATA.classes[n - 1][0]);
  }
});

document.getElementById("save").onclick = () => {
  const rows = [["image_id", "rel_path", "category_group", "label", "batch"]];
  DATA.items.forEach((it) => {
    if (labels[it.image_id]) {
      rows.push([it.image_id, it.rel_path, it.category, labels[it.image_id], DATA.batch]);
    }
  });
  if (rows.length === 1) {
    alert("Ничего не размечено — выгружать нечего.");
    return;
  }
  const csv = rows.map((r) => r.map((v) => `"${String(v).replace(/"/g, '""')}"`).join(",")).join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
  a.download = "labels_" + DATA.task + "_" + DATA.batch + ".csv";
  a.click();
};

render();
</script>
"""
