"""Статическая проверка кода, который нельзя импортировать без GPU.

scripts/train.py и scripts/evaluate.py тянут torch, и на машине без него их
тесты пропускаются целиком. Опечатка в имени или забытый импорт там доезжали
до сервера и всплывали только при запуске — так evaluate.py однажды упал на
NameError: load_component вызывался, но не был импортирован.

pyflakes разбирает код без исполнения и находит такие ошибки за доли секунды.
"""

from pathlib import Path

import pytest

pyflakes_api = pytest.importorskip("pyflakes.api")
from pyflakes import reporter as pyflakes_reporter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TARGETS = sorted([*(ROOT / "posefit").glob("*.py"), *(ROOT / "scripts").glob("*.py")])


class _Collector(pyflakes_reporter.Reporter):
    def __init__(self):
        self.messages: list[str] = []

    def unexpectedError(self, filename, message):
        self.messages.append(f"{filename}: {message}")

    def syntaxError(self, filename, message, lineno, offset, text):
        self.messages.append(f"{filename}:{lineno}: синтаксис: {message}")

    def flake(self, message):
        self.messages.append(str(message))


@pytest.mark.parametrize("path", TARGETS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_no_undefined_names_or_syntax_errors(path):
    collector = _Collector()
    pyflakes_api.checkPath(str(path), collector)
    # Неиспользуемый импорт — косметика; неопределённое имя — падение на сервере.
    fatal = [m for m in collector.messages
             if "undefined name" in m or "синтаксис" in m or "unexpected" in m.lower()]
    assert not fatal, "\n".join(fatal)


def _function(tree, name):
    import ast

    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def test_evaluation_never_tracks_gradients():
    """Регрессия: декоратор no_grad однажды уехал с generate на соседнюю
    функцию при правке, и 50 шагов диффузии копили граф градиентов —
    замер падал по памяти на первом батче. Код при этом был валиден,
    и pyflakes его пропускал.
    """
    import ast

    # generate и загрузка модели живут в posefit/generation.py: ими пользуются
    # и замер, и ноутбук со своими примерами.
    generation = ast.parse((ROOT / "posefit" / "generation.py").read_text(encoding="utf-8"))
    decorators = [ast.unparse(d) for d in _function(generation, "generate").decorator_list]
    assert "torch.no_grad()" in decorators, f"generate без no_grad: {decorators}"

    loader_src = ast.unparse(_function(generation, "load_models"))
    # freeze_except_self_attention размораживает слои внимания — в выводе ей не место.
    assert "freeze_except_self_attention" not in loader_src
    assert "unet.requires_grad_(False)" in loader_src

    tree = ast.parse((ROOT / "scripts" / "evaluate.py").read_text(encoding="utf-8"))
    main_src = ast.unparse(_function(tree, "main"))
    assert "torch.set_grad_enabled(False)" in main_src
    assert "freeze_except_self_attention" not in main_src
