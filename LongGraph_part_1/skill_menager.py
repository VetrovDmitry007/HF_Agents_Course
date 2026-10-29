"""
Универсальный менеджер Skills для AI-агентов.

Модуль отвечает за обнаружение, загрузку и выполнение навыков, расположенных
в каталоге skills. Каждый навык описывается в файле SKILL.md и может содержать
дополнительные ресурсы и Python-скрипты.

Менеджер формирует краткий каталог доступных навыков для передачи в системный
промпт LLM, предоставляет инструменты для загрузки полных инструкций выбранного
навыка, чтения его справочных материалов и запуска разрешённых скриптов.

Модуль не зависит от конкретного агента или графа и может использоваться как
универсальный слой управления Skills в LangChain и LangGraph-приложениях.
"""

import subprocess
from pathlib import Path
import yaml
from langchain_core.tools import BaseTool, tool

SKILLS_DIR = Path(__file__).resolve().parent / "skills"


def load_skill_catalog() -> dict:
    """ Возвращает краткое описание навыков с их путями

    Returns:
        {
        "addition": {
            "description": "Use when the user asks to add numbers or calculate a sum.",
            "path": "skills/addition/SKILL.md",
            },
        "multiplication": {
            "description": "Use when the user asks to multiply numbers or calculate a product.",
            "path": "skills/multiplication/SKILL.md",
            },
        "division": {
            "description": "Use when the user asks to divide one number by another.",
            "path": "skills/division/SKILL.md",
            },
        }

    """
    skills = {}


    for skill_dir in SKILLS_DIR.iterdir():
        skill_file = skill_dir / "SKILL.md"

        if not skill_file.is_file():
            continue

        content = skill_file.read_text(encoding="utf-8")
        _, frontmatter, _ = content.split("---", 2)

        metadata = yaml.safe_load(frontmatter)

        skills[metadata["name"]] = {
            "description": metadata["description"],
            "path": str(skill_file),
        }

    return skills


SKILLS = load_skill_catalog()


def build_skill_catalog() -> str:
    """ Возвращает описание скилов для SystemMessage

    Returns:
        - addition: Используйте, когда пользователь просит сложить числа или вычислить сумму.
        - multiplication: Используйте, когда пользователь просит перемножить числа или вычислить произведение.
        - division: Используйте, когда пользователь просит разделить одно число на другое.

    """
    return "\n".join(
        f"- {name}: {skill['description']}"
        for name, skill in SKILLS.items()
    )


@tool
def load_skill(skill_name: str) -> str:
    """Загрузить полные инструкции для данного навыка."""
    skill = SKILLS.get(skill_name)

    if skill is None:
        return f"Skill '{skill_name}' not found."

    content = Path(skill["path"]).read_text(encoding="utf-8")
    _, _, body = content.split("---", 2)

    return body.strip()


@tool
def run_skill_script(skill_name: str, script_path: str, script_args: list[str]) -> str:
    """Запустить Python-скрипт из каталога scripts выбранного навыка."""

    skill = SKILLS.get(skill_name)

    if skill is None:
        return f"Skill '{skill_name}' not found."

    skill_dir = Path(skill["path"]).parent.resolve()
    scripts_dir = (skill_dir / "scripts").resolve()
    file_path = (skill_dir / script_path).resolve()

    if scripts_dir not in file_path.parents:
        return "Могут выполняться только скрипты из каталога scripts."

    if not file_path.is_file():
        return f"Script '{script_path}' not found."

    result = subprocess.run(
        ["python", str(file_path), *script_args],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        return result.stderr

    return result.stdout.strip()


@tool
def read_skill_resource(skill_name: str, resource_path: str) -> str:
    """Прочитать дополнительную документацию из каталога references навыка."""

    skill = SKILLS.get(skill_name)

    if skill is None:
        return f"Skill '{skill_name}' not found."

    skill_dir = Path(skill["path"]).parent.resolve()
    references_dir = (skill_dir / "references").resolve()
    file_path = (skill_dir / resource_path).resolve()

    if references_dir not in file_path.parents:
        return "Можно читать только файлы из каталога references."

    if not file_path.is_file():
        return f"Resource '{resource_path}' not found."

    return file_path.read_text(encoding="utf-8")


def get_sys_tools() -> dict[str, list[BaseTool]]:
    """ Возвращает список системных инструментов """
    sys_tools = dict()
    sys_tools['selector'] = [load_skill]
    sys_tools['executor'] = [load_skill, run_skill_script, read_skill_resource]
    return sys_tools
