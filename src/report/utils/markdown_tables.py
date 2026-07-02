from __future__ import annotations

import re
from typing import Dict, List, Optional


def parse_markdown_table(content: str) -> Dict[str, Dict[str, str]]:
    table: Dict[str, Dict[str, str]] = {}
    header: List[str] = []
    for line in content.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            header = []
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if all(re.fullmatch(r"-+", cell) for cell in cells if cell):
            continue
        if not header:
            header = cells
            continue
        row_key = cells[0]
        for index, column in enumerate(header[1:], start=1):
            if column and index < len(cells):
                table.setdefault(row_key, {})[column] = cells[index]
    return table


def parse_markdown_sections(content: str) -> Dict[str, Dict[str, Dict[str, str]]]:
    sections: Dict[str, Dict[str, Dict[str, str]]] = {}
    current_title: Optional[str] = None
    current_lines: List[str] = []

    for line in content.splitlines():
        if line.startswith("## "):
            if current_title is not None:
                parsed = parse_markdown_table("\n".join(current_lines))
                if parsed:
                    sections[current_title] = parsed
            current_title = line[3:].strip()
            current_lines = []
            continue
        if current_title is not None:
            current_lines.append(line)

    if current_title is not None:
        parsed = parse_markdown_table("\n".join(current_lines))
        if parsed:
            sections[current_title] = parsed
    return sections
