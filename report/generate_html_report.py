"""
HTML Report Generator

从 tables/ 目录读取 Markdown 评测结果，生成单文件交互式 HTML 报告。
输出文件：tables/report.html
"""
from pathlib import Path
from typing import List, Tuple, Optional


def parse_md_table(filepath: Path) -> List[Tuple[List[str], List[List[str]]]]:
    """解析 Markdown 文件中的所有表格（支持多个表格，带 ## 小节标题）

    Returns:
        [(section_title, headers, rows), ...]
    """
    if not filepath.exists():
        return []

    content = filepath.read_text(encoding="utf-8")
    lines = content.split("\n")

    results = []
    current_section = ""
    current_headers: List[str] = []
    current_rows: List[List[str]] = []
    in_table = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            if in_table and current_headers:
                results.append((current_section, current_headers, current_rows))
            current_section = stripped[3:].strip()
            current_headers = []
            current_rows = []
            in_table = False
        elif stripped.startswith("|") and "---" not in stripped:
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            if not in_table:
                current_headers = cells
                in_table = True
            else:
                current_rows.append(cells)
        elif stripped.startswith("|") and "---" in stripped:
            pass  # separator row, skip
        else:
            if in_table and current_headers:
                results.append((current_section, current_headers, current_rows))
                current_headers = []
                current_rows = []
                in_table = False

    if in_table and current_headers:
        results.append((current_section, current_headers, current_rows))

    return results


def value_to_color(val: float, min_val: float, max_val: float) -> str:
    """将数值映射为红→黄→绿的 RGB 颜色字符串"""
    if max_val == min_val:
        t = 0.5
    else:
        t = (val - min_val) / (max_val - min_val)
    t = max(0.0, min(1.0, t))

    if t < 0.5:
        # 红 → 黄
        r = 220
        g = int(220 * (t * 2))
        b = int(50 * (1 - t * 2))
    else:
        # 黄 → 绿
        r = int(220 * (1 - (t - 0.5) * 2))
        g = 200
        b = int(50 * (t - 0.5) * 2)

    return f"rgb({r},{g},{b})"


def try_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def render_summary_heatmap(summary_file: Path) -> str:
    """生成 SUMMARY 热力图 HTML"""
    tables = parse_md_table(summary_file)
    if not tables:
        return "<p>SUMMARY 文件解析失败</p>"

    _, headers, rows = tables[0]
    # headers[0] = "数据集 \ 模型", headers[1:] = model names
    models = headers[1:]

    # 收集所有数值
    all_vals = []
    for row in rows:
        for cell in row[1:]:
            v = try_float(cell)
            if v is not None:
                all_vals.append(v)

    min_val = min(all_vals) if all_vals else 0.0
    max_val = max(all_vals) if all_vals else 1.0

    # 每行的最大值索引
    row_max_idx = []
    for row in rows:
        vals = [try_float(c) for c in row[1:]]
        m = -1
        best = -1.0
        for i, v in enumerate(vals):
            if v is not None and v > best:
                best = v
                m = i
        row_max_idx.append(m)

    html = ['<table class="summary-table">']
    # header row
    html.append("<thead><tr>")
    html.append(f'<th>{headers[0]}</th>')
    for m in models:
        html.append(f"<th>{m}</th>")
    html.append("</tr></thead>")
    html.append("<tbody>")
    for ri, row in enumerate(rows):
        html.append("<tr>")
        html.append(f'<td class="row-label">{row[0]}</td>')
        for ci, cell in enumerate(row[1:]):
            v = try_float(cell)
            style = ""
            extra_class = ""
            if v is not None:
                color = value_to_color(v, min_val, max_val)
                style = f'style="background:{color}"'
                if ci == row_max_idx[ri]:
                    extra_class = " best"
            html.append(f'<td class="heatmap-cell{extra_class}" {style}>{cell}</td>')
        html.append("</tr>")
    html.append("</tbody></table>")
    return "\n".join(html)


def render_benchmark_table(headers: List[str], rows: List[List[str]], highlight: bool = True) -> str:
    """渲染单个指标表格，可选高亮最高/最低分"""
    if not headers or not rows:
        return ""

    # 逐列找数值列的 min/max
    col_min: List[Optional[float]] = [None] * len(headers)
    col_max: List[Optional[float]] = [None] * len(headers)

    if highlight:
        for col in range(1, len(headers)):
            vals = []
            for row in rows:
                if col < len(row):
                    v = try_float(row[col])
                    if v is not None:
                        vals.append(v)
            if vals:
                col_min[col] = min(vals)
                col_max[col] = max(vals)

    html = ['<table class="data-table">']
    html.append("<thead><tr>")
    for h in headers:
        html.append(f"<th>{h}</th>")
    html.append("</tr></thead><tbody>")

    for row in rows:
        html.append("<tr>")
        for ci, cell in enumerate(row):
            if ci == 0:
                html.append(f'<td class="metric-label">{cell}</td>')
            else:
                v = try_float(cell)
                cls = ""
                if v is not None and highlight:
                    if col_max[ci] is not None and v == col_max[ci] and col_max[ci] != col_min[ci]:
                        cls = ' class="cell-best"'
                    elif col_min[ci] is not None and v == col_min[ci] and col_max[ci] != col_min[ci]:
                        cls = ' class="cell-worst"'
                html.append(f"<td{cls}>{cell}</td>")
        html.append("</tr>")

    html.append("</tbody></table>")
    return "\n".join(html)


def render_benchmark_section(benchmark: str, tables_dir: Path) -> str:
    """生成单个 benchmark 的 Tab 内容 HTML"""
    bench_dir = tables_dir / benchmark

    # 基础指标
    basic_tables = parse_md_table(bench_dir / "基础指标.md")
    basic_html = ""
    if basic_tables:
        _, headers, rows = basic_tables[0]
        basic_html = render_benchmark_table(headers, rows, highlight=True)

    # 其他指标
    other_tables = parse_md_table(bench_dir / "其他指标.md")
    other_html_parts = []
    for section_title, headers, rows in other_tables:
        # 跳过 counts 类统计表（仅含样本数量，无意义）
        if section_title.lower().endswith("_counts"):
            continue
        table_html = render_benchmark_table(headers, rows, highlight=False)
        if section_title:
            other_html_parts.append(f'<h4 class="sub-section">{section_title}</h4>')
        other_html_parts.append(table_html)
    other_html = "\n".join(other_html_parts)

    details_block = ""
    if other_html:
        details_block = f"""
<details class="other-metrics">
  <summary>其他指标（展开查看）</summary>
  <div class="other-metrics-content">
    {other_html}
  </div>
</details>"""

    return f"""
<div class="benchmark-section">
  <h3>基础指标</h3>
  <div class="table-scroll">{basic_html}</div>
  {details_block}
</div>"""


def generate_html_report(tables_dir: str = "tables", output_file: str = None) -> None:
    """生成 HTML 报告主函数"""
    tables_path = Path(tables_dir)
    output_path = Path(output_file) if output_file else tables_path / "report.html"

    # 自动扫描 tables/ 下所有含 Markdown 表格的数据集目录，按名称排序
    benchmarks = sorted(
        d.name for d in tables_path.iterdir()
        if d.is_dir() and any(d.glob("*.md"))
    )

    summary_html = render_summary_heatmap(tables_path / "SUMMARY.md")

    # 生成各 benchmark tab 内容
    tab_buttons = []
    tab_contents = []
    for i, bench in enumerate(benchmarks):
        active_btn = " active" if i == 0 else ""
        active_content = " active" if i == 0 else ""
        tab_buttons.append(
            f'<button class="tab-btn{active_btn}" onclick="switchTab(event, \'{bench}\')">{bench}</button>'
        )
        section_html = render_benchmark_section(bench, tables_path)
        tab_contents.append(
            f'<div id="tab-{bench}" class="tab-content{active_content}">{section_html}</div>'
        )

    tabs_html = "\n".join(tab_buttons)
    contents_html = "\n".join(tab_contents)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ToM 评测报告</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: "Helvetica Neue", Arial, "PingFang SC", "Microsoft YaHei", sans-serif;
         background: #f5f7fa; color: #333; font-size: 14px; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 24px 16px; }}
  h1 {{ font-size: 24px; font-weight: 700; margin-bottom: 6px; color: #1a1a2e; }}
  .subtitle {{ color: #666; margin-bottom: 28px; font-size: 13px; }}
  h2 {{ font-size: 18px; font-weight: 600; margin: 28px 0 14px; color: #222; border-left: 4px solid #4a90d9; padding-left: 10px; }}
  h3 {{ font-size: 15px; font-weight: 600; margin: 16px 0 8px; color: #444; }}
  h4.sub-section {{ font-size: 13px; font-weight: 600; margin: 14px 0 6px; color: #555; }}

  /* Summary heatmap */
  .summary-table {{ border-collapse: collapse; width: 100%; background: #fff;
                    border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,.08); }}
  .summary-table th {{ background: #1a1a2e; color: #fff; padding: 10px 14px;
                        text-align: center; font-size: 12px; white-space: nowrap; }}
  .summary-table td {{ padding: 9px 14px; text-align: center; font-size: 13px; border-bottom: 1px solid #eee; }}
  .summary-table td.row-label {{ font-weight: 600; background: #f0f4ff; text-align: left; white-space: nowrap; }}
  .heatmap-cell {{ font-weight: 500; transition: opacity .15s; }}
  .heatmap-cell.best {{ font-weight: 700; outline: 2px solid #1a7a1a; outline-offset: -2px; }}
  .summary-table tbody tr:hover td {{ opacity: .85; }}

  /* Tabs */
  .tab-bar {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 0; }}
  .tab-btn {{ padding: 8px 18px; border: 1px solid #c8d6e8; border-bottom: none; background: #e8eef7;
              cursor: pointer; border-radius: 6px 6px 0 0; font-size: 13px; font-weight: 500;
              color: #555; transition: background .15s; }}
  .tab-btn:hover {{ background: #d0dcf0; }}
  .tab-btn.active {{ background: #fff; color: #1a1a2e; font-weight: 700; border-color: #b0c0d8; }}
  .tab-wrapper {{ background: #fff; border: 1px solid #b0c0d8; border-radius: 0 6px 6px 6px;
                  padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,.06); }}
  .tab-content {{ display: none; }}
  .tab-content.active {{ display: block; }}

  /* Data tables */
  .table-scroll {{ overflow-x: auto; }}
  .data-table {{ border-collapse: collapse; min-width: 600px; width: 100%; margin-bottom: 8px; }}
  .data-table th {{ background: #2c3e6b; color: #fff; padding: 8px 12px; text-align: center;
                    font-size: 12px; white-space: nowrap; }}
  .data-table td {{ padding: 7px 12px; border-bottom: 1px solid #eee; text-align: center; font-size: 13px; }}
  .data-table td.metric-label {{ font-weight: 500; background: #f6f8ff; text-align: left;
                                  white-space: nowrap; color: #444; }}
  .data-table tbody tr:hover td {{ background: #f0f4ff; }}
  td.cell-best {{ color: #1a6e1a; font-weight: 700; background: #e6f9e6 !important; }}
  td.cell-worst {{ color: #b00; font-weight: 700; background: #fff0f0 !important; }}

  /* Details/折叠 */
  .other-metrics {{ margin-top: 16px; border: 1px solid #dde4f0; border-radius: 6px; overflow: hidden; }}
  .other-metrics summary {{ padding: 10px 16px; background: #eef2fc; cursor: pointer;
                             font-weight: 600; font-size: 13px; color: #3a4a7a;
                             list-style: none; user-select: none; }}
  .other-metrics summary::-webkit-details-marker {{ display: none; }}
  .other-metrics summary::before {{ content: "▶  "; font-size: 10px; }}
  .other-metrics[open] summary::before {{ content: "▼  "; }}
  .other-metrics-content {{ padding: 16px; background: #fafbff; overflow-x: auto; }}
</style>
</head>
<body>
<div class="container">
  <h1>ToM 评测报告</h1>
  <p class="subtitle">Theory of Mind 基准测试 — 多模型横向对比</p>

  <h2>总览：Accuracy 热力图</h2>
  {summary_html}

  <h2>各 Benchmark 详情</h2>
  <div class="tab-bar">
    {tabs_html}
  </div>
  <div class="tab-wrapper">
    {contents_html}
  </div>
</div>

<script>
function switchTab(event, benchName) {{
  document.querySelectorAll('.tab-btn').forEach(function(b) {{ b.classList.remove('active'); }});
  document.querySelectorAll('.tab-content').forEach(function(c) {{ c.classList.remove('active'); }});
  event.currentTarget.classList.add('active');
  document.getElementById('tab-' + benchName).classList.add('active');
}}
</script>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"HTML 报告已生成：{output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="生成交互式 HTML 评测报告")
    parser.add_argument("--tables-dir", default="tables", help="tables 目录路径")
    parser.add_argument("--output", default=None, help="输出 HTML 文件路径（默认: tables/report.html）")
    args = parser.parse_args()

    generate_html_report(tables_dir=args.tables_dir, output_file=args.output)


if __name__ == "__main__":
    main()
