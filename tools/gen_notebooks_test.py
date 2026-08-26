#!/usr/bin/env python3
"""Behavior tests for generated notebook structure and safe argument parsing."""

import gen_notebooks as notebooks


def test_notebook_wraps_cells_in_a_runnable_v4_document():
    cells = [notebooks.md("# Title"), notebooks.code("print('ok')")]
    document = notebooks.notebook(cells)

    assert document["cells"] == cells
    assert document["nbformat"] == 4 and document["nbformat_minor"] == 5
    assert document["metadata"]["kernelspec"]["name"] == "python3"
    assert cells[0]["cell_type"] == "markdown"
    assert cells[1]["execution_count"] is None and cells[1]["outputs"] == []


def test_setup_cell_substitutes_gpu_expression_without_template_marker():
    cell = notebooks.setup_cell('"GPU ready" if HAS_GPU else "CPU only"')
    assert "__GPU_NOTE__" not in cell
    assert '"GPU ready" if HAS_GPU else "CPU only"' in cell
    assert "FAK_BRANCH" in cell and "nvidia-smi" in cell


def test_argument_parser_keeps_check_and_rejects_unknown_write_mode():
    assert notebooks.parse_args(["--check"]) == (True, None)
    assert notebooks.parse_args(["--chek"]) == (False, 2)
    assert notebooks.parse_args(["--help"]) == (False, 0)

