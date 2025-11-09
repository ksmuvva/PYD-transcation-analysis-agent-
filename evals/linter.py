#!/usr/bin/env python3
"""
Linter to prevent LLM-as-judge violations (REQ-EVAL-003).

Scans evals/ package for forbidden LLM-judge patterns.
"""

import ast
import sys
from pathlib import Path
from typing import List, Tuple


class LLMJudgeLinter(ast.NodeVisitor):
    """AST visitor to detect LLM-as-judge violations."""

    FORBIDDEN_IMPORTS = [
        "LLMJudge",
        "GEval",
        "GPTJudge",
        "ClaudeJudge",
        "LLMEvaluator",
    ]

    FORBIDDEN_PATTERNS = [
        "llm.judge",
        "evaluate_with_llm",
        "score_with_model",
        "ai_scorer",
    ]

    def __init__(self, filename: str):
        self.filename = filename
        self.violations: List[Tuple[int, str]] = []

    def visit_Import(self, node: ast.Import):
        """Check import statements."""
        for alias in node.names:
            if any(forbidden in alias.name for forbidden in self.FORBIDDEN_IMPORTS):
                self.violations.append((
                    node.lineno,
                    f"Forbidden import: {alias.name} (REQ-EVAL-003: No LLM-as-judge)"
                ))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Check from-import statements."""
        if node.module:
            for alias in node.names:
                if any(forbidden in alias.name for forbidden in self.FORBIDDEN_IMPORTS):
                    self.violations.append((
                        node.lineno,
                        f"Forbidden import: {node.module}.{alias.name} (REQ-EVAL-003: No LLM-as-judge)"
                    ))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """Check function calls."""
        # Get function name
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        else:
            func_name = ""

        # Check for forbidden patterns
        for pattern in self.FORBIDDEN_PATTERNS:
            if pattern in func_name.lower():
                self.violations.append((
                    node.lineno,
                    f"Forbidden function call: {func_name} (REQ-EVAL-003: No LLM-as-judge)"
                ))

        self.generic_visit(node)


def lint_file(filepath: Path) -> List[Tuple[int, str]]:
    """
    Lint a Python file for LLM-judge violations.

    Args:
        filepath: Path to Python file

    Returns:
        List of (line_number, violation_message) tuples
    """
    try:
        with open(filepath) as f:
            tree = ast.parse(f.read(), filename=str(filepath))

        linter = LLMJudgeLinter(str(filepath))
        linter.visit(tree)

        return linter.violations
    except SyntaxError as e:
        return [(e.lineno or 0, f"Syntax error: {e}")]
    except Exception as e:
        return [(0, f"Error parsing file: {e}")]


def main():
    """Main entry point for linter."""
    if len(sys.argv) < 2:
        print("Usage: python evals/linter.py <file1.py> [file2.py ...]")
        sys.exit(1)

    all_violations = []

    for filepath in sys.argv[1:]:
        path = Path(filepath)
        if not path.exists():
            print(f"File not found: {filepath}")
            continue

        if not path.suffix == ".py":
            continue

        violations = lint_file(path)

        if violations:
            all_violations.extend(violations)
            print(f"\n{filepath}:")
            for line_no, message in violations:
                print(f"  Line {line_no}: {message}")

    if all_violations:
        print(f"\n❌ Found {len(all_violations)} LLM-as-judge violation(s)")
        print("REQ-EVAL-003: All metrics must be pure Python functions, no LLM involvement!")
        sys.exit(1)
    else:
        print("✅ No LLM-as-judge violations found")
        sys.exit(0)


if __name__ == "__main__":
    main()
