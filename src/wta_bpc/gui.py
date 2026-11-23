"""PyQt6 front-end for a Weapon-Target Assignment (WTA) solver.

The GUI deliberately contains no optimization logic. Connect the existing solver
in ``solver_adapter`` near the bottom of this file, or pass a callable to
``WTAWindow(solver=...)``.

Expected payload format passed to the solver adapter:
    {
        "n_targets": int,
        "n_weapons": int,
        "hit_probabilities": list[list[float]],  # [weapon][target]
        "additional_parameters": dict,
    }
"""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any, Callable

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QFrame,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QStyledItemDelegate,
    QDoubleSpinBox,
    QInputDialog,
    QSizePolicy,
)

SolverCallable = Callable[[dict[str, Any]], Any]


STYLESHEET = """
QMainWindow, QWidget#root {
    background: #0a1422;
    color: #e7eef7;
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 10pt;
}
QFrame#header {
    background: #10243a;
    border: 1px solid #1d3c56;
    border-radius: 14px;
}
QLabel#eyebrow {
    color: #62d3c5;
    font-size: 8pt;
    font-weight: 700;
    letter-spacing: 2px;
}
QLabel#mainTitle {
    color: #f4f8fc;
    font-size: 23pt;
    font-weight: 700;
}
QLabel#subtitle {
    color: #a6b8ca;
    font-size: 9pt;
}
QFrame#panel {
    background: #101f31;
    border: 1px solid #21384e;
    border-radius: 12px;
}
QLabel#sectionTitle {
    color: #f1f6fb;
    font-size: 13pt;
    font-weight: 650;
}
QLabel#muted {
    color: #9aadc0;
}
QLabel#metricValue {
    color: #62d3c5;
    font-size: 22pt;
    font-weight: 700;
}
QTabWidget::pane {
    border: 1px solid #21384e;
    border-radius: 10px;
    top: -1px;
    background: #0d1a2a;
}
QTabBar::tab {
    background: #102033;
    color: #9db0c3;
    padding: 11px 20px;
    margin-right: 5px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
}
QTabBar::tab:selected {
    background: #17334a;
    color: #70e0d0;
    border-bottom: 2px solid #62d3c5;
}
QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTableWidget {
    background: #0a1726;
    color: #e7eef7;
    border: 1px solid #2a4359;
    border-radius: 6px;
    padding: 6px;
    selection-background-color: #1d6570;
    selection-color: white;
}
QSpinBox:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus, QTableWidget:focus {
    border: 1px solid #62d3c5;
}
QTableWidget {
    gridline-color: #21384e;
    padding: 3px;
}
QHeaderView::section {
    background: #172b40;
    color: #bbccdc;
    border: none;
    border-right: 1px solid #2a4359;
    border-bottom: 1px solid #2a4359;
    padding: 8px;
    font-weight: 650;
}
QTableCornerButton::section {
    background: #172b40;
    border: none;
}
QPushButton {
    background: #1a354b;
    color: #dce9f4;
    border: 1px solid #34536b;
    border-radius: 7px;
    padding: 9px 15px;
    font-weight: 600;
}
QPushButton:hover {
    background: #24485f;
    border-color: #62d3c5;
}
QPushButton:pressed { background: #142b3d; }
QPushButton:disabled {
    background: #182635;
    color: #687b8d;
    border-color: #263747;
}
QPushButton#primaryButton {
    background: #20a99b;
    color: #061a20;
    border: none;
    padding: 12px 24px;
    font-size: 10pt;
    font-weight: 750;
}
QPushButton#primaryButton:hover { background: #47c5b5; }
QPushButton#primaryButton:disabled { background: #345d60; color: #9bb6b4; }
QPushButton#subtleButton {
    background: transparent;
    border: 1px solid #34536b;
    color: #a9bdcf;
}
QLabel#status {
    color: #a5bacb;
    padding: 4px 2px;
}
"""


class ProbabilityDelegate(QStyledItemDelegate):
    """Cell editor constrained to probabilities in [0, 1]."""

    def createEditor(self, parent, option, index):  # noqa: N802 - Qt API name
        editor = QDoubleSpinBox(parent)
        editor.setRange(0.0, 1.0)
        editor.setDecimals(4)
        editor.setSingleStep(0.01)
        editor.setFrame(False)
        return editor

    def setEditorData(self, editor, index):  # noqa: N802 - Qt API name
        try:
            value = float(index.data(Qt.ItemDataRole.EditRole) or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        editor.setValue(value)

    def setModelData(self, editor, model, index):  # noqa: N802 - Qt API name
        model.setData(index, f"{editor.value():.4f}", Qt.ItemDataRole.EditRole)


class SolverWorker(QThread):
    """Runs the user's solver outside the GUI thread."""

    result_ready = pyqtSignal(object)
    error_raised = pyqtSignal(str)

    def __init__(self, solver: SolverCallable, payload: dict[str, Any]):
        super().__init__()
        self.solver = solver
        self.payload = payload

    def run(self) -> None:
        try:
            result = self.solver(self.payload)
            self.result_ready.emit(result)
        except Exception:
            self.error_raised.emit(traceback.format_exc())


class WTAWindow(QMainWindow):
    def __init__(self, solver: SolverCallable | None = None):
        super().__init__()
        self.solver = solver
        self._worker: SolverWorker | None = None

        self.setWindowTitle("WTA Mission Console")
        self.setMinimumSize(980, 700)
        self.resize(1120, 780)

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(24, 20, 24, 18)
        outer.setSpacing(16)

        outer.addWidget(self._build_header())
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_scenario_tab(), "01  Scenario")
        self.tabs.addTab(self._build_probability_tab(), "02  Hit probabilities")
        self.tabs.addTab(self._build_advanced_tab(), "03  Advanced parameters")
        self.results_tab = self._build_results_tab()
        self.tabs.addTab(self.results_tab, "04  Solver output")
        outer.addWidget(self.tabs, 1)
        outer.addLayout(self._build_footer())

        self._refresh_matrix()
        self._set_status("Ready. Configure the scenario and probability matrix.")

    def _build_header(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("header")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(22, 18, 22, 18)

        left = QVBoxLayout()
        eyebrow = QLabel("OPERATIONS RESEARCH  /  OPTIMIZATION WORKSPACE")
        eyebrow.setObjectName("eyebrow")
        title = QLabel("Weapon–Target Assignment")
        title.setObjectName("mainTitle")
        subtitle = QLabel("Scenario configuration  •  Engagement probability matrix  •  Solver interface")
        subtitle.setObjectName("subtitle")
        left.addWidget(eyebrow)
        left.addSpacing(3)
        left.addWidget(title)
        left.addWidget(subtitle)
        layout.addLayout(left, 1)

        right = QFrame()
        right.setStyleSheet("background:#0b1b2b; border:1px solid #29475e; border-radius:10px;")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(16, 10, 16, 10)
        right_layout.setSpacing(0)
        tag = QLabel("MODEL STATUS")
        tag.setStyleSheet("color:#8fa9bc; font-size:8pt; font-weight:700; letter-spacing:1px; border:none;")
        state = QLabel("INPUT CONFIGURATION")
        state.setStyleSheet("color:#62d3c5; font-size:10pt; font-weight:700; border:none;")
        right_layout.addWidget(tag)
        right_layout.addWidget(state)
        layout.addWidget(right, 0, Qt.AlignmentFlag.AlignVCenter)
        return frame

    def _panel(self, title: str, description: str | None = None) -> tuple[QFrame, QVBoxLayout]:
        frame = QFrame()
        frame.setObjectName("panel")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(10)
        heading = QLabel(title)
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)
        if description:
            note = QLabel(description)
            note.setObjectName("muted")
            note.setWordWrap(True)
            layout.addWidget(note)
        return frame, layout

    def _build_scenario_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        intro = QLabel("Define the dimensions of the assignment problem.")
        intro.setObjectName("muted")
        layout.addWidget(intro)

        panel, panel_layout = self._panel(
            "Scenario dimensions",
            "Set the number of targets and available weapons. The probability matrix will resize automatically, preserving values where possible.",
        )
        row = QHBoxLayout()
        row.setSpacing(14)
        self.target_count = self._make_count_card("TARGETS", "Number of targets")
        self.weapon_count = self._make_count_card("WEAPONS", "Number of weapons")
        row.addWidget(self.target_count[0], 1)
        row.addWidget(self.weapon_count[0], 1)
        panel_layout.addLayout(row)
        layout.addWidget(panel)

        info_panel, info_layout = self._panel("Input convention")
        convention = QLabel(
            "The hit-probability table is organized by weapon (rows) and target (columns). "
            "Each cell P[w, t] is the probability that weapon w successfully hits target t. "
            "Values must be between 0 and 1, inclusive."
        )
        convention.setWordWrap(True)
        convention.setStyleSheet("color:#b5c6d5; line-height:150%;")
        info_layout.addWidget(convention)
        layout.addWidget(info_panel)
        layout.addStretch(1)
        return page

    def _make_count_card(self, label: str, description: str) -> tuple[QFrame, QSpinBox]:
        card = QFrame()
        card.setStyleSheet("background:#0b1929; border:1px solid #2a4359; border-radius:10px;")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(18, 16, 18, 16)
        label_widget = QLabel(label)
        label_widget.setStyleSheet("color:#62d3c5; font-size:8pt; font-weight:750; letter-spacing:1.5px; border:none;")
        value = QSpinBox()
        value.setRange(1, 100)
        value.setValue(4 if label == "TARGETS" else 3)
        value.setMinimumHeight(44)
        value.setStyleSheet("font-size:16pt; font-weight:700; padding:5px 10px;")
        desc = QLabel(description)
        desc.setObjectName("muted")
        desc.setStyleSheet("border:none;")
        card_layout.addWidget(label_widget)
        card_layout.addWidget(value)
        card_layout.addWidget(desc)
        value.valueChanged.connect(self._refresh_matrix)
        return card, value

    def _build_probability_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        top = QHBoxLayout()
        left = QVBoxLayout()
        heading = QLabel("Weapon-to-target hit probability")
        heading.setObjectName("sectionTitle")
        hint = QLabel("Edit each cell directly. Use decimal probabilities, for example 0.85 for 85%.")
        hint.setObjectName("muted")
        left.addWidget(heading)
        left.addWidget(hint)
        top.addLayout(left, 1)

        fill_button = QPushButton("Fill all…")
        fill_button.clicked.connect(self._fill_all_probabilities)
        clear_button = QPushButton("Clear to zero")
        clear_button.setObjectName("subtleButton")
        clear_button.clicked.connect(self._clear_probabilities)
        top.addWidget(fill_button)
        top.addWidget(clear_button)
        layout.addLayout(top)

        self.probability_table = QTableWidget()
        self.probability_table.setItemDelegate(ProbabilityDelegate(self.probability_table))
        self.probability_table.setAlternatingRowColors(True)
        self.probability_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.probability_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.probability_table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.AnyKeyPressed
        )
        self.probability_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.probability_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.probability_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.probability_table, 1)

        note = QLabel("Rows = weapons     /     Columns = targets     /     Allowed range: 0.0000 – 1.0000")
        note.setObjectName("muted")
        layout.addWidget(note)
        return page

    def _build_advanced_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        intro = QLabel("Add solver-specific settings without changing the interface code.")
        intro.setObjectName("muted")
        layout.addWidget(intro)

        self.advanced_table = QTableWidget(0, 2)
        self.advanced_table.setHorizontalHeaderLabels(["Parameter name", "Value"])
        self.advanced_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.advanced_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.advanced_table.verticalHeader().setVisible(False)
        self.advanced_table.setAlternatingRowColors(True)
        self.advanced_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.advanced_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.advanced_table.setMinimumHeight(260)
        layout.addWidget(self.advanced_table, 1)

        buttons = QHBoxLayout()
        add_button = QPushButton("+  Add parameter")
        add_button.clicked.connect(self._add_parameter_row)
        remove_button = QPushButton("Remove selected")
        remove_button.setObjectName("subtleButton")
        remove_button.clicked.connect(self._remove_parameter_rows)
        buttons.addWidget(add_button)
        buttons.addWidget(remove_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        help_panel, help_layout = self._panel("Value parsing")
        help_text = QLabel(
            "Values are interpreted as JSON when possible: 12, 0.95, true, false, null, [1, 2], or {\"mode\": \"exact\"}. "
            "Anything that is not valid JSON is passed as a string. Parameter names must be unique."
        )
        help_text.setObjectName("muted")
        help_text.setWordWrap(True)
        help_layout.addWidget(help_text)
        layout.addWidget(help_panel)
        return page

    def _build_results_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)

        heading_row = QHBoxLayout()
        heading = QLabel("Solver result")
        heading.setObjectName("sectionTitle")
        self.result_status = QLabel("No run yet")
        self.result_status.setObjectName("muted")
        heading_row.addWidget(heading)
        heading_row.addStretch(1)
        heading_row.addWidget(self.result_status)
        layout.addLayout(heading_row)

        self.result_output = QPlainTextEdit()
        self.result_output.setReadOnly(True)
        self.result_output.setPlaceholderText("The solver output will appear here after you press ‘Run solver’. ")
        self.result_output.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.result_output.setStyleSheet("font-family:Consolas, 'Cascadia Code', monospace; font-size:9pt;")
        layout.addWidget(self.result_output, 1)

        clear = QPushButton("Clear output")
        clear.setObjectName("subtleButton")
        clear.clicked.connect(self._clear_output)
        layout.addWidget(clear, 0, Qt.AlignmentFlag.AlignRight)
        return page

    def _build_footer(self) -> QHBoxLayout:
        footer = QHBoxLayout()
        self.status_label = QLabel()
        self.status_label.setObjectName("status")
        footer.addWidget(self.status_label, 1)
        self.run_button = QPushButton("RUN SOLVER   →")
        self.run_button.setObjectName("primaryButton")
        self.run_button.setMinimumWidth(190)
        self.run_button.clicked.connect(self._run_solver)
        footer.addWidget(self.run_button)
        return footer

    def _refresh_matrix(self, *_args) -> None:
        """Resize the matrix while retaining the overlapping old cells."""
        if not hasattr(self, "probability_table"):
            return
        table = self.probability_table
        old_rows, old_cols = table.rowCount(), table.columnCount()
        old_values: dict[tuple[int, int], str] = {}
        for r in range(old_rows):
            for c in range(old_cols):
                item = table.item(r, c)
                if item is not None:
                    old_values[(r, c)] = item.text()

        n_weapons = self.weapon_count[1].value()
        n_targets = self.target_count[1].value()
        table.setRowCount(n_weapons)
        table.setColumnCount(n_targets)
        table.setHorizontalHeaderLabels([f"Target {i + 1:02d}" for i in range(n_targets)])
        table.setVerticalHeaderLabels([f"Weapon {i + 1:02d}" for i in range(n_weapons)])

        for r in range(n_weapons):
            for c in range(n_targets):
                value = old_values.get((r, c), "0.8000")
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(r, c, item)
        self._set_status(f"Scenario dimensions: {n_weapons} weapons × {n_targets} targets.")

    def _fill_all_probabilities(self) -> None:
        value, accepted = QInputDialog.getDouble(
            self, "Fill probability matrix", "Probability for every weapon–target pair:",
            0.8, 0.0, 1.0, 4,
        )
        if not accepted:
            return
        for r in range(self.probability_table.rowCount()):
            for c in range(self.probability_table.columnCount()):
                self.probability_table.item(r, c).setText(f"{value:.4f}")

    def _clear_probabilities(self) -> None:
        for r in range(self.probability_table.rowCount()):
            for c in range(self.probability_table.columnCount()):
                self.probability_table.item(r, c).setText("0.0000")

    def _add_parameter_row(self) -> None:
        row = self.advanced_table.rowCount()
        self.advanced_table.insertRow(row)
        self.advanced_table.setItem(row, 0, QTableWidgetItem(""))
        self.advanced_table.setItem(row, 1, QTableWidgetItem(""))
        self.advanced_table.editItem(self.advanced_table.item(row, 0))

    def _remove_parameter_rows(self) -> None:
        selected = sorted({index.row() for index in self.advanced_table.selectionModel().selectedRows()}, reverse=True)
        if not selected:
            current = self.advanced_table.currentRow()
            if current >= 0:
                selected = [current]
        for row in selected:
            self.advanced_table.removeRow(row)

    def _collect_payload(self) -> dict[str, Any]:
        n_weapons = self.weapon_count[1].value()
        n_targets = self.target_count[1].value()
        probabilities: list[list[float]] = []

        for r in range(n_weapons):
            row_values: list[float] = []
            for c in range(n_targets):
                item = self.probability_table.item(r, c)
                raw = item.text().strip() if item else ""
                try:
                    value = float(raw)
                except ValueError as exc:
                    raise ValueError(f"Invalid probability at Weapon {r + 1}, Target {c + 1}: {raw!r}.") from exc
                if not 0.0 <= value <= 1.0:
                    raise ValueError(
                        f"Probability at Weapon {r + 1}, Target {c + 1} must be between 0 and 1."
                    )
                row_values.append(value)
            probabilities.append(row_values)

        additional: dict[str, Any] = {}
        for row in range(self.advanced_table.rowCount()):
            key_item = self.advanced_table.item(row, 0)
            value_item = self.advanced_table.item(row, 1)
            key = key_item.text().strip() if key_item else ""
            raw_value = value_item.text().strip() if value_item else ""
            if not key and not raw_value:
                continue
            if not key:
                raise ValueError(f"Advanced parameter on row {row + 1} has a value but no name.")
            if key in additional:
                raise ValueError(f"Duplicate advanced parameter name: {key!r}.")
            try:
                parsed_value = json.loads(raw_value)
            except json.JSONDecodeError:
                parsed_value = raw_value
            additional[key] = parsed_value

        return {
            "n_targets": n_targets,
            "n_weapons": n_weapons,
            "hit_probabilities": probabilities,
            "additional_parameters": additional,
        }

    def _run_solver(self) -> None:
        if self.solver is None:
            QMessageBox.information(
                self,
                "Solver not connected",
                "Connect your existing solver in solver_adapter() at the bottom of wta_gui.py, "
                "then restart the application.",
            )
            return
        try:
            payload = self._collect_payload()
        except ValueError as exc:
            QMessageBox.warning(self, "Check the inputs", str(exc))
            return

        self.result_output.clear()
        self.result_status.setText("Running…")
        self._set_status("Solver is running in a background thread…")
        self.run_button.setEnabled(False)
        self.run_button.setText("SOLVER RUNNING…")
        self._worker = SolverWorker(self.solver, payload)
        self._worker.result_ready.connect(self._show_result)
        self._worker.error_raised.connect(self._show_solver_error)
        self._worker.finished.connect(self._solver_finished)
        self._worker.start()

    def _show_result(self, result: Any) -> None:
        try:
            rendered = json.dumps(result, indent=2, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            rendered = repr(result)
        self.result_output.setPlainText(rendered)
        self.result_status.setText("Completed")
        self.tabs.setCurrentWidget(self.results_tab)
        self._set_status("Solver completed successfully.")

    def _show_solver_error(self, details: str) -> None:
        self.result_output.setPlainText(details)
        self.result_status.setText("Error")
        self.tabs.setCurrentWidget(self.results_tab)
        self._set_status("The solver raised an exception. See the output panel for details.")

    def _solver_finished(self) -> None:
        self.run_button.setEnabled(True)
        self.run_button.setText("RUN SOLVER   →")
        self._worker = None

    def _clear_output(self) -> None:
        self.result_output.clear()
        self.result_status.setText("No run yet")

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.warning(
                self,
                "Solver still running",
                "The solver is still running. Wait for it to finish before closing the application.",
            )
            event.ignore()
            return
        event.accept()


def solver_adapter(payload: dict[str, Any]) -> Any:
    """Adapt the GUI payload to the signature of your existing solver.

    Replace the NotImplementedError with your import and function call. For
    example, if your solver is ``solve_wta(n_targets, n_weapons, probability_matrix, ... )``,
    map the payload fields to those arguments here.

    The adapter receives:
      - payload["n_targets"]: integer
      - payload["n_weapons"]: integer
      - payload["hit_probabilities"]: matrix indexed [weapon][target]
      - payload["additional_parameters"]: dictionary of extra parameters

    Return any Python object; JSON-serializable results are displayed as formatted
    JSON, and other objects are displayed using repr().
    """
    # Example integration (adapt names to your actual module and signature):
    # from my_wta_solver import solve_wta
    # return solve_wta(
    #     n_targets=payload["n_targets"],
    #     n_weapons=payload["n_weapons"],
    #     probability_matrix=payload["hit_probabilities"],
    #     **payload["additional_parameters"],
    # )
    raise NotImplementedError("Edit solver_adapter() to call your existing WTA solver.")


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    app.setFont(QFont("Segoe UI", 10))

    window = WTAWindow(solver=solver_adapter)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
