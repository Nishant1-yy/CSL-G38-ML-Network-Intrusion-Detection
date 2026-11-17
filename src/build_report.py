"""Build the project report from the measured outputs in ``results/``."""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
OUTPUT_DIR = ROOT / "report"
OUTPUT_DOCX = OUTPUT_DIR / "23ESKCY036_NishantKumawat_CSL-G38.docx"
SOURCE_FILE = ROOT / "src" / "ids_pipeline.py"
LOGO_FILE = ROOT / "assets" / "skit_logo.png"


def _validate_results(results: dict[str, Any]) -> None:
    if results.get("meta", {}).get("synthetic") is not False:
        raise ValueError("results.json must explicitly record meta.synthetic as false.")
    dataset = results.get("dataset", {})
    train_rows = dataset.get("train_rows")
    test_rows = dataset.get("test_rows")
    if not isinstance(train_rows, int) or not 125_000 <= train_rows <= 126_000:
        raise ValueError(
            "The training row count is outside the requested NSL-KDD range: "
            f"{train_rows!r}."
        )
    if not isinstance(test_rows, int) or not 22_000 <= test_rows <= 23_000:
        raise ValueError(
            "The test row count is outside the requested NSL-KDD range: "
            f"{test_rows!r}."
        )
    if not (RESULTS_DIR / "run_log.txt").is_file():
        raise FileNotFoundError(f"Missing measured run log: {RESULTS_DIR / 'run_log.txt'}")
    for relative_path in results.get("figure_files", []):
        if not (RESULTS_DIR / relative_path).is_file():
            raise FileNotFoundError(f"Missing results figure: {relative_path}")


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def _decimal(value: float | int | None, digits: int = 3) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def _seconds(value: float | int | None) -> str:
    return "N/A" if value is None else f"{value:.2f}"


def _format_setting(value: Any) -> str:
    if isinstance(value, dict):
        return ", ".join(f"{key}={item}" for key, item in value.items())
    return str(value)


def _set_run_font(run: Any, name: str, size: float, bold: bool | None = None) -> None:
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold


def _set_cell_shading(cell: Any, fill: str) -> None:
    properties = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    properties.append(shading)


def _set_repeat_table_header(row: Any) -> None:
    properties = row._tr.get_or_add_trPr()
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    properties.append(repeat)


def _set_cant_split(row: Any) -> None:
    properties = row._tr.get_or_add_trPr()
    cant_split = OxmlElement("w:cantSplit")
    properties.append(cant_split)


def _set_table_cell(cell: Any, value: Any, header: bool, size: float) -> None:
    cell.text = ""
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.paragraph_format.line_spacing = 1.0
    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = paragraph.add_run(str(value))
    _set_run_font(run, "Times New Roman", size, bold=header)
    if header:
        _set_cell_shading(cell, "D9EAF7")


def _add_table(
    document: Document,
    headers: list[str],
    rows: Iterable[Iterable[Any]],
    chapter: int,
    table_number: int,
    caption: str,
    font_size: float = 9,
) -> None:
    caption_paragraph = document.add_paragraph()
    caption_paragraph.paragraph_format.keep_with_next = True
    caption_paragraph.paragraph_format.space_before = Pt(4)
    caption_paragraph.paragraph_format.space_after = Pt(3)
    caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption_run = caption_paragraph.add_run(
        f"Table {chapter}.{table_number}: {caption}"
    )
    _set_run_font(caption_run, "Times New Roman", 10, bold=True)

    data_rows = [list(row) for row in rows]
    table = document.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    table.autofit = True
    for index, header in enumerate(headers):
        _set_table_cell(table.rows[0].cells[index], header, True, font_size)
    _set_repeat_table_header(table.rows[0])
    _set_cant_split(table.rows[0])
    for values in data_rows:
        row = table.add_row()
        _set_cant_split(row)
        for index, value in enumerate(values):
            _set_table_cell(row.cells[index], value, False, font_size)


def _add_figure(
    document: Document,
    image: Path,
    chapter: int,
    figure_number: int,
    caption: str,
    width_cm: float = 15.0,
) -> None:
    if not image.is_file():
        raise FileNotFoundError(f"Missing figure required by report: {image}")
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.keep_with_next = True
    paragraph.paragraph_format.space_before = Pt(2)
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.add_run().add_picture(str(image), width=Cm(width_cm))
    caption_paragraph = document.add_paragraph()
    caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption_paragraph.paragraph_format.keep_together = True
    caption_paragraph.paragraph_format.space_before = Pt(0)
    caption_paragraph.paragraph_format.space_after = Pt(5)
    run = caption_paragraph.add_run(f"Fig. {chapter}.{figure_number}: {caption}")
    _set_run_font(run, "Times New Roman", 10, bold=True)


def _add_body(document: Document, text: str, *, keep_with_next: bool = False) -> Any:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    paragraph.paragraph_format.line_spacing = 1.5
    paragraph.paragraph_format.space_after = Pt(6)
    paragraph.paragraph_format.keep_with_next = keep_with_next
    paragraph.add_run(text)
    return paragraph


def _add_subheading(document: Document, title: str) -> None:
    document.add_heading(title, level=2)


def _add_code(document: Document, text: str) -> None:
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.line_spacing = 1.0
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.paragraph_format.keep_together = False
    run = paragraph.add_run(text)
    _set_run_font(run, "Courier New", 10)


def _add_field(paragraph: Any, instruction: str, fallback: str) -> None:
    begin_run = OxmlElement("w:r")
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    begin_run.append(begin)
    paragraph._p.append(begin_run)

    instruction_run = OxmlElement("w:r")
    instruction_text = OxmlElement("w:instrText")
    instruction_text.set(qn("xml:space"), "preserve")
    instruction_text.text = f" {instruction} "
    instruction_run.append(instruction_text)
    paragraph._p.append(instruction_run)

    separate_run = OxmlElement("w:r")
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    separate_run.append(separate)
    paragraph._p.append(separate_run)

    result_run = OxmlElement("w:r")
    result_text = OxmlElement("w:t")
    result_text.text = fallback
    result_run.append(result_text)
    paragraph._p.append(result_run)

    end_run = OxmlElement("w:r")
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    end_run.append(end)
    paragraph._p.append(end_run)


def _enable_field_updates(document: Document) -> None:
    settings = document.settings._element
    update_fields = settings.find(qn("w:updateFields"))
    if update_fields is None:
        update_fields = OxmlElement("w:updateFields")
        settings.append(update_fields)
    update_fields.set(qn("w:val"), "true")


def _configure_document(document: Document) -> None:
    section = document.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.left_margin = Cm(3.0)
    section.right_margin = Cm(2.4)
    section.top_margin = Cm(2.2)
    section.bottom_margin = Cm(2.2)
    section.different_first_page_header_footer = True

    normal = document.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    normal.font.size = Pt(12)
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    normal.paragraph_format.space_after = Pt(6)

    heading1 = document.styles["Heading 1"]
    heading1.font.name = "Times New Roman"
    heading1._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    heading1.font.size = Pt(16)
    heading1.font.bold = True
    heading1.font.color.rgb = RGBColor(0, 0, 0)
    heading1.paragraph_format.page_break_before = True
    heading1.paragraph_format.keep_with_next = True
    heading1.paragraph_format.line_spacing = 1.0
    heading1.paragraph_format.space_before = Pt(0)
    heading1.paragraph_format.space_after = Pt(12)

    heading2 = document.styles["Heading 2"]
    heading2.font.name = "Times New Roman"
    heading2._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    heading2.font.size = Pt(14)
    heading2.font.bold = True
    heading2.font.color.rgb = RGBColor(0, 0, 0)
    heading2.paragraph_format.keep_with_next = True
    heading2.paragraph_format.line_spacing = 1.0
    heading2.paragraph_format.space_before = Pt(8)
    heading2.paragraph_format.space_after = Pt(5)

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.paragraph_format.line_spacing = 1.0
    _add_field(footer, "PAGE", "1")
    section.first_page_footer.paragraphs[0].text = ""
    _enable_field_updates(document)


def _add_cover(document: Document) -> None:
    lines = [
        ("A", 16, False, False),
        ("Mini Project Report", 20, True, False),
        ("on", 14, False, False),
    ]
    for text, size, bold, _ in lines:
        paragraph = document.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.line_spacing = 1.0
        paragraph.paragraph_format.space_after = Pt(4)
        run = paragraph.add_run(text)
        _set_run_font(run, "Times New Roman", size, bold)

    title_paragraph = document.add_paragraph()
    title_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_paragraph.paragraph_format.space_before = Pt(10)
    title_paragraph.paragraph_format.space_after = Pt(10)
    title_run = title_paragraph.add_run(
        '"MACHINE LEARNING BASED NETWORK INTRUSION DETECTION"'
    )
    _set_run_font(title_run, "Times New Roman", 16, True)
    title_run.font.color.rgb = RGBColor(0, 0, 255)

    cover_lines = [
        ("Project Code: CSL-G38", True),
        ("Submitted in partial fulfilment of the requirements of", False),
        ("Cyber Security Lab (7CIT4-22)", True),
        ("Bachelor of Technology - VII Semester", False),
        ("in", False),
        ("Computer Science & Engineering (Internet of Things)", True),
        ("Session 2026-27", False),
    ]
    for text, bold in cover_lines:
        paragraph = document.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.line_spacing = 1.0
        paragraph.paragraph_format.space_after = Pt(3)
        run = paragraph.add_run(text)
        _set_run_font(run, "Times New Roman", 12, bold)

    if LOGO_FILE.is_file():
        logo_paragraph = document.add_paragraph()
        logo_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        logo_paragraph.paragraph_format.space_after = Pt(4)
        logo_paragraph.add_run().add_picture(str(LOGO_FILE), width=Cm(3.2))

    people = document.add_table(rows=1, cols=2)
    people.alignment = WD_TABLE_ALIGNMENT.CENTER
    people.autofit = True
    entries = [
        (
            "Submitted By",
            "Name: Nishant Kumawat\n"
            "Roll No.: 23ESKCY036\n"
            "Class / Batch: [YOUR BATCH]\n"
            "Semester: VII",
        ),
        (
            "Submitted To",
            "Vikash Dhankar\n"
            "Assistant Professor\n"
            "Department of CSE, SKIT, Jaipur",
        ),
    ]
    for cell, (heading, details) in zip(people.rows[0].cells, entries, strict=True):
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        cell.text = ""
        first = cell.paragraphs[0]
        first.alignment = WD_ALIGN_PARAGRAPH.CENTER
        first.paragraph_format.line_spacing = 1.0
        run = first.add_run(heading + "\n")
        _set_run_font(run, "Times New Roman", 12, True)
        details_run = first.add_run(details)
        _set_run_font(details_run, "Times New Roman", 11)

    for text, size, bold in (
        ("Department of Computer Science & Engineering", 12, True),
        (
            "Swami Keshvanand Institute of Technology, Management & Gramothan, Jaipur",
            11,
            False,
        ),
        ("(Affiliated to Rajasthan Technical University, Kota)", 11, False),
        ("November 2026", 12, True),
    ):
        paragraph = document.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.line_spacing = 1.0
        paragraph.paragraph_format.space_after = Pt(3)
        run = paragraph.add_run(text)
        _set_run_font(run, "Times New Roman", size, bold)


def _add_declaration(document: Document) -> None:
    document.add_heading("Candidate's Declaration and Ethical Declaration", level=1)
    _add_body(
        document,
        "I, Nishant Kumawat, Roll No. 23ESKCY036, a student of B.Tech VII Semester, "
        "CSE (IoT), [YOUR BATCH], declare that this mini project is my own original "
        "work completed under the guidance of Vikash Dhankar. Sources used in the "
        "project are cited in this report. The experiments analyze the pre-collected "
        "NSL-KDD dataset offline; no live traffic was captured and no attack was "
        "launched by this project.",
    )
    _add_body(
        document,
        "I further declare that any attack or testing activity associated with this "
        "work was limited to systems owned by me or for which authorization was "
        "provided; no unauthorized system was targeted. Misuse of computer resources "
        "is an offence under the Information Technology Act, 2000. This project "
        "does not provide or claim authorization for operational testing.",
    )
    for text in (
        "Date: ____________________",
        "Place: Jaipur",
        "Signature: ____________________",
        "Nishant Kumawat (23ESKCY036)",
    ):
        paragraph = document.add_paragraph()
        paragraph.paragraph_format.line_spacing = 1.5
        paragraph.add_run(text)


def _make_abstract(results: dict[str, Any]) -> str:
    dataset = results["dataset"]
    binary = results["experiment_a"]
    binary_metrics = binary["best_metrics"]
    multi = results["experiment_b"]
    multi_metrics = multi["best_metrics"]
    hardest_class = min(
        multi_metrics["per_class"].items(), key=lambda item: item[1]["recall"]
    )
    return (
        "This mini project evaluates machine-learning methods for offline network "
        "intrusion detection using the NSL-KDD benchmark. The experiment used "
        f"{dataset['train_rows']:,} official training records and "
        f"{dataset['test_rows']:,} separate official test records, with "
        f"{dataset['feature_count']} predictive features after excluding difficulty "
        "metadata. Imputation, one-hot encoding, and scaling were fit on the "
        "training file; the official test file was reserved for evaluation. Seven "
        "classifiers were compared in a binary benign-versus-attack experiment and "
        "a five-category experiment. In the binary experiment, "
        f"{binary['best_model']} achieved accuracy {_decimal(binary_metrics['accuracy'])}, "
        f"precision {_decimal(binary_metrics['precision'])}, "
        f"recall {_decimal(binary_metrics['recall'])}, F1 {_decimal(binary_metrics['f1'])}, "
        f"ROC-AUC {_decimal(binary_metrics['auc'])}, and false-alarm rate "
        f"{_decimal(binary_metrics['far'])}. In the five-category experiment, "
        f"{multi['best_model']} produced accuracy "
        f"{_decimal(multi_metrics['accuracy'])} and macro F1 "
        f"{_decimal(multi_metrics['f1_macro'])}. The lowest-recall category for "
        f"that model was {hardest_class[0]} at {_decimal(hardest_class[1]['recall'])}. "
        "Measured runtime and feature-importance values, class-wise results, "
        "confusion matrices, receiver-operating-characteristic curves, and "
        "pipeline checks are included. The workflow analyzed pre-collected data "
        "only: it did not capture live traffic, launch attacks, or test unauthorized "
        "systems. These findings describe performance on one historical benchmark, "
        "not proof of effectiveness on present-day networks. Data age, class "
        "imbalance, and changes in traffic distributions are important limits on "
        "generalization."
    )


def _add_toc(document: Document) -> None:
    document.add_heading("Table of Contents", level=1)
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.line_spacing = 1.0
    _add_field(paragraph, 'TOC \\o "1-2" \\h \\z \\u', "Update this field in Word.")


def _attack_category_rows(dataset: dict[str, Any]) -> list[list[str]]:
    categories = dataset["attack_categories"]
    rows = []
    for name in dataset["target_classes"]:
        members = categories.get(name, [])
        rows.append([name, ", ".join(members) if members else "No attack labels"])
    return rows


def _function_rows(source: str) -> list[list[str]]:
    tree = ast.parse(source)
    rows = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            docstring = ast.get_docstring(node)
            summary = docstring.splitlines()[0] if docstring else "Internal pipeline helper."
            rows.append([node.name, summary])
    return rows


def _condense_source(source: str) -> str:
    lines = source.splitlines()
    tree = ast.parse(source)
    feature_group_constant: tuple[int, int] | None = None
    functions: dict[str, ast.FunctionDef] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == "FEATURE_GROUPS"
                for target in node.targets
            ):
                feature_group_constant = (
                    node.lineno,
                    node.end_lineno or node.lineno,
                )
        elif isinstance(node, ast.FunctionDef):
            functions[node.name] = node

    module_end = functions["_environment"].lineno - 1
    module_lines = lines[:module_end]
    if feature_group_constant is not None:
        start, end = feature_group_constant
        module_lines[start - 1 : end] = ["# omitted in printout, see repo"]

    def function_text(name: str) -> str:
        node = functions[name]
        return "\n".join(lines[node.lineno - 1 : node.end_lineno])

    selected = [
        "\n".join(module_lines),
        "# Environment, model, metric, test, and plotting helpers omitted in printout, see repo.",
        "# plotting code omitted in printout, see repo",
        function_text("run_pipeline"),
        function_text("main"),
    ]
    return "\n\n".join(selected)


def _format_binary_rows(results: dict[str, Any]) -> list[list[str]]:
    rows = []
    for name, result in results["experiment_a"]["comparison"].items():
        metrics = result["metrics"]
        rows.append(
            [
                name,
                _decimal(metrics["accuracy"]),
                _decimal(metrics["precision"]),
                _decimal(metrics["recall"]),
                _decimal(metrics["f1"]),
                _decimal(metrics["auc"]),
                _decimal(metrics["far"]),
                _seconds(result["training_seconds"]),
            ]
        )
    return rows


def _format_multiclass_rows(results: dict[str, Any]) -> list[list[str]]:
    rows = []
    for name, result in results["experiment_b"]["comparison"].items():
        metrics = result["metrics"]
        rows.append(
            [
                name,
                _decimal(metrics["accuracy"]),
                _decimal(metrics["precision_macro"]),
                _decimal(metrics["recall_macro"]),
                _decimal(metrics["f1_macro"]),
                _decimal(metrics["f1_weighted"]),
                _decimal(metrics["auc_ovr_macro"]),
                _seconds(result["training_seconds"]),
            ]
        )
    return rows


def _add_front_matter(document: Document, results: dict[str, Any]) -> None:
    _add_cover(document)

    _add_declaration(document)

    document.add_heading("Acknowledgement", level=1)
    _add_body(
        document,
        "I express my sincere gratitude to Vikash Dhankar, Assistant Professor, "
        "Department of Computer Science and Engineering, SKIT, Jaipur, for guidance "
        "and feedback during this mini project. I thank the Cyber Security Lab and "
        "the Department of Computer Science & Engineering for the academic context "
        "and resources that supported this work. I also acknowledge the Canadian "
        "Institute for Cybersecurity and the NSL-KDD dataset contributors for "
        "making the benchmark available for research and education. The reported "
        "measurements were generated by the local experiment pipeline described "
        "in this report.",
    )

    document.add_heading("Abstract", level=1)
    abstract = _make_abstract(results)
    count = _word_count(abstract)
    print(f"Abstract word count: {count}")
    if not 150 <= count <= 200:
        raise ValueError(f"Abstract must be 150-200 words, got {count}.")
    _add_body(document, abstract)

    _add_toc(document)


def _add_chapter_one(document: Document) -> None:
    document.add_heading("Chapter 1 - Introduction", level=1)
    _add_subheading(document, "Background")
    _add_body(
        document,
        "Network intrusion detection examines communication or host activity for "
        "indicators of malicious behavior. Machine-learning classifiers can learn "
        "patterns from labeled records, but their performance depends on data "
        "quality, feature construction, class balance, and the evaluation design. "
        "A benchmark result is evidence about that benchmark and is not, by itself, "
        "evidence of operational security.",
    )
    _add_subheading(document, "Problem statement")
    _add_body(
        document,
        "This project trains and compares classifiers, including Random Forest and "
        "Support Vector Machine models, on NSL-KDD connection records. It evaluates "
        "binary benign-versus-attack classification and the five conventional "
        "benign/attack-family categories, reporting accuracy, precision, recall, "
        "F1, ROC-AUC, and false-alarm behavior from held-out data.",
    )
    _add_subheading(document, "Objectives and scope")
    _add_body(
        document,
        "The objectives are to prepare data without fitting transformations on "
        "the test file, compare seven classifier families, measure per-class as "
        "well as aggregate performance, visualize errors, and produce a repeatable "
        "record of the experiment. The scope is offline analysis of the official "
        "NSL-KDD train and test files. The work does not capture packets, monitor "
        "a live network, block traffic, or establish production readiness.",
    )
    _add_subheading(document, "Syllabus connection and ethics")
    _add_body(
        document,
        "The work is aligned with Cyber Security Lab Experiment 7 - Intrusion "
        "Detection System. The dataset is pre-collected and analyzed locally. No "
        "attack traffic was generated, no live target was contacted, and no "
        "unauthorized system was tested. Dataset provenance and limitations are "
        "disclosed; model predictions are not treated as calibrated risk scores.",
    )
    _add_subheading(document, "Report organization")
    _add_body(
        document,
        "The report introduces IDS concepts and the benchmark, records the tools "
        "and workflow, documents implementation and measured tests, discusses "
        "results and validity limits, and closes with conclusions and future work.",
    )


def _add_chapter_two(document: Document, results: dict[str, Any]) -> None:
    document.add_heading("Chapter 2 - Background and Theory", level=1)
    _add_subheading(document, "Intrusion detection systems")
    _add_body(
        document,
        "An Intrusion Detection System (IDS) identifies activity that may violate "
        "a security policy and raises an alert. An Intrusion Prevention System "
        "(IPS) additionally attempts to block or contain activity. A Network IDS "
        "(NIDS) analyzes network traffic; a Host IDS (HIDS) examines events on "
        "an individual host. Signature detection matches known patterns, while "
        "anomaly detection flags departures from a learned or specified baseline. "
        "The classifier in this project is an offline benchmark experiment, not "
        "an operational IDS or IPS.",
    )
    _add_subheading(document, "Machine learning and NSL-KDD")
    _add_body(
        document,
        "Supervised learning uses labeled examples to estimate a mapping from "
        "traffic features to class labels. NSL-KDD is a cleaned benchmark derived "
        "from KDD Cup 99. Its connection records provide basic, content, "
        "time-based traffic, and host-based traffic attributes. The difficulty "
        "metadata field is not used as a predictor. The family labels group "
        "individual attacks into Denial of Service (DOS), Probe, Remote to Local "
        "(R2L), and User to Root (U2R), alongside Benign traffic.",
    )
    dataset = results["dataset"]
    _add_table(
        document,
        ["Feature group", "Count", "Included attributes"],
        [
            [item["group"], item["count"], ", ".join(item["features"])]
            for item in dataset["feature_groups"]
        ],
        2,
        1,
        "NSL-KDD predictor feature groups",
        font_size=8,
    )
    _add_table(
        document,
        ["Target category", "Associated labels"],
        _attack_category_rows(dataset),
        2,
        2,
        "Target categories represented in the experiment",
        font_size=8,
    )
    _add_subheading(document, "Preprocessing and classifiers")
    _add_body(
        document,
        "Numeric values are median-imputed and standardized; categorical values "
        "are imputed with their most frequent training value and one-hot encoded. "
        "Unknown test categories are handled by the encoder. The fitted "
        "preprocessor is learned from the official training file only. The "
        "comparison includes linear Logistic Regression, Decision Tree, Random "
        "Forest, Linear SVM, distance-weighted K-Nearest Neighbors, Gaussian "
        "Naive Bayes, and Histogram Gradient Boosting.",
    )
    _add_table(
        document,
        ["Classifier", "Recorded settings"],
        [[name, _format_setting(settings)] for name, settings in results["settings"].items()],
        2,
        3,
        "Classifier families and experiment settings",
        font_size=8,
    )
    _add_subheading(document, "Evaluation metrics")
    for formula in (
        "Accuracy = (TP + TN) / (TP + TN + FP + FN)",
        "Precision = TP / (TP + FP)",
        "Recall = TP / (TP + FN)",
        "F1 = 2 × Precision × Recall / (Precision + Recall)",
        "False-alarm rate = FP / (FP + TN)",
        "ROC plots true-positive rate against false-positive rate as the decision threshold varies; AUC is the area under that curve.",
        "Macro averaging takes the unweighted mean across classes; weighted averaging weights each class by its support.",
    ):
        paragraph = document.add_paragraph()
        paragraph.paragraph_format.left_indent = Cm(0.7)
        paragraph.paragraph_format.line_spacing = 1.2
        paragraph.paragraph_format.space_after = Pt(4)
        paragraph.add_run(formula)
    _add_body(
        document,
        "Here TP, TN, FP, and FN denote true positives, true negatives, false "
        "positives, and false negatives. Precision and recall are reported for "
        "the attack-positive binary task; multiclass summaries use macro and "
        "weighted averages alongside per-class measures.",
    )


def _add_chapter_three(document: Document, results: dict[str, Any]) -> None:
    document.add_heading("Chapter 3 - Tools and Environment", level=1)
    _add_body(
        document,
        "The local Python experiment uses pandas and NumPy for tabular data, "
        "scikit-learn for preprocessing and classifiers, and Matplotlib for "
        "figures. The environment details below were recorded by the pipeline "
        "during this run.",
    )
    environment = results["meta"]["environment"]
    env_rows = [
        ["Operating system", environment["os"]],
        ["CPU", environment["cpu"]],
        ["Logical CPU count", environment["cpu_cores"]],
        ["Memory", f"{environment['ram_gib']} GiB" if environment["ram_gib"] else "Unavailable"],
        ["Python", environment["python"]],
    ]
    env_rows.extend(
        [[f"{package} version", package_version] for package, package_version in environment["libraries"].items()]
    )
    _add_table(
        document,
        ["Environment item", "Recorded value"],
        env_rows,
        3,
        1,
        "Run environment captured from the experiment process",
    )
    _add_figure(
        document,
        RESULTS_DIR / "figures" / "fig_workflow.png",
        3,
        1,
        "Measured data-to-results workflow",
        width_cm=15.2,
    )
    _add_body(
        document,
        "The workflow used local copies of the pre-collected NSL-KDD files. It "
        "performed no packet capture, live traffic monitoring, attack generation, "
        "or interaction with an external target.",
    )


def _add_chapter_four(document: Document) -> None:
    document.add_heading("Chapter 4 - Codes", level=1)
    source = SOURCE_FILE.read_text(encoding="utf-8")
    _add_table(
        document,
        ["Function", "Responsibility"],
        _function_rows(source),
        4,
        1,
        "Pipeline functions and responsibilities",
        font_size=8,
    )
    _add_subheading(document, "Pipeline source listing")
    _add_body(
        document,
        "The listing below shows the core execution path used to generate the "
        "measurements. The long feature-group constant, helper definitions, and "
        "plotting implementation are marked as omitted in printout, see repo; "
        "their measured outputs and figures are included in this report.",
    )
    _add_code(document, _condense_source(source))


def _add_chapter_five(document: Document, results: dict[str, Any]) -> None:
    document.add_heading("Chapter 5 - Results and Testing", level=1)
    _add_subheading(document, "Run log excerpt")
    log_lines = (RESULTS_DIR / "run_log.txt").read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()
    _add_code(document, "\n".join(log_lines[:24]))

    dataset = results["dataset"]
    _add_subheading(document, "Dataset statistics")
    _add_table(
        document,
        ["Statistic", "Measured value"],
        [
            ["Official training records", f"{dataset['train_rows']:,}"],
            ["Official held-out test records", f"{dataset['test_rows']:,}"],
            ["Predictor features after metadata exclusion", dataset["feature_count"]],
            ["Raw feature columns before exclusion", dataset["raw_feature_columns"]],
            ["Difficulty field excluded", dataset["difficulty_field_excluded"]],
        ],
        5,
        1,
        "Dataset and predictor counts from results.json",
    )
    distribution_rows = []
    for category in dataset["target_classes"]:
        distribution_rows.append(
            [
                category,
                dataset["class_distribution"]["train"][category],
                dataset["class_distribution"]["test"][category],
            ]
        )
    _add_table(
        document,
        ["Category", "Training records", "Test records"],
        distribution_rows,
        5,
        2,
        "Observed five-category class distribution",
    )
    _add_figure(
        document,
        RESULTS_DIR / "figures" / "fig_class_distribution.png",
        5,
        1,
        "Training and official test class distribution",
        width_cm=14.5,
    )

    _add_subheading(document, "Classifier settings")
    _add_table(
        document,
        ["Classifier", "Settings from the recorded run"],
        [[name, _format_setting(settings)] for name, settings in results["settings"].items()],
        5,
        3,
        "Classifier configurations recorded by the pipeline",
        font_size=8,
    )

    experiment_a = results["experiment_a"]
    _add_subheading(document, "Experiment A - Binary classification")
    _add_body(
        document,
        "The binary target distinguishes Benign records from Attack records. "
        "The following values were computed on the official held-out test file; "
        "training time is the measured fit duration recorded for each classifier.",
    )
    _add_table(
        document,
        ["Classifier", "Accuracy", "Precision", "Recall", "F1", "AUC", "FAR", "Train s"],
        _format_binary_rows(results),
        5,
        4,
        "Binary experiment held-out test metrics",
        font_size=8,
    )
    _add_figure(
        document,
        RESULTS_DIR / "figures" / "fig_binary_comparison.png",
        5,
        2,
        "Binary metric comparison across classifiers",
        width_cm=15,
    )
    _add_figure(
        document,
        RESULTS_DIR / "figures" / "fig_binary_confusion_matrices.png",
        5,
        3,
        "Binary confusion matrices across classifiers",
        width_cm=15,
    )
    _add_figure(
        document,
        RESULTS_DIR / "figures" / "fig_binary_roc.png",
        5,
        4,
        "Binary one-vs-rest ROC curves",
        width_cm=13.2,
    )

    experiment_b = results["experiment_b"]
    _add_subheading(document, "Experiment B - Five-category classification")
    _add_table(
        document,
        [
            "Classifier",
            "Accuracy",
            "Macro P",
            "Macro R",
            "Macro F1",
            "Weighted F1",
            "OvR AUC",
            "Train s",
        ],
        _format_multiclass_rows(results),
        5,
        5,
        "Five-category held-out test comparison",
        font_size=8,
    )
    best_metrics = experiment_b["best_metrics"]
    _add_body(
        document,
        f"The highest macro-F1 classifier was {experiment_b['best_model']} with "
        f"accuracy {_decimal(best_metrics['accuracy'])}, macro precision "
        f"{_decimal(best_metrics['precision_macro'])}, macro recall "
        f"{_decimal(best_metrics['recall_macro'])}, and macro F1 "
        f"{_decimal(best_metrics['f1_macro'])}.",
    )
    _add_table(
        document,
        ["Class", "Precision", "Recall", "F1", "Support"],
        [
            [
                name,
                _decimal(values["precision"]),
                _decimal(values["recall"]),
                _decimal(values["f1"]),
                values["support"],
            ]
            for name, values in best_metrics["per_class"].items()
        ],
        5,
        6,
        f"Per-class results for {experiment_b['best_model']}",
    )
    _add_figure(
        document,
        RESULTS_DIR / "figures" / "fig_five_class_confusion_matrix.png",
        5,
        5,
        f"Five-category confusion matrix for {experiment_b['best_model']}",
        width_cm=13.2,
    )
    _add_figure(
        document,
        RESULTS_DIR / "figures" / "fig_feature_importance.png",
        5,
        6,
        f"Feature importance for {experiment_b['best_model']}",
        width_cm=14,
    )

    _add_subheading(document, "Pipeline test cases")
    _add_table(
        document,
        ["Test", "Check", "Status", "Observed evidence"],
        [
            [case["id"], case["test"], case["status"], case["observed"]]
            for case in results["test_cases"]
        ],
        5,
        7,
        "Executed pipeline checks",
        font_size=8,
    )

    _add_subheading(document, "Generated observations and discussion")
    binary = experiment_a["comparison"]
    lowest_far = min(binary.items(), key=lambda item: item[1]["metrics"]["far"])
    best_binary = experiment_a["best_model"]
    best_binary_metrics = binary[best_binary]["metrics"]
    hardest_class = min(
        best_metrics["per_class"].items(), key=lambda item: item[1]["recall"]
    )
    easiest_class = max(
        best_metrics["per_class"].items(), key=lambda item: item[1]["recall"]
    )
    _add_body(
        document,
        f"In Experiment A, {best_binary} had the highest attack-class F1 "
        f"({_decimal(best_binary_metrics['f1'])}) under the report's selection "
        f"rule. {lowest_far[0]} had the lowest measured false-alarm rate "
        f"({_decimal(lowest_far[1]['metrics']['far'])}), which is a distinct "
        "operational trade-off and does not by itself identify the best overall "
        "classifier.",
    )
    _add_body(
        document,
        f"For the selected five-category model, {hardest_class[0]} was the "
        f"hardest category by recall ({_decimal(hardest_class[1]['recall'])}); "
        f"{easiest_class[0]} had the highest recall "
        f"({_decimal(easiest_class[1]['recall'])}). The per-class table and "
        "confusion matrix show where class errors occur rather than hiding them "
        "inside an aggregate score.",
    )
    first_feature = experiment_b["top_features"][0]
    _add_body(
        document,
        f"The highest-ranked transformed feature under the selected model's "
        f"recorded importance method was {first_feature['feature']} "
        f"({_decimal(first_feature['importance'])}). Feature importance is "
        "model-dependent and describes this fitted benchmark model; it does not "
        "establish causality or a universal indicator of malicious behavior.",
    )

    _add_subheading(document, "Security analysis")
    binary_matrix = binary[best_binary]["confusion_matrix"]
    true_negative, false_positive = binary_matrix[0]
    false_negative, true_positive = binary_matrix[1]
    _add_body(
        document,
        f"For the selected binary classifier, the held-out confusion matrix "
        f"contains {false_positive:,} benign records predicted as attacks and "
        f"{false_negative:,} attacks predicted as benign, alongside "
        f"{true_negative:,} true negatives and {true_positive:,} true positives. "
        "False alarms can create analyst workload, while missed attacks can "
        "leave malicious activity undetected. These trade-offs require a "
        "deployment-specific threshold and validation process; this experiment "
        "does not authorize or recommend an operational deployment.",
    )
    _add_subheading(document, "Threats to validity")
    knn_limit = results["settings"]["K-Nearest Neighbors"].get("training_row_limit")
    knn_statement = (
        f" K-Nearest Neighbors used a stratified training subset capped at "
        f"{knn_limit:,} records as recorded in the settings."
        if knn_limit
        else ""
    )
    _add_body(
        document,
        "NSL-KDD is a historical, curated benchmark and may not represent current "
        "network services, collection procedures, or adversary behavior. A "
        "single official test file does not establish performance across sites "
        "or time. Class imbalance makes macro and per-class measures important; "
        "preprocessing was fit on training data, but the benchmark's feature and "
        "label design still constrains external validity."
        + knn_statement
        + " No live traffic or independent modern dataset was evaluated.",
    )


def _add_chapter_six(document: Document, results: dict[str, Any]) -> None:
    document.add_heading("Chapter 6 - Conclusion and Future Scope", level=1)
    binary = results["experiment_a"]
    multi = results["experiment_b"]
    binary_metrics = binary["best_metrics"]
    multi_metrics = multi["best_metrics"]
    _add_body(
        document,
        f"The experiment processed {results['dataset']['train_rows']:,} official "
        f"training records and {results['dataset']['test_rows']:,} held-out test "
        f"records. The best binary classifier by attack F1 was {binary['best_model']} "
        f"(F1 {_decimal(binary_metrics['f1'])}, AUC {_decimal(binary_metrics['auc'])}); "
        f"the highest macro-F1 five-category classifier was {multi['best_model']} "
        f"(macro F1 {_decimal(multi_metrics['f1_macro'])}). The results establish "
        "a reproducible benchmark comparison, not live-network detection "
        "effectiveness.",
    )
    _add_subheading(document, "Future scope")
    _add_body(
        document,
        "Future work can evaluate independent and more recent labeled datasets, "
        "capture-level or temporal holdouts, drift, calibration, class-specific "
        "operating thresholds, and false-positive workload. Any live testing "
        "would require explicit authorization, controlled infrastructure, and "
        "separate safety review. The current report makes no production-readiness "
        "claim.",
    )


def _add_references(document: Document) -> None:
    document.add_heading("References", level=1)
    references = [
        "Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. (2009). A Detailed "
        "Analysis of the KDD CUP 99 Data Set. IEEE Symposium on Computational "
        "Intelligence for Security and Defense Applications (CISDA).",
        "Canadian Institute for Cybersecurity. NSL-KDD dataset page. "
        "https://www.unb.ca/cic/datasets/nsl.html",
        "Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. "
        "Journal of Machine Learning Research, 12, 2825-2830.",
        "Breiman, L. (2001). Random Forests. Machine Learning, 45, 5-32.",
        "Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine "
        "Learning, 20, 273-297.",
        "Denning, D. E. (1987). An Intrusion-Detection Model. IEEE Transactions "
        "on Software Engineering, SE-13(2), 222-232.",
        "National Institute of Standards and Technology. (2007). Guide to "
        "Intrusion Detection and Prevention Systems (IDPS), Special Publication "
        "800-94.",
    ]
    for item in references:
        paragraph = document.add_paragraph()
        paragraph.paragraph_format.left_indent = Cm(0.6)
        paragraph.paragraph_format.first_line_indent = Cm(-0.6)
        paragraph.paragraph_format.line_spacing = 1.3
        paragraph.paragraph_format.space_after = Pt(7)
        paragraph.add_run(item)


def build_report() -> Path:
    results_path = RESULTS_DIR / "results.json"
    if not results_path.is_file():
        raise FileNotFoundError(
            f"Missing {results_path}; run src.ids_pipeline before building the report."
        )
    results = json.loads(results_path.read_text(encoding="utf-8"))
    _validate_results(results)

    document = Document()
    _configure_document(document)
    _add_front_matter(document, results)
    _add_chapter_one(document)
    _add_chapter_two(document, results)
    _add_chapter_three(document, results)
    _add_chapter_four(document)
    _add_chapter_five(document, results)
    _add_chapter_six(document, results)
    _add_references(document)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    document.save(OUTPUT_DOCX)
    print(f"Wrote {OUTPUT_DOCX}")
    return OUTPUT_DOCX


if __name__ == "__main__":
    build_report()
