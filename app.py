from pathlib import Path
from uuid import uuid4

from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from app.services.pdf_extractor import extract_pdf_text
from app.services.section_detector import detect_sections
from app.services.research_extractor import extract_research_insights
from app.services.summarizer import generate_summaries
from app.services.metadata_extractor import extract_metadata


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
ALLOWED_EXTENSIONS = {"pdf"}
MAX_UPLOAD_SIZE = 16 * 1024 * 1024


def _non_overwriting_path(directory: Path, filename: str) -> Path:
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    return directory / f"{candidate.stem}-{uuid4().hex[:8]}{candidate.suffix}"


def create_app() -> Flask:
    flask_app = Flask(__name__)
    flask_app.config.update(
        SECRET_KEY="researchmate-development-key",
        MAX_CONTENT_LENGTH=MAX_UPLOAD_SIZE,
        UPLOAD_FOLDER=str(UPLOAD_DIR),
    )
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    @flask_app.get("/")
    def index():
        return render_template("index.html")

    @flask_app.post("/upload")
    def upload():
        uploaded_file = request.files.get("paper")

        if uploaded_file is None or not uploaded_file.filename:
            flash("Please select a PDF file.", "error")
            return redirect(url_for("index"))

        filename = secure_filename(uploaded_file.filename)
        if not filename or Path(filename).suffix.lower().lstrip(".") not in ALLOWED_EXTENSIONS:
            flash("Only PDF files are supported.", "error")
            return redirect(url_for("index"))

        saved_path = _non_overwriting_path(UPLOAD_DIR, filename)
        uploaded_file.save(saved_path)

        try:
            extraction = extract_pdf_text(saved_path)
        except ValueError as error:
            saved_path.unlink(missing_ok=True)
            flash(str(error), "error")
            return redirect(url_for("index"))

        sections = detect_sections(extraction["pages"]) if extraction["processed"] else []
        metadata = extract_metadata(
            extraction["pages"],
            sections,
            extraction.get("document_metadata"),
        )
        insights = extract_research_insights(sections) if sections else {}
        return render_template(
            "result.html",
            filename=filename,
            page_count=extraction["page_count"],
            text=extraction["text"],
            text_length=extraction["text_length"],
            processed=extraction["processed"],
            extraction_status=extraction["extraction_status"],
            sections=sections,
            metadata=metadata,
            insights=insights,
            summaries=generate_summaries(sections, insights) if sections else {},
        )

    return flask_app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
