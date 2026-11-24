"""
Validate the GreenDC corpus, PDFs and catalogue, and report whether any
updates (rebuilds, re-downloads, integrations) are required.

Usage:
    python validate_corpus.py              # offline checks
    python validate_corpus.py --deep       # also open every PDF and compare text
    python validate_corpus.py --online     # also query OpenAlex for retractions
    python validate_corpus.py --report data/validation_report.json

Exit code is 0 when there are no ERRORs, 1 otherwise (warnings don't fail).
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

DATA_DIR = "data"
CORPUS_FILE = os.path.join(DATA_DIR, "corpus.jsonl")
CATALOGUE_FILE = os.path.join(DATA_DIR, "documents_catalogue.json")
NEW_DOCS_FILE = os.path.join(DATA_DIR, "new_documents.json")
PDF_FOLDER = os.path.join(DATA_DIR, "pdfs")
NEW_PDF_FOLDER = os.path.join(DATA_DIR, "new_pdfs")

REQUIRED_FIELDS = ["document_id", "openalex_id", "title", "year",
                   "authors", "source", "pdf_url", "full_text"]
MIN_TEXT_CHARS = 1000
MIN_YEAR, MAX_YEAR = 1990, 2026

# Terms that indicate the document is on-topic for a green data centre corpus.
DOMAIN_TERMS = [
    "data center", "data centre", "datacenter", "datacentre", "cloud",
    "server", "energy", "cooling", "carbon", "water", "power usage",
    "pue", "gpu", "hpc", "sustainab", "emission", "renewable",
]


class Report:
    def __init__(self):
        self.errors, self.warnings, self.info = [], [], []
        self.actions = []

    def error(self, msg): self.errors.append(msg)
    def warn(self, msg): self.warnings.append(msg)
    def note(self, msg): self.info.append(msg)
    def action(self, msg):
        if msg not in self.actions:
            self.actions.append(msg)

    def print(self):
        for label, items in (("INFO", self.info), ("WARNING", self.warnings),
                             ("ERROR", self.errors)):
            if items:
                print(f"\n=== {label} ({len(items)}) ===")
                for item in items:
                    print(" -", item)
        print("\n=== UPDATES REQUIRED ===")
        if self.actions:
            for a in self.actions:
                print(" *", a)
        else:
            print(" None. Corpus, PDFs and catalogue are consistent.")
        print(f"\nSummary: {len(self.errors)} errors, "
              f"{len(self.warnings)} warnings")


def norm_title(title):
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def text_quality(text):
    """Return (printable_ratio, replacement_char_count, alpha_ratio)."""
    if not text:
        return 0.0, 0, 0.0
    printable = sum(ch.isprintable() or ch in "\n\t" for ch in text)
    alpha = sum(ch.isalpha() for ch in text)
    return printable / len(text), text.count("�"), alpha / len(text)


# ---------------------------------------------------------------- corpus
def check_corpus(rep):
    docs = []
    if not os.path.exists(CORPUS_FILE):
        rep.error(f"{CORPUS_FILE} not found")
        return docs

    with open(CORPUS_FILE, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                rep.warn(f"corpus line {n}: blank line")
                continue
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError as e:
                rep.error(f"corpus line {n}: invalid JSON ({e})")

    rep.note(f"corpus.jsonl: {len(docs)} documents")

    ids = Counter(d.get("document_id") for d in docs)
    oa = Counter(d.get("openalex_id") for d in docs if d.get("openalex_id"))
    dois = Counter(d.get("doi").lower() for d in docs if d.get("doi"))
    titles = defaultdict(list)
    no_license = []

    for d in docs:
        did = d.get("document_id") or "<no id>"
        missing = [k for k in REQUIRED_FIELDS if not d.get(k)]
        if missing:
            rep.warn(f"{did}: missing/empty fields {missing}")
        if not re.fullmatch(r"GD\d+", did):
            rep.error(f"{did}: document_id does not match GD<number>")

        year = d.get("year")
        if not isinstance(year, int) or not MIN_YEAR <= year <= MAX_YEAR:
            rep.warn(f"{did}: suspicious year {year!r}")
        if not d.get("license"):
            no_license.append(did)

        text = d.get("full_text") or ""
        if len(text) < MIN_TEXT_CHARS:
            rep.error(f"{did}: full_text only {len(text)} chars")
        printable, repl, alpha = text_quality(text)
        if printable < 0.95 or repl > 50 or alpha < 0.5:
            rep.warn(f"{did}: poor text extraction (printable={printable:.2f},"
                     f" alpha={alpha:.2f}, U+FFFD={repl}) - may need OCR")

        sample = ((d.get("title") or "") + " " + text[:20000]).lower()
        hits = sum(term in sample for term in DOMAIN_TERMS)
        if hits < 3:
            rep.warn(f"{did}: possibly off-topic ({hits} domain terms) - "
                     f"\"{(d.get('title') or '')[:70]}\"")

        titles[norm_title(d.get("title"))].append(did)

    if no_license:
        rep.warn(f"{len(no_license)} documents have no license recorded: "
                 f"{no_license[0]}..{no_license[-1]}")

    for k, c in ids.items():
        if c > 1:
            rep.error(f"duplicate document_id {k} ({c}x)")
    for k, c in oa.items():
        if c > 1:
            rep.error(f"duplicate openalex_id {k} ({c}x)")
    for k, c in dois.items():
        if c > 1:
            rep.error(f"duplicate DOI {k} ({c}x)")
    for t, group in titles.items():
        if t and len(group) > 1:
            rep.warn(f"duplicate title in {group}: \"{t[:70]}\"")

    # Gaps in the GD numbering are fine but worth knowing about.
    nums = sorted(int(i[2:]) for i in ids if i and re.fullmatch(r"GD\d+", i))
    if nums:
        gaps = sorted(set(range(1, nums[-1] + 1)) - set(nums))
        if gaps:
            rep.note(f"ID gaps (not an error): {gaps[:20]}"
                     f"{' ...' if len(gaps) > 20 else ''}")

    if any(c > 1 for c in ids.values()) or any(c > 1 for c in oa.values()):
        rep.action("De-duplicate corpus.jsonl, then re-run build_catalog.py")
    return docs


# ------------------------------------------------------------------ pdfs
def check_pdfs(rep, docs, deep):
    if not os.path.isdir(PDF_FOLDER):
        rep.error(f"{PDF_FOLDER} not found")
        return
    on_disk = {f[:-4] for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")}
    expected = {d["document_id"] for d in docs if d.get("document_id")}

    missing = sorted(expected - on_disk, key=lambda x: int(x[2:]))
    orphans = sorted(on_disk - expected)
    rep.note(f"pdfs/: {len(on_disk)} files")
    if missing:
        rep.error(f"{len(missing)} corpus documents have no GD<n>.pdf: "
                  f"{missing[:10]}{' ...' if len(missing) > 10 else ''}")
    # update_catalog.py saves PDFs under their title; map them back to GD ids.
    renames = {}
    if os.path.exists(NEW_DOCS_FILE):
        with open(NEW_DOCS_FILE, encoding="utf-8") as f:
            for n in json.load(f):
                stem = os.path.basename(n.get("document_path") or "")[:-4]
                if stem in orphans and n.get("document_id") in missing:
                    renames[stem] = n["document_id"]
    if renames:
        rep.warn(f"{len(renames)} PDFs are saved under their title instead "
                 f"of GD<n>.pdf (e.g. {next(iter(renames))}.pdf -> "
                 f"{next(iter(renames.values()))}.pdf)")
        rep.action(f"Rename {len(renames)} title-named PDFs in data/pdfs to "
                   f"GD<n>.pdf (mapping in new_documents.json)")
    other = [o for o in orphans if o not in renames]
    if other:
        rep.warn(f"{len(other)} PDFs not in corpus: {other}")

    for did in sorted(expected & on_disk):
        path = os.path.join(PDF_FOLDER, did + ".pdf")
        with open(path, "rb") as f:
            head = f.read(5)
        if head == b"versi":  # git-lfs pointer file
            rep.error(f"{did}: PDF is a Git LFS pointer - run `git lfs pull`")
        elif head != b"%PDF-":
            rep.error(f"{did}: file is not a PDF (starts with {head!r})")
        elif os.path.getsize(path) < 10_000:
            rep.warn(f"{did}: PDF is very small ({os.path.getsize(path)} B)")

    still_missing = [m for m in missing if m not in renames.values()]
    if still_missing:
        rep.action(f"Re-download {len(still_missing)} missing PDFs "
                   f"(pdf_url in corpus.jsonl)")

    if not deep:
        return
    try:
        import fitz
    except ImportError:
        rep.warn("--deep requested but PyMuPDF (fitz) is not installed")
        return
    fitz.TOOLS.mupdf_display_errors(False)
    fitz.TOOLS.mupdf_display_warnings(False)
    by_id = {d["document_id"]: d for d in docs}
    for did in sorted(expected & on_disk, key=lambda x: int(x[2:])):
        try:
            with fitz.open(os.path.join(PDF_FOLDER, did + ".pdf")) as pdf:
                pages = pdf.page_count
                text = "".join(p.get_text() + "\n" for p in pdf)
        except Exception as e:
            rep.error(f"{did}: PDF cannot be opened ({e})")
            continue
        stored = by_id[did].get("full_text") or ""
        if pages == 0:
            rep.error(f"{did}: PDF has 0 pages")
        if stored and abs(len(text) - len(stored)) / len(stored) > 0.05:
            rep.warn(f"{did}: PDF text ({len(text)} chars) differs from "
                     f"corpus full_text ({len(stored)} chars) - PDF changed?")


# ------------------------------------------------------------- catalogue
def check_catalogue(rep, docs):
    if not os.path.exists(CATALOGUE_FILE):
        rep.error(f"{CATALOGUE_FILE} not found")
        rep.action("Run build_catalog.py")
        return
    with open(CATALOGUE_FILE, encoding="utf-8") as f:
        try:
            cat = json.load(f)
        except json.JSONDecodeError as e:
            rep.error(f"catalogue is not valid JSON ({e})")
            rep.action("Run build_catalog.py")
            return

    rep.note(f"documents_catalogue.json: {len(cat)} documents")
    corpus_ids = {d["document_id"] for d in docs if d.get("document_id")}
    cat_ids = set(cat)
    stale = False

    if corpus_ids - cat_ids:
        rep.error(f"in corpus but not catalogue: "
                  f"{sorted(corpus_ids - cat_ids)}")
        stale = True
    if cat_ids - corpus_ids:
        rep.error(f"in catalogue but not corpus: "
                  f"{sorted(cat_ids - corpus_ids)}")
        stale = True

    null_paths, bad_paths, mismatched = [], [], []
    topic_counts = Counter()
    for d in docs:
        entry = cat.get(d.get("document_id"))
        if not entry:
            continue
        meta = entry.get("metadata", {})
        if entry.get("full_text") != d.get("full_text") or \
                meta.get("name") != (d.get("title") or "Unknown title"):
            mismatched.append(d["document_id"])
        path = meta.get("document_path")
        if path is None:
            null_paths.append(d["document_id"])
        elif not os.path.exists(path):
            bad_paths.append(d["document_id"])
        topic_counts.update(meta.get("topics", []))

    if mismatched:
        rep.error(f"{len(mismatched)} catalogue entries differ from corpus "
                  f"(title/full_text): {mismatched[:15]}")
        stale = True
    if null_paths:
        rep.warn(f"{len(null_paths)} catalogue entries have document_path="
                 f"null although the PDF may exist: {null_paths[:15]}"
                 f"{' ...' if len(null_paths) > 15 else ''}")
        stale = True
    if bad_paths:
        rep.error(f"document_path points to missing file: {bad_paths}")
        stale = True

    if os.path.getmtime(CORPUS_FILE) > os.path.getmtime(CATALOGUE_FILE) + 1:
        rep.warn("corpus.jsonl is newer than documents_catalogue.json")
        stale = True

    rep.note("topic distribution: " + ", ".join(
        f"{t}={c}" for t, c in topic_counts.most_common()))
    if stale:
        rep.action("Re-run build_catalog.py to regenerate the catalogue")


# --------------------------------------------------- pending new documents
def check_new_documents(rep, docs):
    if not os.path.exists(NEW_DOCS_FILE):
        return
    with open(NEW_DOCS_FILE, encoding="utf-8") as f:
        new_docs = json.load(f)
    corpus_oa = {d.get("openalex_id") for d in docs}
    corpus_ids = {d.get("document_id") for d in docs}
    status = Counter((n.get("download_status"), n.get("review_status"))
                     for n in new_docs)
    rep.note("new_documents.json status (download, review): " +
             ", ".join(f"{k}={v}" for k, v in status.items()))

    pending = []
    for n in new_docs:
        integrated = n.get("download_status") == "integrated"
        keep = n.get("review_status") in (None, "keep")
        in_corpus = n.get("openalex_id") in corpus_oa
        if integrated and not in_corpus:
            rep.error(f"marked integrated but not in corpus: "
                      f"{n.get('document_id')} \"{n.get('title', '')[:60]}\"")
        if integrated and n.get("document_id") not in corpus_ids:
            rep.warn(f"integrated doc id {n.get('document_id')} not in corpus")
        if not integrated and keep and not in_corpus:
            pending.append(n.get("title", "")[:60])
    if pending:
        rep.warn(f"{len(pending)} new documents not yet integrated")
        rep.action(f"Review/integrate {len(pending)} pending entries in "
                   f"new_documents.json, then re-run build_catalog.py")

    if os.path.isdir(NEW_PDF_FOLDER) and os.listdir(NEW_PDF_FOLDER):
        rep.warn(f"{len(os.listdir(NEW_PDF_FOLDER))} files waiting in "
                 f"{NEW_PDF_FOLDER}")


# --------------------------------------------------------- online checks
def check_online(rep, docs):
    try:
        import requests
    except ImportError:
        rep.warn("--online requested but requests is not installed")
        return
    ids = [d["openalex_id"].rsplit("/", 1)[-1] for d in docs
           if d.get("openalex_id")]
    by_oa = {d["openalex_id"].rsplit("/", 1)[-1]: d for d in docs
             if d.get("openalex_id")}
    seen = set()
    for i in range(0, len(ids), 50):
        batch = ids[i:i + 50]
        try:
            r = requests.get(
                "https://api.openalex.org/works",
                params={"filter": "openalex_id:" + "|".join(batch),
                        "per_page": 50,
                        "select": "id,is_retracted,open_access,"
                                  "publication_year,title"},
                headers={"User-Agent": "GreenDCResearchProject/1.0"},
                timeout=30)
            r.raise_for_status()
        except Exception as e:
            rep.warn(f"OpenAlex request failed: {e}")
            return
        for w in r.json().get("results", []):
            wid = w["id"].rsplit("/", 1)[-1]
            seen.add(wid)
            d = by_oa.get(wid)
            if not d:
                continue
            if w.get("is_retracted"):
                rep.error(f"{d['document_id']}: paper is RETRACTED")
                rep.action("Remove retracted papers from the corpus")
            if not (w.get("open_access") or {}).get("is_oa"):
                rep.warn(f"{d['document_id']}: no longer listed as open "
                         f"access - check licence")
            if w.get("publication_year") != d.get("year"):
                rep.warn(f"{d['document_id']}: year changed "
                         f"{d.get('year')} -> {w.get('publication_year')}")
    missing = set(ids) - seen
    if missing:
        rep.warn(f"{len(missing)} OpenAlex IDs no longer resolve "
                 f"(merged/deleted): {sorted(missing)[:10]}")
    rep.note(f"OpenAlex: checked {len(seen)} works online")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--deep", action="store_true",
                    help="open every PDF and compare its text to the corpus")
    ap.add_argument("--online", action="store_true",
                    help="check OpenAlex for retractions/metadata changes")
    ap.add_argument("--report", help="write the results to a JSON file")
    args = ap.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    rep = Report()
    docs = check_corpus(rep)
    check_pdfs(rep, docs, args.deep)
    check_catalogue(rep, docs)
    check_new_documents(rep, docs)
    if args.online:
        check_online(rep, docs)
    rep.print()

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump({"errors": rep.errors, "warnings": rep.warnings,
                       "info": rep.info, "actions": rep.actions}, f, indent=2)
        print("Report written to", args.report)
    sys.exit(1 if rep.errors else 0)


if __name__ == "__main__":
    main()
