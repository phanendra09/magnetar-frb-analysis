"""Extract PDF pages as images and preview text for manual review."""

from pathlib import Path

import fitz

repo_root = Path(__file__).resolve().parent
pdf_path = repo_root / "Magnetar.pdf"
output_dir = repo_root / "pdf_review"
output_dir.mkdir(exist_ok=True)

doc = fitz.open(pdf_path)
print(f"PDF: {pdf_path}")
print(f"Pages: {doc.page_count}")
print(f"Title: {doc.metadata.get('title', 'N/A')}")
print(f"Author: {doc.metadata.get('author', 'N/A')}")
print(f"Producer: {doc.metadata.get('producer', 'N/A')}")
print()

for page_num in range(doc.page_count):
    page = doc[page_num]
    pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
    img_path = output_dir / f"paper_page_{page_num + 1}.png"
    pix.save(img_path)
    print(f"Page {page_num + 1}: saved {img_path} ({pix.width}x{pix.height})")

print("\n" + "=" * 60)
print("TEXT CONTENT REVIEW")
print("=" * 60)
for page_num in range(doc.page_count):
    page = doc[page_num]
    text = page.get_text("text")
    print(f"\n--- PAGE {page_num + 1} ---")
    print(text[:500])
    print(f"... [{len(text)} total chars on this page]")

doc.close()
