import os
import json
import pytesseract
from pdf2image import convert_from_path
from langdetect import detect_langs
from googletrans import Translator


def detect_languages(text):
    """Detect dominant languages in a string."""
    try:
        langs = detect_langs(text)
        return [str(lang.lang) for lang in langs]
    except Exception:
        return ["unknown"]


def translate_text(text, target_lang):
    """Translate text to target language (en or mr)."""
    translator = Translator()
    try:
        return translator.translate(text, dest=target_lang).text
    except Exception:
        return text  # fallback if translation fails


def process_pdf(pdf_path, output_json="output.json"):
    # Convert all PDF pages to images
    images = convert_from_path(pdf_path)

    result = {"pages": {}}

    for i, img in enumerate(images, start=1):
        # OCR on each page
        text = pytesseract.image_to_string(img, lang="mar+eng").strip()

        if not text:
            continue

        langs = detect_languages(text)
        page_entry = {
            "original_text": text,
            "detected_languages": langs,
            "translations": {}
        }

        # Translation rules
        if "mr" in langs and "en" not in langs:
            page_entry["translations"]["english"] = translate_text(text, "en")

        elif "en" in langs and "mr" not in langs:
            page_entry["translations"]["marathi"] = translate_text(text, "mr")

        elif "en" in langs and "mr" in langs:
            page_entry["translations"]["english"] = translate_text(text, "en")
            page_entry["translations"]["marathi"] = translate_text(text, "mr")

        result["pages"][f"page_{i}"] = page_entry

    # Save JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"✅ OCR & translation completed. Processed {len(result['pages'])} pages. Saved to {output_json}")


if __name__ == "__main__":
    pdf_path = input("Enter PDF path: ").strip()
    process_pdf(pdf_path)
