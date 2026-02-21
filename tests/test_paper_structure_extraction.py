"""Tests for paper section detection and section-aware splitting."""
import pytest
from src.ingestion.splitter import detect_sections, split_text_with_sections


SAMPLE_PAPER_TEXT = """\
Abstract
This paper presents a novel approach to transformer architectures.
We propose a new attention mechanism that reduces computational cost.

1. Introduction
Transformers have revolutionized NLP since their introduction in 2017.
The original transformer model uses quadratic self-attention which limits scalability.

2. Methods
We propose a modified attention mechanism with linear complexity.
Our approach replaces the softmax attention with a kernel-based approximation.

3. Results
Our method achieves state-of-the-art on GLUE benchmark.
We observe a 2x speedup compared to standard transformers.

4. Conclusion
We have shown that linear attention can match quadratic attention.
Future work includes extending this to vision transformers.

References
[1] Vaswani et al. Attention Is All You Need. 2017.
[2] Devlin et al. BERT: Pre-training of Deep Bidirectional Transformers. 2019."""


class TestDetectSections:
    def test_detects_standard_sections(self):
        sections = detect_sections(SAMPLE_PAPER_TEXT)
        types = [s[1] for s in sections]
        assert "abstract" in types
        assert "introduction" in types
        assert "methods" in types
        assert "results" in types
        assert "conclusion" in types
        assert "references" in types

    def test_returns_correct_number_of_sections(self):
        sections = detect_sections(SAMPLE_PAPER_TEXT)
        assert len(sections) == 6

    def test_returns_empty_for_unstructured_text(self):
        text = "Just some random text without any section headers. " * 20
        sections = detect_sections(text)
        assert sections == []

    def test_handles_uppercase_headers(self):
        text = "ABSTRACT\nSome content about the paper.\n\nINTRODUCTION\nMore content here."
        sections = detect_sections(text)
        types = [s[1] for s in sections]
        assert "abstract" in types
        assert "introduction" in types

    def test_handles_numbered_headers(self):
        text = "1 Introduction\nContent.\n\n2 Methods\nMore content."
        sections = detect_sections(text)
        types = [s[1] for s in sections]
        assert "introduction" in types
        assert "methods" in types

    def test_handles_dotted_numbered_headers(self):
        text = "1. Introduction\nContent.\n\n2. Methodology\nMore content."
        sections = detect_sections(text)
        types = [s[1] for s in sections]
        assert "introduction" in types
        assert "methods" in types

    def test_section_boundaries_cover_full_text(self):
        sections = detect_sections(SAMPLE_PAPER_TEXT)
        if sections:
            # Last section should end at the text length
            assert sections[-1][3] == len(SAMPLE_PAPER_TEXT)
            # Each section end should be the next section's start
            for i in range(len(sections) - 1):
                assert sections[i][3] == sections[i + 1][2]

    def test_ignores_long_lines(self):
        """Lines longer than 80 chars should not be treated as section headers."""
        text = "Introduction " + "x" * 100 + "\nSome content."
        sections = detect_sections(text)
        assert sections == []

    def test_related_work_maps_to_introduction(self):
        text = "Related Work\nSurvey of prior approaches."
        sections = detect_sections(text)
        assert len(sections) == 1
        assert sections[0][1] == "introduction"

    def test_discussion_maps_to_results(self):
        text = "Results and Discussion\nWe found significant improvements."
        sections = detect_sections(text)
        assert len(sections) == 1
        assert sections[0][1] == "results"


class TestSplitTextWithSections:
    def test_returns_documents_with_chunk_type(self):
        docs = split_text_with_sections(
            SAMPLE_PAPER_TEXT,
            metadata={"source": "test.pdf", "title": "Test Paper", "page": 0},
        )
        assert len(docs) > 0
        assert all("chunk_type" in d.metadata for d in docs)
        assert all("section" in d.metadata for d in docs)

    def test_chunk_types_are_valid(self):
        docs = split_text_with_sections(
            SAMPLE_PAPER_TEXT,
            metadata={"source": "test.pdf", "title": "Test Paper", "page": 0},
        )
        valid_types = {"abstract", "introduction", "methods", "results", "conclusion", "references", "general"}
        for doc in docs:
            assert doc.metadata["chunk_type"] in valid_types

    def test_fallback_to_general_on_unstructured(self):
        plain_text = "No sections here, just plain text about machine learning. " * 50
        docs = split_text_with_sections(
            plain_text,
            metadata={"source": "test.pdf", "title": "Test", "page": 0},
        )
        assert len(docs) > 0
        assert all(d.metadata["chunk_type"] == "general" for d in docs)
        assert all(d.metadata["section"] == "" for d in docs)

    def test_preserves_base_metadata(self):
        docs = split_text_with_sections(
            SAMPLE_PAPER_TEXT,
            metadata={"source": "test.pdf", "title": "Test Paper", "page": 0, "total_pages": 5},
        )
        for doc in docs:
            assert doc.metadata["source"] == "test.pdf"
            assert doc.metadata["title"] == "Test Paper"
            assert doc.metadata["total_pages"] == 5

    def test_empty_text_returns_empty(self):
        docs = split_text_with_sections("", metadata={"source": "test.pdf"})
        assert docs == []

    def test_respects_chunk_size(self):
        docs = split_text_with_sections(
            SAMPLE_PAPER_TEXT,
            metadata={"source": "test.pdf"},
            chunk_size=500,
            chunk_overlap=50,
        )
        for doc in docs:
            # Chunks should generally respect chunk_size (may slightly exceed due to splitter behavior)
            assert len(doc.page_content) <= 600  # Allow some margin

    def test_abstract_chunks_have_abstract_type(self):
        docs = split_text_with_sections(
            SAMPLE_PAPER_TEXT,
            metadata={"source": "test.pdf"},
        )
        abstract_docs = [d for d in docs if d.metadata["chunk_type"] == "abstract"]
        assert len(abstract_docs) > 0
        # Abstract content should be about the paper's contribution
        for doc in abstract_docs:
            assert "transformer" in doc.page_content.lower() or "abstract" in doc.page_content.lower()
