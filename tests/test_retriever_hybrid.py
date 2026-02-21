"""Tests for hybrid BM25 + vector retrieval."""
import pytest
from unittest.mock import Mock, MagicMock, patch
from langchain_core.documents import Document


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            page_content="Gradient-based learning applied to document recognition using neural networks.",
            metadata={
                "title": "Gradient-based learning applied to document recognition",
                "authors": "LeCun et al.",
                "page": 1,
                "source": "doc1.pdf"
            }
        ),
        Document(
            page_content="Methods for robust learning with noisy labels in computer vision.",
            metadata={
                "title": "Wang_Symmetric_Cross_Entropy_for_Robust_Learning_With_Noisy_Labels",
                "authors": "Wang et al.",
                "page": 8,
                "source": "doc2.pdf"
            }
        ),
        Document(
            page_content="Cross-modal retrieval using deep learning techniques.",
            metadata={
                "title": "Hu_Learning_Cross-Modal_Retrieval_With_Noisy_Labels",
                "authors": "Hu et al.",
                "page": 8,
                "source": "doc3.pdf"
            }
        ),
    ]


def test_create_retriever_vector_only():
    """Test creating a vector-only retriever."""
    from src.core.retriever import create_retriever

    # Mock vectorstore
    mock_vs = Mock()
    mock_retriever = Mock()
    mock_vs.as_retriever.return_value = mock_retriever

    # Create vector-only retriever
    result = create_retriever(mock_vs, use_hybrid=False)

    assert result == mock_retriever
    mock_vs.as_retriever.assert_called_once_with(search_kwargs={"k": 5})


def test_create_retriever_hybrid_requires_documents():
    """Test that hybrid retriever gracefully falls back when documents are missing."""
    from src.core.retriever import create_retriever
    import warnings

    mock_vs = Mock()
    mock_vs.as_retriever.return_value = Mock()

    # Should warn but not raise an error (graceful fallback)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = create_retriever(mock_vs, documents=None, use_hybrid=True)
        
        # Should have warned
        assert len(w) == 1
        assert "Hybrid retrieval requested" in str(w[0].message)
        assert "Falling back to vector-only" in str(w[0].message)
    
    # Should return vector retriever (fallback)
    assert result is not None


@patch('src.core.retriever.BM25Retriever')
@patch('src.core.retriever.SimpleEnsembleRetriever')
def test_create_retriever_hybrid(mock_ensemble, mock_bm25, sample_documents):
    """Test creating a hybrid BM25 + vector retriever."""
    from src.core.retriever import create_retriever

    mock_vs = Mock()
    mock_vs_retriever = Mock()
    mock_vs.as_retriever.return_value = mock_vs_retriever

    mock_bm25_instance = Mock()
    mock_bm25.from_documents.return_value = mock_bm25_instance

    mock_ensemble_instance = Mock()
    mock_ensemble.return_value = mock_ensemble_instance

    # Create hybrid retriever
    result = create_retriever(
        mock_vs,
        documents=sample_documents,
        k=5,
        use_hybrid=True
    )

    # Verify vector retriever was created
    mock_vs.as_retriever.assert_called_once_with(search_kwargs={"k": 5})

    # Verify BM25 retriever was created from documents
    mock_bm25.from_documents.assert_called_once_with(sample_documents)
    assert mock_bm25_instance.k == 5

    # Verify ensemble was created with correct weights
    mock_ensemble.assert_called_once()
    call_kwargs = mock_ensemble.call_args[1]
    assert call_kwargs['weights'] == [0.6, 0.4]
    assert len(call_kwargs['retrievers']) == 2

    assert result == mock_ensemble_instance


def test_retrieve_invokes_retriever():
    """Test that retrieve() calls the retriever's invoke method."""
    from src.core.retriever import retrieve

    mock_retriever = Mock()
    expected_docs = [
        Document(page_content="test", metadata={"title": "Test"})
    ]
    mock_retriever.invoke.return_value = expected_docs

    result = retrieve(mock_retriever, "test query")

    mock_retriever.invoke.assert_called_once_with("test query")
    assert result == expected_docs


@patch('src.core.retriever.BM25Retriever')
@patch('src.core.retriever.SimpleEnsembleRetriever')
def test_hybrid_retriever_ensemble_weights(mock_ensemble, mock_bm25, sample_documents):
    """Test that ensemble is configured with correct 60/40 weights."""
    from src.core.retriever import create_retriever

    mock_vs = Mock()
    mock_vs.as_retriever.return_value = Mock()
    mock_bm25.from_documents.return_value = Mock()

    create_retriever(mock_vs, documents=sample_documents, use_hybrid=True)

    # Verify the correct weights were passed
    call_args = mock_ensemble.call_args
    assert call_args[1]['weights'] == [0.6, 0.4], "Expected 60% vector, 40% BM25"


def test_retrieve_returns_list_of_documents():
    """Test that retrieve returns a list of Document objects."""
    from src.core.retriever import retrieve

    mock_retriever = Mock()
    docs = [
        Document(page_content="content1", metadata={"title": "Doc1"}),
        Document(page_content="content2", metadata={"title": "Doc2"}),
    ]
    mock_retriever.invoke.return_value = docs

    result = retrieve(mock_retriever, "query")

    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(doc, Document) for doc in result)
