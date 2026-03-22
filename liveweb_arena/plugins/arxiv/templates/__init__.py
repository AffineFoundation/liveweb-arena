"""ArXiv question templates."""

from .paper_info import ArxivPaperInfoTemplate

from .author_extrema import ArxivAuthorExtremaTemplate
from .multi_author_filter import ArxivMultiAuthorFilterTemplate
from .title_length_extrema import ArxivTitleLengthExtremaTemplate
from .category_comparison import ArxivCategoryComparisonTemplate

__all__ = [
    "ArxivPaperInfoTemplate",
    "ArxivAuthorExtremaTemplate",
    "ArxivMultiAuthorFilterTemplate",
    "ArxivTitleLengthExtremaTemplate",
    "ArxivCategoryComparisonTemplate",
]
