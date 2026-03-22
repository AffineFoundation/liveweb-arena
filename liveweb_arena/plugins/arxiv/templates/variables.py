"""Category pool, constants, and metric definitions for ArXiv templates."""

from dataclasses import dataclass
from typing import List, Tuple

# How many papers the agent must inspect for top-N templates.
# Min category volume is 8 papers/day → 1.6× safety margin at top_n=5.
TOP_N_CHOICES = [3, 4, 5]

# Fixed window size for category_comparison (always uses the max).
TOP_N = max(TOP_N_CHOICES)

# Rank positions for paper_info (Nth newest paper).
RANK_CHOICES = [1, 2, 3]

RANK_LABELS = {
    1: "most recent",
    2: "second most recent",
    3: "third most recent",
}


@dataclass(frozen=True)
class Category:
    """An ArXiv category with display name."""
    code: str       # e.g., "cs.AI"
    name: str       # e.g., "Artificial Intelligence"
    group: str      # e.g., "cs" (for pairing across groups)

    @property
    def listing_url(self) -> str:
        """URL for the new-submissions listing page."""
        return f"https://arxiv.org/list/{self.code}/new"


# Diverse categories across ArXiv's subject groups.
# Selection criteria:
# - Daily new-submission volume ≥ 8 papers/day (verified 2026-03-19)
# - Stable existence on arxiv.org
# - Playwright-accessible listing pages
# - No aliases (cs.SY→eess.SY, cs.NA→math.NA excluded)
#
# Excluded for low volume (< 8 new papers/day, verified 2026-03-19):
# q-fin.ST (~1), q-bio.NC (~3), q-bio.QM (~5), cs.DB (~5), cs.IT (~5),
# cs.MA (~5), math.DS (~5), physics.flu-dyn (~5), astro-ph.EP (~5),
# eess.IV (~6), cs.NI (~6), math.MP (~6), math.FA (~7), cs.CE (~7),
# cs.SI (~7), cs.CY (~7), nucl-ex (~7), stat.AP (~7).
CATEGORIES: List[Category] = [
    # Computer Science (11)
    Category("cs.AI", "Artificial Intelligence", "cs"),
    Category("cs.CL", "Computation and Language", "cs"),
    Category("cs.CV", "Computer Vision", "cs"),
    Category("cs.LG", "Machine Learning", "cs"),
    Category("cs.SE", "Software Engineering", "cs"),
    Category("cs.CR", "Cryptography and Security", "cs"),
    Category("cs.RO", "Robotics", "cs"),
    Category("cs.DS", "Data Structures and Algorithms", "cs"),
    Category("cs.HC", "Human-Computer Interaction", "cs"),
    Category("cs.IR", "Information Retrieval", "cs"),
    Category("cs.GT", "Computer Science and Game Theory", "cs"),
    # Mathematics (9)
    Category("math.CO", "Combinatorics", "math"),
    Category("math.PR", "Probability", "math"),
    Category("math.OC", "Optimization and Control", "math"),
    Category("math.NA", "Numerical Analysis", "math"),
    Category("math.AG", "Algebraic Geometry", "math"),
    Category("math.AP", "Analysis of PDEs", "math"),
    Category("math.NT", "Number Theory", "math"),
    Category("math.DG", "Differential Geometry", "math"),
    Category("math.GR", "Group Theory", "math"),
    # Physics (16)
    Category("hep-th", "High Energy Physics - Theory", "physics"),
    Category("hep-ph", "High Energy Physics - Phenomenology", "physics"),
    Category("quant-ph", "Quantum Physics", "physics"),
    Category("gr-qc", "General Relativity and Quantum Cosmology", "physics"),
    Category("astro-ph.CO", "Cosmology and Nongalactic Astrophysics", "physics"),
    Category("astro-ph.GA", "Astrophysics of Galaxies", "physics"),
    Category("astro-ph.HE", "High Energy Astrophysical Phenomena", "physics"),
    Category("astro-ph.SR", "Solar and Stellar Astrophysics", "physics"),
    Category("astro-ph.IM", "Instrumentation and Methods for Astrophysics", "physics"),
    Category("cond-mat.str-el", "Strongly Correlated Electrons", "physics"),
    Category("cond-mat.mes-hall", "Mesoscale and Nanoscale Physics", "physics"),
    Category("cond-mat.mtrl-sci", "Materials Science", "physics"),
    Category("cond-mat.stat-mech", "Statistical Mechanics", "physics"),
    Category("cond-mat.supr-con", "Superconductivity", "physics"),
    Category("cond-mat.soft", "Soft Condensed Matter", "physics"),
    Category("physics.optics", "Optics", "physics"),
    # Statistics (2)
    Category("stat.ML", "Statistics - Machine Learning", "stat"),
    Category("stat.ME", "Statistics - Methodology", "stat"),
    # Electrical Engineering (3)
    Category("eess.SP", "Signal Processing", "eess"),
    Category("eess.SY", "Systems and Control", "eess"),
    Category("eess.AS", "Audio and Speech Processing", "eess"),
]


def build_category_pairs() -> List[Tuple[Category, Category]]:
    """Build all cross-group category pairs for comparison templates.

    Pairs categories across groups (e.g., cs vs math) so comparisons are
    interesting — categories within the same group tend to have similar volumes.
    """
    pairs: List[Tuple[Category, Category]] = []
    n = len(CATEGORIES)
    for i in range(n):
        for j in range(i + 1, n):
            if CATEGORIES[i].group != CATEGORIES[j].group:
                pairs.append((CATEGORIES[i], CATEGORIES[j]))
    return pairs


CATEGORY_PAIRS: List[Tuple[Category, Category]] = build_category_pairs()
