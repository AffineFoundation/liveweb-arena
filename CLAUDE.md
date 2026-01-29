# Claude Code Memory

## Development Guidelines

1. **Occam's Razor** - Keep code minimal while maintaining quality
2. **Engineering First** - Every change should improve overall project structure
3. **Zero Redundancy** - No redundant code allowed
4. **Fix Root Cause** - Never patch over problems, always solve at the root
5. **File Size** - Keep files under 500 lines
6. **Import Style** - Use absolute imports for cross-package (`liveweb_arena.core.xxx`), relative for same package
7. **Commit Rules** - Only commit when explicitly asked; keep messages concise
8. **Template Testing** - Every new question template must be tested via `eval.py` with multiple seeds to verify the entire evaluation pipeline works correctly. Use 10-minute timeout for evaluations.

## Template Design Guidelines

**Core Principle**: Templates must test real web interaction ability, NOT memorization.

### 1. Anti-Memorization Design

Fixed question pool + fixed answers = memorizable. Models can "cheat" by recalling Q&A pairs from training data without actually browsing.

**Design strategies to prevent memorization:**
- **Dynamic data**: Answers that change over time (e.g., counts that grow as new content is added)
- **Computation required**: Aggregation, comparison, or calculation that cannot be pre-memorized
- **Obscure queries**: Information rarely covered in training data (e.g., 7th billed actor vs lead actor)
- **Large entity pools**: Combinatorial space too large to enumerate all possible Q&A pairs

**Risk assessment**: Prefer templates where answers are dynamic or require real-time computation. Avoid templates with small fixed entity sets and static attributes.

### 2. Verifiability

- Every question must have a clear path: Template -> API endpoint -> Ground truth
- API response and website display must share the same data source
- Validation tolerance accounts for timing-related differences (data may change between agent browsing and ground truth fetch) and format variations, not for agent capability errors

### 3. Solvability

- Target website must be publicly accessible without authentication
- Required information must be visible on the page
- Expected steps should be minimal for theoretical completion, with reasonable buffer (not extremely tight limits)
- **NO navigation hints in questions** - Questions should only contain the question itself. No URLs, symbols, selectors, or any navigation shortcuts. Finding the correct source to get the answer is part of the agent's capability being tested.

### 4. Difficulty Stratification

- **Easy**: Single-hop, direct URL navigation, one data point extraction
- **Medium**: Search required, or multiple data points from same page
- **Hard**: Multi-page navigation, cross-reference, or aggregation across multiple sources

### 5. Template Validation Checklist

**Testing method**: Run `eval.py` with ONE template and ONE seed, then analyze the full output against each checkpoint below. Do not batch test - analyze each case individually.

When testing a template, check these in order:

**Step 1: GT Calculation** - Does ground truth compute successfully?
- ✅ PASS: GT returns a concrete value (e.g., `GT = 42`, `GT = "Bitcoin"`)
- ❌ FAIL: GT returns error (e.g., `"data not found"`, `"API failed"`)
- If GT fails, the template is **broken** - fix before proceeding

**Step 2: GT Data Source** - Is GT using page-bound cache data?
- ✅ PASS: GT is computed from `api_data` collected when agent visits the page
- ❌ FAIL: GT fetches data independently (causes agent sees X, GT uses Y)
- Check: Look for `[GT] Visit xxx → +N items` in logs, GT must use this collected data

**Step 3: Data Visibility** - Can the agent see the data needed to answer?
- ✅ PASS: Required data appears in page accessibility tree or visible content
- ❌ FAIL: Data exists in API but not displayed on page (agent cannot see it)
- Check: Is the data in the cached page? Does scrolling/pagination reveal it?

**Step 4: Theoretical Solvability** - Can a perfect agent complete this task?
- ✅ PASS: Clear path exists from start URL to answer
- ❌ FAIL: Requires authentication, broken links, or impossible navigation

**Validation Result Interpretation:**
- Agent timeout/wrong answer + GT success = Agent capability issue (OK, template works)
- GT failure = Template mechanism issue (NOT OK, must fix)
- GT uses different data than page shows = Data binding issue (NOT OK, must fix)
- Data not visible on page = Template design issue (NOT OK, must fix)
