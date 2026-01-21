# Claude Code Memory

## Development Guidelines

1. **Occam's Razor** - Keep code minimal while maintaining quality
2. **Engineering First** - Every change should improve overall project structure
3. **Zero Redundancy** - No redundant code allowed
4. **Fix Root Cause** - Never patch over problems, always solve at the root
5. **Test Driven** - Every bug fix must have a corresponding test case
6. **File Size** - Keep files under 500 lines
7. **Import Style** - Use absolute imports for cross-package (`liveweb_arena.core.xxx`), relative for same package
8. **Commit Rules** - Only commit when explicitly asked; keep messages concise
9. **Template Testing** - Every new question template must be tested via `eval.py` with multiple seeds to verify the entire evaluation pipeline works correctly
