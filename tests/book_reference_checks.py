"""Guard corrected complexity and references to unpublished material."""
from pathlib import Path
import re
root=Path(__file__).resolve().parents[1]
for page in (root/'docs').rglob('*.md'):
    text=page.read_text()
    assert not re.search(r"Chapter 1\b|Appendix F\b", text), page
text=(root/'docs/part3/13-the-kv-cache-manager.md').read_text()
assert 'O(N) per step and O(N^2) across N generated tokens' in text
assert 'caching does not make total dense attention linear' in text
print('PASS: published chapter references and per-step/total KV complexity')
