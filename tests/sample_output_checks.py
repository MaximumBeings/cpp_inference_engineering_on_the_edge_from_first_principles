"""Verify visible tensor examples against real stdout and reject pass-only samples."""
from pathlib import Path
import os
import re
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]
summary=re.compile(r'(?:\d+/\d+ checks passed|ALL CHECKS PASSED|ALL PASS)')
count=0
for page in (ROOT/'docs').rglob('*.md'):
    for output in re.findall(r'^```text\n(.*?)^```',page.read_text(),re.M|re.S):
        lines=[line.strip() for line in output.splitlines() if line.strip()]
        assert not (lines and all(summary.fullmatch(line) for line in lines)),f'Pass-only output: {page}'
        count+=1
page=ROOT/'docs/part0/02-the-tensor-engine.md'
sections=re.split(r'^## \d+\.\d+ ',page.read_text(),flags=re.M)[1:3]
for index,section in enumerate(sections,1):
    code=re.search(r'^```cpp\n(.*?)^```',section,re.M|re.S)[1]
    expected=re.search(r'^```text\n(.*?)^```',section,re.M|re.S)[1]
    with tempfile.TemporaryDirectory() as directory:
        source=Path(directory)/'example.cpp';binary=Path(directory)/'example'
        source.write_text(code)
        subprocess.run([os.environ.get('CXX','g++'),'-std=c++23','-O2','-Wall','-Wextra',
            '-DMDSPAN_IMPL_STANDARD_NAMESPACE=std','-DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental',
            '-I'+str(ROOT/'_vendor_mdspan/include'),str(source),'-o',str(binary)],check=True)
        actual=subprocess.check_output([str(binary)],text=True)
        assert actual==expected,f'Sample stdout mismatch in section 2.{index}'
print(f'Inspected {count} captured text blocks; no pass-only samples remain. Sections 2.1 and 2.2 match actual stdout.')
