#!/usr/bin/env python3
"""Check illustration coverage, accessibility, source embeds and built image links."""
from pathlib import Path
from html.parser import HTMLParser
import json
import re
import sys
import xml.etree.ElementTree as ET
from urllib.parse import unquote, urlsplit

ROOT=Path(__file__).resolve().parents[2]
manifest=json.loads((ROOT/'tools/illustrations/manifest.json').read_text())
assert len(manifest)==31
count=0
for chapter in manifest:
    page=ROOT/chapter['page']
    content=page.read_text()
    sections=re.findall(r'^## (\d+\.\d+) ',content,re.M)
    assert sections==[c['section'] for c in chapter['concepts']],page
    assert content.count('<!-- edge:opener:start -->')==1,page
    for asset in [chapter['opener']]+[c['asset'] for c in chapter['concepts']]:
        path=ROOT/asset
        node=ET.parse(path).getroot()
        assert node.attrib.get('role')=='img',path
        assert node.attrib.get('aria-labelledby')=='title desc',path
        for tag in ['title','desc']:
            value=node.find('{http://www.w3.org/2000/svg}'+tag)
            assert value is not None and value.text,path
        assert not node.findall('.//{http://www.w3.org/2000/svg}script'),path
        count+=1
    template=ROOT/f"chapter{chapter['chapter']}_template.md"
    if template.exists():
        assert re.findall(r'<!-- edge:.*?:start -->.*?<!-- edge:.*?:end -->',content,re.S)==re.findall(r'<!-- edge:.*?:start -->.*?<!-- edge:.*?:end -->',template.read_text(),re.S),template
appendix_count=0
for entry in json.loads((ROOT/'tools/illustrations/appendices.json').read_text()):
    assert (ROOT/entry['opener']).is_file()
    content=(ROOT/entry['page']).read_text()
    assert content.count('<!-- edge:opener:start -->')==1
    assert re.findall(r'^## ([A-E]\.\d+) ',content,re.M)==[c['section'] for c in entry['concepts']]
    for concept in entry['concepts']:
        assert (ROOT/concept['asset']).is_file()
        appendix_count+=1
assert appendix_count==20
assets=list((ROOT/'docs/images').rglob('*.svg'))
assert len(assets)==234
for asset in assets:
    root=ET.parse(asset).getroot()
    assert root.get('viewBox').split()[2]=='760',asset
    assert root.get('aria-labelledby')=='title desc',asset
    rectangles=root.findall('.//{http://www.w3.org/2000/svg}rect')
    assert rectangles[0].get('fill')=='#181e30',asset
    assert root.find('{http://www.w3.org/2000/svg}title').text,asset
    assert root.find('{http://www.w3.org/2000/svg}desc').text,asset
print('All 234 figures use the shared dark style and 760-unit canvas; all appendix sections are covered.')
assert count==171,count
print('31 chapter openers and 140 section figures pass coverage, XML and accessible-label checks.')

class Images(HTMLParser):
    def __init__(self):super().__init__();self.images=[]
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=='img' and '.svg' in attrs.get('src',''):self.images.append(attrs)

if len(sys.argv)>1:
    site=Path(sys.argv[1]).resolve()
    found=0
    for page in site.rglob('*.html'):
        parser=Images();parser.feed(page.read_text())
        for image in parser.images:
            src=urlsplit(image['src'])
            if src.scheme:continue
            path=(page.parent/unquote(src.path)).resolve()
            assert path.is_relative_to(site) and path.is_file(),(page,image)
            assert image.get('alt'),(page,image)
            found+=1
    assert found==234,found
    print('All 234 SVG embeds resolve in the built book and have alt text.')
