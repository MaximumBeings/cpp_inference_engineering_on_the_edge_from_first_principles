#!/usr/bin/env python3
"""Build chapter openers and section diagrams, keeping source templates in sync."""
from pathlib import Path
import html
import hashlib
import json
import re
import runpy
import sys
import textwrap
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))
from concepts import CONCEPTS, APPENDIX_CONCEPTS
ROOT = Path(__file__).resolve().parents[2]
SPECS = runpy.run_path(str(ROOT / 'tools/build_diagrams.py'))['SPECS']
INK, MUTED, TEAL, ORANGE = '#182d38', '#526674', '#0e756d', '#b94c27'

def esc(value):
    return html.escape(str(value), quote=True)

def txt(x,y,value,size=18,color=INK,weight=400,anchor='start'):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{esc(value)}</text>'

def wrap(x,y,value,size=18,width=50,color=INK,weight=400,step=None):
    return ''.join(txt(x,y+i*(step or size*1.4),line,size,color,weight) for i,line in enumerate(textwrap.wrap(value,width)))

def rect(x,y,w,h,fill='#fff',stroke='#cbd5d8',r=12):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'

def arrow(x1,y1,x2,y2):
    return f'<path d="M{x1} {y1} L{x2} {y2}" fill="none" stroke="#738993" stroke-width="2" marker-end="url(#arrow)"/>'

def svg(title,description,body,height=610,dark=False):
    bg='#181e30'
    result=f'''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="{height}" viewBox="0 0 760 {height}" role="img" aria-labelledby="title desc">
<title id="title">{esc(title)}</title><desc id="desc">{esc(description)}</desc>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="#738993"/></marker></defs>
<g font-family="Arial, Helvetica, sans-serif">{rect(1,1,758,height-2,bg,'#d8dedb' if not dark else '#3a455c',16)}{body}</g></svg>'''
    # Apply the same dark palette to every figure, including overview cards.
    node=ET.fromstring(result)
    ns='{http://www.w3.org/2000/svg}'
    text_colors={'#182d38':'#f5f4fa','#526674':'#bdc9df','#0e756d':'#8ed5c4','#b94c27':'#f3b486'}
    fills={'#fff':'#26324a','#ffffff':'#26324a','#fbfaf7':'#181e30','#e8f1ed':'#253d40','#faeee6':'#44332e','#eaf2ef':'#253d40','#eef0ed':'#303b50','#0e756d':'#245650'}
    strokes={'#cbd5d8':'#5c6c88','#d8dedb':'#3a455c','#d2e1db':'#426b66','#eef0ed':'#303b50'}
    for element in node.iter():
        if element.tag==ns+'text':
            element.set('fill',text_colors.get(element.get('fill'),element.get('fill','#f5f4fa')))
        elif element.tag==ns+'rect':
            element.set('fill',fills.get(element.get('fill'),element.get('fill','#26324a')))
        if element.get('stroke') in strokes:element.set('stroke',strokes[element.get('stroke')])
    ET.register_namespace('', 'http://www.w3.org/2000/svg')
    return ET.tostring(node,encoding='unicode')

def heading(title,section):
    count=len(textwrap.wrap(title,46))
    body=txt(32,36,f'CONCEPT {section}  /  C++ INFERENCE AT THE EDGE',13,TEAL,700)
    body+=wrap(32,78,title,27,46,weight=600,step=34)
    return body, max(138, 78+(count-1)*34+48)

def note(body,y,caption):
    count=len(textwrap.wrap(caption,75))
    body+=rect(32,y,696,34+count*24,'#eaf2ef','#d2e1db')
    body+=wrap(48,y+29,caption,16,75,TEAL,step=24)
    return body,y+54+count*24

def custom_diagram(section,top):
    b=''
    if section=='3.2':
        b+=rect(280,top,200,60,'#e8f1ed')+txt(380,top+38,'Input x',22,TEAL,600,'middle')
        b+=arrow(330,top+64,192,top+108)+arrow(430,top+64,568,top+108)
        for x,label in [(32,'SiLU(W_gate x)'),(408,'W_up x')]:
            b+=rect(x,top+112,320,74)+txt(x+160,top+157,label,21,INK,500,'middle')
        b+=arrow(192,top+190,330,top+236)+arrow(568,top+190,430,top+236)
        b+=rect(230,top+240,300,66,'#faeee6')+txt(380,top+281,'Elementwise multiply',21,ORANGE,600,'middle')
        b+=arrow(380,top+310,380,top+344)
        b+=rect(230,top+348,300,66)+txt(380,top+389,'W_down projection',21,INK,500,'middle')
        return b,top+442
    if section=='4.3':
        for i in range(8):
            x=32+i*87
            b+=rect(x,top+58,80,84,'#e8f1ed' if i<4 else '#faeee6')
            b+=txt(x+40,top+110,f'b{7-i}',23,TEAL if i<4 else ORANGE,600,'middle')
        b+=txt(206,top+29,'HIGH NIBBLE',16,TEAL,700,'middle')+txt(554,top+29,'LOW NIBBLE',16,ORANGE,700,'middle')
        b+=arrow(206,top+148,206,top+200)+arrow(554,top+148,554,top+200)
        b+=rect(32,top+206,320,82)+txt(192,top+253,'B = (byte >> 4) & 0x0F',20,INK,500,'middle')
        b+=rect(408,top+206,320,82)+txt(568,top+253,'A = byte & 0x0F',20,INK,500,'middle')
        b+=txt(32,top+335,'The scale and signed-value convention belong to the block format.',17,TEAL)
        return b,top+363
    if section in {'2.3','13.3','14.1'}:
        if section=='2.3':
            for row in range(3):
                for col in range(4):
                    x=160+col*110;y=top+row*92
                    b+=rect(x,y,84,68,'#e8f1ed')+txt(x+42,y+43,row*4+col,21,INK,600,'middle')
            b+=arrow(160,top-16,565,top-16)+txt(160,top-30,'Contiguous row traversal',17,TEAL)
            b+=arrow(130,top+5,130,top+244)
            b+=txt(32,top+315,'Column traversal jumps by 4 elements in this row-major layout.',17,ORANGE)
            return b,top+346
        import math
        for i in range(8):
            angle=(i-2)*math.pi/4;x=210+112*math.cos(angle);y=top+164+112*math.sin(angle)
            b+=rect(x-36,y-24,72,48,'#e8f1ed' if i!=0 else '#faeee6')+txt(x,y+7,f'slot {i}',15,INK,600,'middle')
        b+=txt(210,top+162,'PHYSICAL',16,TEAL,600,'middle')+txt(210,top+188,'SLOTS',16,TEAL,600,'middle')
        for i,line in enumerate(['logical positions keep increasing','position 8 -> slot 0','position 9 -> slot 1','position 10 -> slot 2']):
            b+=wrap(400,top+70+i*65,line,18,28,step=25)
        b+=txt(32,top+332,'Slot reuse does not imply reusing the old positional encoding.',17,TEAL)
        return b,top+362
    if section=='13.2':
        for i in range(3):
            y=top+i*112
            b+=rect(32,y,220,76)+txt(142,y+45,f'Logical page {i}',20,INK,500,'middle')
        for j,block in enumerate([2,7,9]):
            y=top+j*112
            b+=rect(508,y,220,76,'#e8f1ed')+txt(618,y+45,f'Physical block {block}',19,TEAL,500,'middle')
        for i,destination in enumerate([1,0,2]):b+=arrow(258,top+i*112+38,502,top+destination*112+38)
        b+=txt(32,top+372,'Block table: [7, 2, 9]. Token offset is retained inside each block.',17,TEAL)
        return b,top+400
    if section in {'3.3','30.3'}:
        b+=f'<circle cx="260" cy="{top+160}" r="120" fill="none" stroke="#b7c6cc" stroke-width="2"/>'
        b+=f'<path d="M110 {top+160} H420 M260 {top+306} V{top+8}" fill="none" stroke="#a3b2ba" stroke-width="1.5"/>'
        b+=arrow(260,top+160,380,top+160)+arrow(260,top+160,260,top+40)
        b+=f'<path d="M324 {top+160} A64 64 0 0 0 260 {top+96}" fill="none" stroke="{ORANGE}" stroke-width="3" marker-end="url(#arrow)"/>'
        b+=txt(394,top+167,'[1,0]',19,TEAL,600)+txt(275,top+32,'[0,1]',19,TEAL,600)
        b+=wrap(482,top+90,'Illustrative rotation through 90 degrees',21,19,ORANGE,600,29)
        b+=wrap(482,top+210,'The vector length remains one.',18,22,TEAL,500,26)
        b+=txt(32,top+350,'RoPE uses a different angle for each position and coordinate pair.',17,TEAL)
        return b,top+380
    return None

def render(spec,section):
    kind,title,items,caption=spec
    b,top=heading(title,section)
    custom=custom_diagram(section,top)
    if custom:
        drawing,end=custom
        b+=drawing
    elif kind=='flow':
        # Clockwise order keeps each dependency direction explicit.
        coords=[(32,top),(408,top),(408,top+148),(32,top+148)]
        for i,label in enumerate(items):
            x,y=coords[i]
            b+=rect(x,y,320,116,'#fff')+txt(x+18,y+30,f'{i+1:02}',14,ORANGE,700)
            b+=wrap(x+18,y+62,label,20,25,weight=500,step=27)
        if len(items)>1:b+=arrow(356,top+56,402,top+56)
        if len(items)>2:b+=arrow(568,top+120,568,top+142)
        if len(items)>3:b+=arrow(404,top+204,358,top+204)
        end=top+288
    elif kind=='compare':
        heights=[]
        for column,value in enumerate(items):
            pieces=value.split('|')
            heights.append(76+sum(len(textwrap.wrap(p,28))*26+14 for p in pieces[1:]))
        h=max(heights)
        for col,value in enumerate(items):
            x=32+col*376; pieces=value.split('|')
            b+=rect(x,top,320,h,'#e8f1ed' if col==0 else '#faeee6')
            b+=wrap(x+18,top+34,pieces[0],20,25,TEAL if col==0 else ORANGE,600,27)
            y=top+90
            for p in pieces[1:]:
                b+=wrap(x+18,y,p,18,28,step=26)
                y+=len(textwrap.wrap(p,28))*26+14
        end=top+h+24
    elif kind=='math':
        for i,label in enumerate(items):
            y=top+i*76
            b+=rect(32,y,696,60,'#fff' if i<len(items)-1 else '#e8f1ed')
            b+=txt(54,y+36,f'{i+1:02}',15,ORANGE,700)
            b+=wrap(110,y+37,label,21,49,step=26)
        end=top+len(items)*76+8
    elif kind=='mapping':
        for i,value in enumerate(items):
            left,right=value.split('|',1); y=top+i*92
            b+=rect(32,y,286,74,'#fff')+rect(408,y,320,74,'#e8f1ed')
            b+=wrap(48,y+31,left,18,25,step=25)+wrap(424,y+31,right,18,28,TEAL,step=25)
            b+=arrow(322,y+37,402,y+37)
        end=top+len(items)*92+8
    elif kind=='memory':
        for i,label in enumerate(items):
            x=32+(i%2)*376;y=top+(i//2)*124
            b+=rect(x,y,320,96,'#e8f1ed' if i%2==0 else '#faeee6')
            b+=txt(x+16,y+27,f'BLOCK {i+1}',12,MUTED,600)
            b+=wrap(x+16,y+58,label,19,26,step=26)
        b+=txt(32,top+283,'One representation; explicit boundaries and ownership',17,TEAL,600)
        end=top+310
    elif kind=='grid':
        for row in range(3):
            for col in range(4):
                x=32+col*82;y=top+row*74;selected=(row,col)==(1,2)
                b+=rect(x,y,70,62,'#0e756d' if selected else '#fff')
                b+=txt(x+35,y+38,row*4+col,21,'#fff' if selected else INK,600,'middle')
        for i,label in enumerate(items):b+=wrap(414,top+35+i*53,label,18,29,step=25)
        b+=txt(32,top+269,'Flat addresses increase from left to right in each row.',17,TEAL)
        end=top+300
    elif kind=='bars':
        for i,value in enumerate(items):
            label,count=value.split('|');y=top+i*86
            b+=txt(32,y+22,label,18,INK,500)
            b+=rect(32,y+38,570,24,'#eef0ed','#eef0ed',5)
            b+=rect(32,y+38,float(count)*57,24,'#0e756d','#0e756d',5)
            b+=txt(624,y+58,count,18,TEAL,600)
        end=top+len(items)*86+10
    elif kind=='timeline':
        for i,value in enumerate(items):
            parts=value.split('|'); y=top+i*127
            b+=txt(32,y+22,parts[0],18,TEAL,600)
            for j,label in enumerate(parts[1:]):
                x=32+j*242
                b+=rect(x,y+40,212,72,'#fff')
                b+=wrap(x+12,y+69,label,17,21,step=23)
                if j<2:b+=arrow(x+216,y+76,x+236,y+76)
        end=top+len(items)*127+4
    else:
        raise ValueError(kind)
    b,height=note(b,end,caption)
    return svg(title,caption,b,height)


def render_overview(spec,key):
    title,description,nodes,caption=spec
    b=txt(32,36,'C++ INFERENCE AT THE EDGE  /  OVERVIEW',13,TEAL,700)
    title_lines=len(textwrap.wrap(title,46))
    b+=wrap(32,78,title,27,46,weight=600,step=34)
    subtitle_y=78+(title_lines-1)*34+38
    b+=wrap(32,subtitle_y,description,18,72,MUTED,step=26)
    top=subtitle_y+(len(textwrap.wrap(description,72))-1)*26+42
    coords=[(32,top),(408,top),(408,top+214),(32,top+214)]
    for i,(label,detail) in enumerate(nodes):
        x,y=coords[i]
        b+=rect(x,y,320,184,'#fff')
        b+=txt(x+18,y+27,f'{i+1:02}',13,ORANGE,700)
        b+=wrap(x+18,y+59,label,21,25,weight=600,step=28)
        b+=wrap(x+18,y+113,detail,18,28,MUTED,step=25)
    b+=arrow(356,top+78,402,top+78)
    b+=arrow(568,top+188,568,top+208)
    b+=arrow(404,top+292,358,top+292)
    b,height=note(b,top+424,caption)
    return svg(title,description+' '+caption,b,height)

def opener(key,page):
    raw=re.search(r'^# (.+)',page.read_text(),re.M)[1]
    title=re.sub(r'^(Chapter \d+|Appendix [A-E]):\s*','',raw).split(' -- ')[0]
    lines=textwrap.wrap(title,36)
    accent=['#8ed5c4','#f3b486','#b8b4ec'][(int(key) if key.isdigit() else ord(key))%3]
    b=txt(32,38,f'C++ INFERENCE AT THE EDGE  /  '+(f'CHAPTER {int(key):02}' if key.isdigit() else f'APPENDIX {key.upper()}'),13,'#bdc9df',700)
    b+='<path d="M32 57 H728" stroke="#45516c"/>'
    b+=wrap(32,105,title,34,36,'#f5f4fa',500,43)
    top=105+(len(lines)-1)*43+56
    # Four silicon tiles form a concept-specific circuit rather than a stock photograph.
    labels=[n[0] for n in SPECS[key][2]]
    for i,label in enumerate(labels):
        x=52+(i%2)*368;y=top+(i//2)*128
        b+=rect(x,y,288,98,'#26324a','#5c6c88')
        for p in range(7):
            px=x+24+p*37
            b+=f'<path d="M{px} {y-9} V{y} M{px} {y+98} V{y+107}" stroke="#60708b" stroke-width="2"/>'
        b+=txt(x+18,y+26,f'{i+1:02}',12,accent,700)
        b+=wrap(x+18,y+58,label,20,24,'#f5f4fa',500,27)
    b+=f'<path d="M340 {top+49} H420 M196 {top+98} V{top+128} M564 {top+98} V{top+128} M340 {top+177} H420" stroke="{accent}" stroke-width="3" fill="none"/>'
    bottom=top+276
    b+=wrap(32,bottom,SPECS[key][0],22,57,accent,600,29)
    bottom+=len(textwrap.wrap(SPECS[key][0],57))*29+25
    b+=txt(32,bottom,'Original conceptual chapter artwork  /  schematic, not a benchmark',13,'#aab8cf')
    return svg(title,SPECS[key][1],b,bottom+30,True)

def figure(src,title,caption,marker):
    # Version the image URL so readers do not retain stale diagrams after a deploy.
    asset=ROOT/'docs'/src.removeprefix('../')
    src+='?v='+hashlib.sha256(asset.read_bytes()).hexdigest()[:12]
    # Markdown links are resolved by MkDocs, including nested directory URLs.
    return f'''<!-- edge:{marker}:start -->
<figure class="chapter-figure" markdown="1">

[![{esc(title)}]({src})]({src})

<figcaption>{esc(caption)}</figcaption>
</figure>
<!-- edge:{marker}:end -->
'''

def main():
    assets=[];manifest=[]
    chapters=sorted((ROOT/'docs').glob('part*/*.md'))
    assert len(chapters)==31
    for page in chapters:
        key=page.name[:2]
        headers=re.findall(r'^## (\d+\.\d+) (.+)$',page.read_text(),re.M)
        assert len(headers)==len(CONCEPTS[key]),(key,len(headers),len(CONCEPTS[key]))
        paths=[]
        opener_path=ROOT/f'docs/images/chapter-openers/{key}.svg'
        opener_path.parent.mkdir(parents=True,exist_ok=True)
        opener_path.write_text(opener(key,page)); assets.append(opener_path)
        for index,((section,heading_text),spec) in enumerate(zip(headers,CONCEPTS[key]),1):
            path=ROOT/f'docs/images/technical/ch{key}/{key}-{index:02}.svg'
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(render(spec,section));assets.append(path)
            paths.append((section,heading_text,spec,path))
        template=ROOT/f'chapter{key}_template.md'
        for target in [page]+([template] if template.exists() else []):
            value=re.sub(r'<!-- edge:.*?:start -->.*?<!-- edge:.*?:end -->\s*','',target.read_text(),flags=re.S)
            at=value.index('\n')
            value=value[:at]+'\n\n'+figure(f'../images/chapter-openers/{key}.svg',f'Chapter {int(key)}: '+SPECS[key][0],'Chapter roadmap. Open the image for a full-size view.','opener')+'\n'+value[at:].lstrip('\n')
            for section,heading_text,spec,path in paths:
                header=f'## {section} {heading_text}'
                assert header in value,(target,header)
                fig=figure(f'../images/technical/ch{key}/{path.name}',spec[1],spec[3],f'concept-{section}')
                value=value.replace(header+'\n',header+'\n\n'+fig+'\n',1)
            target.write_text(value)
        manifest.append({'chapter':key,'page':str(page.relative_to(ROOT)),'opener':str(opener_path.relative_to(ROOT)),'concepts':[{'section':section,'asset':str(path.relative_to(ROOT)),'layout':spec[0]} for section,_,spec,path in paths]})
    appendices=[]
    appendix_count=0
    for page in sorted((ROOT/'docs/appendix').glob('*.md')):
        key=page.name[0]
        path=ROOT/f'docs/images/chapter-openers/appendix-{key}.svg'
        path.write_text(opener(key,page));assets.append(path)
        headers=re.findall(r'^## ([A-E]\.\d+) (.+)$',page.read_text(),re.M)
        assert len(headers)==len(APPENDIX_CONCEPTS[key]),key
        paths=[]
        for i,((section,label),spec) in enumerate(zip(headers,APPENDIX_CONCEPTS[key]),1):
            asset=ROOT/f'docs/images/technical/appendix-{key}/{key}-{i:02}.svg'
            asset.parent.mkdir(parents=True,exist_ok=True)
            asset.write_text(render(spec,section));assets.append(asset)
            paths.append((section,label,spec,asset));appendix_count+=1
        template=ROOT/f'appendix_{key}_template.md'
        for target in [page,template]:
            content=re.sub(r'<!-- edge:.*?:start -->.*?<!-- edge:.*?:end -->\s*','',target.read_text(),flags=re.S)
            at=content.index('\n')
            content=content[:at]+'\n\n'+figure(f'../images/chapter-openers/appendix-{key}.svg',f'Appendix {key.upper()}: '+SPECS[key][0],'Appendix roadmap. Open the image for a full-size view.','opener')+'\n'+content[at:].lstrip('\n')
            for section,label,spec,asset in paths:
                header=f'## {section} {label}'
                content=content.replace(header+'\n',header+'\n\n'+figure(f'../images/technical/appendix-{key}/{asset.name}',spec[1],spec[3],f'concept-{section}')+'\n',1)
            target.write_text(content)
        appendices.append({'page':str(page.relative_to(ROOT)),'opener':str(path.relative_to(ROOT)),'concepts':[{'section':section,'asset':str(asset.relative_to(ROOT))} for section,_,_,asset in paths]})
    (ROOT/'tools/illustrations/appendices.json').write_text(json.dumps(appendices,indent=2)+'\n')
    for path in assets:
        root=ET.parse(path).getroot()
        assert root.find('{http://www.w3.org/2000/svg}title') is not None
        assert root.find('{http://www.w3.org/2000/svg}desc') is not None
    (ROOT/'tools/illustrations/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Generated {len(chapters)} chapter openers and {len(assets)-len(chapters)-len(appendices)-appendix_count} section diagrams, plus {len(appendices)} appendix openers and {appendix_count} appendix concepts.')

if __name__=='__main__':main()
