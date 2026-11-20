from pathlib import Path
import html
import re
import textwrap
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
# Chapter-specific inputs explain the actual data transformations.
SPECS = {
'index': ('From model bytes to the next token','An edge engine must fit weights, live state and working memory on the device.',
 [('Model file','Mapped weights and metadata'),('Token IDs','Tokenizer and chat template'),('Forward pass','Layers, attention and KV state'),('Next token','Sampling and text decoding')],
 'Memory budget = weights + KV cache + activations + runtime overhead'),
'getting-started':('Choose the toolchain for the target','Compile for the processor that will execute the binary.',
 [('Portable C++','GCC 13+ and C++23'),('CPU specialization','AVX2 on x86 or NEON on Arm'),('Library support','Pinned mdspan and SQLite'),('GPU checks','nvcc, target architecture and driver')],
 'A successful cross-compile still needs execution on the target or an emulator.'),
'02':('A tensor view describes existing storage','Shape and strides determine which address a multidimensional index selects.',
 [('Arena storage','Allocate a contiguous buffer'),('Extents','Set rows and columns'),('Layout and strides','Translate indices to offsets'),('mdspan view','Access without copying data')],
 'Row-major offset(i, j) = i × columns + j'),
'03':('Inside a transformer block','Residual paths carry the input around attention and the feed-forward network.',
 [('Normalize','RMSNorm scales features'),('Attend','RoPE, Q/K/V and KV state'),('Add residual','Combine attention with input'),('Feed forward','RMSNorm, SwiGLU, residual')],
 'Attention(Q, K, V) = softmax(QKᵀ / √d) V'),
'04':('Represent a floating value with an integer','The scale and zero point define the affine mapping.',
 [('Choose range','Measure minimum and maximum'),('Compute metadata','Scale s and zero point z'),('Round and clamp','q = clamp(round(x / s) + z)'),('Reconstruct','x̂ = s × (q − z)')],
 'Quantization error depends on scale, clipping and the distribution of values.'),
'05':('Load model tensors without copying their payload','The file reader validates metadata before exposing mapped tensor ranges.',
 [('File header','Magic, version and counts'),('Tensor descriptors','Name, type, shape and offset'),('Mapped storage','Aligned ranges in the file'),('Tensor view','Decode the stored type')],
 'Validate every size and byte range before reading a tensor.'),
'06':('Use precision where it matters','Different tensor classes have different error sensitivity and storage cost.',
 [('Tensor class','Embedding, attention or FFN'),('Choose format','Q4, Q8 or floating point'),('Dispatch kernel','Match the stored block type'),('Accumulate','Combine into the layer output')],
 'The memory budget is the sum of each tensor’s actual encoded size.'),
'07':('Quantize a vector in a rotated coordinate system','Use a reconstruction stage and a residual correction for inner products.',
 [('Rotate','Spread energy across coordinates'),('MSE quantizer','Map values to a codebook'),('Residual sketch','QJL corrects the dot estimate'),('Estimate','Combine both terms')],
 'Keep the MSE reconstruction, residual and query in compatible coordinates.'),
'08':('The roofline bounds attainable throughput','Operational intensity determines whether memory bandwidth or compute limits a kernel.',
 [('Count work','Floating-point operations'),('Count traffic','Bytes transferred from memory'),('Compute intensity','I = FLOPs / bytes'),('Bound throughput','min(peak FLOP/s, bandwidth × I)')],
 'Decode GEMV often has low intensity; batching can improve weight reuse.'),
'09':('A dot product across SIMD lanes','Accumulate independent products in a vector register before reducing them.',
 [('Load lanes','Read adjacent elements'),('Multiply','Pair corresponding values'),('Accumulate','Keep partial sums in lanes'),('Reduce','Combine into one scalar')],
 'Vector width, alignment, tail elements and ISA support all matter.'),
'10':('Share work without racing on state','Workers coordinate ownership and publication through explicit synchronization.',
 [('Partition work','Assign independent ranges'),('Run workers','Each owns its output region'),('Synchronize','Join, barrier or atomic handoff'),('Consume results','Read after publication')],
 'A data race is undefined behavior; volatile does not provide thread synchronization.'),
'11':('Parallelize the forward pass at independent outputs','Row and head partitions avoid concurrent updates to the same result.',
 [('Projection rows','Distribute output rows'),('Attention heads','Assign independent heads'),('Worker barrier','Complete the current phase'),('Next operation','Use the assembled output')],
 'Keep reduction order fixed when bitwise reproducibility is required.'),
'12':('From text to model token IDs','Pre-tokenization and BPE must match the vocabulary used to train the model.',
 [('Input text','UTF-8 bytes and special tokens'),('Pre-tokenize','Split text into merge regions'),('Ranked merges','Apply vocabulary merge order'),('Token IDs','Feed the model’s embeddings')],
 'Decoding can span multiple tokens for one UTF-8 character.'),
'13':('The KV cache preserves attention history','New tokens append keys and values; each query reads the retained history.',
 [('Append K and V','Store the current token'),('Map logical slots','Resolve pages or ring positions'),('Read history','Attend to retained positions'),('Manage capacity','Grow, evict or reject explicitly')],
 'KV bytes = layers × tokens × KV heads × head dimension × 2 × bytes per element'),
'14':('Reuse and limit live KV state','Windowing, requantization and prefix reuse trade capacity for work or precision.',
 [('Identify prefix','Find reusable cached state'),('Keep active window','Retain required token positions'),('Compress state','Requantize with metadata'),('Track ownership','Release pages when unused')],
 'Prefix sharing requires compatible model state and careful reference lifetimes.'),
'15':('Produce a token from a real checkpoint','Model metadata and tokenizer behavior must agree with the forward pass.',
 [('Inspect GGUF','Architecture and tensor shapes'),('Load vocabulary','Tokens, merges and template'),('Run layers','Qwen2 block and KV cache'),('Project logits','Choose the first output token')],
 'A synthetic self-test and a real checkpoint run validate different contracts.'),
'16':('Turn the forward pass into a command-line engine','Parse input strictly and make capacity and numerical failures visible.',
 [('CLI and config','Validate numeric options'),('Prefill','Process the prompt tokens'),('Decode loop','Forward, sample, stream, append'),('Stop and report','EOS, token limit or error')],
 'A repeated token receives one repetition penalty per sampling step.'),
'17':('Measure the full request lifecycle','Warm-up and the measured decode interval must be explicitly separated.',
 [('Start request','Record the clock origin'),('First token','TTFT includes prompt work'),('Steady decode','Measure inter-token latency'),('Summarize','Median, tails, TPS and memory')],
 'Report model, hardware, flags and measurement boundaries with every comparison.'),
'18':('A manufacturing inspection decision path','Model output is one input to a disposition policy.',
 [('Capture frame','Acquire and timestamp image'),('Visual inference','Extract defect evidence'),('Disposition','Accept, reject or human review'),('Log decision','Persist evidence and status')],
 'Confidence and model output require validation before actuating a production workflow.'),
'19':('Turn shelf images into inventory evidence','Keep observed items linked to their store, image and expected planogram.',
 [('Shelf photo','Capture store and time'),('Extract items','Read positions and counts'),('Compare planogram','Find gaps and misplaced stock'),('Publish findings','Inventory system and review')],
 'Separate observed image evidence from inferred stock availability.'),
'20':('Medical triage with a human review gate','The examples illustrate workflow plumbing; clinical deployment requires separate validation.',
 [('DICOM input','Windowing and image metadata'),('Model findings','Structured preliminary output'),('Review queue','Confidence and escalation'),('Human decision','Clinician reviews the evidence')],
 'Automated triage output requires clinical review and deployment-specific validation.'),
'21':('Route evidence through a structured pipeline','Documents, claims and camera images need different preprocessing and decision policies.',
 [('Input evidence','Document, claim photo or frame'),('Select pipeline','OCR, vision or color correction'),('Validate fields','Consistency and uncertainty'),('Route result','Review, record or action')],
 'Preserve the source evidence and record uncertainty with each derived result.'),
'22':('Combine visual findings with context','Security, accessibility and provenance examples use explicit evidence and review.',
 [('Capture evidence','Image, place and timestamp'),('Extract findings','Objects, text and conditions'),('Check context','Rules, history and provenance'),('Review outcome','Human assessment and audit')],
 'A model’s description alone does not establish identity, authenticity or intent.'),
'23':('Verify a point-of-sale claim against evidence','Reconciliation compares independent signals before making a disposition.',
 [('Capture item','Photo, receipt or document'),('Extract signals','Visual features and fields'),('Cross-check','Transaction and policy context'),('Escalate or record','Retain the evidence trail')],
 'An anomaly is a review signal; it does not by itself prove counterfeit or fraud.'),
'24':('Translate an edit request into image operations','A validated operation plan connects natural language with the image processor.',
 [('User request','Desired image adjustment'),('Parse plan','Operation names and parameters'),('Validate','Bounds and supported actions'),('Apply edit','Run the image operations')],
 'Validate the plan before it reaches cv::Mat operations.'),
'25':('Sample camera streams into reviewable evidence','Frame sampling and timestamps connect model findings to the source recording.',
 [('Video stream','Timed frame samples'),('Scene findings','Tags and event cues'),('Link evidence','Keep source time and identity'),('Review record','Incident or personal log')],
 'Keep the original recording and its provenance alongside generated descriptions.'),
'26':('Cross-check financial document fields','Independent checks expose inconsistencies for further review.',
 [('Capture document','Check or invoice image'),('Extract fields','Amount, payee, MICR and text'),('Compare signals','Checksums and duplicate tests'),('Review findings','Escalation with evidence')],
 'Matching fields do not establish authenticity; preserve uncertainty and provenance.'),
'27':('Fit inference into a market-data latency budget','Deadline handling is part of the decision path.',
 [('Market event','Timestamp the incoming tick'),('Compute features','Book state and signal inputs'),('Run inference','Measure remaining budget'),('Deadline decision','Act, abstain or discard stale work')],
 'An offline accuracy score does not establish live latency or trading performance.'),
'28':('Assess a claim using multiple consistency checks','Photo evidence and policy context feed a reviewable disposition.',
 [('Claim evidence','Photos and incident details'),('Extract damage','Structured observations'),('Cross-check','Duplicates, timestamps, policy'),('Disposition','Review and record reasons')],
 'A fraud flag is an uncertainty-bearing signal for review.'),
'29':('Monitor transactions with explicit windows','Aggregate event history before combining anomaly signals.',
 [('Transaction','Account, amount, time and place'),('Window state','Velocity and rolling totals'),('Combine signals','Structuring and travel checks'),('Review queue','Record the alert and evidence')],
 'An alert needs contextual review; thresholds alone cannot establish money laundering.'),
'30':('Connect the math to kernel behavior','Derivations should predict the quantities the implementation measures.',
 [('FLOPs and bytes','Count arithmetic and traffic'),('Numerical stability','Track error and finite values'),('Geometry','Rotations and relative position'),('Sensitivity','Curvature and quantization error')],
 'Use the mathematical assumptions explicitly when interpreting measurements.'),
'31':('Continuous batching schedules active requests','Retire completed work and admit new requests at iteration boundaries.',
 [('Waiting requests','Admission and token budget'),('Prefill chunks','Bound prompt work per step'),('Decode iteration','Advance active sequences'),('Retire and admit','Free slots and schedule again')],
 'KV capacity, per-step work and fairness constrain admission.'),
'32':('Tiled attention keeps the score matrix out of global memory','Each query retains running softmax statistics while K/V tiles pass through shared memory.',
 [('Load K/V tile','Cooperate through shared memory'),('Compute tile scores','QKᵀ scaled by √d'),('Update online state','Running maximum, sum and output'),('Finalize row','Normalize accumulated output')],
 'Validate partial tiles and query blocks against the CPU reference on a physical GPU.'),
'a':('Cross-compile, then execute the target binary','Compilation and execution are distinct stages of target validation.',
 [('Host build','GCC and CMake configuration'),('Arm toolchain','Target compiler and libraries'),('Target binary','Check architecture and linkage'),('Execute tests','Arm device or QEMU emulator')],
 'Registered binfmt emulation can execute an Arm ELF without an explicit CTest emulator.'),
'b':('Use the quiz to test a chain of reasoning','Work from the constraints to a defensible implementation decision.',
 [('Read constraints','Model, device and workload'),('Choose principle','Memory, math or concurrency'),('Predict behavior','State assumptions and bounds'),('Check solution','Compare reasoning and result')],
 'Explain why an answer follows from the constraints.'),
'c':('Choose an optimization from the limiting resource','Measure the bottleneck before specializing the implementation.',
 [('Memory capacity','Weights and KV must fit'),('Memory traffic','Decode and cache reuse'),('Compute budget','SIMD and parallel work'),('Serving state','Windowing and scheduling')],
 'Each optimization must preserve the model’s numerical and state contracts.'),
'd':('Profile a controlled workload','A measurement only answers the question defined by its boundaries.',
 [('Fix the inputs','Model, prompt and generation'),('Warm up','Separate startup from steady work'),('Measure','Latency, counters and memory'),('Compare','Record hardware and flags')],
 'Repeat measurements and report distributions rather than one selected timing.'),
'e':('Translate familiar framework concepts into engine parts','The representation changes while the underlying responsibilities remain.',
 [('Tensor API','Storage, shape and layout'),('Model loader','Metadata and typed weights'),('Tokenizer','Vocabulary and byte handling'),('Serving runtime','KV state and scheduling')],
 'Check the book’s implemented scope before assuming framework feature parity.'),
}
PALETTE={'bg':'#fbfaf7','ink':'#182d38','muted':'#526674','line':'#cbd5d8','teal':'#a33516','orange':'#c0441f'}
def esc(s): return html.escape(s,quote=True)
def text(x,y,content,size=20,color=None,weight=400):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color or PALETTE["ink"]}" font-weight="{weight}">{esc(content)}</text>'
def lines(x,y,content,size=18,width=24,color=None):
    return ''.join(text(x,y+i*25,line,size,color) for i,line in enumerate(textwrap.wrap(content,width)))
def rect(x,y,w,h,fill='#fff',stroke='#cbd5d8',r=14):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
def arrow(x1,y1,x2,y2):
    return f'<path d="M{x1} {y1} H{x2}" stroke="#738993" stroke-width="2.5" marker-end="url(#arrow)" fill="none"/>'
def base(title,desc,body,h=470):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1120 {h}" role="img" aria-labelledby="title desc">
<title id="title">{esc(title)}</title><desc id="desc">{esc(desc)}</desc>
<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="#738993"/></marker></defs>
<g font-family="Roboto, Arial, sans-serif">{rect(1,1,1118,h-2,PALETTE['bg'],'#d8dedb',18)}
<path d="M32 38 H74" stroke="#c0441f" stroke-width="5" stroke-linecap="round"/>
{lines(32,83,title,29,65)}
{lines(32,140 if len(textwrap.wrap(title,65)) > 1 else 124,desc,18,106,PALETTE['muted'])}
{body}</g></svg>'''
def pipeline(spec):
    title,desc,nodes,note=spec
    b=''
    for i,(label,detail) in enumerate(nodes):
        x=32+i*270
        b+=rect(x,183,246,145)
        b+=f'<path d="M{x+1} 202 V309" stroke="{PALETTE["teal"] if i%2==0 else PALETTE["orange"]}" stroke-width="4"/>'
        b+=lines(x+17,219,label,22,19)
        b+=lines(x+17,270,detail,17,23,PALETTE['muted'])
        if i<3:b+=arrow(x+247,255,x+266,255)
    b+=rect(32,360,1056,78,'#fff0e8','#e9b9a5',12)+lines(50,392,note,19,95,PALETTE['teal'])
    return base(title,desc,b)
def matrix(spec):
    b=''
    for r in range(3):
        for c in range(4):
            x,y=54+c*100,190+r*58
            b+=rect(x,y,88,46,'#ffe4d6' if (r,c)!=(1,2) else '#f9dfca')
            b+=text(x+30,y+30,str(r*4+c),20)
    b+=text(54,390,'3 × 4 view, row-major storage',20)
    b+=arrow(468,270,550,270)
    b+=rect(580,192,474,172)+text(606,230,'Element at [1, 2]',24,weight=600)
    b+=text(606,276,'offset = 1 × 4 + 2 = 6',24,PALETTE['teal'])
    b+=text(606,322,'The view owns no copied payload.',19,PALETTE['muted'])
    return base(spec[0],spec[1],b)
def roofline(spec):
    b=f'<path d="M80 352 V182 M80 352 H648" stroke="#738993" stroke-width="2"/>'
    b+=f'<path d="M105 334 L362 211 H638" fill="none" stroke="#a33516" stroke-width="5"/>'
    b+=text(120,311,'Bandwidth bound',18)+text(392,196,'Compute ceiling',18)
    b+=text(220,395,'Operational intensity (FLOPs / byte)',19)
    b+=text(86,171,'Attainable FLOP/s',18)
    b+=rect(706,185,350,180)+text(728,221,'Two limits',24,weight=600)
    b+=lines(728,258,'Throughput ≤ bandwidth × I',19,30)
    b+=lines(728,309,'Throughput ≤ peak compute',19,30)
    b+=text(80,437,'Schematic axes show the limiting regimes; no measured timings are implied.',17,PALETTE['muted'])
    return base(spec[0],spec[1],b)
def simd(spec):
    b=text(36,198,'Input A',19)+text(36,260,'Input B',19)
    for i in range(8):
        x=150+i*88
        for y,prefix in ((170,'a'),(232,'b')):
            b+=rect(x,y,72,44,'#fff0e8')+text(x+20,y+28,f'{prefix}{i}',19)
        b+=text(x+20,313,'×',27)
    b+=rect(150,335,688,54,'#f9e7d9')+text(178,370,'Accumulate lane products, then reduce the eight partial sums',19)
    b+=text(36,431,'Example: eight float lanes in an AVX2 register. Handle the remaining elements separately.',18,PALETTE['muted'])
    return base(spec[0],spec[1],b)
def cache(spec):
    b=''
    for i in range(8):
        x=42+i*130
        b+=rect(x,188,112,68,'#ffe4d6')+text(x+18,230,f'K token {i}',18)
        b+=rect(x,266,112,68,'#f9e7d9')+text(x+18,308,f'V token {i}',18)
    b+=text(44,380,'A new query attends to the retained K/V positions.',22,PALETTE['teal'])
    b+=text(44,419,'Pages and windows change storage management; the model still needs correct logical positions.',17,PALETTE['muted'])
    return base(spec[0],spec[1],b)
def attention(spec):
    b=rect(32,183,220,151)+text(54,220,'Query row',23,weight=600)+lines(54,263,'One thread owns its running state',18,19)
    b+=arrow(260,252,302,252)
    b+=rect(322,183,300,151,'#fff0e8')+text(344,220,'Shared-memory tile',23,weight=600)
    b+=lines(344,263,'Load up to 32 keys and values; synchronize threads',18,26)
    b+=arrow(630,252,672,252)
    b+=rect(696,183,390,151)+text(718,220,'Online softmax state',23,weight=600)
    b+=text(718,266,'Running max m, sum ℓ, output o',20)
    b+=text(718,306,'Rescale old state for each new tile.',18,PALETTE['muted'])
    b+=text(36,382,'Repeat over K/V tiles → normalize o / ℓ → write one output row',22,PALETTE['teal'])
    b+=text(36,424,'A100 validation covers partial 32-key tiles, partial 128-thread blocks and d ≤ 64.',18,PALETTE['muted'])
    return base(spec[0],spec[1],b)
def batching(spec):
    b=''
    tracks=[('Request A',[(0,2,'Prefill'),(2,3,'Decode'),(3,4,'Done')]),
            ('Request B',[(0,1,'Wait'),(1,2,'Prefill'),(2,4,'Decode')]),
            ('Request C',[(0,2,'Wait'),(2,3,'Prefill'),(3,4,'Decode')])]
    for i,(label,segments) in enumerate(tracks):
        y=190+i*64
        b+=text(36,y+31,label,20)
        for a,z,label in segments:
            x=190+a*215
            fill={'Prefill':'#ffe4d6','Decode':'#f9e7d9','Wait':'#efefea','Done':'#f4d6c7'}[label]
            b+=rect(x,y,(z-a)*215-12,48,fill)+text(x+15,y+31,label,19)
    b+=text(190,418,'Scheduling iterations → admit or retire work at boundaries',21,PALETTE['teal'])
    return base(spec[0],spec[1],b)
CUSTOM={'02':matrix,'08':roofline,'09':simd,'13':cache,'31':batching,'32':attention}
def prose(value):
    for old, new in {'×': '*', 'ᵀ': '^T', '√': 'sqrt ', '−': '-', '’': "'", '≤': '<=', '→': '->'}.items():
        value = value.replace(old, new)
    return unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode()

def main():
    pages=[ROOT/'docs/index.md',ROOT/'docs/getting-started.md']
    pages+=sorted((ROOT/'docs').glob('part*/*.md'))
    pages+=sorted((ROOT/'docs/appendix').glob('*.md'))
    for page in pages:
        key=page.stem if page.parent==ROOT/'docs' else page.name[:2] if page.parent.name!='appendix' else page.name[0]
        spec=SPECS[key]
        folder=ROOT/'docs/images'
        folder.mkdir(exist_ok=True)
        asset=f'{key}-overview.svg'
        (folder/asset).write_text(CUSTOM.get(key,pipeline)(spec))
        relative='images/' if page.parent==ROOT/'docs' else '../images/'
        figure=f'''<!-- chapter-figure -->
<figure class="chapter-figure" markdown="1">

![{esc(prose(spec[0]+'. '+spec[1]))}]({relative}{asset}){{: loading="lazy" width="1120" height="470" }}

  <figcaption>{esc(prose(spec[3]))}</figcaption>
</figure>
<!-- /chapter-figure -->

'''
        template=ROOT/f'chapter{key}_template.md' if key.isdigit() else ROOT/f'appendix_{key}_template.md'
        targets=[page]+([template] if template.is_file() else [])
        for target in targets:
            content=re.sub(r'<!-- chapter-figure -->.*?<!-- /chapter-figure -->\s*','',target.read_text(),flags=re.S)
            # Keep the existing heading sequence intact for chapter verifiers.
            at=content.find('\n## ')
            if at<0:
                at=content.find('\n',content.find('\n')+1)
            content=content[:at].rstrip()+'\n\n'+figure+content[at:].lstrip('\n')
            target.write_text(content)
    print(f'Generated and embedded {len(pages)} accessible SVG figures.')
if __name__=='__main__':
    main()
