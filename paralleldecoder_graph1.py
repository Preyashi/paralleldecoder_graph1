# ============================================================
# GRAPH RAG + MMR RETRIEVAL + CONFIDENCE PROPAGATION
# SINGLE GOOGLE COLAB SCRIPT
# ============================================================

# ============================================================
# INSTALLS
# ============================================================

!pip -q uninstall -y transformers sentence-transformers peft trl accelerate
!pip -q install transformers==4.41.2
!pip -q install sentence-transformers==2.7.0
!pip -q install accelerate==0.30.1
!pip -q install networkx scikit-learn scipy

# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import networkx as nx
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from textwrap import dedent
import re

# ============================================================
# SMALL GENERATION MODEL
# ============================================================

MODEL_NAME = "google/flan-t5-base"

generator = pipeline(
    task="text2text-generation",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    device_map="auto"
)

# ============================================================
# EMBEDDING MODEL
# ============================================================

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ============================================================
# KNOWLEDGE BASE
# MANY SMALL CHUNKS
# ============================================================

knowledge_docs = [

    # ATTENTION
    "Attention mechanisms allow transformers to focus on important tokens.",
    "Query vectors represent what the current token is searching for.",
    "Key vectors represent token identities used for matching.",
    "Value vectors contain the actual token information passed forward.",
    "Scaled dot-product attention computes similarity between queries and keys.",
    "Attention scores are normalized using softmax.",
    "Scaling by square root of key dimension stabilizes gradients.",
    "Self-attention allows tokens to interact with other tokens.",
    "Multi-head attention enables multiple representation subspaces.",
    "Attention captures long-range dependencies in sequences.",

    # FEEDFORWARD
    "Transformer feedforward networks contain two linear layers.",
    "A nonlinear activation is placed between feedforward layers.",
    "Feedforward layers process each token independently.",
    "Feedforward networks expand representation dimensionality.",
    "Transformer blocks combine attention and feedforward modules.",
    "Residual connections stabilize deep transformer training.",
    "Layer normalization improves optimization stability.",
    "Feedforward networks increase representational capacity.",
    "GELU is commonly used in transformer feedforward layers.",
    "Transformer blocks alternate attention and feedforward computation.",

    # KV CACHE
    "KV caching stores previously computed key vectors.",
    "KV caching stores previously computed value vectors.",
    "Autoregressive decoding predicts one token at a time.",
    "KV cache avoids recomputing attention over previous tokens.",
    "KV caching significantly accelerates transformer inference.",
    "Without KV caching transformers recompute previous attention states.",
    "KV cache reduces decoding latency.",
    "Decoder-only transformers heavily benefit from KV caching.",
    "KV cache memory grows with sequence length.",
    "Efficient KV cache management is important for long contexts.",

    # INFERENCE OPTIMIZATION
    "Inference optimization improves transformer serving efficiency.",
    "Batching increases hardware utilization during inference.",
    "Quantization reduces numerical precision for faster inference.",
    "4-bit quantization greatly reduces memory usage.",
    "8-bit quantization balances accuracy and efficiency.",
    "Pruning removes less important model weights.",
    "Tensor parallelism distributes computation across devices.",
    "Flash attention optimizes memory access patterns.",
    "Memory bandwidth often bottlenecks transformer inference.",
    "Inference acceleration reduces latency and serving cost.",

    # EXTRA DIVERSITY
    "Transformers process sequences in parallel during training.",
    "Decoder models generate text autoregressively.",
    "Encoder-decoder transformers are common in translation.",
    "Large language models are built using transformer architectures.",
    "Positional embeddings encode token order information.",
    "Rotary embeddings improve long-context extrapolation.",
    "Softmax converts scores into probability distributions.",
    "Embeddings map tokens into dense vector spaces.",
    "Transformer scaling laws relate performance to compute.",
    "Attention complexity grows quadratically with sequence length."
]

# ============================================================
# CHUNKING
# ============================================================

chunks = knowledge_docs

print("\nBuilding retrieval corpus...")
print(f"\nTotal chunks: {len(chunks)}")

# ============================================================
# EMBEDDINGS
# ============================================================

chunk_embeddings = embedder.encode(
    chunks,
    convert_to_numpy=True,
    normalize_embeddings=True
)

# ============================================================
# GRAPH
# ============================================================

G = nx.DiGraph()

G.add_edges_from([
    ("attention", "feedforward"),
    ("attention", "kv_cache"),
    ("feedforward", "inference_optimization"),
    ("kv_cache", "inference_optimization")
])

print("\nDependency Graph:")
for edge in G.edges():
    print(edge)

execution_order = list(nx.topological_sort(G))

print("\nExecution Order:")
print(execution_order)

# ============================================================
# NODE TASKS
# ============================================================

node_tasks = {
    "attention": """
Explain:
- query key value
- scaled dot-product attention
- intuition behind attention
""",

    "feedforward": """
Explain:
- FFN role
- transformer block structure
- nonlinear transformations
""",

    "kv_cache": """
Explain:
- KV caching
- autoregressive decoding
- avoiding recomputation
""",

    "inference_optimization": """
Explain:
- batching
- quantization
- memory optimization
- inference acceleration
"""
}

# ============================================================
# MMR RETRIEVAL
# ============================================================

def mmr_retrieval(query, k=3, lambda_param=0.7):

    q_emb = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    similarities = cosine_similarity(q_emb, chunk_embeddings)[0]

    selected = []
    candidate_indices = list(range(len(chunks)))

    first_idx = np.argmax(similarities)
    selected.append(first_idx)
    candidate_indices.remove(first_idx)

    while len(selected) < k and candidate_indices:

        mmr_scores = []

        for idx in candidate_indices:

            relevance = similarities[idx]

            diversity = max([
                cosine_similarity(
                    [chunk_embeddings[idx]],
                    [chunk_embeddings[s]]
                )[0][0]
                for s in selected
            ])

            score = (
                lambda_param * relevance
                -
                (1 - lambda_param) * diversity
            )

            mmr_scores.append((score, idx))

        mmr_scores.sort(reverse=True)

        best_idx = mmr_scores[0][1]

        selected.append(best_idx)
        candidate_indices.remove(best_idx)

    retrieved_chunks = [chunks[i] for i in selected]

    return retrieved_chunks

# ============================================================
# GENERATION
# ============================================================

def generate_text(prompt):

    out = generator(
        prompt,
        max_new_tokens=120,
        do_sample=False
    )

    return out[0]["generated_text"]

# ============================================================
# CLEAN OUTPUT
# ============================================================

def clean_output(text):

    lines = text.split("\n")

    bad_patterns = [
        "rules:",
        "instructions:",
        "task:",
        "retrieved knowledge:",
        "parent context:",
        "write a clean",
        "bullet point",
        "formatting"
    ]

    cleaned = []

    for line in lines:

        lower = line.lower().strip()

        if any(p in lower for p in bad_patterns):
            continue

        cleaned.append(line)

    text = "\n".join(cleaned)

    text = re.sub(r"\s+", " ", text)

    return text.strip()

# ============================================================
# CONFIDENCE METRICS
# ============================================================

def semantic_coherence(output, retrieved_docs):

    if len(retrieved_docs) == 0:
        return 0.0

    output_emb = embedder.encode(
        [output],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    doc_embs = embedder.encode(
        retrieved_docs,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    sims = cosine_similarity(output_emb, doc_embs)[0]

    return float(np.mean(sims))

def coverage_score(output, task):

    bullet_items = re.findall(r"- (.*)", task)

    output_lower = output.lower()

    hits = 0

    for item in bullet_items:

        words = item.lower().split()

        if any(w in output_lower for w in words):
            hits += 1

    return hits / max(len(bullet_items), 1)

def validator_score(output):

    if len(output.split()) < 20:
        return 0.2

    if "i don't know" in output.lower():
        return 0.0

    return 1.0

def information_density(output):

    words = output.split()

    unique = set(words)

    return len(unique) / max(len(words), 1)

def prompt_leakage_penalty(output):

    bad = [
        "rules:",
        "instructions:",
        "task:",
        "retrieved knowledge:",
        "parent context:"
    ]

    output_lower = output.lower()

    count = 0

    for b in bad:
        if b in output_lower:
            count += 1

    return min(count * 0.25, 1.0)

# ============================================================
# NODE EXECUTION
# ============================================================

node_outputs = {}
node_confidences = {}

for node in execution_order:

    print("\n" + "="*70)
    print(f"GENERATING NODE: {node}")
    print("="*70)

    retrieved = mmr_retrieval(node, k=3)

    parent_text = ""

    for parent in G.predecessors(node):
        parent_text += node_outputs[parent] + "\n"

    prompt = dedent(f"""
    You are a technical educator teaching a class.
    You focus on explaining the below topic.

    Topic:
    {node}

    Content to cover:
    {node_tasks[node]}

    Known Knowledge:
    {parent_text}

    Retrieved evidences:
    {'\n'.join(retrieved)}

    Rules:
    Never do repetitions.
    Make sure to avoid hallucination.
    No need to re-explain the Known knowledge.

    Write a good technical explanation for your topic.
    """)

    raw_output = generate_text(prompt)

    output = raw_output

    # ========================================================
    # CONFIDENCE
    # ========================================================

    coherence = semantic_coherence(output, retrieved)

    coverage = coverage_score(output, node_tasks[node])

    validator = validator_score(output)

    density = information_density(output)

    leakage = prompt_leakage_penalty(output)

    confidence = (
        0.35 * coherence +
        0.25 * coverage +
        0.20 * validator +
        0.20 * density
        -
        0.20 * leakage
    )

    confidence = max(0.0, min(1.0, confidence))

    node_outputs[node] = output

    node_confidences[node] = {
        "semantic_coherence": round(coherence, 3),
        "coverage_score": round(coverage, 3),
        "validator_score": round(validator, 3),
        "information_density": round(density, 3),
        "prompt_leakage_penalty": round(leakage, 3),
        "final_confidence": round(confidence, 3)
    }

    print("\nRETRIEVED CHUNKS:")
    for r in retrieved:
        print("-", r)

    print("\nCONFIDENCE COMPONENTS:")
    print(node_confidences[node])

    print("\nFINAL OUTPUT:\n")
    print(output)

# ============================================================
# FINAL DOCUMENT
# ============================================================

print("\n" + "#"*80)
print("FINAL MERGED DOCUMENT")
print("#"*80)

for node in execution_order:

    print(f"\n\n## {node.title()}\n")
    print(node_outputs[node])

# ============================================================
# CONFIDENCE SUMMARY
# ============================================================

print("\n" + "#"*80)
print("CONFIDENCE SUMMARY")
print("#"*80)

for node, conf in node_confidences.items():

    print(f"\nNODE: {node}")

    for k, v in conf.items():
        print(f"{k}: {v}")

print("\nExperiment complete.")
