# =============================================================================
# MODEL-09 — CURRENT 9-MODEL BENCHMARK DEPLOYMENT
#
# STREAMLIT DEPLOYMENT
#
# PIPELINE:
#
# RAW HIV-1 PROTEIN SEQUENCE
#          ↓
# ESM-2 facebook/esm2_t33_650M_UR50D
#          ↓
# 1280-D RESIDUE EMBEDDINGS
#          ↓
# COMPLETE 48-AA NON-OVERLAPPING CHUNKS
#          ↓
# MEAN + MAX
#          ↓
# 2560-D TOKEN REPRESENTATION
#          ↓
# PAD TO 91 TOKENS
#          ↓
# TRAIN-ONLY STANDARDIZATION
#          ↓
# MODEL-09
#          ↓
# SIGMOID PROBABILITY
#          ↓
# FROZEN VALIDATION THRESHOLD = 0.71
#          ↓
# NON-RECOMBINANT / RECOMBINANT
#
# CURRENT 9-MODEL BENCHMARK ARTIFACTS
#
# MODEL-09_Bidirectional_Attention_Transformer_Encoder.pt
# MODEL-09_BENCHMARK_TRAIN_MEAN.npy
# MODEL-09_BENCHMARK_TRAIN_STD.npy
# MODEL-09_BENCHMARK_FROZEN_THRESHOLD.txt
#
# =============================================================================


# =============================================================================
# 1. IMPORTS
# =============================================================================

import os
import re
import time
from pathlib import Path

import numpy as np
import streamlit as st

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, EsmModel


# =============================================================================
# 2. STREAMLIT PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="MODEL-09 HIV-1 Recombinant Classifier",
    page_icon="🧬",
    layout="wide"
)


# =============================================================================
# 3. DEPLOYMENT CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(
    os.environ.get(
        "PROJECT_ROOT",
        "."
    )
)

ARTIFACT_DIR = (
    PROJECT_ROOT
    / "artifacts"
)


# -----------------------------------------------------------------------------
# CURRENT BENCHMARK PARAMETERS
# -----------------------------------------------------------------------------

ESM_MODEL_NAME = (
    "facebook/esm2_t33_650M_UR50D"
)

ESM2_DIMENSION = 1280

CHUNK_SIZE = 48

CHUNK_STRIDE = 48

INPUT_DIM = 2560

TOKEN_LENGTH = 91

MODEL_DIM = 96

ATTENTION_HEADS = 4

BASE_DROPOUT = 0.30

ATTENTION_DROPOUT = 0.25

REPRESENTATION_NOISE = 0.015

SEED = 42


# -----------------------------------------------------------------------------
# ARTIFACT FILES
# -----------------------------------------------------------------------------

MODEL_PATH = (
    ARTIFACT_DIR
    / "MODEL-09_Bidirectional_Attention_Transformer_Encoder.pt"
)

TRAIN_MEAN_PATH = (
    ARTIFACT_DIR
    / "MODEL-09_BENCHMARK_TRAIN_MEAN.npy"
)

TRAIN_STD_PATH = (
    ARTIFACT_DIR
    / "MODEL-09_BENCHMARK_TRAIN_STD.npy"
)

THRESHOLD_PATH = (
    ARTIFACT_DIR
    / "MODEL-09_BENCHMARK_FROZEN_THRESHOLD.txt"
)


# =============================================================================
# 4. DEVICE
# =============================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =============================================================================
# 5. RANDOM SEED
# =============================================================================

torch.manual_seed(
    SEED
)

np.random.seed(
    SEED
)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        SEED
    )


# =============================================================================
# 6. ARTIFACT VALIDATION
# =============================================================================

def validate_artifacts():

    required = {

        "MODEL-09 checkpoint":
            MODEL_PATH,

        "Training mean":
            TRAIN_MEAN_PATH,

        "Training std":
            TRAIN_STD_PATH,

        "Frozen threshold":
            THRESHOLD_PATH

    }

    missing = []

    for name, path in required.items():

        if not path.exists():

            missing.append(
                f"{name}: {path}"
            )

    if missing:

        raise FileNotFoundError(
            "Required MODEL-09 deployment artifacts are missing:\n\n"
            + "\n".join(
                missing
            )
            + "\n\n"
            "Expected directory:\n"
            f"{ARTIFACT_DIR}"
        )


# =============================================================================
# 7. LOAD STANDARDIZATION ARTIFACTS
# =============================================================================

@st.cache_resource
def load_standardization():

    mean = np.load(
        TRAIN_MEAN_PATH
    ).astype(
        np.float32
    )

    std = np.load(
        TRAIN_STD_PATH
    ).astype(
        np.float32
    )

    if mean.shape != (
        1,
        1,
        INPUT_DIM
    ):

        raise RuntimeError(
            "Invalid training mean shape: "
            f"{mean.shape}. "
            f"Expected (1, 1, {INPUT_DIM})."
        )

    if std.shape != (
        1,
        1,
        INPUT_DIM
    ):

        raise RuntimeError(
            "Invalid training std shape: "
            f"{std.shape}. "
            f"Expected (1, 1, {INPUT_DIM})."
        )

    if not np.all(
        np.isfinite(mean)
    ):

        raise RuntimeError(
            "Training mean contains "
            "non-finite values."
        )

    if not np.all(
        np.isfinite(std)
    ):

        raise RuntimeError(
            "Training std contains "
            "non-finite values."
        )

    if np.any(
        std <= 0
    ):

        raise RuntimeError(
            "Training std contains "
            "zero or negative values."
        )

    return (
        mean,
        std
    )


# =============================================================================
# 8. LOAD FROZEN THRESHOLD
# =============================================================================

@st.cache_resource
def load_frozen_threshold():

    with open(
        THRESHOLD_PATH,
        "r"
    ) as f:

        raw = f.read().strip()

    try:

        threshold = float(
            raw
        )

    except ValueError:

        raise RuntimeError(
            "Frozen threshold artifact does not "
            "contain a valid floating-point value."
        )

    if not (
        0.0
        <= threshold
        <= 1.0
    ):

        raise RuntimeError(
            f"Invalid frozen threshold: {threshold}"
        )

    return float(
        threshold
    )


# =============================================================================
# 9. LOCAL ATTENTION BLOCK
# =============================================================================

class LocalAttentionBlock(
    nn.Module
):

    def __init__(
        self,
        dim=MODEL_DIM,
        heads=ATTENTION_HEADS,
        dropout=ATTENTION_DROPOUT
    ):

        super().__init__()

        self.norm1 = nn.LayerNorm(
            dim
        )

        self.attn = nn.MultiheadAttention(
            dim,
            heads,
            dropout=dropout,
            batch_first=True
        )

        self.dropout1 = nn.Dropout(
            dropout
        )

        self.norm2 = nn.LayerNorm(
            dim
        )

        self.ff = nn.Sequential(

            nn.Linear(
                dim,
                dim * 2
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                dim * 2,
                dim
            ),

            nn.Dropout(
                dropout
            )

        )

    def forward(
        self,
        x
    ):

        z = self.norm1(
            x
        )

        attention_output, _ = (
            self.attn(
                z,
                z,
                z,
                need_weights=False
            )
        )

        x = (
            x
            +
            self.dropout1(
                attention_output
            )
        )

        x = (
            x
            +
            self.ff(
                self.norm2(
                    x
                )
            )
        )

        return x


# =============================================================================
# 10. GLOBAL ATTENTION BLOCK
# =============================================================================

class GlobalAttentionBlock(
    LocalAttentionBlock
):

    pass


# =============================================================================
# 11. ATTENTION POOLING
# =============================================================================

class AttentionPooling(
    nn.Module
):

    def __init__(
        self,
        dim
    ):

        super().__init__()

        self.score = nn.Sequential(

            nn.Linear(
                dim,
                dim // 2
            ),

            nn.Tanh(),

            nn.Linear(
                dim // 2,
                1
            )

        )

    def forward(
        self,
        x
    ):

        scores = (
            self.score(
                x
            )
            .squeeze(-1)
        )

        weights = torch.softmax(
            scores,
            dim=1
        )

        pooled = torch.sum(
            x
            *
            weights.unsqueeze(-1),
            dim=1
        )

        return (
            pooled,
            weights
        )


# =============================================================================
# 12. MODEL-09
# =============================================================================

class BidirectionalAttentionTransformerEncoder(
    nn.Module
):

    def __init__(
        self,
        input_dim=INPUT_DIM,
        model_dim=MODEL_DIM,
        heads=ATTENTION_HEADS,
        max_tokens=TOKEN_LENGTH
    ):

        super().__init__()

        self.input_projection = nn.Sequential(

            nn.LayerNorm(
                input_dim
            ),

            nn.Linear(
                input_dim,
                model_dim
            ),

            nn.GELU(),

            nn.Dropout(
                BASE_DROPOUT
            )

        )

        self.position_embedding = nn.Parameter(

            torch.zeros(
                1,
                max_tokens,
                model_dim
            )

        )

        nn.init.normal_(
            self.position_embedding,
            std=0.02
        )

        self.local_attention = (
            LocalAttentionBlock()
        )

        self.global_attention = (
            GlobalAttentionBlock()
        )

        self.pool = AttentionPooling(
            model_dim
        )

        self.classifier = nn.Sequential(

            nn.LayerNorm(
                model_dim
            ),

            nn.Linear(
                model_dim,
                48
            ),

            nn.GELU(),

            nn.Dropout(
                BASE_DROPOUT
            ),

            nn.Linear(
                48,
                1
            )

        )

    def forward(
        self,
        x,
        training_noise=False
    ):

        x = self.input_projection(
            x
        )

        if (
            self.training
            and training_noise
            and REPRESENTATION_NOISE > 0
        ):

            x = (
                x
                +
                torch.randn_like(x)
                *
                REPRESENTATION_NOISE
            )

        T = x.size(1)

        if T > TOKEN_LENGTH:

            raise ValueError(
                f"Input contains {T} tokens. "
                f"Maximum supported token length is "
                f"{TOKEN_LENGTH}."
            )

        x = (
            x
            +
            self.position_embedding[
                :,
                :T
            ]
        )

        x = self.local_attention(
            x
        )

        x = self.global_attention(
            x
        )

        pooled, attention = (
            self.pool(
                x
            )
        )

        logits = (
            self.classifier(
                pooled
            )
            .squeeze(-1)
        )

        return (
            logits,
            attention
        )


# =============================================================================
# 13. LOAD MODEL-09 CHECKPOINT
# =============================================================================

@st.cache_resource
def load_model():

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model = (
        BidirectionalAttentionTransformerEncoder()
    )

    if (
        isinstance(
            checkpoint,
            dict
        )
        and
        "model_state_dict"
        in checkpoint
    ):

        state_dict = (
            checkpoint[
                "model_state_dict"
            ]
        )

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    return model


# =============================================================================
# 14. LOAD ESM-2
# =============================================================================

@st.cache_resource
def load_esm():

    tokenizer = (
        AutoTokenizer.from_pretrained(
            ESM_MODEL_NAME
        )
    )

    esm_model = (
        EsmModel.from_pretrained(
            ESM_MODEL_NAME
        )
    )

    esm_model = esm_model.to(
        DEVICE
    )

    esm_model.eval()

    return (
        tokenizer,
        esm_model
    )


# =============================================================================
# 15. CLEAN PROTEIN SEQUENCE
# =============================================================================

def clean_sequence(
    sequence
):

    sequence = (
        sequence
        .upper()
        .strip()
    )

    sequence = re.sub(
        r"\s+",
        "",
        sequence
    )

    sequence = sequence.replace(
        ">",
        ""
    )

    if not sequence:

        raise ValueError(
            "Protein sequence is empty."
        )

    valid_amino_acids = set(
        "ACDEFGHIKLMNPQRSTVWYBXZJUO"
    )

    invalid = sorted(
        set(sequence)
        -
        valid_amino_acids
    )

    if invalid:

        raise ValueError(
            "Invalid amino-acid characters found: "
            +
            ", ".join(
                invalid
            )
        )

    return sequence


# =============================================================================
# 16. EXTRACT ESM-2 RESIDUE EMBEDDINGS
# =============================================================================

def extract_residue_embeddings(
    sequence,
    tokenizer,
    esm_model
):

    sequence = clean_sequence(
        sequence
    )

    encoded = tokenizer(
        sequence,
        return_tensors="pt",
        add_special_tokens=True
    )

    encoded = {
        key: value.to(
            DEVICE
        )
        for key, value
        in encoded.items()
    }

    with torch.no_grad():

        outputs = esm_model(
            **encoded
        )

    hidden = (
        outputs.last_hidden_state
        .squeeze(0)
    )

    # ESM special tokens:
    # position 0 = BOS
    # final position = EOS
    #
    # Remove both to obtain
    # residue-level embeddings.

    residue_embeddings = (
        hidden[
            1:-1
        ]
    )

    residue_embeddings = (
        residue_embeddings
        .float()
        .cpu()
        .numpy()
    )

    if residue_embeddings.shape[0] != len(
        sequence
    ):

        raise RuntimeError(
            "ESM-2 residue count does not "
            "match input sequence length.\n"
            f"Sequence length: {len(sequence)}\n"
            f"Embedding residues: "
            f"{residue_embeddings.shape[0]}"
        )

    if residue_embeddings.shape[1] != (
        ESM2_DIMENSION
    ):

        raise RuntimeError(
            "Unexpected ESM-2 embedding "
            f"dimension: {residue_embeddings.shape[1]}. "
            f"Expected {ESM2_DIMENSION}."
        )

    return (
        sequence,
        residue_embeddings
    )


# =============================================================================
# 17. RESIDUE → COMPLETE 48-AA TOKENS
# =============================================================================

def residue_to_tokens(
    residue_embeddings
):

    residues = (
        residue_embeddings.shape[0]
    )

    raw_tokens = []

    start = 0

    while (
        start
        +
        CHUNK_SIZE
        <= residues
    ):

        chunk = (
            residue_embeddings[
                start:
                start
                +
                CHUNK_SIZE
            ]
        )

        mean_embedding = (
            chunk.mean(
                axis=0
            )
        )

        max_embedding = (
            chunk.max(
                axis=0
            )
        )

        token = np.concatenate(
            [
                mean_embedding,
                max_embedding
            ],
            axis=0
        )

        raw_tokens.append(
            token
        )

        start += CHUNK_STRIDE

    if not raw_tokens:

        raise ValueError(
            "Protein sequence is shorter than "
            f"one complete {CHUNK_SIZE}-aa chunk."
        )

    tokens = np.asarray(
        raw_tokens,
        dtype=np.float32
    )

    if tokens.shape[1] != (
        INPUT_DIM
    ):

        raise RuntimeError(
            "Unexpected token dimension: "
            f"{tokens.shape[1]}. "
            f"Expected {INPUT_DIM}."
        )

    return tokens


# =============================================================================
# 18. PAD / TRUNCATE TO 91 TOKENS
# =============================================================================

def pad_to_benchmark_length(
    tokens
):

    raw_count = (
        tokens.shape[0]
    )

    if raw_count > TOKEN_LENGTH:

        raise ValueError(
            f"Input produces {raw_count} complete "
            f"48-aa tokens, which exceeds the "
            f"benchmark maximum of {TOKEN_LENGTH}."
        )

    padded = np.zeros(
        (
            TOKEN_LENGTH,
            INPUT_DIM
        ),
        dtype=np.float32
    )

    padded[
        :raw_count
    ] = tokens

    return padded


# =============================================================================
# 19. TRAIN-ONLY STANDARDIZATION
# =============================================================================

def standardize_tokens(
    tokens,
    train_mean,
    train_std
):

    standardized = (
        tokens
        -
        train_mean
    ) / train_std

    standardized = np.asarray(
        standardized,
        dtype=np.float32
    )

    if not np.all(
        np.isfinite(
            standardized
        )
    ):

        raise RuntimeError(
            "Standardized token matrix "
            "contains non-finite values."
        )

    return standardized


# =============================================================================
# 20. COMPLETE PREPROCESSING PIPELINE
# =============================================================================

def preprocess_sequence(
    sequence,
    tokenizer,
    esm_model,
    train_mean,
    train_std
):

    clean_seq, residue_embeddings = (
        extract_residue_embeddings(
            sequence,
            tokenizer,
            esm_model
        )
    )

    raw_tokens = (
        residue_to_tokens(
            residue_embeddings
        )
    )

    raw_token_count = (
        raw_tokens.shape[0]
    )

    padded_tokens = (
        pad_to_benchmark_length(
            raw_tokens
        )
    )

    standardized_tokens = (
        standardize_tokens(
            padded_tokens,
            train_mean,
            train_std
        )
    )

    tensor = torch.tensor(
        standardized_tokens,
        dtype=torch.float32,
        device=DEVICE
    ).unsqueeze(0)

    return {

        "sequence":
            clean_seq,

        "sequence_length":
            len(clean_seq),

        "residue_embeddings":
            residue_embeddings,

        "raw_tokens":
            raw_tokens,

        "raw_token_count":
            raw_token_count,

        "padded_tokens":
            padded_tokens,

        "standardized_tokens":
            standardized_tokens,

        "tensor":
            tensor

    }


# =============================================================================
# 21. MODEL PREDICTION
# =============================================================================

def predict(
    model,
    tensor,
    threshold
):

    model.eval()

    with torch.no_grad():

        logits, attention = (
            model(
                tensor,
                training_noise=False
            )
        )

        probability = (
            torch.sigmoid(
                logits
            )
            .item()
        )

    prediction = int(
        probability
        >= threshold
    )

    if prediction == 1:

        label = (
            "RECOMBINANT"
        )

    else:

        label = (
            "NON-RECOMBINANT"
        )

    attention = (
        attention
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
    )

    return {

        "probability":
            float(
                probability
            ),

        "prediction":
            prediction,

        "label":
            label,

        "attention":
            attention

    }


# =============================================================================
# 22. INITIALIZATION
# =============================================================================

try:

    validate_artifacts()

    train_mean, train_std = (
        load_standardization()
    )

    frozen_threshold = (
        load_frozen_threshold()
    )

    model = load_model()

    tokenizer, esm_model = (
        load_esm()
    )

    initialization_error = None

except Exception as e:

    initialization_error = str(
        e
    )


# =============================================================================
# 23. SIDEBAR
# =============================================================================

with st.sidebar:

    st.header(
        "MODEL-09 Configuration"
    )

    st.write(
        f"ESM-2 dimension: "
        f"`{ESM2_DIMENSION}`"
    )

    st.write(
        f"Chunk size: "
        f"`{CHUNK_SIZE}`"
    )

    st.write(
        f"Chunk stride: "
        f"`{CHUNK_STRIDE}`"
    )

    st.write(
        f"Token dimension: "
        f"`{INPUT_DIM}`"
    )

    st.write(
        f"Token length: "
        f"`{TOKEN_LENGTH}`"
    )

    st.write(
        f"MODEL dimension: "
        f"`{MODEL_DIM}`"
    )

    st.write(
        f"Attention heads: "
        f"`{ATTENTION_HEADS}`"
    )

    st.write(
        f"Device: "
        f"`{DEVICE}`"
    )

    st.divider()

    if initialization_error is None:

        st.success(
            "Deployment artifacts loaded"
        )

        st.metric(
            "Frozen threshold",
            f"{frozen_threshold:.4f}"
        )

    else:

        st.error(
            "Deployment initialization failed"
        )


# =============================================================================
# 24. MAIN TITLE
# =============================================================================

st.title(
    "🧬 MODEL-09 HIV-1 Recombinant Classifier"
)

st.caption(
    "Current 9-model benchmark deployment"
)


# =============================================================================
# 25. DEPLOYMENT STATUS
# =============================================================================

if initialization_error is not None:

    st.error(
        "MODEL-09 could not be initialized."
    )

    st.code(
        initialization_error
    )

    st.info(
        "Ensure the following files are present "
        "inside the artifacts/ directory:\n\n"
        "MODEL-09_Bidirectional_Attention_Transformer_Encoder.pt\n"
        "MODEL-09_BENCHMARK_TRAIN_MEAN.npy\n"
        "MODEL-09_BENCHMARK_TRAIN_STD.npy\n"
        "MODEL-09_BENCHMARK_FROZEN_THRESHOLD.txt"
    )

    st.stop()


# =============================================================================
# 26. MODEL INFORMATION
# =============================================================================

with st.expander(
    "Deployment pipeline"
):

    st.markdown(
        """
**Raw protein sequence**

↓

**ESM-2 t33 650M**

↓

**1280-D residue embeddings**

↓

**Complete 48-aa non-overlapping chunks**

↓

**Mean + Max pooling**

↓

**2560-D token representation**

↓

**Pad to 91 tokens**

↓

**Train-only benchmark standardization**

↓

**MODEL-09 Bidirectional Attention Transformer**

↓

**Sigmoid probability**

↓

**Frozen validation threshold**

↓

**Recombinant / Non-Recombinant**
"""
    )


# =============================================================================
# 27. INPUT
# =============================================================================

st.subheader(
    "Protein Sequence"
)

sequence_input = st.text_area(

    "Paste an HIV-1 protein sequence",

    height=220,

    placeholder=(
        "Example:\n"
        "MRVMGTQKNYSLLWRWGIMIFGILMACSANNLWVTVYYGVPVW..."
    )

)


# =============================================================================
# 28. PREDICTION BUTTON
# =============================================================================

predict_button = st.button(
    "🔬 Run MODEL-09 Prediction",
    type="primary",
    use_container_width=True
)


# =============================================================================
# 29. PREDICTION
# =============================================================================

if predict_button:

    if not sequence_input.strip():

        st.warning(
            "Please provide a protein sequence."
        )

        st.stop()

    try:

        start_time = time.time()

        with st.spinner(
            "Running ESM-2 and MODEL-09..."
        ):

            processed = (
                preprocess_sequence(
                    sequence_input,
                    tokenizer,
                    esm_model,
                    train_mean,
                    train_std
                )
            )

            result = predict(
                model,
                processed[
                    "tensor"
                ],
                frozen_threshold
            )

        elapsed = (
            time.time()
            -
            start_time
        )

        # ---------------------------------------------------------------------
        # RESULTS
        # ---------------------------------------------------------------------

        st.divider()

        st.subheader(
            "MODEL-09 Prediction"
        )

        if result[
            "prediction"
        ] == 1:

            st.error(
                "### RECOMBINANT"
            )

        else:

            st.success(
                "### NON-RECOMBINANT"
            )

        col1, col2, col3 = st.columns(
            3
        )

        with col1:

            st.metric(
                "Recombinant Probability",
                f"{result['probability']:.8f}"
            )

        with col2:

            st.metric(
                "Frozen Threshold",
                f"{frozen_threshold:.4f}"
            )

        with col3:

            st.metric(
                "Sequence Length",
                f"{processed['sequence_length']:,} aa"
            )

        # ---------------------------------------------------------------------
        # REPRESENTATION INFORMATION
        # ---------------------------------------------------------------------

        st.subheader(
            "Representation"
        )

        rep1, rep2, rep3 = st.columns(
            3
        )

        with rep1:

            st.metric(
                "ESM-2 Residues",
                f"{processed['residue_embeddings'].shape[0]:,}"
            )

        with rep2:

            st.metric(
                "Complete 48-aa Tokens",
                str(
                    processed[
                        "raw_token_count"
                    ]
                )
            )

        with rep3:

            st.metric(
                "Final Tokens",
                str(
                    TOKEN_LENGTH
                )
            )

        st.caption(
            "Input mode: raw protein sequence → "
            "ESM-2 → complete 48-aa tokens → "
            "2560-D mean+max representation → "
            "91-token benchmark representation."
        )

        # ---------------------------------------------------------------------
        # PROBABILITY INTERPRETATION
        # ---------------------------------------------------------------------

        st.subheader(
            "Prediction Probability"
        )

        st.progress(
            min(
                max(
                    result[
                        "probability"
                    ],
                    0.0
                ),
                1.0
            )
        )

        st.write(
            f"Probability = "
            f"**{result['probability']:.8f}**"
        )

        st.write(
            f"Decision rule: "
            f"probability ≥ "
            f"**{frozen_threshold:.4f}** "
            f"→ Recombinant"
        )

        # ---------------------------------------------------------------------
        # ATTENTION
        # ---------------------------------------------------------------------

        with st.expander(
            "MODEL-09 attention information"
        ):

            attention = result[
                "attention"
            ]

            st.write(
                "Attention vector shape:",
                attention.shape
            )

            top_indices = np.argsort(
                attention
            )[::-1][
                :10
            ]

            attention_rows = []

            for idx in top_indices:

                attention_rows.append({

                    "Token":
                        int(
                            idx
                        )
                        + 1,

                    "Attention":
                        float(
                            attention[
                                idx
                            ]
                        )

                })

            st.dataframe(
                attention_rows,
                use_container_width=True
            )

        # ---------------------------------------------------------------------
        # TECHNICAL DETAILS
        # ---------------------------------------------------------------------

        with st.expander(
            "Technical processing details"
        ):

            st.write(
                "Model:",
                "MODEL-09 Bidirectional Attention Transformer Encoder"
            )

            st.write(
                "ESM-2:",
                ESM_MODEL_NAME
            )

            st.write(
                "ESM-2 dimension:",
                ESM2_DIMENSION
            )

            st.write(
                "Chunk size:",
                CHUNK_SIZE
            )

            st.write(
                "Chunk stride:",
                CHUNK_STRIDE
            )

            st.write(
                "Token dimension:",
                INPUT_DIM
            )

            st.write(
                "Benchmark token length:",
                TOKEN_LENGTH
            )

            st.write(
                "Model dimension:",
                MODEL_DIM
            )

            st.write(
                "Attention heads:",
                ATTENTION_HEADS
            )

            st.write(
                "Frozen threshold:",
                f"{frozen_threshold:.10f}"
            )

            st.write(
                "Device:",
                str(
                    DEVICE
                )
            )

            st.write(
                "Inference time:",
                f"{elapsed:.2f} seconds"
            )

    except Exception as e:

        st.error(
            "Prediction failed."
        )

        st.exception(
            e
        )


# =============================================================================
# 30. FOOTER
# =============================================================================

st.divider()

st.caption(
    "MODEL-09 current 9-model benchmark deployment | "
    "Frozen 537-sequence homology-aware benchmark | "
    "ESM-2 residue embeddings | "
    "48-aa complete chunks | "
    "2560-D mean+max tokens | "
    "91-token representation"
)
