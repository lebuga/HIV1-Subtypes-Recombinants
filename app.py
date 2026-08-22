# =============================================================================
# MODEL-09 — AUTHORITATIVE HIV-1 RECOMBINANT CLASSIFIER
#
# DEPLOYMENT VERSION
#
# VERIFIED REPRESENTATION:
#
# RAW AMINO-ACID SEQUENCE
#          ↓
# ESM-2 t33 650M
#          ↓
# RESIDUE-LEVEL EMBEDDINGS (1280-D)
#          ↓
# 48-AA NON-OVERLAPPING CHUNKS
#          ↓
#      MEAN + MAX
#          ↓
# 2560-D TOKEN
#          ↓
# 91 TOKENS
#          ↓
# FROZEN TRAINING STANDARDIZATION
#          ↓
# MODEL-09 TEACHER
#          ↓
# RECOMBINANT PROBABILITY
#
# IMPORTANT:
#   - NO retraining
#   - NO historical 0.88 threshold
#   - MEAN + MAX, NOT MEAN + STD
#   - Training mean/std are frozen artifacts
# =============================================================================


import os

import numpy as np
import streamlit as st

import torch
import torch.nn as nn

from transformers import (
    AutoTokenizer,
    AutoModel
)


# =============================================================================
# 1. MODEL CONFIGURATION
# =============================================================================

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"

ESM_DIM = 1280

CHUNK_SIZE = 48
CHUNK_STRIDE = 48

TOKEN_FEATURE_DIM = 2560
TOKEN_LENGTH = 91

MODEL_DIM = 96
ATTENTION_HEADS = 4

ATTENTION_DROPOUT = 0.25
BASE_DROPOUT = 0.30

DEFAULT_THRESHOLD = 0.50

REPRESENTATION_TYPE = "MEAN + MAX"

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =============================================================================
# 2. STREAMLIT PAGE
# =============================================================================

st.set_page_config(
    page_title="MODEL-09 HIV-1 Recombinant Classifier",
    page_icon="🧬",
    layout="wide"
)


# =============================================================================
# 3. APPLICATION TITLE
# =============================================================================

st.title(
    "MODEL-09 HIV-1 Recombinant Classifier"
)

st.write(
    """
    Whole-protein HIV-1 recombinant-status prediction using
    the verified MODEL-09 attention-based teacher pipeline.
    """
)


# =============================================================================
# 4. LOCATE DEPLOYMENT FILES
# =============================================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

TEACHER_PATH = os.path.join(
    BASE_DIR,
    "MODEL-09_Bidirectional_Attention_Transformer_Encoder.pt"
)

TRAIN_MEAN_PATH = os.path.join(
    BASE_DIR,
    "MODEL09_TRAIN_MEAN.npy"
)

TRAIN_STD_PATH = os.path.join(
    BASE_DIR,
    "MODEL09_TRAIN_STD.npy"
)


# =============================================================================
# 5. LOCAL ATTENTION BLOCK
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
            embed_dim=dim,
            num_heads=heads,
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

        attention_output, _ = self.attn(
            z,
            z,
            z,
            need_weights=False
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
                self.norm2(x)
            )
        )

        return x


# =============================================================================
# 6. GLOBAL ATTENTION BLOCK
# =============================================================================

class GlobalAttentionBlock(
    LocalAttentionBlock
):

    pass


# =============================================================================
# 7. ATTENTION POOLING
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

        scores = self.score(
            x
        ).squeeze(-1)

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
# 8. MODEL-09 TEACHER
# =============================================================================

class BidirectionalAttentionTransformerEncoder(
    nn.Module
):

    def __init__(
        self,
        input_dim=TOKEN_FEATURE_DIM,
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

        # Deployment must never use training noise.
        if (
            self.training
            and training_noise
        ):

            raise RuntimeError(
                "training_noise is not allowed "
                "during deployment."
            )

        T = x.size(1)

        if T > TOKEN_LENGTH:

            raise RuntimeError(
                f"Input has {T} tokens but "
                f"MODEL-09 supports maximum "
                f"{TOKEN_LENGTH} tokens."
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
            self.pool(x)
        )

        logits = self.classifier(
            pooled
        ).squeeze(-1)

        return (
            logits,
            attention
        )


# =============================================================================
# 9. LOAD TRAINING STANDARDIZATION
# =============================================================================

@st.cache_resource
def load_standardization():

    if not os.path.isfile(
        TRAIN_MEAN_PATH
    ):

        raise FileNotFoundError(
            "MODEL09_TRAIN_MEAN.npy not found:\n"
            f"{TRAIN_MEAN_PATH}"
        )

    if not os.path.isfile(
        TRAIN_STD_PATH
    ):

        raise FileNotFoundError(
            "MODEL09_TRAIN_STD.npy not found:\n"
            f"{TRAIN_STD_PATH}"
        )

    train_mean = np.load(
        TRAIN_MEAN_PATH
    ).astype(
        np.float32
    )

    train_std = np.load(
        TRAIN_STD_PATH
    ).astype(
        np.float32
    )

    if train_mean.shape != (
        TOKEN_FEATURE_DIM,
    ):

        raise RuntimeError(
            "TRAIN_MEAN shape mismatch.\n"
            f"Found: {train_mean.shape}\n"
            f"Expected: ({TOKEN_FEATURE_DIM},)"
        )

    if train_std.shape != (
        TOKEN_FEATURE_DIM,
    ):

        raise RuntimeError(
            "TRAIN_STD shape mismatch.\n"
            f"Found: {train_std.shape}\n"
            f"Expected: ({TOKEN_FEATURE_DIM},)"
        )

    # Prevent zero division.
    train_std = np.where(
        train_std < 1e-8,
        1.0,
        train_std
    )

    return (
        train_mean,
        train_std
    )


# =============================================================================
# 10. LOAD TEACHER
# =============================================================================

@st.cache_resource
def load_teacher():

    if not os.path.isfile(
        TEACHER_PATH
    ):

        raise FileNotFoundError(
            "MODEL-09 teacher checkpoint not found:\n"
            f"{TEACHER_PATH}"
        )

    checkpoint = torch.load(
        TEACHER_PATH,
        map_location="cpu",
        weights_only=False
    )

    if not isinstance(
        checkpoint,
        dict
    ):

        raise RuntimeError(
            "Teacher checkpoint is not a dictionary."
        )

    if (
        "model_state_dict"
        not in checkpoint
    ):

        raise RuntimeError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )

    model = (
        BidirectionalAttentionTransformerEncoder()
    )

    state_dict = checkpoint[
        "model_state_dict"
    ]

    missing, unexpected = (
        model.load_state_dict(
            state_dict,
            strict=False
        )
    )

    if missing or unexpected:

        raise RuntimeError(
            "MODEL-09 checkpoint mismatch.\n\n"
            f"Missing keys: {missing}\n\n"
            f"Unexpected keys: {unexpected}"
        )

    model.eval()

    model.cpu()

    model.to(
        DEVICE
    )

    return (
        model,
        checkpoint
    )


# =============================================================================
# 11. LOAD ESM-2
# =============================================================================

@st.cache_resource
def load_esm():

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    esm_model = (
        AutoModel.from_pretrained(
            MODEL_NAME
        )
    )

    esm_model.eval()

    esm_model.to(
        DEVICE
    )

    return (
        tokenizer,
        esm_model
    )


# =============================================================================
# 12. CLEAN SEQUENCE
# =============================================================================

def clean_sequence(
    sequence
):

    sequence = (
        sequence
        .replace(
            "\n",
            ""
        )
        .replace(
            "\r",
            ""
        )
        .replace(
            " ",
            ""
        )
        .replace(
            "\t",
            ""
        )
        .upper()
    )

    return sequence


# =============================================================================
# 13. VALIDATE SEQUENCE
# =============================================================================
#
# X IS ALLOWED.
#
# X = unknown/ambiguous amino acid.
#
# =============================================================================

def validate_sequence(
    sequence
):

    allowed = set(
        "ACDEFGHIKLMNPQRSTVWYX"
    )

    invalid = sorted(
        set(sequence)
        -
        allowed
    )

    if invalid:

        return (
            False,
            "Invalid amino-acid character(s): "
            +
            ", ".join(invalid)
        )

    if len(sequence) == 0:

        return (
            False,
            "The sequence is empty."
        )

    return (
        True,
        ""
    )


# =============================================================================
# 14. ESM-2 CHUNK EMBEDDING
# =============================================================================

@torch.inference_mode()
def embed_esm_chunk(
    sequence_chunk,
    tokenizer,
    esm_model
):

    encoded = tokenizer(
        sequence_chunk,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=CHUNK_SIZE + 2
    )

    input_ids = (
        encoded[
            "input_ids"
        ]
        .to(DEVICE)
    )

    attention_mask = (
        encoded[
            "attention_mask"
        ]
        .to(DEVICE)
    )

    outputs = esm_model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

    hidden = (
        outputs.last_hidden_state
    )

    residue_embeddings = hidden[
        0,
        1:-1,
        :
    ]

    residue_embeddings = (
        residue_embeddings
        .float()
        .cpu()
        .numpy()
    )

    return residue_embeddings


# =============================================================================
# 15. FULL RESIDUE EMBEDDINGS
# =============================================================================

def extract_residue_embeddings(
    sequence,
    tokenizer,
    esm_model
):

    sequence_length = len(
        sequence
    )

    embedding_sum = np.zeros(
        (
            sequence_length,
            ESM_DIM
        ),
        dtype=np.float32
    )

    embedding_count = np.zeros(
        sequence_length,
        dtype=np.float32
    )

    starts = list(
        range(
            0,
            sequence_length,
            CHUNK_STRIDE
        )
    )

    total_chunks = len(
        starts
    )

    progress = st.progress(
        0,
        text=(
            "Computing ESM-2 "
            "residue embeddings..."
        )
    )

    for i, start in enumerate(
        starts,
        1
    ):

        end = min(
            start + CHUNK_SIZE,
            sequence_length
        )

        chunk = sequence[
            start:end
        ]

        emb = embed_esm_chunk(
            chunk,
            tokenizer,
            esm_model
        )

        expected_shape = (
            end - start,
            ESM_DIM
        )

        if emb.shape != (
            expected_shape
        ):

            raise RuntimeError(
                "Unexpected ESM-2 "
                f"embedding shape: {emb.shape}; "
                f"expected {expected_shape}"
            )

        embedding_sum[
            start:end
        ] += emb

        embedding_count[
            start:end
        ] += 1.0

        progress.progress(
            i / total_chunks,
            text=(
                f"ESM-2 chunk "
                f"{i}/{total_chunks}"
            )
        )

    progress.empty()

    if np.any(
        embedding_count == 0
    ):

        raise RuntimeError(
            "Some residues were not covered "
            "by ESM-2."
        )

    residue_embeddings = (
        embedding_sum
        /
        embedding_count[:, None]
    )

    return residue_embeddings


# =============================================================================
# 16. VERIFIED MODEL-09 REPRESENTATION
# =============================================================================
#
# THIS IS THE CRITICAL CORRECTION.
#
# Each 48-aa chunk produces:
#
#     MEAN = 1280
#     MAX  = 1280
#
# Therefore:
#
#     MEAN + MAX = 2560
#
# DO NOT CHANGE MAX TO STD.
#
# =============================================================================

def build_model09_representation(
    residue_embeddings
):

    n_residues = (
        residue_embeddings.shape[0]
    )

    tokens = []

    for start in range(
        0,
        n_residues,
        CHUNK_STRIDE
    ):

        end = min(
            start + CHUNK_SIZE,
            n_residues
        )

        chunk = residue_embeddings[
            start:end
        ]

        # -------------------------------------------------------------
        # VERIFIED MEAN POOL
        # -------------------------------------------------------------

        mean_pool = np.mean(
            chunk,
            axis=0
        )

        # -------------------------------------------------------------
        # VERIFIED MAX POOL
        # -------------------------------------------------------------

        max_pool = np.max(
            chunk,
            axis=0
        )

        # -------------------------------------------------------------
        # EXACT 2560-D TOKEN
        # -------------------------------------------------------------

        token = np.concatenate(
            [
                mean_pool,
                max_pool
            ],
            axis=0
        )

        tokens.append(
            token
        )

    if len(tokens) == 0:

        raise ValueError(
            f"Sequence has only "
            f"{n_residues} residues; "
            f"cannot create a "
            f"{CHUNK_SIZE}-residue chunk."
        )

    tokens = np.asarray(
        tokens,
        dtype=np.float32
    )

    raw_token_count = (
        tokens.shape[0]
    )

    # -------------------------------------------------------------
    # TOKEN LENGTH = 91
    # -------------------------------------------------------------

    if raw_token_count > TOKEN_LENGTH:

        tokens = tokens[
            :TOKEN_LENGTH
        ]

    elif raw_token_count < TOKEN_LENGTH:

        padding = np.zeros(
            (
                TOKEN_LENGTH
                -
                raw_token_count,
                TOKEN_FEATURE_DIM
            ),
            dtype=np.float32
        )

        tokens = np.concatenate(
            [
                tokens,
                padding
            ],
            axis=0
        )

    if tokens.shape != (
        TOKEN_LENGTH,
        TOKEN_FEATURE_DIM
    ):

        raise RuntimeError(
            "MODEL-09 representation "
            "shape mismatch.\n"
            f"Found: {tokens.shape}\n"
            f"Expected: "
            f"({TOKEN_LENGTH}, "
            f"{TOKEN_FEATURE_DIM})"
        )

    return (
        tokens,
        raw_token_count
    )


# =============================================================================
# 17. STANDARDIZATION
# =============================================================================

def standardize(
    tokens,
    train_mean,
    train_std
):

    standardized = (
        tokens
        -
        train_mean[None, :]
    ) / train_std[None, :]

    if not np.all(
        np.isfinite(
            standardized
        )
    ):

        raise RuntimeError(
            "Standardized representation "
            "contains non-finite values."
        )

    return standardized.astype(
        np.float32
    )


# =============================================================================
# 18. MODEL PREDICTION
# =============================================================================

@torch.inference_mode()
def predict(
    standardized_tokens,
    teacher,
    threshold
):

    x = torch.from_numpy(
        standardized_tokens
    ).unsqueeze(
        0
    )

    x = x.to(
        DEVICE
    )

    logits, attention = teacher(
        x,
        training_noise=False
    )

    logit = (
        logits
        .detach()
        .cpu()
        .item()
    )

    probability = float(
        torch.sigmoid(
            logits
        )
        .detach()
        .cpu()
        .item()
    )

    prediction = (
        "RECOMBINANT"
        if probability >= threshold
        else "NON-RECOMBINANT"
    )

    attention = (
        attention
        .detach()
        .cpu()
        .numpy()[0]
    )

    return (
        logit,
        probability,
        prediction,
        attention
    )


# =============================================================================
# 19. FORENSIC DIAGNOSTIC
# =============================================================================

def forensic_diagnostic(
    residue_embeddings,
    tokens,
    standardized,
    train_mean,
    train_std,
    raw_token_count
):

    st.subheader(
        "Forensic Deployment Diagnostic"
    )

    st.write(
        "This diagnostic confirms that Streamlit is using "
        "the verified MODEL-09 MEAN + MAX representation."
    )

    # -------------------------------------------------------------
    # REPRESENTATION CONFIGURATION
    # -------------------------------------------------------------

    st.markdown(
        "### Representation configuration"
    )

    diagnostic_data = {

        "Representation":
            REPRESENTATION_TYPE,

        "ESM dimension":
            ESM_DIM,

        "Chunk size":
            CHUNK_SIZE,

        "Chunk stride":
            CHUNK_STRIDE,

        "Token feature dimension":
            TOKEN_FEATURE_DIM,

        "Token length":
            TOKEN_LENGTH,

        "Raw token count":
            raw_token_count,

        "Residue embedding shape":
            str(
                residue_embeddings.shape
            ),

        "Final token shape":
            str(
                tokens.shape
            ),

        "Standardized shape":
            str(
                standardized.shape
            )
    }

    for key, value in (
        diagnostic_data.items()
    ):

        st.write(
            f"**{key}:** `{value}`"
        )

    # -------------------------------------------------------------
    # RAW TOKEN STATISTICS
    # -------------------------------------------------------------

    st.markdown(
        "### Raw MEAN + MAX token statistics"
    )

    mean_part = tokens[
        :,
        :ESM_DIM
    ]

    max_part = tokens[
        :,
        ESM_DIM:
    ]

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            "**MEAN portion**"
        )

        st.write(
            f"Min: `{mean_part.min():.8f}`"
        )

        st.write(
            f"Max: `{mean_part.max():.8f}`"
        )

        st.write(
            f"Mean: `{mean_part.mean():.8f}`"
        )

        st.write(
            f"Std: `{mean_part.std():.8f}`"
        )

    with col2:

        st.write(
            "**MAX portion**"
        )

        st.write(
            f"Min: `{max_part.min():.8f}`"
        )

        st.write(
            f"Max: `{max_part.max():.8f}`"
        )

        st.write(
            f"Mean: `{max_part.mean():.8f}`"
        )

        st.write(
            f"Std: `{max_part.std():.8f}`"
        )

    # -------------------------------------------------------------
    # STANDARDIZATION STATISTICS
    # -------------------------------------------------------------

    st.markdown(
        "### Standardization diagnostic"
    )

    st.write(
        f"Training mean shape: "
        f"`{train_mean.shape}`"
    )

    st.write(
        f"Training std shape: "
        f"`{train_std.shape}`"
    )

    st.write(
        f"Standardized min: "
        f"`{standardized.min():.8f}`"
    )

    st.write(
        f"Standardized max: "
        f"`{standardized.max():.8f}`"
    )

    st.write(
        f"Standardized mean: "
        f"`{standardized.mean():.8f}`"
    )

    st.write(
        f"Standardized std: "
        f"`{standardized.std():.8f}`"
    )

    st.write(
        f"All finite: "
        f"`{np.all(np.isfinite(standardized))}`"
    )

    # -------------------------------------------------------------
    # FIRST TOKEN
    # -------------------------------------------------------------

    st.markdown(
        "### First token diagnostic"
    )

    st.write(
        "First 10 MEAN features:"
    )

    st.code(
        np.array2string(
            tokens[
                0,
                :10
            ],
            precision=8
        )
    )

    st.write(
        "First 10 MAX features:"
    )

    st.code(
        np.array2string(
            tokens[
                0,
                ESM_DIM:
                ESM_DIM + 10
            ],
            precision=8
        )
    )

    st.write(
        "First 10 standardized features:"
    )

    st.code(
        np.array2string(
            standardized[
                0,
                :10
            ],
            precision=8
        )
    )


# =============================================================================
# 20. SIDEBAR
# =============================================================================

with st.sidebar:

    st.header(
        "MODEL-09 Configuration"
    )

    st.write(
        f"ESM-2: `{MODEL_NAME}`"
    )

    st.write(
        f"ESM dimension: `{ESM_DIM}`"
    )

    st.write(
        f"Chunk size: `{CHUNK_SIZE}`"
    )

    st.write(
        f"Chunk stride: `{CHUNK_STRIDE}`"
    )

    st.write(
        f"Representation: `{REPRESENTATION_TYPE}`"
    )

    st.write(
        f"Token dimension: `{TOKEN_FEATURE_DIM}`"
    )

    st.write(
        f"Token length: `{TOKEN_LENGTH}`"
    )

    st.write(
        f"Model dimension: `{MODEL_DIM}`"
    )

    st.write(
        f"Attention heads: `{ATTENTION_HEADS}`"
    )

    st.write(
        f"Device: `{DEVICE}`"
    )

    st.divider()

    threshold = st.slider(
        "Classification threshold",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_THRESHOLD,
        step=0.01
    )

    st.caption(
        "Deployment threshold is 0.50 by default. "
        "The historical 0.88 threshold is not used."
    )


# =============================================================================
# 21. LOAD ALL ARTIFACTS
# =============================================================================

try:

    train_mean, train_std = (
        load_standardization()
    )

    teacher, checkpoint = (
        load_teacher()
    )

    tokenizer, esm_model = (
        load_esm()
    )

    st.success(
        "MODEL-09 deployment artifacts loaded successfully."
    )

except Exception as e:

    st.error(
        "Failed to load MODEL-09 deployment artifacts."
    )

    st.exception(
        e
    )

    st.stop()


# =============================================================================
# 22. INPUT
# =============================================================================

st.subheader(
    "Enter HIV-1 protein sequence"
)

sequence_input = st.text_area(
    "Raw amino-acid sequence",
    height=250,
    placeholder=(
        "Paste an HIV-1 amino-acid sequence here..."
    )
)

run_prediction = st.button(
    "Run MODEL-09 Prediction",
    type="primary"
)


# =============================================================================
# 23. MAIN PREDICTION PIPELINE
# =============================================================================

if run_prediction:

    # -------------------------------------------------------------------------
    # CLEAN
    # -------------------------------------------------------------------------

    sequence = clean_sequence(
        sequence_input
    )

    # -------------------------------------------------------------------------
    # VALIDATE
    # -------------------------------------------------------------------------

    valid, error = (
        validate_sequence(
            sequence
        )
    )

    if not valid:

        st.error(
            error
        )

        st.stop()

    st.info(
        f"Sequence length: "
        f"{len(sequence):,} amino acids"
    )

    try:

        # =====================================================================
        # STEP 1 — ESM-2 RESIDUE EMBEDDINGS
        # =====================================================================

        with st.spinner(
            "Computing ESM-2 residue embeddings..."
        ):

            residue_embeddings = (
                extract_residue_embeddings(
                    sequence,
                    tokenizer,
                    esm_model
                )
            )

        st.success(
            "ESM-2 preprocessing complete."
        )

        st.write(
            f"Residue embedding shape: "
            f"`{residue_embeddings.shape}`"
        )

        # =====================================================================
        # STEP 2 — EXACT MEAN + MAX REPRESENTATION
        # =====================================================================

        with st.spinner(
            "Building verified MODEL-09 MEAN + MAX representation..."
        ):

            tokens, raw_token_count = (
                build_model09_representation(
                    residue_embeddings
                )
            )

        st.success(
            "MEAN + MAX representation generated."
        )

        st.write(
            f"Raw token count: `{raw_token_count}`"
        )

        st.write(
            f"Final MODEL-09 input: "
            f"`{tokens.shape}`"
        )

        # =====================================================================
        # STEP 3 — FROZEN STANDARDIZATION
        # =====================================================================

        with st.spinner(
            "Applying frozen training standardization..."
        ):

            standardized = (
                standardize(
                    tokens,
                    train_mean,
                    train_std
                )
            )

        st.success(
            "Frozen training standardization applied."
        )

        # =====================================================================
        # STEP 4 — MODEL-09 TEACHER
        # =====================================================================

        with st.spinner(
            "Running MODEL-09 teacher..."
        ):

            (
                logit,
                probability,
                prediction,
                attention
            ) = predict(
                standardized,
                teacher,
                threshold
            )

        # =====================================================================
        # RESULTS
        # =====================================================================

        st.divider()

        st.subheader(
            "MODEL-09 Prediction"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Recombinant Probability",
                f"{probability:.6f}"
            )

        with col2:

            if (
                prediction
                ==
                "RECOMBINANT"
            ):

                st.error(
                    prediction
                )

            else:

                st.success(
                    prediction
                )

        st.write(
            f"Classification threshold: "
            f"`{threshold:.2f}`"
        )

        # =====================================================================
        # FORENSIC DIAGNOSTIC
        # =====================================================================

        with st.expander(
            "🔬 Forensic Deployment Diagnostic",
            expanded=False
        ):

            forensic_diagnostic(
                residue_embeddings,
                tokens,
                standardized,
                train_mean,
                train_std,
                raw_token_count
            )

            st.markdown(
                "### Teacher output"
            )

            st.write(
                f"**Logit:** `{logit:.9f}`"
            )

            st.write(
                f"**Probability:** "
                f"`{probability:.9f}`"
            )

            st.write(
                f"**Prediction:** "
                f"`{prediction}`"
            )

            st.write(
                f"**Threshold:** "
                f"`{threshold:.2f}`"
            )

            st.write(
                f"**Teacher parameters:** "
                f"`{sum(p.numel() for p in teacher.parameters()):,}`"
            )

            st.write(
                f"**Device:** `{DEVICE}`"
            )

        # =====================================================================
        # TECHNICAL DETAILS
        # =====================================================================

        with st.expander(
            "Technical details"
        ):

            st.write(
                f"Input sequence length: "
                f"`{len(sequence)}` aa"
            )

            st.write(
                f"Representation: "
                f"`{REPRESENTATION_TYPE}`"
            )

            st.write(
                f"Residue embedding shape: "
                f"`{residue_embeddings.shape}`"
            )

            st.write(
                f"Raw token count: "
                f"`{raw_token_count}`"
            )

            st.write(
                f"Final token shape: "
                f"`{tokens.shape}`"
            )

            st.write(
                f"Standardized shape: "
                f"`{standardized.shape}`"
            )

            st.write(
                f"MODEL-09 logit: "
                f"`{logit:.9f}`"
            )

            st.write(
                f"MODEL-09 probability: "
                f"`{probability:.9f}`"
            )

            st.write(
                f"MODEL-09 parameters: "
                f"`{sum(p.numel() for p in teacher.parameters()):,}`"
            )

            st.write(
                f"Device: `{DEVICE}`"
            )

            if isinstance(
                checkpoint,
                dict
            ):

                st.write(
                    "Checkpoint metadata:"
                )

                metadata_keys = [
                    "model_id",
                    "model_name",
                    "esm2_dimension",
                    "chunk_size",
                    "chunk_stride",
                    "input_dim",
                    "token_length",
                    "model_dim",
                    "attention_heads",
                    "seed"
                ]

                for key in metadata_keys:

                    if key in checkpoint:

                        st.write(
                            f"- {key}: "
                            f"`{checkpoint[key]}`"
                        )

        # =====================================================================
        # ATTENTION WEIGHTS
        # =====================================================================

        with st.expander(
            "MODEL-09 attention weights"
        ):

            st.line_chart(
                attention
            )

    except Exception as e:

        st.error(
            "MODEL-09 prediction failed."
        )

        st.exception(
            e
        )


# =============================================================================
# 24. FOOTER
# =============================================================================

st.divider()

st.caption(
    "MODEL-09 deployment | "
    "Verified representation: MEAN + MAX | "
    "ESM-2 t33 650M | "
    "48-aa chunks | "
    "91-token input | "
    "2560-D features"
)
