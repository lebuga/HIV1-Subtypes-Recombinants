# =============================================================================
# MODEL-09 — FINAL STREAMLIT DEPLOYMENT
#
# VERIFIED FORENSIC DEPLOYMENT PIPELINE
#
# RAW HIV-1 PROTEIN SEQUENCE
#          ↓
# ESM-2 t33 650M
#          ↓
# 1280-D residue embeddings
#          ↓
# 48-aa non-overlapping chunks
#          ↓
# MEAN + MAX POOLING
#          ↓
# 2560-D tokens
#          ↓
# 91-token representation
#          ↓
# TRAIN MEAN / TRAIN STD
#          ↓
# ORIGINAL MODEL-09 TEACHER
#          ↓
# RECOMBINANT PROBABILITY
#
# IMPORTANT:
#   - NO TRAINING IS PERFORMED
#   - NO KNOWLEDGE DISTILLATION
#   - NO QUANTIZATION
#   - NO historical 0.88 threshold
#   - Exact teacher checkpoint is used
#   - X is accepted as an ambiguous amino acid
# =============================================================================


# =============================================================================
# 1. IMPORTS
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
# 2. MODEL-09 CONFIGURATION
# =============================================================================

MODEL_NAME = (
    "facebook/esm2_t33_650M_UR50D"
)

ESM_DIM = 1280

CHUNK_SIZE = 48
CHUNK_STRIDE = 48

TOKEN_FEATURE_DIM = 2560
TOKEN_LENGTH = 91

MODEL_DIM = 96
ATTENTION_HEADS = 4

ATTENTION_DROPOUT = 0.25
BASE_DROPOUT = 0.30

# Deployment threshold.
#
# We intentionally do NOT use the historical 0.88 threshold.
DEFAULT_THRESHOLD = 0.50


# =============================================================================
# 3. DEVICE
# =============================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =============================================================================
# 4. FILE LOCATIONS
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
# 5. STREAMLIT PAGE
# =============================================================================

st.set_page_config(
    page_title="MODEL-09 HIV-1 Recombinant Classifier",
    page_icon="🧬",
    layout="wide"
)


# =============================================================================
# 6. HEADER
# =============================================================================

st.title(
    "MODEL-09 HIV-1 Recombinant Classifier"
)

st.write(
    """
    Whole-protein HIV-1 recombinant-status prediction using
    the original MODEL-09 attention-based teacher model.
    """
)

st.caption(
    "Verified representation: ESM-2 residue embeddings → "
    "48-aa chunks → MEAN + MAX → 2560-D tokens → "
    "91 tokens → frozen training standardization → MODEL-09."
)


# =============================================================================
# 7. MODEL ARCHITECTURE
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
# GLOBAL ATTENTION
# =============================================================================

class GlobalAttentionBlock(
    LocalAttentionBlock
):

    pass


# =============================================================================
# ATTENTION POOLING
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
# MODEL-09 TEACHER
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

        # Training noise is NEVER allowed during deployment.
        if (
            self.training
            and training_noise
        ):

            raise RuntimeError(
                "training_noise is not allowed during deployment."
            )

        T = x.size(1)

        x = (
            x
            +
            self.position_embedding[
                :, :T
            ]
        )

        x = self.local_attention(
            x
        )

        x = self.global_attention(
            x
        )

        pooled, attention = self.pool(
            x
        )

        logits = self.classifier(
            pooled
        ).squeeze(-1)

        return (
            logits,
            attention
        )


# =============================================================================
# 8. LOAD TRAINING STANDARDIZATION
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

    if not np.all(
        np.isfinite(train_mean)
    ):

        raise RuntimeError(
            "TRAIN_MEAN contains non-finite values."
        )

    if not np.all(
        np.isfinite(train_std)
    ):

        raise RuntimeError(
            "TRAIN_STD contains non-finite values."
        )

    # Protect against division by zero.
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
# 9. LOAD ORIGINAL MODEL-09 TEACHER
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
            "Teacher checkpoint does not contain "
            "'model_state_dict'."
        )

    model = (
        BidirectionalAttentionTransformerEncoder()
    )

    missing, unexpected = (
        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ],
            strict=False
        )
    )

    if missing or unexpected:

        raise RuntimeError(
            "MODEL-09 checkpoint mismatch.\n\n"
            f"Missing keys:\n{missing}\n\n"
            f"Unexpected keys:\n{unexpected}"
        )

    model.eval()

    model.to(
        DEVICE
    )

    return (
        model,
        checkpoint
    )


# =============================================================================
# 10. LOAD ESM-2
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
# 11. SEQUENCE CLEANING
# =============================================================================

def clean_sequence(
    sequence
):

    sequence = (
        sequence
        .replace("\n", "")
        .replace("\r", "")
        .replace(" ", "")
        .replace("\t", "")
        .upper()
    )

    return sequence


# =============================================================================
# 12. SEQUENCE VALIDATION
# =============================================================================

def validate_sequence(
    sequence
):

    # Standard amino acids + X.
    #
    # X = unknown / ambiguous amino acid.
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
# 13. ESM-2 CHUNK EMBEDDING
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

    input_ids = encoded[
        "input_ids"
    ].to(
        DEVICE
    )

    attention_mask = encoded[
        "attention_mask"
    ].to(
        DEVICE
    )

    outputs = esm_model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

    hidden = (
        outputs.last_hidden_state
    )

    # Remove ESM special tokens.
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
# 14. FULL RESIDUE EMBEDDINGS
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
            "Computing ESM-2 residue embeddings..."
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

        if emb.shape != expected_shape:

            raise RuntimeError(
                "Unexpected ESM-2 embedding shape.\n"
                f"Found: {emb.shape}\n"
                f"Expected: {expected_shape}"
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
                f"ESM-2 embedding chunk "
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
# 15. BUILD EXACT MODEL-09 REPRESENTATION
#
# IMPORTANT:
#
# THIS IS THE VERIFIED CORRECT REPRESENTATION:
#
#     MEAN + MAX
#
# NOT:
#
#     MEAN + STD
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
        # VERIFIED ORIGINAL POOLING
        # -------------------------------------------------------------

        mean_pool = np.mean(
            chunk,
            axis=0
        )

        max_pool = np.max(
            chunk,
            axis=0
        )

        # -------------------------------------------------------------
        # 1280 MEAN + 1280 MAX = 2560
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
            f"Sequence has only {n_residues} residues; "
            f"cannot create a {CHUNK_SIZE}-residue chunk."
        )

    tokens = np.asarray(
        tokens,
        dtype=np.float32
    )

    raw_token_count = (
        tokens.shape[0]
    )

    # -----------------------------------------------------------------
    # FORCE EXACT 91 TOKENS
    # -----------------------------------------------------------------

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
            "MODEL-09 representation shape mismatch.\n"
            f"Found: {tokens.shape}\n"
            f"Expected: "
            f"({TOKEN_LENGTH}, {TOKEN_FEATURE_DIM})"
        )

    return (
        tokens,
        raw_token_count
    )


# =============================================================================
# 16. STANDARDIZATION
# =============================================================================

def standardize(
    tokens,
    train_mean,
    train_std
):

    standardized = (
        tokens
        -
        train_mean[
            None,
            :
        ]
    ) / train_std[
        None,
        :
    ]

    if not np.all(
        np.isfinite(
            standardized
        )
    ):

        raise RuntimeError(
            "Standardized representation contains "
            "non-finite values."
        )

    return standardized.astype(
        np.float32
    )


# =============================================================================
# 17. PREDICTION
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

    probability = torch.sigmoid(
        logits
    ).item()

    if probability >= threshold:

        prediction = (
            "RECOMBINANT"
        )

    else:

        prediction = (
            "NON-RECOMBINANT"
        )

    attention = (
        attention
        .detach()
        .cpu()
        .numpy()[0]
    )

    return (
        probability,
        prediction,
        attention
    )


# =============================================================================
# 18. SIDEBAR
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
        "Pooling: `MEAN + MAX`"
    )

    st.write(
        f"Token dimension: "
        f"`{TOKEN_FEATURE_DIM}`"
    )

    st.write(
        f"Token length: "
        f"`{TOKEN_LENGTH}`"
    )

    st.write(
        f"Model dimension: "
        f"`{MODEL_DIM}`"
    )

    st.write(
        f"Attention heads: "
        f"`{ATTENTION_HEADS}`"
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
        "The historical 0.88 threshold is "
        "not used in this deployment."
    )

    st.divider()

    st.write(
        f"Device: `{DEVICE}`"
    )


# =============================================================================
# 19. LOAD ALL DEPLOYMENT ARTIFACTS
# =============================================================================

try:

    with st.spinner(
        "Loading MODEL-09 deployment artifacts..."
    ):

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
# 20. INPUT
# =============================================================================

st.subheader(
    "Enter HIV-1 protein sequence"
)

st.write(
    """
    Paste a raw amino-acid protein sequence below.
    Whitespace and line breaks are automatically removed.
    The ambiguous amino-acid symbol **X** is accepted.
    """
)

sequence_input = st.text_area(
    "Raw amino-acid sequence",
    height=250,
    placeholder=(
        "Paste HIV-1 protein sequence here..."
    )
)


run_prediction = st.button(
    "Run MODEL-09 Prediction",
    type="primary"
)


# =============================================================================
# 21. PREDICTION PIPELINE
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

    valid, error = validate_sequence(
        sequence
    )

    if not valid:

        st.error(
            error
        )

        st.stop()

    # -------------------------------------------------------------------------
    # BASIC INPUT INFORMATION
    # -------------------------------------------------------------------------

    st.info(
        f"Sequence length: "
        f"{len(sequence):,} amino acids"
    )

    try:

        # =====================================================================
        # STEP 1 — ESM-2 RESIDUE EMBEDDINGS
        # =====================================================================

        st.subheader(
            "1. ESM-2 preprocessing"
        )

        with st.spinner(
            "Extracting ESM-2 residue embeddings..."
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
        # STEP 2 — MODEL-09 MEAN + MAX
        # =====================================================================

        st.subheader(
            "2. MODEL-09 representation"
        )

        with st.spinner(
            "Building MEAN + MAX representation..."
        ):

            tokens, raw_token_count = (
                build_model09_representation(
                    residue_embeddings
                )
            )

        st.success(
            "MODEL-09 representation complete."
        )

        st.write(
            f"Raw token count: "
            f"`{raw_token_count}`"
        )

        st.write(
            f"Final model input: "
            f"`{tokens.shape}`"
        )


        # =====================================================================
        # STEP 3 — STANDARDIZATION
        # =====================================================================

        st.subheader(
            "3. Training standardization"
        )

        with st.spinner(
            "Applying frozen training standardization..."
        ):

            standardized = standardize(
                tokens,
                train_mean,
                train_std
            )

        st.success(
            "Training standardization applied."
        )

        st.write(
            f"Standardized representation: "
            f"`{standardized.shape}`"
        )


        # =====================================================================
        # STEP 4 — MODEL-09 TEACHER
        # =====================================================================

        st.subheader(
            "4. MODEL-09 prediction"
        )

        with st.spinner(
            "Running MODEL-09 teacher..."
        ):

            (
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

        col1, col2 = st.columns(
            2
        )

        with col1:

            st.metric(
                "Recombinant Probability",
                f"{probability:.6f}"
            )

        with col2:

            if prediction == "RECOMBINANT":

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

        st.caption(
            "Probability is the sigmoid output of the "
            "original MODEL-09 teacher."
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
                f"ESM-2 model: "
                f"`{MODEL_NAME}`"
            )

            st.write(
                f"Residue embedding shape: "
                f"`{residue_embeddings.shape}`"
            )

            st.write(
                f"Pooling: `MEAN + MAX`"
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
                f"Device: "
                f"`{DEVICE}`"
            )

            parameter_count = sum(
                p.numel()
                for p in teacher.parameters()
            )

            st.write(
                f"MODEL-09 parameters: "
                f"`{parameter_count:,}`"
            )

            st.write(
                "Training performed during prediction: "
                "`NO`"
            )

            st.write(
                "Historical 0.88 threshold used: "
                "`NO`"
            )

            st.write(
                "Training representation: "
                "`MEAN + MAX`"
            )

            st.write(
                "Training standardization: "
                "`frozen training mean/std`"
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
                            f"- `{key}`: "
                            f"`{checkpoint[key]}`"
                        )


        # =====================================================================
        # ATTENTION
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
