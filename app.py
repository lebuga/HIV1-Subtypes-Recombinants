# =============================================================================
# MODEL-09 — CURRENT 9-MODEL BENCHMARK DEPLOYMENT
#
# HIV-1 RECOMBINANT CLASSIFIER
#
# CURRENT BENCHMARK PIPELINE:
#
# RAW HIV-1 PROTEIN SEQUENCE
#        ↓
# ESM-2 t33 650M
# facebook/esm2_t33_650M_UR50D
#        ↓
# 1280-D RESIDUE EMBEDDINGS
#        ↓
# COMPLETE 48-AA NON-OVERLAPPING CHUNKS
#        ↓
# MEAN + MAX
#        ↓
# 2560-D TOKEN REPRESENTATION
#        ↓
# PAD / TRUNCATE
#        ↓
# 91 TOKENS × 2560
#        ↓
# TRAIN-ONLY STANDARDIZATION
#        ↓
# MODEL-09
# Bidirectional Attention Transformer Encoder
#        ↓
# SIGMOID PROBABILITY
#        ↓
# FROZEN VALIDATION THRESHOLD = 0.71
#        ↓
# NON-RECOMBINANT / RECOMBINANT
#
# IMPORTANT:
#   This is the CURRENT 9-MODEL BENCHMARK deployment.
#
#   It is NOT the previous authoritative MODEL-09 forensic deployment.
#
# Deployment artifacts:
#
#   MODEL-09_Bidirectional_Attention_Transformer_Encoder.pt
#   MODEL-09_BENCHMARK_TRAIN_MEAN.npy
#   MODEL-09_BENCHMARK_TRAIN_STD.npy
#   MODEL-09_BENCHMARK_FROZEN_THRESHOLD.txt
#
# These files may exist either:
#
#   repository/
#       artifacts/
#
# OR:
#
#   repository/
#       app.py
#       MODEL-09_*.*
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
# 3. GLOBAL REPRODUCIBILITY / NUMERICAL SETTINGS
# =============================================================================

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =============================================================================
# 4. CURRENT BENCHMARK CONFIGURATION
# =============================================================================

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"

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

FROZEN_THRESHOLD_FALLBACK = 0.71


# =============================================================================
# 5. DEVICE
# =============================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =============================================================================
# 6. APPLICATION ROOT
# =============================================================================

APP_DIR = Path(
    __file__
).resolve().parent


# =============================================================================
# 7. ARTIFACT LOCATION RESOLUTION
# =============================================================================
#
# The user uploaded the MODEL-09 files directly to the repository root.
#
# Nevertheless, this code supports BOTH:
#
#   repository/artifacts/
#
# and:
#
#   repository/
#
# This prevents future path problems.
# =============================================================================

ARTIFACTS_CANDIDATES = [

    APP_DIR / "artifacts",

    APP_DIR

]


# =============================================================================
# 8. REQUIRED ARTIFACT FILENAMES
# =============================================================================

MODEL_FILENAME = (
    "MODEL-09_Bidirectional_Attention_Transformer_Encoder.pt"
)

TRAIN_MEAN_FILENAME = (
    "MODEL-09_BENCHMARK_TRAIN_MEAN.npy"
)

TRAIN_STD_FILENAME = (
    "MODEL-09_BENCHMARK_TRAIN_STD.npy"
)

THRESHOLD_FILENAME = (
    "MODEL-09_BENCHMARK_FROZEN_THRESHOLD.txt"
)


# =============================================================================
# 9. FIND DEPLOYMENT ARTIFACT DIRECTORY
# =============================================================================

def locate_artifact_directory():

    for candidate in ARTIFACTS_CANDIDATES:

        model_path = (
            candidate
            / MODEL_FILENAME
        )

        mean_path = (
            candidate
            / TRAIN_MEAN_FILENAME
        )

        std_path = (
            candidate
            / TRAIN_STD_FILENAME
        )

        threshold_path = (
            candidate
            / THRESHOLD_FILENAME
        )

        if (
            model_path.is_file()
            and
            mean_path.is_file()
            and
            std_path.is_file()
            and
            threshold_path.is_file()
        ):

            return candidate

    return None


# =============================================================================
# 10. RESOLVE ARTIFACT DIRECTORY
# =============================================================================

ARTIFACT_DIR = locate_artifact_directory()


# =============================================================================
# 11. ARTIFACT PATHS
# =============================================================================

if ARTIFACT_DIR is not None:

    MODEL_PATH = (
        ARTIFACT_DIR
        / MODEL_FILENAME
    )

    TRAIN_MEAN_PATH = (
        ARTIFACT_DIR
        / TRAIN_MEAN_FILENAME
    )

    TRAIN_STD_PATH = (
        ARTIFACT_DIR
        / TRAIN_STD_FILENAME
    )

    THRESHOLD_PATH = (
        ARTIFACT_DIR
        / THRESHOLD_FILENAME
    )

else:

    MODEL_PATH = (
        APP_DIR
        / MODEL_FILENAME
    )

    TRAIN_MEAN_PATH = (
        APP_DIR
        / TRAIN_MEAN_FILENAME
    )

    TRAIN_STD_PATH = (
        APP_DIR
        / TRAIN_STD_FILENAME
    )

    THRESHOLD_PATH = (
        APP_DIR
        / THRESHOLD_FILENAME
    )


# =============================================================================
# 12. ARTIFACT VERIFICATION
# =============================================================================

def verify_deployment_artifacts():

    required = {

        "MODEL-09 checkpoint":
            MODEL_PATH,

        "training mean":
            TRAIN_MEAN_PATH,

        "training std":
            TRAIN_STD_PATH,

        "frozen threshold":
            THRESHOLD_PATH

    }

    missing = []

    for name, path in required.items():

        if not path.is_file():

            missing.append(
                f"{name}: {path}"
            )

    if missing:

        message = (
            "Required MODEL-09 deployment artifacts "
            "are missing:\n\n"
            +
            "\n".join(
                missing
            )
            +
            "\n\nExpected files:\n"
            +
            MODEL_FILENAME
            +
            "\n"
            +
            TRAIN_MEAN_FILENAME
            +
            "\n"
            +
            TRAIN_STD_FILENAME
            +
            "\n"
            +
            THRESHOLD_FILENAME
        )

        raise FileNotFoundError(
            message
        )


# =============================================================================
# 13. VERIFY ARTIFACTS
# =============================================================================

try:

    verify_deployment_artifacts()

    ARTIFACTS_OK = True

except Exception as exc:

    ARTIFACTS_OK = False

    ARTIFACT_ERROR = str(
        exc
    )


# =============================================================================
# 14. LOAD FROZEN THRESHOLD
# =============================================================================

def load_frozen_threshold():

    try:

        value = float(
            THRESHOLD_PATH
            .read_text()
            .strip()
        )

        if not (
            0.0
            <
            value
            <
            1.0
        ):

            raise ValueError(
                "Frozen threshold must be between 0 and 1."
            )

        return value

    except Exception:

        return FROZEN_THRESHOLD_FALLBACK


# =============================================================================
# 15. LOAD TRAINING STANDARDIZATION ARTIFACTS
# =============================================================================

@st.cache_resource(
    show_spinner=False
)
def load_standardization_artifacts():

    train_mean = np.load(
        TRAIN_MEAN_PATH
    )

    train_std = np.load(
        TRAIN_STD_PATH
    )

    train_mean = np.asarray(
        train_mean,
        dtype=np.float32
    )

    train_std = np.asarray(
        train_std,
        dtype=np.float32
    )

    expected_shape = (
        1,
        1,
        INPUT_DIM
    )

    if train_mean.shape != expected_shape:

        raise RuntimeError(
            "Training mean has incorrect shape. "
            f"Expected {expected_shape}, "
            f"got {train_mean.shape}."
        )

    if train_std.shape != expected_shape:

        raise RuntimeError(
            "Training std has incorrect shape. "
            f"Expected {expected_shape}, "
            f"got {train_std.shape}."
        )

    if not np.all(
        np.isfinite(
            train_mean
        )
    ):

        raise RuntimeError(
            "Training mean contains non-finite values."
        )

    if not np.all(
        np.isfinite(
            train_std
        )
    ):

        raise RuntimeError(
            "Training std contains non-finite values."
        )

    if np.any(
        train_std <= 0
    ):

        raise RuntimeError(
            "Training std contains zero or negative values."
        )

    return (
        train_mean,
        train_std
    )


# =============================================================================
# 16. MODEL COMPONENTS
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
# 17. GLOBAL ATTENTION
# =============================================================================

class GlobalAttentionBlock(
    LocalAttentionBlock
):

    pass


# =============================================================================
# 18. ATTENTION POOLING
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
# 19. MODEL-09 ARCHITECTURE
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
            LocalAttentionBlock(
                dim=model_dim,
                heads=heads,
                dropout=ATTENTION_DROPOUT
            )
        )

        self.global_attention = (
            GlobalAttentionBlock(
                dim=model_dim,
                heads=heads,
                dropout=ATTENTION_DROPOUT
            )
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
            and
            training_noise
            and
            REPRESENTATION_NOISE > 0
        ):

            x = (
                x
                +
                torch.randn_like(
                    x
                )
                *
                REPRESENTATION_NOISE
            )

        T = x.size(1)

        if T > TOKEN_LENGTH:

            raise RuntimeError(
                f"Input contains {T} tokens, "
                f"but MODEL-09 supports maximum "
                f"{TOKEN_LENGTH} tokens."
            )

        x = (
            x
            +
            self.position_embedding[
                :,
                :T,
                :
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
# 20. LOAD MODEL CHECKPOINT
# =============================================================================

@st.cache_resource(
    show_spinner=False
)
def load_model():

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu"
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

        metadata = checkpoint

    else:

        state_dict = checkpoint

        metadata = {}


    # -------------------------------------------------------------------------
    # Verify checkpoint metadata where available
    # -------------------------------------------------------------------------

    metadata_checks = {

        "input_dim":
            INPUT_DIM,

        "token_length":
            TOKEN_LENGTH,

        "model_dim":
            MODEL_DIM,

        "attention_heads":
            ATTENTION_HEADS

    }

    for key, expected in (
        metadata_checks.items()
    ):

        if key in metadata:

            actual = metadata[
                key
            ]

            if actual != expected:

                raise RuntimeError(

                    f"Checkpoint metadata mismatch "
                    f"for '{key}'. "
                    f"Expected {expected}, "
                    f"got {actual}."

                )


    # -------------------------------------------------------------------------
    # Build exact architecture
    # -------------------------------------------------------------------------

    model = (
        BidirectionalAttentionTransformerEncoder(
            input_dim=INPUT_DIM,
            model_dim=MODEL_DIM,
            heads=ATTENTION_HEADS,
            max_tokens=TOKEN_LENGTH
        )
    )


    # -------------------------------------------------------------------------
    # Load weights
    # -------------------------------------------------------------------------

    try:

        model.load_state_dict(
            state_dict,
            strict=True
        )

    except Exception as exc:

        raise RuntimeError(
            "MODEL-09 checkpoint could not be loaded "
            "into the current benchmark architecture.\n\n"
            + str(exc)
        )


    model.to(
        DEVICE
    )

    model.eval()

    return (
        model,
        metadata
    )


# =============================================================================
# 21. LOAD ESM-2
# =============================================================================

@st.cache_resource(
    show_spinner=True
)
def load_esm2():

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    model = (
        EsmModel.from_pretrained(
            MODEL_NAME
        )
    )

    model.to(
        DEVICE
    )

    model.eval()

    return (
        tokenizer,
        model
    )


# =============================================================================
# 22. SEQUENCE CLEANING
# =============================================================================

def clean_protein_sequence(
    sequence
):

    if sequence is None:

        raise ValueError(
            "No sequence was provided."
        )

    sequence = str(
        sequence
    )

    # Remove FASTA header lines.
    lines = []

    for line in sequence.splitlines():

        line = line.strip()

        if not line:
            continue

        if line.startswith(">"):
            continue

        lines.append(
            line
        )

    sequence = "".join(
        lines
    )

    # Remove whitespace.
    sequence = re.sub(
        r"\s+",
        "",
        sequence
    )

    # Uppercase.
    sequence = sequence.upper()

    # Allow standard amino-acid alphabet plus X/B/Z/U/O.
    allowed = set(
        "ACDEFGHIKLMNPQRSTVWY"
        "XBZOU"
    )

    invalid = sorted(
        set(sequence)
        -
        allowed
    )

    if invalid:

        raise ValueError(
            "Invalid amino-acid characters detected: "
            +
            ", ".join(
                invalid
            )
        )

    if len(sequence) == 0:

        raise ValueError(
            "The protein sequence is empty."
        )

    return sequence


# =============================================================================
# 23. ESM-2 RESIDUE EMBEDDING EXTRACTION
# =============================================================================
#
# ESM-2 has a maximum token length limitation.
#
# We therefore process long proteins in overlapping amino-acid windows.
#
# IMPORTANT:
#   The benchmark residue embeddings were generated from the full protein
#   processing pipeline. This deployment implementation preserves the same
#   residue-level dimensionality and downstream tokenization.
#
# For proteins longer than the model context, residue embeddings are
# reconstructed using overlapping windows and averaged at overlapping
# residue positions.
# =============================================================================

ESM_MAX_RESIDUES = 1022

ESM_WINDOW_STRIDE = 896


def extract_residue_embeddings(
    sequence,
    tokenizer,
    esm_model
):

    sequence_length = len(
        sequence
    )

    accumulated = None

    counts = None

    start_positions = list(
        range(
            0,
            sequence_length,
            ESM_WINDOW_STRIDE
        )
    )

    for start in start_positions:

        end = min(
            start
            +
            ESM_MAX_RESIDUES,
            sequence_length
        )

        fragment = sequence[
            start:end
        ]

        if not fragment:
            continue

        encoded = tokenizer(
            fragment,
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
        )

        # Remove CLS and EOS.
        residue_hidden = (
            hidden[
                0,
                1:-1,
                :
            ]
        )

        residue_hidden = (
            residue_hidden
            .detach()
            .float()
            .cpu()
            .numpy()
        )

        expected_length = (
            end - start
        )

        if (
            residue_hidden.shape[0]
            !=
            expected_length
        ):

            raise RuntimeError(

                "ESM-2 residue embedding length "
                "does not match input fragment length. "
                f"Expected {expected_length}, "
                f"got {residue_hidden.shape[0]}."

            )

        if accumulated is None:

            accumulated = np.zeros(
                (
                    sequence_length,
                    ESM2_DIMENSION
                ),
                dtype=np.float32
            )

            counts = np.zeros(
                sequence_length,
                dtype=np.float32
            )

        accumulated[
            start:end
        ] += residue_hidden

        counts[
            start:end
        ] += 1.0

        # Once the end of the sequence is covered,
        # no further window is needed.
        if end >= sequence_length:
            break

    if accumulated is None:

        raise RuntimeError(
            "No ESM-2 residue embeddings were generated."
        )

    counts = np.maximum(
        counts,
        1.0
    )

    residue_embeddings = (
        accumulated
        /
        counts[:, None]
    )

    if residue_embeddings.shape != (
        sequence_length,
        ESM2_DIMENSION
    ):

        raise RuntimeError(

            "Unexpected residue embedding shape: "
            f"{residue_embeddings.shape}. "
            f"Expected "
            f"({sequence_length}, "
            f"{ESM2_DIMENSION})."

        )

    return residue_embeddings.astype(
        np.float32
    )


# =============================================================================
# 24. RESIDUE → 2560-D COMPLETE TOKENIZATION
# =============================================================================
#
# Each complete 48-residue chunk:
#
#     48 × 1280
#
# becomes:
#
#     mean(48 residues) = 1280
#     max (48 residues) = 1280
#
# concatenated:
#
#     2560-D token
#
# Incomplete terminal chunks are discarded.
# =============================================================================

def residue_to_tokens(
    residue_embeddings
):

    residue_embeddings = np.asarray(
        residue_embeddings,
        dtype=np.float32
    )

    if residue_embeddings.ndim != 2:

        raise ValueError(
            "Residue embeddings must be 2-D."
        )

    if residue_embeddings.shape[1] != (
        ESM2_DIMENSION
    ):

        raise ValueError(

            "Unexpected ESM-2 dimension. "
            f"Expected {ESM2_DIMENSION}, "
            f"got {residue_embeddings.shape[1]}."

        )

    residue_count = (
        residue_embeddings.shape[0]
    )

    complete_tokens = (
        residue_count
        //
        CHUNK_SIZE
    )

    if complete_tokens < 1:

        raise ValueError(

            f"Protein contains only "
            f"{residue_count} residues. "
            f"At least {CHUNK_SIZE} residues "
            f"are required."

        )

    usable_residues = (
        complete_tokens
        *
        CHUNK_SIZE
    )

    x = residue_embeddings[
        :usable_residues
    ]

    x = x.reshape(
        complete_tokens,
        CHUNK_SIZE,
        ESM2_DIMENSION
    )

    mean_features = (
        np.mean(
            x,
            axis=1
        )
    )

    max_features = (
        np.max(
            x,
            axis=1
        )
    )

    tokens = np.concatenate(
        [
            mean_features,
            max_features
        ],
        axis=1
    )

    if tokens.shape != (
        complete_tokens,
        INPUT_DIM
    ):

        raise RuntimeError(

            "Unexpected token representation shape: "
            f"{tokens.shape}."

        )

    return tokens.astype(
        np.float32
    )


# =============================================================================
# 25. PAD / TRUNCATE TO 91 TOKENS
# =============================================================================

def fit_token_length(
    tokens
):

    tokens = np.asarray(
        tokens,
        dtype=np.float32
    )

    token_count = (
        tokens.shape[0]
    )

    if token_count >= TOKEN_LENGTH:

        return (
            tokens[
                :TOKEN_LENGTH
            ],
            token_count,
            "truncated"
        )

    padded = np.zeros(
        (
            TOKEN_LENGTH,
            INPUT_DIM
        ),
        dtype=np.float32
    )

    padded[
        :token_count
    ] = tokens

    return (
        padded,
        token_count,
        "zero-padded"
    )


# =============================================================================
# 26. APPLY TRAIN-ONLY STANDARDIZATION
# =============================================================================

def standardize_tokens(
    tokens,
    train_mean,
    train_std
):

    tokens = np.asarray(
        tokens,
        dtype=np.float32
    )

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
            "Standardized token matrix contains "
            "non-finite values."
        )

    return standardized


# =============================================================================
# 27. COMPLETE DEPLOYMENT PREPROCESSING
# =============================================================================

def build_model_input(
    sequence,
    tokenizer,
    esm_model,
    train_mean,
    train_std
):

    cleaned = (
        clean_protein_sequence(
            sequence
        )
    )

    residue_embeddings = (
        extract_residue_embeddings(
            cleaned,
            tokenizer,
            esm_model
        )
    )

    tokens = (
        residue_to_tokens(
            residue_embeddings
        )
    )

    raw_token_count = (
        tokens.shape[0]
    )

    fixed_tokens, _, padding_mode = (
        fit_token_length(
            tokens
        )
    )

    standardized = (
        standardize_tokens(
            fixed_tokens,
            train_mean,
            train_std
        )
    )

    model_input = torch.from_numpy(
        standardized
    ).unsqueeze(
        0
    )

    model_input = (
        model_input.to(
            DEVICE
        )
    )

    return {

        "sequence":
            cleaned,

        "sequence_length":
            len(cleaned),

        "residue_embeddings":
            residue_embeddings,

        "raw_tokens":
            tokens,

        "raw_token_count":
            raw_token_count,

        "fixed_tokens":
            fixed_tokens,

        "padding_mode":
            padding_mode,

        "standardized":
            standardized,

        "model_input":
            model_input

    }


# =============================================================================
# 28. MODEL PREDICTION
# =============================================================================

def predict_model09(
    model,
    model_input,
    threshold
):

    model.eval()

    with torch.no_grad():

        logits, attention = (
            model(
                model_input,
                training_noise=False
            )
        )

        probability = (
            torch.sigmoid(
                logits
            )
            .detach()
            .cpu()
            .item()
        )

    prediction = int(
        probability
        >=
        threshold
    )

    label = (
        "RECOMBINANT"
        if prediction == 1
        else
        "NON-RECOMBINANT"
    )

    return {

        "logit":
            float(
                logits.detach()
                .cpu()
                .item()
            ),

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
            .detach()
            .cpu()
            .numpy()[0]

    }


# =============================================================================
# 29. APPLICATION HEADER
# =============================================================================

st.title(
    "🧬 MODEL-09 HIV-1 Recombinant Classifier"
)

st.caption(
    "Current 9-model benchmark deployment"
)


# =============================================================================
# 30. DEPLOYMENT INFORMATION
# =============================================================================

with st.expander(
    "Deployment configuration",
    expanded=False
):

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            "**Model:** MODEL-09"
        )

        st.write(
            "**ESM-2:** "
            f"`{MODEL_NAME}`"
        )

        st.write(
            f"**ESM dimension:** "
            f"`{ESM2_DIMENSION}`"
        )

        st.write(
            f"**Chunk size:** "
            f"`{CHUNK_SIZE}`"
        )

    with col2:

        st.write(
            f"**Token dimension:** "
            f"`{INPUT_DIM}`"
        )

        st.write(
            f"**Token length:** "
            f"`{TOKEN_LENGTH}`"
        )

        st.write(
            f"**Model dimension:** "
            f"`{MODEL_DIM}`"
        )

        st.write(
            f"**Attention heads:** "
            f"`{ATTENTION_HEADS}`"
        )

    st.write(
        f"**Device:** `{DEVICE}`"
    )

    if ARTIFACT_DIR is not None:

        st.write(
            f"**Artifact directory:** "
            f"`{ARTIFACT_DIR}`"
        )


# =============================================================================
# 31. ARTIFACT FAILURE SCREEN
# =============================================================================

if not ARTIFACTS_OK:

    st.error(
        "MODEL-09 could not be initialized."
    )

    st.code(
        ARTIFACT_ERROR
    )

    st.stop()


# =============================================================================
# 32. LOAD THRESHOLD
# =============================================================================

try:

    FROZEN_THRESHOLD = (
        load_frozen_threshold()
    )

except Exception as exc:

    st.error(
        "Could not load the frozen MODEL-09 threshold."
    )

    st.exception(
        exc
    )

    st.stop()


# =============================================================================
# 33. LOAD STANDARDIZATION
# =============================================================================

try:

    train_mean, train_std = (
        load_standardization_artifacts()
    )

except Exception as exc:

    st.error(
        "Could not load MODEL-09 training standardization artifacts."
    )

    st.exception(
        exc
    )

    st.stop()


# =============================================================================
# 34. LOAD MODEL
# =============================================================================

try:

    with st.spinner(
        "Loading MODEL-09 checkpoint..."
    ):

        model, checkpoint_metadata = (
            load_model()
        )

except Exception as exc:

    st.error(
        "MODEL-09 checkpoint could not be initialized."
    )

    st.exception(
        exc
    )

    st.stop()


# =============================================================================
# 35. LOAD ESM-2 ONLY WHEN NEEDED
# =============================================================================
#
# Loading ESM-2 consumes substantial memory.
#
# We therefore defer it until the user actually requests a prediction.
# =============================================================================


# =============================================================================
# 36. INPUT SECTION
# =============================================================================

st.subheader(
    "Enter an HIV-1 protein sequence"
)

st.write(
    "Paste a raw amino-acid sequence or FASTA-formatted "
    "protein sequence below."
)

sequence_input = st.text_area(
    "HIV-1 protein sequence",
    height=220,
    placeholder=(
        "Example:\n"
        ">HIV-1_protein\n"
        "MRVMGTQKNYSLLWRWGIMIFGILMACSANN..."
    )
)


# =============================================================================
# 37. PREDICTION BUTTON
# =============================================================================

predict_button = st.button(
    "🔬 Classify Sequence",
    type="primary",
    use_container_width=True
)


# =============================================================================
# 38. RUN PREDICTION
# =============================================================================

if predict_button:

    if not sequence_input.strip():

        st.warning(
            "Please enter an HIV-1 protein sequence."
        )

        st.stop()

    try:

        # ---------------------------------------------------------------------
        # Load ESM-2
        # ---------------------------------------------------------------------

        with st.spinner(
            "Loading ESM-2..."
        ):

            tokenizer, esm_model = (
                load_esm2()
            )


        # ---------------------------------------------------------------------
        # Build model input
        # ---------------------------------------------------------------------

        with st.spinner(
            "Processing protein sequence..."
        ):

            start_time = time.time()

            processed = (
                build_model_input(
                    sequence_input,
                    tokenizer,
                    esm_model,
                    train_mean,
                    train_std
                )
            )

            preprocessing_time = (
                time.time()
                -
                start_time
            )


        # ---------------------------------------------------------------------
        # Prediction
        # ---------------------------------------------------------------------

        with st.spinner(
            "Running MODEL-09..."
        ):

            prediction_start = (
                time.time()
            )

            result = (
                predict_model09(
                    model,
                    processed[
                        "model_input"
                    ],
                    FROZEN_THRESHOLD
                )
            )

            prediction_time = (
                time.time()
                -
                prediction_start
            )


        # ---------------------------------------------------------------------
        # Display main result
        # ---------------------------------------------------------------------

        st.divider()

        st.subheader(
            "MODEL-09 Prediction"
        )

        if result["prediction"] == 1:

            st.error(
                "🧬 RECOMBINANT"
            )

        else:

            st.success(
                "🧬 NON-RECOMBINANT"
            )


        # ---------------------------------------------------------------------
        # Probability
        # ---------------------------------------------------------------------

        probability = (
            result[
                "probability"
            ]
        )

        col1, col2, col3 = (
            st.columns(3)
        )

        with col1:

            st.metric(
                "Recombinant Probability",
                f"{probability:.6f}"
            )

        with col2:

            st.metric(
                "Frozen Threshold",
                f"{FROZEN_THRESHOLD:.2f}"
            )

        with col3:

            st.metric(
                "Sequence Length",
                f"{processed['sequence_length']:,} aa"
            )


        # ---------------------------------------------------------------------
        # Probability bar
        # ---------------------------------------------------------------------

        st.progress(
            min(
                max(
                    probability,
                    0.0
                ),
                1.0
            )
        )


        # ---------------------------------------------------------------------
        # Processing details
        # ---------------------------------------------------------------------

        with st.expander(
            "Prediction details",
            expanded=True
        ):

            st.write(
                "**Input sequence length:** "
                f"{processed['sequence_length']} aa"
            )

            st.write(
                "**Raw complete 48-aa tokens:** "
                f"{processed['raw_token_count']}"
            )

            st.write(
                "**MODEL-09 input:** "
                f"{TOKEN_LENGTH} × {INPUT_DIM}"
            )

            st.write(
                "**Token handling:** "
                f"{processed['padding_mode']}"
            )

            st.write(
                "**ESM-2 residue embedding shape:** "
                f"{processed['residue_embeddings'].shape}"
            )

            st.write(
                "**Token matrix shape:** "
                f"{processed['fixed_tokens'].shape}"
            )

            st.write(
                "**Standardized matrix shape:** "
                f"{processed['standardized'].shape}"
            )

            st.write(
                "**Logit:** "
                f"{result['logit']:.6f}"
            )

            st.write(
                "**Preprocessing time:** "
                f"{preprocessing_time:.2f} seconds"
            )

            st.write(
                "**MODEL-09 inference time:** "
                f"{prediction_time:.4f} seconds"
            )

            st.write(
                "**Device:** "
                f"{DEVICE}"
            )


        # ---------------------------------------------------------------------
        # Attention visualization
        # ---------------------------------------------------------------------

        attention = (
            result[
                "attention"
            ]
        )

        with st.expander(
            "MODEL-09 attention distribution",
            expanded=False
        ):

            st.write(
                "Attention weights across the 91-token "
                "MODEL-09 representation."
            )

            st.bar_chart(
                attention
            )


        # ---------------------------------------------------------------------
        # Deployment provenance
        # ---------------------------------------------------------------------

        with st.expander(
            "Model provenance",
            expanded=False
        ):

            st.write(
                "This deployment uses the current "
                "9-model benchmark MODEL-09 checkpoint."
            )

            st.write(
                f"Frozen validation threshold: "
                f"`{FROZEN_THRESHOLD:.10f}`"
            )

            st.write(
                "Training-only standardization statistics "
                "are loaded from the frozen benchmark artifacts."
            )

            st.write(
                "No benchmark test data are used during "
                "deployment preprocessing."
            )

            st.write(
                "No model retraining occurs during inference."
            )


    except Exception as exc:

        st.error(
            "Prediction failed."
        )

        st.exception(
            exc
        )

        st.info(
            "Check that the input is a valid amino-acid "
            "protein sequence and that all four MODEL-09 "
            "deployment artifacts are present."
        )


# =============================================================================
# 39. FOOTER
# =============================================================================

st.divider()

st.caption(
    "MODEL-09 | Current 9-model benchmark | "
    "ESM-2 → 48-aa mean+max tokens → "
    "91-token representation → "
    "train-only standardization → "
    "Bidirectional Attention Transformer"
)
