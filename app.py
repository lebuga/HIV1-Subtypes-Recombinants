# =============================================================================
# MODEL-09 HIV-1 RECOMBINANT CLASSIFIER
# CURRENT 9-MODEL BENCHMARK DEPLOYMENT
#
# DEPLOYMENT REPRESENTATION
#
# RAW HIV-1 PROTEIN SEQUENCE
#        ↓
# ESM-2 t33 650M
#        ↓
# residue embeddings: 1280-D
#        ↓
# complete 48-aa chunks
#        ↓
# mean + max
#        ↓
# 2560-D token representation
#        ↓
# pad/truncate to 91 tokens
#        ↓
# TRAIN-ONLY STANDARDIZATION
#        ↓
# (91, 2560)
#        ↓
# ONE batch dimension
#        ↓
# (1, 91, 2560)
#        ↓
# MODEL-09
#        ↓
# sigmoid probability
#        ↓
# frozen validation threshold = 0.71
#        ↓
# NON-RECOMBINANT / RECOMBINANT
#
# IMPORTANT:
# This app corresponds to the CURRENT 9-MODEL BENCHMARK.
#
# Benchmark:
#   Train = 376
#   Validation = 81
#   Test = 80
#   Total = 537
#
# Representation:
#   ESM-2 dimension = 1280
#   Chunk size      = 48
#   Chunk stride    = 48
#   Token dimension = 2560
#   Token length    = 91
#
# Deployment artifacts:
#   artifacts/
#       MODEL-09_Bidirectional_Attention_Transformer_Encoder.pt
#       MODEL-09_BENCHMARK_TRAIN_MEAN.npy
#       MODEL-09_BENCHMARK_TRAIN_STD.npy
#       MODEL-09_BENCHMARK_FROZEN_THRESHOLD.txt
#
# =============================================================================


# =============================================================================
# 1. IMPORTS
# =============================================================================

import os
import sys
import time
import math
import warnings
from pathlib import Path

import numpy as np
import streamlit as st

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 2. GLOBAL CONFIGURATION
# =============================================================================

APP_VERSION = "MODEL-09 CURRENT 9-MODEL BENCHMARK"

SEED = 42

ESM_MODEL_NAME = (
    "facebook/esm2_t33_650M_UR50D"
)

ESM2_DIMENSION = 1280

CHUNK_SIZE = 48

CHUNK_STRIDE = 48

TOKEN_FEATURE_DIM = 2560

TOKEN_LENGTH = 91

MODEL_DIM = 96

ATTENTION_HEADS = 4

BASE_DROPOUT = 0.30

ATTENTION_DROPOUT = 0.25

REPRESENTATION_NOISE = 0.015

FROZEN_THRESHOLD_DEFAULT = 0.71


# =============================================================================
# 3. RANDOM SEED
# =============================================================================

def set_seed(seed=42):

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# =============================================================================
# 4. DEVICE
# =============================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =============================================================================
# 5. STREAMLIT PAGE
# =============================================================================

st.set_page_config(

    page_title=(
        "MODEL-09 HIV-1 Recombinant Classifier"
    ),

    page_icon="🧬",

    layout="wide"
)


# =============================================================================
# 6. TITLE
# =============================================================================

st.title(
    "🧬 MODEL-09 HIV-1 Recombinant Classifier"
)

st.caption(
    "Current 9-model benchmark deployment"
)


# =============================================================================
# 7. PROJECT / ARTIFACT PATH DISCOVERY
# =============================================================================

def locate_artifacts():

    """
    Robust artifact discovery for Streamlit Cloud / GitHub.

    Primary expected location:

        ./artifacts/

    Also checks a few safe alternatives.
    """

    app_root = Path(
        __file__
    ).resolve().parent

    candidates = [

        app_root / "artifacts",

        Path.cwd() / "artifacts",

        app_root,

    ]

    required_files = [

        "MODEL-09_Bidirectional_Attention_Transformer_Encoder.pt",

        "MODEL-09_BENCHMARK_TRAIN_MEAN.npy",

        "MODEL-09_BENCHMARK_TRAIN_STD.npy",

        "MODEL-09_BENCHMARK_FROZEN_THRESHOLD.txt",

    ]

    for directory in candidates:

        if not directory.exists():

            continue

        if all(
            (
                directory / filename
            ).is_file()
            for filename in required_files
        ):

            return directory

    # Return primary location even when incomplete.
    # The validation function will produce the detailed error.

    return candidates[0]


ARTIFACT_DIR = locate_artifacts()


# =============================================================================
# 8. ARTIFACT PATHS
# =============================================================================

CHECKPOINT_PATH = (
    ARTIFACT_DIR
    /
    "MODEL-09_Bidirectional_Attention_Transformer_Encoder.pt"
)

TRAIN_MEAN_PATH = (
    ARTIFACT_DIR
    /
    "MODEL-09_BENCHMARK_TRAIN_MEAN.npy"
)

TRAIN_STD_PATH = (
    ARTIFACT_DIR
    /
    "MODEL-09_BENCHMARK_TRAIN_STD.npy"
)

THRESHOLD_PATH = (
    ARTIFACT_DIR
    /
    "MODEL-09_BENCHMARK_FROZEN_THRESHOLD.txt"
)


# =============================================================================
# 9. ARTIFACT VALIDATION
# =============================================================================

def verify_artifacts():

    required = {

        "MODEL-09 checkpoint":
            CHECKPOINT_PATH,

        "Training mean":
            TRAIN_MEAN_PATH,

        "Training std":
            TRAIN_STD_PATH,

        "Frozen threshold":
            THRESHOLD_PATH,

    }

    missing = []

    for name, path in required.items():

        if not path.is_file():

            missing.append(
                f"{name}: {path}"
            )

    if missing:

        raise FileNotFoundError(

            "Required MODEL-09 deployment artifacts "
            "are missing:\n\n"
            +
            "\n".join(missing)
            +
            "\n\nExpected directory:\n"
            +
            str(ARTIFACT_DIR)
        )

    return True


# =============================================================================
# 10. LOAD TRAIN STANDARDIZATION ARTIFACTS
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_standardization():

    mean = np.load(
        TRAIN_MEAN_PATH
    )

    std = np.load(
        TRAIN_STD_PATH
    )

    mean = np.asarray(
        mean,
        dtype=np.float32
    )

    std = np.asarray(
        std,
        dtype=np.float32
    )

    if mean.size != TOKEN_FEATURE_DIM:

        raise ValueError(
            "Training mean contains "
            f"{mean.size} values. "
            f"Expected {TOKEN_FEATURE_DIM}."
        )

    if std.size != TOKEN_FEATURE_DIM:

        raise ValueError(
            "Training std contains "
            f"{std.size} values. "
            f"Expected {TOKEN_FEATURE_DIM}."
        )

    mean = mean.reshape(
        1,
        TOKEN_FEATURE_DIM
    )

    std = std.reshape(
        1,
        TOKEN_FEATURE_DIM
    )

    std = np.where(
        std < 1e-8,
        1.0,
        std
    )

    if not np.all(
        np.isfinite(mean)
    ):

        raise ValueError(
            "Training mean contains NaN/Inf."
        )

    if not np.all(
        np.isfinite(std)
    ):

        raise ValueError(
            "Training std contains NaN/Inf."
        )

    return mean, std


# =============================================================================
# 11. LOAD FROZEN THRESHOLD
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_frozen_threshold():

    with open(
        THRESHOLD_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        text = f.read().strip()

    try:

        threshold = float(
            text
        )

    except Exception:

        raise ValueError(
            "Could not parse frozen threshold "
            f"from {THRESHOLD_PATH}"
        )

    if not (
        0.0
        <= threshold
        <= 1.0
    ):

        raise ValueError(
            "Frozen threshold must be between "
            f"0 and 1. Received {threshold}"
        )

    return threshold


# =============================================================================
# 12. LOCAL ATTENTION BLOCK
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

        # x MUST be:
        #
        # [batch, tokens, model_dim]
        #
        # e.g.
        #
        # [1, 91, 96]

        if x.ndim != 3:

            raise RuntimeError(
                "LocalAttentionBlock expected a "
                "3-D tensor "
                "(batch,tokens,features), "
                f"received {x.ndim}-D: "
                f"{tuple(x.shape)}"
            )

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
                self.norm2(
                    x
                )
            )
        )

        return x


# =============================================================================
# 13. GLOBAL ATTENTION BLOCK
# =============================================================================

class GlobalAttentionBlock(
    LocalAttentionBlock
):

    pass


# =============================================================================
# 14. ATTENTION POOLING
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

        if x.ndim != 3:

            raise RuntimeError(
                "AttentionPooling expected 3-D "
                f"input, received {x.shape}"
            )

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
# 15. MODEL-09
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

        # ---------------------------------------------------------------------
        # HARD MODEL INPUT CHECK
        # ---------------------------------------------------------------------

        if x.ndim != 3:

            raise RuntimeError(

                "MODEL-09 requires a 3-D input tensor "
                "(batch,tokens,features). "

                f"Received {x.ndim}-D tensor "
                f"with shape {tuple(x.shape)}."
            )

        if x.shape[1] != TOKEN_LENGTH:

            raise RuntimeError(

                "MODEL-09 requires exactly "
                f"{TOKEN_LENGTH} tokens. "

                f"Received {x.shape[1]}."
            )

        if x.shape[2] != TOKEN_FEATURE_DIM:

            raise RuntimeError(

                "MODEL-09 requires exactly "
                f"{TOKEN_FEATURE_DIM} features. "

                f"Received {x.shape[2]}."
            )

        # ---------------------------------------------------------------------
        # INPUT PROJECTION
        # ---------------------------------------------------------------------

        x = self.input_projection(
            x
        )

        # ---------------------------------------------------------------------
        # TRAINING REPRESENTATION NOISE
        #
        # Disabled during deployment.
        # ---------------------------------------------------------------------

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

        # ---------------------------------------------------------------------
        # POSITION EMBEDDING
        # ---------------------------------------------------------------------

        T = x.size(1)

        if T > self.position_embedding.size(1):

            raise RuntimeError(
                "Sequence token length exceeds "
                "MODEL-09 positional embedding size."
            )

        x = (

            x

            +

            self.position_embedding[
                :,
                :T
            ]
        )

        # ---------------------------------------------------------------------
        # LOCAL ATTENTION
        # ---------------------------------------------------------------------

        x = self.local_attention(
            x
        )

        # ---------------------------------------------------------------------
        # GLOBAL ATTENTION
        # ---------------------------------------------------------------------

        x = self.global_attention(
            x
        )

        # ---------------------------------------------------------------------
        # ATTENTION POOLING
        # ---------------------------------------------------------------------

        pooled, attention = (
            self.pool(x)
        )

        # ---------------------------------------------------------------------
        # CLASSIFIER
        # ---------------------------------------------------------------------

        logits = self.classifier(
            pooled
        ).squeeze(-1)

        return (
            logits,
            attention
        )


# =============================================================================
# 16. LOAD MODEL CHECKPOINT
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_model():

    checkpoint = torch.load(

        CHECKPOINT_PATH,

        map_location=DEVICE,

        weights_only=False
    )

    model = (
        BidirectionalAttentionTransformerEncoder()
    )

    # -------------------------------------------------------------------------
    # Support both:
    #
    # checkpoint["model_state_dict"]
    #
    # and direct state_dict checkpoints.
    # -------------------------------------------------------------------------

    if isinstance(
        checkpoint,
        dict
    ) and (
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

    if not isinstance(
        state_dict,
        dict
    ):

        raise RuntimeError(
            "MODEL-09 checkpoint does not contain "
            "a valid state dictionary."
        )

    # -------------------------------------------------------------------------
    # Remove possible DataParallel prefix.
    # -------------------------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in (
        state_dict.items()
    ):

        if key.startswith(
            "module."
        ):

            key = key[
                len("module.") :
            ]

        cleaned_state_dict[
            key
        ] = value

    # -------------------------------------------------------------------------
    # Load weights strictly.
    # -------------------------------------------------------------------------

    try:

        model.load_state_dict(
            cleaned_state_dict,
            strict=True
        )

    except RuntimeError as e:

        raise RuntimeError(
            "MODEL-09 checkpoint architecture does not "
            "match the current deployment architecture.\n\n"
            +
            str(e)
        )

    model = model.to(
        DEVICE
    )

    model.eval()

    return model


# =============================================================================
# 17. INPUT SEQUENCE CLEANING
# =============================================================================

VALID_AMINO_ACIDS = set(
    "ACDEFGHIKLMNPQRSTVWY"
)


def clean_protein_sequence(
    sequence
):

    if sequence is None:

        raise ValueError(
            "No protein sequence was supplied."
        )

    sequence = str(
        sequence
    )

    # Remove FASTA headers.

    lines = sequence.splitlines()

    cleaned_lines = []

    for line in lines:

        line = line.strip()

        if not line:

            continue

        if line.startswith(">"):

            continue

        cleaned_lines.append(
            line
        )

    sequence = "".join(
        cleaned_lines
    )

    # Remove whitespace.

    sequence = "".join(
        sequence.split()
    )

    sequence = sequence.upper()

    if not sequence:

        raise ValueError(
            "The protein sequence is empty."
        )

    invalid = sorted(
        set(sequence)
        -
        VALID_AMINO_ACIDS
    )

    if invalid:

        raise ValueError(

            "Invalid amino-acid characters found: "
            +
            ", ".join(invalid)
            +
            "\n\nAllowed amino acids:\n"
            +
            "".join(
                sorted(
                    VALID_AMINO_ACIDS
                )
            )
        )

    return sequence


# =============================================================================
# 18. LOAD ESM-2
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_esm2():

    try:

        from transformers import (
            AutoTokenizer,
            AutoModel
        )

    except Exception as e:

        raise RuntimeError(
            "Transformers is not installed.\n\n"
            "Install with:\n"
            "pip install transformers"
        ) from e

    tokenizer = (
        AutoTokenizer.from_pretrained(
            ESM_MODEL_NAME
        )
    )

    model = (
        AutoModel.from_pretrained(
            ESM_MODEL_NAME
        )
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    return (
        tokenizer,
        model
    )


# =============================================================================
# 19. ESM-2 RESIDUE EMBEDDING
# =============================================================================

def extract_residue_embeddings(
    sequence,
    tokenizer,
    esm_model
):

    """
    Produce one 1280-D ESM-2 representation per amino acid.

    Special BOS/EOS positions are removed.
    """

    sequence = clean_protein_sequence(
        sequence
    )

    # ESM-2 maximum native sequence length
    # requires chunking for whole proteins.

    MAX_ESM_RESIDUES = 1022

    all_embeddings = []

    start = 0

    with torch.no_grad():

        while start < len(sequence):

            chunk = sequence[
                start:
                start + MAX_ESM_RESIDUES
            ]

            encoded = tokenizer(
                chunk,
                return_tensors="pt",
                add_special_tokens=True
            )

            encoded = {
                key: value.to(DEVICE)
                for key, value
                in encoded.items()
            }

            outputs = esm_model(
                **encoded
            )

            hidden = (
                outputs.last_hidden_state
            )

            # Remove BOS/EOS.

            residue_hidden = hidden[
                0,
                1:
                1 + len(chunk)
            ]

            residue_hidden = (
                residue_hidden
                .detach()
                .cpu()
                .float()
                .numpy()
            )

            if residue_hidden.shape != (
                len(chunk),
                ESM2_DIMENSION
            ):

                raise RuntimeError(

                    "Unexpected ESM-2 residue embedding "
                    f"shape {residue_hidden.shape}. "

                    f"Expected "
                    f"({len(chunk)}, {ESM2_DIMENSION})."
                )

            all_embeddings.append(
                residue_hidden
            )

            start += len(chunk)

    residue_embeddings = np.concatenate(
        all_embeddings,
        axis=0
    )

    if residue_embeddings.shape != (
        len(sequence),
        ESM2_DIMENSION
    ):

        raise RuntimeError(

            "Final residue embedding shape mismatch. "

            f"Received {residue_embeddings.shape}. "

            f"Expected "
            f"({len(sequence)}, {ESM2_DIMENSION})."
        )

    return (
        sequence,
        residue_embeddings
    )


# =============================================================================
# 20. RESIDUE → COMPLETE 48-AA TOKENS
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

            "Residue embeddings must be 2-D: "
            "(residues,1280). "

            f"Received {residue_embeddings.shape}."
        )

    if residue_embeddings.shape[1] != (
        ESM2_DIMENSION
    ):

        raise ValueError(

            "Expected residue embedding dimension "
            f"{ESM2_DIMENSION}. "

            f"Received {residue_embeddings.shape[1]}."
        )

    tokens = []

    # IMPORTANT:
    #
    # Complete chunks only.
    #
    # No partial final chunk.

    for start in range(
        0,
        residue_embeddings.shape[0]
        - CHUNK_SIZE
        + 1,
        CHUNK_STRIDE
    ):

        chunk = residue_embeddings[
            start:
            start + CHUNK_SIZE
        ]

        if chunk.shape[0] != CHUNK_SIZE:

            continue

        mean_vector = np.mean(
            chunk,
            axis=0
        )

        max_vector = np.max(
            chunk,
            axis=0
        )

        token = np.concatenate(
            [
                mean_vector,
                max_vector
            ],
            axis=0
        )

        tokens.append(
            token
        )

    if not tokens:

        raise ValueError(

            "Protein sequence is too short to produce "
            "a complete 48-aa token."
        )

    tokens = np.asarray(
        tokens,
        dtype=np.float32
    )

    if tokens.shape[1] != (
        TOKEN_FEATURE_DIM
    ):

        raise RuntimeError(

            "Token feature dimension mismatch. "

            f"Expected {TOKEN_FEATURE_DIM}. "

            f"Received {tokens.shape[1]}."
        )

    return tokens


# =============================================================================
# 21. PAD / TRUNCATE TO 91 TOKENS
# =============================================================================

def normalize_token_length(
    tokens
):

    tokens = np.asarray(
        tokens,
        dtype=np.float32
    )

    if tokens.ndim != 2:

        raise ValueError(
            "Tokens must be 2-D."
        )

    if tokens.shape[1] != (
        TOKEN_FEATURE_DIM
    ):

        raise ValueError(

            "Expected token feature dimension "
            f"{TOKEN_FEATURE_DIM}. "

            f"Received {tokens.shape[1]}."
        )

    raw_tokens = tokens.shape[0]

    # -------------------------------------------------------------------------
    # Truncate
    # -------------------------------------------------------------------------

    if raw_tokens > TOKEN_LENGTH:

        tokens = tokens[
            :TOKEN_LENGTH
        ]

    # -------------------------------------------------------------------------
    # Pad
    # -------------------------------------------------------------------------

    elif raw_tokens < TOKEN_LENGTH:

        padding = np.zeros(

            (
                TOKEN_LENGTH - raw_tokens,
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

            "Token-length normalization failed. "

            f"Received {tokens.shape}. "

            f"Expected "
            f"({TOKEN_LENGTH},{TOKEN_FEATURE_DIM})."
        )

    return (
        tokens,
        raw_tokens
    )


# =============================================================================
# 22. TRAIN-ONLY STANDARDIZATION
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

    if tokens.shape != (
        TOKEN_LENGTH,
        TOKEN_FEATURE_DIM
    ):

        raise ValueError(

            "Expected standardized input source shape "

            f"({TOKEN_LENGTH},{TOKEN_FEATURE_DIM}). "

            f"Received {tokens.shape}."
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
            "Standardized token representation "
            "contains NaN or infinite values."
        )

    return standardized


# =============================================================================
# 23. CRITICAL MODEL-09 PREDICTION FUNCTION
# =============================================================================

def predict_model09(
    model,
    token_matrix,
    train_mean,
    train_std,
    frozen_threshold
):

    """
    CRITICAL SHAPE CONTRACT:

        token_matrix before batching:
            (91, 2560)

        model input:
            (1, 91, 2560)

    NEVER:
            (1, 1, 91, 2560)
    """

    # -------------------------------------------------------------------------
    # Convert tensor → numpy if necessary
    # -------------------------------------------------------------------------

    if isinstance(
        token_matrix,
        torch.Tensor
    ):

        token_matrix = (

            token_matrix

            .detach()

            .cpu()

            .numpy()
        )

    token_matrix = np.asarray(
        token_matrix,
        dtype=np.float32
    )

    # -------------------------------------------------------------------------
    # SHAPE NORMALIZATION
    # -------------------------------------------------------------------------

    if token_matrix.ndim == 4:

        # Specifically recover the accidental:
        #
        # (1,1,91,2560)

        if (

            token_matrix.shape[0] == 1

            and

            token_matrix.shape[1] == 1

        ):

            token_matrix = (
                token_matrix[0, 0]
            )

        else:

            raise ValueError(

                "Unsupported 4-D MODEL-09 input: "

                f"{token_matrix.shape}. "

                "Expected (91,2560) "
                "or (1,91,2560)."
            )

    elif token_matrix.ndim == 3:

        if token_matrix.shape[0] == 1:

            token_matrix = (
                token_matrix[0]
            )

        else:

            raise ValueError(

                "MODEL-09 deployment accepts one "
                "sequence at a time. "

                f"Received {token_matrix.shape}."
            )

    elif token_matrix.ndim == 2:

        pass

    else:

        raise ValueError(

            "MODEL-09 token matrix must be "
            "2-D or 3-D. "

            f"Received {token_matrix.shape}."
        )

    # -------------------------------------------------------------------------
    # Verify unbatched representation
    # -------------------------------------------------------------------------

    if token_matrix.shape[1] != (
        TOKEN_FEATURE_DIM
    ):

        raise ValueError(

            "MODEL-09 requires "
            f"{TOKEN_FEATURE_DIM} features. "

            f"Received {token_matrix.shape[1]}."
        )

    # -------------------------------------------------------------------------
    # Ensure exactly 91 tokens
    # -------------------------------------------------------------------------

    token_matrix, raw_token_count = (
        normalize_token_length(
            token_matrix
        )
    )

    # -------------------------------------------------------------------------
    # TRAIN-ONLY STANDARDIZATION
    # -------------------------------------------------------------------------

    standardized = standardize_tokens(

        token_matrix,

        train_mean,

        train_std
    )

    # -------------------------------------------------------------------------
    # Convert to tensor
    #
    # Current shape:
    #
    #     (91,2560)
    #
    # -------------------------------------------------------------------------

    model_input = torch.from_numpy(
        standardized
    ).float()

    # -------------------------------------------------------------------------
    # HARD CHECK BEFORE BATCHING
    # -------------------------------------------------------------------------

    if model_input.ndim != 2:

        raise RuntimeError(

            "Expected 2-D token tensor before "
            "batching. "

            f"Received {model_input.ndim}-D "
            f"{tuple(model_input.shape)}."
        )

    if tuple(
        model_input.shape
    ) != (
        TOKEN_LENGTH,
        TOKEN_FEATURE_DIM
    ):

        raise RuntimeError(

            "Unexpected pre-batch shape: "

            f"{tuple(model_input.shape)}. "

            f"Expected "
            f"({TOKEN_LENGTH},{TOKEN_FEATURE_DIM})."
        )

    # -------------------------------------------------------------------------
    # ADD EXACTLY ONE BATCH DIMENSION
    #
    # (91,2560)
    #       ↓
    # (1,91,2560)
    # -------------------------------------------------------------------------

    model_input = (
        model_input
        .unsqueeze(0)
        .to(DEVICE)
    )

    # -------------------------------------------------------------------------
    # CRITICAL FINAL SHAPE ASSERTION
    # -------------------------------------------------------------------------

    if model_input.ndim != 3:

        raise RuntimeError(

            "MODEL-09 requires a 3-D tensor. "

            f"Received {model_input.ndim}-D "
            f"{tuple(model_input.shape)}."
        )

    expected_shape = (
        1,
        TOKEN_LENGTH,
        TOKEN_FEATURE_DIM
    )

    if tuple(
        model_input.shape
    ) != expected_shape:

        raise RuntimeError(

            "MODEL-09 final input shape mismatch. "

            f"Expected {expected_shape}. "

            f"Received "
            f"{tuple(model_input.shape)}."
        )

    # -------------------------------------------------------------------------
    # INFERENCE
    # -------------------------------------------------------------------------

    model.eval()

    with torch.no_grad():

        logits, attention = model(

            model_input,

            training_noise=False
        )

        probability = (

            torch.sigmoid(
                logits
            )

            .detach()

            .cpu()

            .numpy()

            .reshape(-1)[0]
        )

    probability = float(
        probability
    )

    # -------------------------------------------------------------------------
    # FROZEN VALIDATION THRESHOLD
    # -------------------------------------------------------------------------

    frozen_threshold = float(
        frozen_threshold
    )

    prediction = int(

        probability
        >=
        frozen_threshold
    )

    return {

        "probability":
            probability,

        "prediction":
            prediction,

        "threshold":
            frozen_threshold,

        "raw_token_count":
            raw_token_count,

        "attention":
            attention
            .detach()
            .cpu()
            .numpy()
            .reshape(-1),

        "model_input_shape":
            tuple(
                model_input.shape
            )
    }


# =============================================================================
# 24. INITIALIZE MODEL
# =============================================================================

try:

    verify_artifacts()

    train_mean, train_std = (
        load_standardization()
    )

    frozen_threshold = (
        load_frozen_threshold()
    )

    model = load_model()

    initialization_error = None

except Exception as e:

    model = None

    train_mean = None

    train_std = None

    frozen_threshold = None

    initialization_error = e


# =============================================================================
# 25. SIDEBAR
# =============================================================================

with st.sidebar:

    st.header(
        "MODEL-09 Configuration"
    )

    st.write(
        f"**Device:** `{DEVICE}`"
    )

    st.write(
        f"**ESM-2:** `{ESM_MODEL_NAME}`"
    )

    st.write(
        f"**ESM dimension:** `{ESM2_DIMENSION}`"
    )

    st.write(
        f"**Chunk size:** `{CHUNK_SIZE}`"
    )

    st.write(
        f"**Chunk stride:** `{CHUNK_STRIDE}`"
    )

    st.write(
        f"**Token dimension:** "
        f"`{TOKEN_FEATURE_DIM}`"
    )

    st.write(
        f"**Token length:** `{TOKEN_LENGTH}`"
    )

    st.write(
        f"**Model dimension:** `{MODEL_DIM}`"
    )

    st.write(
        f"**Attention heads:** "
        f"`{ATTENTION_HEADS}`"
    )

    st.divider()

    st.write(
        "**Frozen threshold:**"
    )

    if frozen_threshold is not None:

        st.code(
            f"{frozen_threshold:.10f}"
        )

    else:

        st.code(
            "UNAVAILABLE"
        )


# =============================================================================
# 26. INITIALIZATION FAILURE
# =============================================================================

if initialization_error is not None:

    st.error(
        "MODEL-09 could not be initialized."
    )

    st.code(
        str(initialization_error)
    )

    st.info(
        "Expected repository structure:"
    )

    st.code(
        """
app.py

artifacts/
├── MODEL-09_Bidirectional_Attention_Transformer_Encoder.pt
├── MODEL-09_BENCHMARK_TRAIN_MEAN.npy
├── MODEL-09_BENCHMARK_TRAIN_STD.npy
└── MODEL-09_BENCHMARK_FROZEN_THRESHOLD.txt
""".strip()
    )

    st.stop()


# =============================================================================
# 27. MODEL READY
# =============================================================================

st.success(
    "MODEL-09 loaded successfully."
)

st.write(
    f"Frozen validation threshold: "
    f"**{frozen_threshold:.4f}**"
)

st.write(
    f"Deployment device: **{DEVICE}**"
)


# =============================================================================
# 28. INPUT MODE
# =============================================================================

st.header(
    "Protein Sequence Input"
)

input_mode = st.radio(

    "Choose input mode",

    [
        "Paste protein sequence",
        "Upload FASTA file"
    ]
)


# =============================================================================
# 29. PASTE INPUT
# =============================================================================

sequence_text = ""


if input_mode == (
    "Paste protein sequence"
):

    sequence_text = st.text_area(

        "Paste HIV-1 protein sequence",

        height=250,

        placeholder=(
            "Example:\n"
            "MRVMGTQKNYSLLWRWGIMIFGILMACSANNLW..."
        )
    )


# =============================================================================
# 30. FASTA INPUT
# =============================================================================

else:

    uploaded_file = st.file_uploader(

        "Upload FASTA file",

        type=[
            "fasta",
            "fa",
            "faa",
            "txt"
        ]
    )

    if uploaded_file is not None:

        sequence_text = (
            uploaded_file
            .getvalue()
            .decode(
                "utf-8",
                errors="ignore"
            )
        )


# =============================================================================
# 31. PREDICT BUTTON
# =============================================================================

predict_button = st.button(

    "🧬 Predict Recombinant Status",

    type="primary",

    use_container_width=True
)


# =============================================================================
# 32. PREDICTION PIPELINE
# =============================================================================

if predict_button:

    if not sequence_text.strip():

        st.warning(
            "Please provide a protein sequence."
        )

        st.stop()

    # -------------------------------------------------------------------------
    # Clean sequence
    # -------------------------------------------------------------------------

    try:

        sequence = clean_protein_sequence(
            sequence_text
        )

    except Exception as e:

        st.error(
            "Invalid protein sequence."
        )

        st.code(
            str(e)
        )

        st.stop()

    # -------------------------------------------------------------------------
    # Sequence information
    # -------------------------------------------------------------------------

    st.subheader(
        "Input QC"
    )

    col1, col2, col3 = st.columns(
        3
    )

    with col1:

        st.metric(
            "Protein length",
            f"{len(sequence):,} aa"
        )

    with col2:

        st.metric(
            "Expected token length",
            TOKEN_LENGTH
        )

    with col3:

        st.metric(
            "Token feature dimension",
            TOKEN_FEATURE_DIM
        )

    # -------------------------------------------------------------------------
    # ESM-2
    # -------------------------------------------------------------------------

    try:

        with st.spinner(
            "Loading ESM-2..."
        ):

            tokenizer, esm_model = (
                load_esm2()
            )

    except Exception as e:

        st.error(
            "Could not load ESM-2."
        )

        st.code(
            str(e)
        )

        st.stop()

    # -------------------------------------------------------------------------
    # Residue embeddings
    # -------------------------------------------------------------------------

    try:

        with st.spinner(
            "Generating ESM-2 residue embeddings..."
        ):

            start_time = time.time()

            (
                sequence,
                residue_embeddings
            ) = extract_residue_embeddings(

                sequence,

                tokenizer,

                esm_model
            )

            embedding_time = (
                time.time()
                -
                start_time
            )

    except Exception as e:

        st.error(
            "ESM-2 embedding generation failed."
        )

        st.exception(e)

        st.stop()

    # -------------------------------------------------------------------------
    # Embedding QC
    # -------------------------------------------------------------------------

    if not np.all(
        np.isfinite(
            residue_embeddings
        )
    ):

        st.error(
            "ESM-2 embeddings contain NaN/Inf."
        )

        st.stop()

    st.success(
        "ESM-2 residue embeddings generated."
    )

    st.write(
        f"Residue embedding shape: "
        f"`{residue_embeddings.shape}`"
    )

    st.write(
        f"Embedding time: "
        f"`{embedding_time:.2f} seconds`"
    )

    # -------------------------------------------------------------------------
    # Tokenization
    # -------------------------------------------------------------------------

    try:

        tokens = residue_to_tokens(
            residue_embeddings
        )

        raw_token_count = (
            tokens.shape[0]
        )

        (
            tokens_91,
            _
        ) = normalize_token_length(
            tokens
        )

    except Exception as e:

        st.error(
            "Token construction failed."
        )

        st.exception(e)

        st.stop()

    # -------------------------------------------------------------------------
    # Token QC
    # -------------------------------------------------------------------------

    st.subheader(
        "Benchmark Representation"
    )

    col1, col2, col3 = st.columns(
        3
    )

    with col1:

        st.metric(
            "Complete raw tokens",
            raw_token_count
        )

    with col2:

        st.metric(
            "Final tokens",
            TOKEN_LENGTH
        )

    with col3:

        st.metric(
            "Features/token",
            TOKEN_FEATURE_DIM
        )

    st.write(
        f"Token matrix before standardization: "
        f"`{tokens_91.shape}`"
    )

    # -------------------------------------------------------------------------
    # MODEL-09 prediction
    # -------------------------------------------------------------------------

    try:

        with st.spinner(
            "Running MODEL-09..."
        ):

            result = predict_model09(

                model,

                tokens_91,

                train_mean,

                train_std,

                frozen_threshold
            )

    except Exception as e:

        st.error(
            "Prediction failed."
        )

        st.exception(e)

        st.write(
            "MODEL-09 expected final input:"
        )

        st.code(
            "(1, 91, 2560)"
        )

        st.stop()

    # -------------------------------------------------------------------------
    # FINAL RESULT
    # -------------------------------------------------------------------------

    probability = (
        result[
            "probability"
        ]
    )

    prediction = (
        result[
            "prediction"
        ]
    )

    threshold = (
        result[
            "threshold"
        ]
    )

    # -------------------------------------------------------------------------
    # Classification
    # -------------------------------------------------------------------------

    if prediction == 1:

        label = (
            "RECOMBINANT"
        )

    else:

        label = (
            "NON-RECOMBINANT"
        )

    st.divider()

    st.header(
        "MODEL-09 Prediction"
    )

    if prediction == 1:

        st.error(
            f"### {label}"
        )

    else:

        st.success(
            f"### {label}"
        )

    # -------------------------------------------------------------------------
    # Probability
    # -------------------------------------------------------------------------

    st.metric(

        "Recombinant Probability",

        f"{probability:.8f}"
    )

    # -------------------------------------------------------------------------
    # Threshold
    # -------------------------------------------------------------------------

    st.write(
        f"Frozen validation threshold: "
        f"`{threshold:.10f}`"
    )

    # -------------------------------------------------------------------------
    # Input mode
    # -------------------------------------------------------------------------

    st.write(
        "**Input mode:** "
        "`raw protein sequence → ESM-2 → "
        "48-aa complete chunks → mean+max → "
        "91×2560 benchmark representation`"
    )

    # -------------------------------------------------------------------------
    # Shape verification
    # -------------------------------------------------------------------------

    st.subheader(
        "Deployment Shape Verification"
    )

    st.code(
        f"""
Residue embeddings : {residue_embeddings.shape}
Raw token matrix   : {tokens.shape}
Final token matrix : {tokens_91.shape}
MODEL-09 input     : {result["model_input_shape"]}

Expected MODEL-09 input:
(1, 91, 2560)
""".strip()
    )

    if (
        result[
            "model_input_shape"
        ]
        ==
        (
            1,
            91,
            2560
        )
    ):

        st.success(
            "✓ MODEL-09 input shape verified: "
            "(1, 91, 2560)"
        )

    else:

        st.error(
            "MODEL-09 input shape verification failed."
        )

    # -------------------------------------------------------------------------
    # Interpretation
    # -------------------------------------------------------------------------

    st.divider()

    st.subheader(
        "Interpretation"
    )

    st.write(
        "The reported probability is the MODEL-09 "
        "sigmoid probability for the recombinant class."
    )

    st.write(
        f"The classification uses the frozen validation "
        f"threshold of {threshold:.4f}."
    )

    st.caption(
        "This deployment uses the current 9-model "
        "benchmark artifacts and train-only "
        "standardization statistics."
    )


# =============================================================================
# 33. FOOTER
# =============================================================================

st.divider()

st.caption(
    "MODEL-09 | Current 9-model benchmark | "
    "ESM-2 t33 650M | 1280-D residue embeddings | "
    "48-aa complete chunks | mean + max | "
    "2560-D tokens | 91-token representation"
)
