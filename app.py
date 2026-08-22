# =============================================================================
# MODEL-09 HIV-1 RECOMBINANT CLASSIFIER
# CURRENT 9-MODEL BENCHMARK DEPLOYMENT
#
# Pipeline:
#
# RAW HIV-1 PROTEIN SEQUENCE
#          |
#          v
#       ESM-2
# facebook/esm2_t33_650M_UR50D
#          |
#          v
# 1280-D residue embeddings
#          |
#          v
# COMPLETE 48-AA NON-OVERLAPPING CHUNKS
#          |
#          v
# MEAN + MAX POOLING
#          |
#          v
# 2560-D TOKENS
#          |
#          v
# PAD/TRUNCATE TO 91 TOKENS
#          |
#          v
# TRAIN-ONLY STANDARDIZATION
#          |
#          v
# (1, 91, 2560)
#          |
#          v
# MODEL-09
# Bidirectional Attention Transformer Encoder
#          |
#          v
# SIGMOID
#          |
#          v
# FROZEN VALIDATION THRESHOLD
#          |
#          v
# RECOMBINANT / NON-RECOMBINANT
#
# CURRENT BENCHMARK ARTIFACTS:
#
# artifacts/
#   MODEL-09_Bidirectional_Attention_Transformer_Encoder.pt
#   MODEL-09_BENCHMARK_TRAIN_MEAN.npy
#   MODEL-09_BENCHMARK_TRAIN_STD.npy
#   MODEL-09_BENCHMARK_FROZEN_THRESHOLD.txt
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

from transformers import AutoTokenizer
from transformers import AutoModel


# =============================================================================
# 2. STREAMLIT PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="MODEL-09 HIV-1 Recombinant Classifier",
    page_icon="🧬",
    layout="wide"
)


# =============================================================================
# 3. GLOBAL CONFIGURATION
# =============================================================================

MODEL_ID = (
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

# ESM-2 processing window.
#
# ESM-2 has special tokens and a practical maximum sequence length.
# We process long proteins in overlapping windows and remove special-token
# positions before reconstructing residue-level embeddings.

ESM_WINDOW = 1022

ESM_OVERLAP = 126

ESM_STRIDE = (
    ESM_WINDOW
    - ESM_OVERLAP
)


# =============================================================================
# 4. DEVICE
# =============================================================================

if torch.cuda.is_available():

    DEVICE = torch.device(
        "cuda"
    )

else:

    DEVICE = torch.device(
        "cpu"
    )


# =============================================================================
# 5. REPRODUCIBILITY
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
# 6. PROJECT / ARTIFACT PATHS
# =============================================================================

PROJECT_ROOT = Path(
    os.environ.get(
        "PROJECT_ROOT",
        "."
    )
).resolve()

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
)


CHECKPOINT_PATH = (
    OUTPUT_DIR
    / "MODEL-09_Bidirectional_Attention_Transformer_Encoder.pt"
)


TRAIN_MEAN_PATH = (
    OUTPUT_DIR
    / "MODEL-09_BENCHMARK_TRAIN_MEAN.npy"
)


TRAIN_STD_PATH = (
    OUTPUT_DIR
    / "MODEL-09_BENCHMARK_TRAIN_STD.npy"
)


THRESHOLD_PATH = (
    OUTPUT_DIR
    / "MODEL-09_BENCHMARK_FROZEN_THRESHOLD.txt"
)


# =============================================================================
# 7. REQUIRED ARTIFACT VERIFICATION
# =============================================================================

REQUIRED_ARTIFACTS = {

    "MODEL-09 checkpoint":
        CHECKPOINT_PATH,

    "Training mean":
        TRAIN_MEAN_PATH,

    "Training std":
        TRAIN_STD_PATH,

    "Frozen threshold":
        THRESHOLD_PATH

}


def verify_artifacts():

    missing = []

    for name, path in (
        REQUIRED_ARTIFACTS.items()
    ):

        if not path.exists():

            missing.append(
                f"{name}: {path}"
            )

    if missing:

        message = (
            "Required MODEL-09 deployment artifacts are missing:\n\n"
            + "\n".join(
                missing
            )
            + "\n\nExpected directory:\n"
            + str(
                OUTPUT_DIR
            )
        )

        raise FileNotFoundError(
            message
        )

    return True


# =============================================================================
# 8. MODEL COMPONENTS
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
# 9. GLOBAL ATTENTION
# =============================================================================

class GlobalAttentionBlock(
    LocalAttentionBlock
):

    pass


# =============================================================================
# 10. ATTENTION POOLING
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
# 11. MODEL-09
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

        # ---------------------------------------------------------------------
        # CRITICAL MODEL INPUT SHAPE GUARD
        #
        # Expected:
        #
        # (batch, tokens, features)
        #
        # e.g.
        #
        # (1, 91, 2560)
        #
        # ---------------------------------------------------------------------

        if x.ndim != 3:

            raise RuntimeError(
                "MODEL-09 requires a 3-D tensor "
                "(batch, tokens, features), "
                f"but received {x.ndim}-D tensor "
                f"with shape {tuple(x.shape)}."
            )

        if x.shape[-1] != INPUT_DIM:

            raise RuntimeError(
                "MODEL-09 feature dimension mismatch. "
                f"Expected {INPUT_DIM}, "
                f"received {x.shape[-1]}."
            )

        if x.shape[1] > TOKEN_LENGTH:

            raise RuntimeError(
                "MODEL-09 token dimension exceeds "
                f"maximum {TOKEN_LENGTH}: "
                f"received {x.shape[1]}."
            )

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

        logits = self.classifier(
            pooled
        ).squeeze(-1)

        return (
            logits,
            attention
        )


# =============================================================================
# 12. LOAD ESM-2
# =============================================================================

@st.cache_resource(
    show_spinner=False
)
def load_esm2():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID
    )

    model = AutoModel.from_pretrained(
        MODEL_ID
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
# 13. LOAD MODEL-09
# =============================================================================

@st.cache_resource(
    show_spinner=False
)
def load_model09():

    verify_artifacts()

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu"
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
# 14. LOAD TRAIN STANDARDIZATION
# =============================================================================

@st.cache_resource(
    show_spinner=False
)
def load_standardization():

    verify_artifacts()

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

    # -------------------------------------------------------------------------
    # Expected:
    #
    # (1, 1, 2560)
    #
    # -------------------------------------------------------------------------

    if train_mean.size != INPUT_DIM:

        raise RuntimeError(
            "Training mean has incorrect "
            f"number of values: {train_mean.size}. "
            f"Expected {INPUT_DIM}."
        )

    if train_std.size != INPUT_DIM:

        raise RuntimeError(
            "Training std has incorrect "
            f"number of values: {train_std.size}. "
            f"Expected {INPUT_DIM}."
        )

    train_mean = train_mean.reshape(
        1,
        1,
        INPUT_DIM
    )

    train_std = train_std.reshape(
        1,
        1,
        INPUT_DIM
    )

    train_std = np.maximum(
        train_std,
        1e-8
    )

    if not np.all(
        np.isfinite(
            train_mean
        )
    ):

        raise RuntimeError(
            "Training mean contains "
            "NaN or infinite values."
        )

    if not np.all(
        np.isfinite(
            train_std
        )
    ):

        raise RuntimeError(
            "Training std contains "
            "NaN or infinite values."
        )

    return (
        train_mean,
        train_std
    )


# =============================================================================
# 15. LOAD FROZEN THRESHOLD
# =============================================================================

@st.cache_resource(
    show_spinner=False
)
def load_threshold():

    verify_artifacts()

    text = (
        THRESHOLD_PATH
        .read_text()
        .strip()
    )

    # Extract first floating-point number.
    match = re.search(
        r"[-+]?(?:\d*\.\d+|\d+\.?)",
        text
    )

    if match is None:

        raise RuntimeError(
            "Could not read a numeric threshold "
            f"from {THRESHOLD_PATH}"
        )

    threshold = float(
        match.group(
            0
        )
    )

    if not (
        0.0
        <
        threshold
        <
        1.0
    ):

        raise RuntimeError(
            "Frozen threshold must be between "
            f"0 and 1. Found {threshold}."
        )

    return threshold


# =============================================================================
# 16. PROTEIN SEQUENCE VALIDATION
# =============================================================================

STANDARD_AMINO_ACIDS = set(
    "ACDEFGHIKLMNPQRSTVWY"
)

ALLOWED_AMINO_ACIDS = (
    STANDARD_AMINO_ACIDS
    |
    {"X"}
)


def clean_protein_sequence(
    sequence
):

    if sequence is None:

        raise ValueError(
            "Protein sequence is empty."
        )

    sequence = str(
        sequence
    )

    cleaned_lines = []

    for line in sequence.splitlines():

        line = line.strip()

        if not line:

            continue

        # Ignore FASTA header.
        if line.startswith(">"):

            continue

        cleaned_lines.append(
            line
        )

    sequence = "".join(
        cleaned_lines
    )

    sequence = "".join(
        sequence.split()
    )

    sequence = sequence.upper()

    if not sequence:

        raise ValueError(
            "Protein sequence is empty."
        )

    invalid = sorted(
        set(sequence)
        -
        ALLOWED_AMINO_ACIDS
    )

    if invalid:

        raise ValueError(
            "Invalid amino-acid characters found: "
            +
            ", ".join(
                invalid
            )
            +
            "\n\nAllowed amino acids:\n"
            +
            "".join(
                sorted(
                    ALLOWED_AMINO_ACIDS
                )
            )
            +
            "\n\nX is accepted as an ambiguous "
            "amino acid and is retained unchanged."
        )

    if len(sequence) < CHUNK_SIZE:

        raise ValueError(
            f"Protein sequence contains "
            f"{len(sequence)} residues. "
            f"At least {CHUNK_SIZE} residues "
            "are required."
        )

    return sequence


# =============================================================================
# 17. ESM-2 RESIDUE EMBEDDINGS
# =============================================================================

def extract_esm2_residue_embeddings(
    sequence,
    tokenizer,
    esm_model
):

    sequence = clean_protein_sequence(
        sequence
    )

    sequence_length = len(
        sequence
    )

    # -------------------------------------------------------------------------
    # Short sequence
    # -------------------------------------------------------------------------

    if sequence_length <= ESM_WINDOW:

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
            [0]
        )

        # Remove <cls> and <eos>.
        residue_embeddings = (
            hidden[
                1:
                1 + sequence_length
            ]
        )

        residue_embeddings = (
            residue_embeddings
            .detach()
            .float()
            .cpu()
            .numpy()
        )

        if residue_embeddings.shape != (
            sequence_length,
            ESM2_DIMENSION
        ):

            raise RuntimeError(
                "Unexpected ESM-2 residue "
                "embedding shape: "
                f"{residue_embeddings.shape}. "
                f"Expected "
                f"({sequence_length}, "
                f"{ESM2_DIMENSION})."
            )

        return residue_embeddings


    # -------------------------------------------------------------------------
    # Long sequence
    #
    # We process overlapping windows and average the embeddings of residues
    # that occur in multiple windows.
    # -------------------------------------------------------------------------

    embedding_sum = np.zeros(
        (
            sequence_length,
            ESM2_DIMENSION
        ),
        dtype=np.float64
    )

    embedding_count = np.zeros(
        sequence_length,
        dtype=np.float64
    )

    starts = list(
        range(
            0,
            sequence_length,
            ESM_STRIDE
        )
    )

    progress = st.progress(
        0,
        text=(
            "Extracting ESM-2 residue embeddings..."
        )
    )

    total_windows = len(
        starts
    )

    for window_index, start in enumerate(
        starts
    ):

        end = min(
            start
            +
            ESM_WINDOW,
            sequence_length
        )

        fragment = sequence[
            start:end
        ]

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
            [0]
        )

        fragment_length = len(
            fragment
        )

        fragment_embeddings = (
            hidden[
                1:
                1 + fragment_length
            ]
        )

        fragment_embeddings = (
            fragment_embeddings
            .detach()
            .float()
            .cpu()
            .numpy()
        )

        if fragment_embeddings.shape != (
            fragment_length,
            ESM2_DIMENSION
        ):

            raise RuntimeError(
                "Unexpected ESM-2 fragment "
                "shape: "
                f"{fragment_embeddings.shape}. "
                f"Expected "
                f"({fragment_length}, "
                f"{ESM2_DIMENSION})."
            )

        embedding_sum[
            start:end
        ] += fragment_embeddings

        embedding_count[
            start:end
        ] += 1.0

        progress.progress(
            int(
                (
                    window_index
                    +
                    1
                )
                /
                total_windows
                *
                100
            ),
            text=(
                f"ESM-2 window "
                f"{window_index + 1}/"
                f"{total_windows}"
            )
        )

    progress.empty()

    if np.any(
        embedding_count == 0
    ):

        missing = np.where(
            embedding_count == 0
        )[0]

        raise RuntimeError(
            "Some residues did not receive "
            "ESM-2 embeddings. "
            f"Number missing: {len(missing)}."
        )

    residue_embeddings = (
        embedding_sum
        /
        embedding_count[:, None]
    )

    residue_embeddings = (
        residue_embeddings
        .astype(
            np.float32
        )
    )

    if not np.all(
        np.isfinite(
            residue_embeddings
        )
    ):

        raise RuntimeError(
            "ESM-2 residue embeddings contain "
            "NaN or infinite values."
        )

    return residue_embeddings


# =============================================================================
# 18. RESIDUE → 2560-D TOKENS
# =============================================================================

def residue_embeddings_to_tokens(
    residue_embeddings
):

    residue_embeddings = np.asarray(
        residue_embeddings,
        dtype=np.float32
    )

    if residue_embeddings.ndim != 2:

        raise ValueError(
            "Residue embeddings must be 2-D. "
            f"Received shape "
            f"{residue_embeddings.shape}."
        )

    if residue_embeddings.shape[1] != (
        ESM2_DIMENSION
    ):

        raise ValueError(
            "ESM-2 embedding dimension mismatch. "
            f"Expected {ESM2_DIMENSION}, "
            f"received "
            f"{residue_embeddings.shape[1]}."
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
            "Protein does not contain "
            "a complete 48-aa chunk."
        )

    usable_residues = (
        complete_tokens
        *
        CHUNK_SIZE
    )

    residues = residue_embeddings[
        :usable_residues
    ]

    chunks = residues.reshape(
        complete_tokens,
        CHUNK_SIZE,
        ESM2_DIMENSION
    )

    mean_features = (
        chunks.mean(
            axis=1
        )
    )

    max_features = (
        chunks.max(
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

    if tokens.shape[1] != (
        INPUT_DIM
    ):

        raise RuntimeError(
            "Token feature dimension mismatch. "
            f"Expected {INPUT_DIM}, "
            f"received {tokens.shape[1]}."
        )

    return tokens.astype(
        np.float32
    )


# =============================================================================
# 19. PAD/TRUNCATE TO 91 TOKENS
# =============================================================================

def pad_or_truncate_tokens(
    tokens
):

    tokens = np.asarray(
        tokens,
        dtype=np.float32
    )

    if tokens.ndim != 2:

        raise ValueError(
            "Token matrix must be 2-D."
        )

    if tokens.shape[1] != (
        INPUT_DIM
    ):

        raise ValueError(
            "Token feature dimension must be "
            f"{INPUT_DIM}."
        )

    raw_tokens = (
        tokens.shape[0]
    )

    if raw_tokens >= TOKEN_LENGTH:

        final_tokens = (
            tokens[
                :TOKEN_LENGTH
            ]
        )

    else:

        final_tokens = np.zeros(
            (
                TOKEN_LENGTH,
                INPUT_DIM
            ),
            dtype=np.float32
        )

        final_tokens[
            :raw_tokens
        ] = tokens

    if final_tokens.shape != (
        TOKEN_LENGTH,
        INPUT_DIM
    ):

        raise RuntimeError(
            "Final token representation has "
            f"incorrect shape "
            f"{final_tokens.shape}. "
            f"Expected "
            f"({TOKEN_LENGTH}, "
            f"{INPUT_DIM})."
        )

    return final_tokens


# =============================================================================
# 20. STANDARDIZE USING TRAINING ARTIFACTS ONLY
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
        INPUT_DIM
    ):

        raise ValueError(
            "Expected token matrix shape "
            f"({TOKEN_LENGTH}, {INPUT_DIM}), "
            f"received {tokens.shape}."
        )

    # train_mean/std are:
    #
    # (1, 1, 2560)
    #
    # Convert to:
    #
    # (2560,)
    #

    mean = train_mean.reshape(
        INPUT_DIM
    )

    std = train_std.reshape(
        INPUT_DIM
    )

    std = np.maximum(
        std,
        1e-8
    )

    standardized = (
        tokens
        -
        mean[None, :]
    ) / std[None, :]

    standardized = (
        standardized
        .astype(
            np.float32
        )
    )

    if not np.all(
        np.isfinite(
            standardized
        )
    ):

        raise RuntimeError(
            "Standardized MODEL-09 input contains "
            "NaN or infinite values."
        )

    return standardized


# =============================================================================
# 21. FINAL MODEL INPUT
# =============================================================================

def make_model_input(
    standardized_tokens
):

    x = np.asarray(
        standardized_tokens,
        dtype=np.float32
    )

    # -------------------------------------------------------------------------
    # MUST be:
    #
    # (91, 2560)
    #
    # -------------------------------------------------------------------------

    if x.ndim != 2:

        raise RuntimeError(
            "Before batching, MODEL-09 representation "
            "must be 2-D. "
            f"Received {x.ndim}-D."
        )

    if x.shape != (
        TOKEN_LENGTH,
        INPUT_DIM
    ):

        raise RuntimeError(
            "Before batching, expected "
            f"({TOKEN_LENGTH}, {INPUT_DIM}), "
            f"received {x.shape}."
        )

    # -------------------------------------------------------------------------
    # Add EXACTLY ONE batch dimension.
    #
    # (91, 2560)
    #
    # becomes
    #
    # (1, 91, 2560)
    # -------------------------------------------------------------------------

    x = np.expand_dims(
        x,
        axis=0
    )

    if x.ndim != 3:

        raise RuntimeError(
            "MODEL-09 input must be exactly "
            "3-D after batching. "
            f"Received {x.ndim}-D."
        )

    if x.shape != (
        1,
        TOKEN_LENGTH,
        INPUT_DIM
    ):

        raise RuntimeError(
            "MODEL-09 final input shape mismatch. "
            f"Expected "
            f"(1, {TOKEN_LENGTH}, {INPUT_DIM}), "
            f"received {x.shape}."
        )

    return torch.from_numpy(
        x
    )


# =============================================================================
# 22. COMPLETE PREPROCESSING PIPELINE
# =============================================================================

def build_model09_input(
    sequence,
    tokenizer,
    esm_model,
    train_mean,
    train_std
):

    sequence = clean_protein_sequence(
        sequence
    )

    residue_embeddings = (
        extract_esm2_residue_embeddings(
            sequence,
            tokenizer,
            esm_model
        )
    )

    tokens = (
        residue_embeddings_to_tokens(
            residue_embeddings
        )
    )

    raw_token_count = (
        tokens.shape[0]
    )

    final_tokens = (
        pad_or_truncate_tokens(
            tokens
        )
    )

    standardized = (
        standardize_tokens(
            final_tokens,
            train_mean,
            train_std
        )
    )

    model_input = (
        make_model_input(
            standardized
        )
    )

    return (
        model_input,
        residue_embeddings,
        tokens,
        final_tokens,
        raw_token_count
    )


# =============================================================================
# 23. MODEL PREDICTION
# =============================================================================

def predict_model09(
    model,
    model_input,
    threshold
):

    # -------------------------------------------------------------------------
    # ABSOLUTE FINAL SHAPE CHECK
    # -------------------------------------------------------------------------

    if model_input.ndim != 3:

        raise AssertionError(
            "MODEL-09 prediction input is not 3-D. "
            f"Received {model_input.ndim}-D "
            f"shape={tuple(model_input.shape)}."
        )

    expected_shape = (
        1,
        TOKEN_LENGTH,
        INPUT_DIM
    )

    if tuple(
        model_input.shape
    ) != expected_shape:

        raise AssertionError(
            "MODEL-09 prediction input shape "
            f"must be {expected_shape}, "
            f"received "
            f"{tuple(model_input.shape)}."
        )

    model_input = model_input.to(
        DEVICE,
        dtype=torch.float32
    )

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

    prediction = int(
        probability
        >=
        threshold
    )

    if prediction == 1:

        label = (
            "RECOMBINANT"
        )

    else:

        label = (
            "NON-RECOMBINANT"
        )

    return {
        "probability":
            probability,

        "prediction":
            prediction,

        "label":
            label,

        "threshold":
            float(threshold),

        "attention":
            attention
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
    }


# =============================================================================
# 24. ATTENTION INTERPRETATION
# =============================================================================

def attention_summary(
    attention
):

    attention = np.asarray(
        attention,
        dtype=np.float32
    )

    if attention.size != TOKEN_LENGTH:

        return None

    attention = np.nan_to_num(
        attention,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    total = (
        attention.sum()
    )

    if total > 0:

        attention = (
            attention
            /
            total
        )

    top_indices = np.argsort(
        attention
    )[::-1][:10]

    rows = []

    for index in top_indices:

        start = (
            index
            *
            CHUNK_SIZE
            +
            1
        )

        end = (
            min(
                (
                    index + 1
                )
                *
                CHUNK_SIZE,
                999999999
            )
        )

        rows.append({

            "Token":
                int(index + 1),

            "Approx_Residue_Start":
                int(start),

            "Approx_Residue_End":
                int(end),

            "Attention":
                float(
                    attention[index]
                )
        })

    return rows


# =============================================================================
# 25. APPLICATION HEADER
# =============================================================================

st.title(
    "🧬 MODEL-09 HIV-1 Recombinant Classifier"
)

st.caption(
    "Current 9-model benchmark deployment"
)


# =============================================================================
# 26. INITIALIZE DEPLOYMENT
# =============================================================================

try:

    verify_artifacts()

except Exception as e:

    st.error(
        "MODEL-09 could not be initialized."
    )

    st.code(
        str(e)
    )

    st.stop()


# =============================================================================
# 27. LOAD ARTIFACTS
# =============================================================================

try:

    with st.spinner(
        "Loading MODEL-09..."
    ):

        model = load_model09()

        train_mean, train_std = (
            load_standardization()
        )

        frozen_threshold = (
            load_threshold()
        )

except Exception as e:

    st.error(
        "MODEL-09 could not be initialized."
    )

    st.exception(
        e
    )

    st.stop()


# =============================================================================
# 28. LOAD ESM-2 ONLY WHEN NEEDED
# =============================================================================

st.success(
    "MODEL-09 deployment artifacts loaded successfully."
)


# =============================================================================
# 29. DEPLOYMENT INFORMATION
# =============================================================================

with st.expander(
    "Deployment configuration",
    expanded=False
):

    st.write(
        f"**Device:** `{DEVICE}`"
    )

    st.write(
        f"**ESM-2:** `{MODEL_ID}`"
    )

    st.write(
        f"**ESM-2 dimension:** `{ESM2_DIMENSION}`"
    )

    st.write(
        f"**Chunk size:** `{CHUNK_SIZE}`"
    )

    st.write(
        f"**Chunk stride:** `{CHUNK_STRIDE}`"
    )

    st.write(
        f"**Token dimension:** `{INPUT_DIM}`"
    )

    st.write(
        f"**Token length:** `{TOKEN_LENGTH}`"
    )

    st.write(
        f"**MODEL-09 dimension:** `{MODEL_DIM}`"
    )

    st.write(
        f"**Attention heads:** `{ATTENTION_HEADS}`"
    )

    st.write(
        f"**Frozen threshold:** `{frozen_threshold:.10f}`"
    )

    st.write(
        f"**Checkpoint:** `{CHECKPOINT_PATH}`"
    )

    st.write(
        f"**Training mean:** `{TRAIN_MEAN_PATH}`"
    )

    st.write(
        f"**Training std:** `{TRAIN_STD_PATH}`"
    )

    st.write(
        f"**Threshold:** `{THRESHOLD_PATH}`"
    )


# =============================================================================
# 30. INPUT
# =============================================================================

st.subheader(
    "Enter HIV-1 protein sequence"
)

sequence_input = st.text_area(
    "Protein sequence",
    height=250,
    placeholder=(
        "Paste an amino-acid sequence here.\n\n"
        "FASTA headers beginning with > are accepted.\n"
        "Spaces and line breaks are automatically removed.\n"
        "Ambiguous X residues are accepted."
    )
)


# =============================================================================
# 31. EXAMPLE / CLEAR HELP
# =============================================================================

st.info(
    "Allowed amino acids: "
    "ACDEFGHIKLMNPQRSTVWY "
    "plus X for ambiguous residues."
)


# =============================================================================
# 32. PREDICTION BUTTON
# =============================================================================

predict_button = st.button(
    "🔬 Predict recombinant status",
    type="primary",
    use_container_width=True
)


# =============================================================================
# 33. PREDICTION PIPELINE
# =============================================================================

if predict_button:

    # -------------------------------------------------------------------------
    # Validate raw sequence
    # -------------------------------------------------------------------------

    try:

        sequence = (
            clean_protein_sequence(
                sequence_input
            )
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
    # Sequence statistics
    # -------------------------------------------------------------------------

    sequence_length = len(
        sequence
    )

    x_count = sequence.count(
        "X"
    )

    x_fraction = (
        x_count
        /
        sequence_length
    )


    # -------------------------------------------------------------------------
    # Display sequence QC
    # -------------------------------------------------------------------------

    st.subheader(
        "Sequence QC"
    )

    col1, col2, col3 = (
        st.columns(3)
    )

    with col1:

        st.metric(
            "Protein length",
            f"{sequence_length:,} aa"
        )

    with col2:

        st.metric(
            "Ambiguous X residues",
            f"{x_count:,}"
        )

    with col3:

        st.metric(
            "X fraction",
            f"{x_fraction:.2%}"
        )


    if x_count > 0:

        st.warning(
            f"The input contains {x_count} "
            "ambiguous X residue(s). "
            "X residues are preserved unchanged."
        )


    # -------------------------------------------------------------------------
    # Load ESM-2
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
            "ESM-2 could not be loaded."
        )

        st.exception(
            e
        )

        st.stop()


    # -------------------------------------------------------------------------
    # Build MODEL-09 representation
    # -------------------------------------------------------------------------

    try:

        start_time = (
            time.time()
        )

        with st.spinner(
            "Generating ESM-2 residue embeddings and MODEL-09 representation..."
        ):

            (
                model_input,
                residue_embeddings,
                raw_tokens,
                final_tokens,
                raw_token_count
            ) = build_model09_input(

                sequence,
                tokenizer,
                esm_model,
                train_mean,
                train_std

            )

        preprocessing_time = (
            time.time()
            -
            start_time
        )

    except Exception as e:

        st.error(
            "MODEL-09 preprocessing failed."
        )

        st.exception(
            e
        )

        st.stop()


    # -------------------------------------------------------------------------
    # Representation QC
    # -------------------------------------------------------------------------

    st.subheader(
        "MODEL-09 representation"
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    with col1:

        st.metric(
            "Residue embedding",
            str(
                tuple(
                    residue_embeddings.shape
                )
            )
        )

    with col2:

        st.metric(
            "Raw 48-aa tokens",
            raw_token_count
        )

    with col3:

        st.metric(
            "Final token matrix",
            str(
                tuple(
                    final_tokens.shape
                )
            )
        )

    with col4:

        st.metric(
            "Model input",
            str(
                tuple(
                    model_input.shape
                )
            )
        )


    # -------------------------------------------------------------------------
    # ABSOLUTE 4-D PROTECTION
    # -------------------------------------------------------------------------

    if model_input.ndim != 3:

        st.error(
            "Internal tensor construction error."
        )

        st.code(
            f"Received shape: "
            f"{tuple(model_input.shape)}\n"
            f"Expected: "
            f"(1, {TOKEN_LENGTH}, {INPUT_DIM})"
        )

        st.stop()


    # -------------------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------------------

    try:

        with st.spinner(
            "Running MODEL-09 prediction..."
        ):

            prediction_start = (
                time.time()
            )

            result = predict_model09(
                model,
                model_input,
                frozen_threshold
            )

            prediction_time = (
                time.time()
                -
                prediction_start
            )

    except Exception as e:

        st.error(
            "Prediction failed."
        )

        st.exception(
            e
        )

        st.stop()


    # -------------------------------------------------------------------------
    # RESULT
    # -------------------------------------------------------------------------

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


    col1, col2, col3 = (
        st.columns(3)
    )

    with col1:

        st.metric(
            "Recombinant probability",
            f"{result['probability']:.8f}"
        )

    with col2:

        st.metric(
            "Frozen threshold",
            f"{result['threshold']:.8f}"
        )

    with col3:

        st.metric(
            "Binary prediction",
            result["label"]
        )


    # -------------------------------------------------------------------------
    # Probability interpretation
    # -------------------------------------------------------------------------

    probability = (
        result["probability"]
    )

    threshold = (
        result["threshold"]
    )

    st.write(
        f"The MODEL-09 probability is "
        f"**{probability:.6f}**."
    )

    st.write(
        f"The frozen validation threshold is "
        f"**{threshold:.6f}**."
    )

    if probability >= threshold:

        st.write(
            "Because the probability is greater than "
            "or equal to the frozen threshold, "
            "MODEL-09 classifies this sequence as "
            "**RECOMBINANT**."
        )

    else:

        st.write(
            "Because the probability is below "
            "the frozen threshold, MODEL-09 "
            "classifies this sequence as "
            "**NON-RECOMBINANT**."
        )


    # -------------------------------------------------------------------------
    # Timing
    # -------------------------------------------------------------------------

    with st.expander(
        "Inference diagnostics",
        expanded=False
    ):

        st.write(
            f"Preprocessing time: "
            f"{preprocessing_time:.2f} seconds"
        )

        st.write(
            f"MODEL-09 inference time: "
            f"{prediction_time:.4f} seconds"
        )

        st.write(
            f"Residues: "
            f"{sequence_length}"
        )

        st.write(
            f"Raw complete 48-aa tokens: "
            f"{raw_token_count}"
        )

        st.write(
            f"Final tokens: "
            f"{TOKEN_LENGTH}"
        )

        st.write(
            f"Feature dimension: "
            f"{INPUT_DIM}"
        )

        st.write(
            f"Final tensor shape: "
            f"{tuple(model_input.shape)}"
        )


    # -------------------------------------------------------------------------
    # Attention information
    # -------------------------------------------------------------------------

    attention_rows = (
        attention_summary(
            result["attention"]
        )
    )

    if attention_rows is not None:

        st.subheader(
            "MODEL-09 attention summary"
        )

        st.caption(
            "These are the highest-attention 48-aa token regions. "
            "They should be interpreted as model attention, "
            "not as independent biological evidence of recombination."
        )

        import pandas as pd

        attention_df = pd.DataFrame(
            attention_rows
        )

        st.dataframe(
            attention_df,
            use_container_width=True,
            hide_index=True
        )


# =============================================================================
# 34. FOOTER
# =============================================================================

st.divider()

st.caption(
    "MODEL-09 — Current 9-model benchmark | "
    "ESM-2 650M | 1280-D residue embeddings | "
    "48-aa mean+max tokens | 91 tokens | "
    "2560-D input | frozen train-only standardization | "
    "frozen validation threshold"
)
