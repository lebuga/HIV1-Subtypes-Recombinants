# ==================================================================================================
# MODEL-09 — CURRENT 9-MODEL BENCHMARK STREAMLIT DEPLOYMENT
#
# CURRENT BENCHMARK PIPELINE
#
# RAW HIV-1 PROTEIN SEQUENCE
#        ↓
# ESM-2 t33 650M
#        ↓
# residue-level embeddings
#        ↓
# complete non-overlapping 48-aa chunks
#        ↓
# mean + max pooling
#        ↓
# 2560-D token representation
#        ↓
# pad / truncate to 91 tokens
#        ↓
# TRAIN-ONLY benchmark standardization
#        ↓
# MODEL-09
# Bidirectional Attention Transformer Encoder
#        ↓
# sigmoid probability
#        ↓
# frozen validation threshold = 0.71
#        ↓
# RECOMBINANT / NON-RECOMBINANT
#
# REQUIRED REPOSITORY ARTIFACTS
#
# artifacts/
#   MODEL-09_Bidirectional_Attention_Transformer_Encoder.pt
#   MODEL-09_BENCHMARK_TRAIN_MEAN.npy
#   MODEL-09_BENCHMARK_TRAIN_STD.npy
#   MODEL-09_BENCHMARK_FROZEN_THRESHOLD.txt
#
# IMPORTANT:
# - X is accepted as an amino-acid character.
# - MultiheadAttention receives exactly 3-D tensors:
#       [batch, tokens, model_dim]
# - No 4-D tensor is passed to attention.
# - This is the CURRENT 9-MODEL BENCHMARK pipeline.
# ==================================================================================================


# ==================================================================================================
# 1. IMPORTS
# ==================================================================================================

from pathlib import Path
import os
import re
import hashlib
import time

import numpy as np
import streamlit as st

import torch
import torch.nn as nn

from transformers import AutoTokenizer, EsmModel


# ==================================================================================================
# 2. STREAMLIT PAGE
# ==================================================================================================

st.set_page_config(
    page_title="MODEL-09 HIV-1 Recombinant Classifier",
    page_icon="🧬",
    layout="wide"
)


# ==================================================================================================
# 3. BENCHMARK CONSTANTS
# ==================================================================================================

ESM_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"

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

FROZEN_THRESHOLD_DEFAULT = 0.71

SEED = 42


# ==================================================================================================
# 4. DEVICE
# ==================================================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================================================================================================
# 5. REPOSITORY / ARTIFACT PATH RESOLUTION
# ==================================================================================================

# Streamlit Cloud normally runs with the repository as the current working directory.
# However, do NOT assume that os.getcwd() is the repository root.
#
# This function searches several safe locations and returns the first directory
# containing the required deployment artifacts.

REQUIRED_ARTIFACT_NAMES = [
    "MODEL-09_Bidirectional_Attention_Transformer_Encoder.pt",
    "MODEL-09_BENCHMARK_TRAIN_MEAN.npy",
    "MODEL-09_BENCHMARK_TRAIN_STD.npy",
    "MODEL-09_BENCHMARK_FROZEN_THRESHOLD.txt",
]


def find_repository_root():

    candidates = []

    try:
        candidates.append(
            Path(__file__).resolve().parent
        )
    except Exception:
        pass

    candidates.append(
        Path.cwd()
    )

    candidates.append(
        Path("/mount/src/hiv1-subtypes-recombinants")
    )

    candidates.append(
        Path("/app")
    )

    seen = set()

    for candidate in candidates:

        try:
            candidate = candidate.resolve()
        except Exception:
            continue

        if str(candidate) in seen:
            continue

        seen.add(str(candidate))

        if candidate.exists():
            return candidate

    return Path.cwd()


PROJECT_ROOT = find_repository_root()


def locate_artifacts():

    possible_artifact_dirs = [

        PROJECT_ROOT / "artifacts",

        PROJECT_ROOT,

        Path.cwd() / "artifacts",

        Path.cwd(),

        Path("/mount/src/hiv1-subtypes-recombinants/artifacts"),

        Path("/mount/src/hiv1-subtypes-recombinants"),

    ]

    checked = []

    for directory in possible_artifact_dirs:

        try:
            directory = directory.resolve()
        except Exception:
            continue

        if directory in checked:
            continue

        checked.append(directory)

        if not directory.exists():
            continue

        if all(
            (directory / name).is_file()
            for name in REQUIRED_ARTIFACT_NAMES
        ):

            return directory

    return None


ARTIFACT_DIR = locate_artifacts()


# ==================================================================================================
# 6. ARTIFACT ERROR REPORT
# ==================================================================================================

def artifact_error_message():

    lines = []

    lines.append(
        "Required MODEL-09 deployment artifacts could not be found."
    )

    lines.append("")

    lines.append(
        f"Repository root detected:\n{PROJECT_ROOT}"
    )

    lines.append("")

    lines.append(
        "Expected directory:"
    )

    lines.append(
        str(PROJECT_ROOT / "artifacts")
    )

    lines.append("")

    lines.append(
        "Required files:"
    )

    for name in REQUIRED_ARTIFACT_NAMES:

        lines.append(
            f"  {name}"
        )

    lines.append("")

    lines.append(
        "Checked locations:"
    )

    possible_dirs = [

        PROJECT_ROOT / "artifacts",
        PROJECT_ROOT,
        Path.cwd() / "artifacts",
        Path.cwd(),
        Path("/mount/src/hiv1-subtypes-recombinants/artifacts"),
        Path("/mount/src/hiv1-subtypes-recombinants"),
    ]

    for directory in possible_dirs:

        lines.append(
            f"  {directory}"
        )

    return "\n".join(lines)


# ==================================================================================================
# 7. LOAD DEPLOYMENT ARTIFACTS
# ==================================================================================================

if ARTIFACT_DIR is None:

    st.error(
        "MODEL-09 could not be initialized."
    )

    st.code(
        artifact_error_message(),
        language="text"
    )

    st.stop()


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


# ==================================================================================================
# 8. VERIFY ARTIFACT FILES
# ==================================================================================================

missing_artifacts = []

for path in [

    CHECKPOINT_PATH,
    TRAIN_MEAN_PATH,
    TRAIN_STD_PATH,
    THRESHOLD_PATH

]:

    if not path.is_file():

        missing_artifacts.append(
            str(path)
        )


if missing_artifacts:

    st.error(
        "MODEL-09 deployment artifacts are incomplete."
    )

    st.code(
        "\n".join(
            missing_artifacts
        ),
        language="text"
    )

    st.stop()


# ==================================================================================================
# 9. LOAD STANDARDIZATION ARTIFACTS
# ==================================================================================================

try:

    TRAIN_MEAN = np.load(
        TRAIN_MEAN_PATH
    ).astype(
        np.float32
    )

    TRAIN_STD = np.load(
        TRAIN_STD_PATH
    ).astype(
        np.float32
    )

except Exception as exc:

    st.error(
        "Failed to load MODEL-09 standardization artifacts."
    )

    st.exception(exc)

    st.stop()


# ==================================================================================================
# 10. STANDARDIZATION SHAPE VERIFICATION
# ==================================================================================================

if TRAIN_MEAN.shape != (
    1,
    1,
    INPUT_DIM
):

    st.error(
        "MODEL-09 training mean has an unexpected shape."
    )

    st.code(
        f"Expected: (1, 1, {INPUT_DIM})\n"
        f"Received: {TRAIN_MEAN.shape}",
        language="text"
    )

    st.stop()


if TRAIN_STD.shape != (
    1,
    1,
    INPUT_DIM
):

    st.error(
        "MODEL-09 training std has an unexpected shape."
    )

    st.code(
        f"Expected: (1, 1, {INPUT_DIM})\n"
        f"Received: {TRAIN_STD.shape}",
        language="text"
    )

    st.stop()


if not np.all(
    np.isfinite(TRAIN_MEAN)
):

    st.error(
        "Training mean contains non-finite values."
    )

    st.stop()


if not np.all(
    np.isfinite(TRAIN_STD)
):

    st.error(
        "Training std contains non-finite values."
    )

    st.stop()


if np.any(
    TRAIN_STD <= 0
):

    st.error(
        "Training std contains zero or negative values."
    )

    st.stop()


# ==================================================================================================
# 11. LOAD FROZEN THRESHOLD
# ==================================================================================================

try:

    threshold_text = (
        THRESHOLD_PATH
        .read_text(
            encoding="utf-8"
        )
        .strip()
    )

    FROZEN_THRESHOLD = float(
        threshold_text
    )

except Exception as exc:

    st.error(
        "Could not read MODEL-09 frozen threshold."
    )

    st.exception(exc)

    st.stop()


if not (
    0.0 < FROZEN_THRESHOLD < 1.0
):

    st.error(
        f"Invalid frozen threshold: {FROZEN_THRESHOLD}"
    )

    st.stop()


# ==================================================================================================
# 12. LOCAL ATTENTION BLOCK
# ==================================================================================================

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

        self.norm = nn.LayerNorm(
            dim
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        x
    ):

        # ------------------------------------------------------------------------------------------
        # CRITICAL SHAPE GUARD
        #
        # MultiheadAttention with batch_first=True requires:
        #
        # [batch, sequence_length, embedding_dimension]
        #
        # Therefore x MUST be 3-D.
        # ------------------------------------------------------------------------------------------

        if x.dim() != 3:

            raise RuntimeError(
                "LocalAttentionBlock expected a 3-D tensor "
                f"[batch, tokens, dim], but received "
                f"shape={tuple(x.shape)}"
            )

        z = self.norm(
            x
        )

        attended, _ = self.attn(
            z,
            z,
            z,
            need_weights=False
        )

        return (
            x
            +
            self.dropout(
                attended
            )
        )


# ==================================================================================================
# 13. GLOBAL ATTENTION BLOCK
# ==================================================================================================

class GlobalAttentionBlock(
    LocalAttentionBlock
):

    pass


# ==================================================================================================
# 14. ATTENTION POOLING
# ==================================================================================================

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

        if x.dim() != 3:

            raise RuntimeError(
                "AttentionPooling expected a 3-D tensor "
                f"[batch, tokens, dim], got {tuple(x.shape)}"
            )

        scores = self.score(
            x
        ).squeeze(
            -1
        )

        weights = torch.softmax(
            scores,
            dim=1
        )

        pooled = torch.sum(
            x
            *
            weights.unsqueeze(
                -1
            ),
            dim=1
        )

        return (
            pooled,
            weights
        )


# ==================================================================================================
# 15. MODEL-09 ARCHITECTURE
# ==================================================================================================

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

        # ------------------------------------------------------------------------------------------
        # MODEL-09 INPUT CONTRACT
        #
        # x:
        #   [batch, 91, 2560]
        #
        # Absolutely no 4-D tensor is allowed.
        # ------------------------------------------------------------------------------------------

        if x.dim() != 3:

            raise RuntimeError(
                "MODEL-09 expected input shape "
                "[batch, tokens, features]. "
                f"Received {tuple(x.shape)}"
            )

        if x.size(-1) != INPUT_DIM:

            raise RuntimeError(
                f"MODEL-09 expected input feature dimension "
                f"{INPUT_DIM}, received {x.size(-1)}"
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

        if T > TOKEN_LENGTH:

            raise RuntimeError(
                f"MODEL-09 received {T} tokens, "
                f"maximum is {TOKEN_LENGTH}"
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

        logits = self.classifier(
            pooled
        ).squeeze(
            -1
        )

        return (
            logits,
            attention
        )


# ==================================================================================================
# 16. LOAD MODEL CHECKPOINT
# ==================================================================================================

@st.cache_resource
def load_model():

    model = (
        BidirectionalAttentionTransformerEncoder()
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False
    )

    if isinstance(
        checkpoint,
        dict
    ) and "model_state_dict" in checkpoint:

        state_dict = (
            checkpoint[
                "model_state_dict"
            ]
        )

    else:

        state_dict = checkpoint

    # Remove DataParallel prefix if present.

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith(
            "module."
        ):

            key = key[
                len("module.") :
            ]

        cleaned_state_dict[
            key
        ] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True
    )

    model.eval()

    model.to(
        DEVICE
    )

    return model


# ==================================================================================================
# 17. LOAD ESM-2
# ==================================================================================================

@st.cache_resource
def load_esm():

    tokenizer = AutoTokenizer.from_pretrained(
        ESM_MODEL_NAME
    )

    esm_model = EsmModel.from_pretrained(
        ESM_MODEL_NAME
    )

    esm_model.eval()

    esm_model.to(
        DEVICE
    )

    return (
        tokenizer,
        esm_model
    )


# ==================================================================================================
# 18. SAFE PROTEIN NORMALIZATION
# ==================================================================================================

# Standard ESM-2 amino-acid alphabet.
#
# IMPORTANT:
# X is deliberately accepted.
#
# This prevents the previous:
#
# "Invalid amino-acid characters found: X"
#
# error.
#
# We do NOT silently replace X with another amino acid.

ALLOWED_AMINO_ACIDS = set(
    "ACDEFGHIKLMNPQRSTVWYX"
)


def normalize_protein_sequence(
    sequence
):

    if sequence is None:

        raise ValueError(
            "No protein sequence was supplied."
        )

    sequence = str(
        sequence
    )

    # Remove FASTA header lines.

    lines = (
        sequence
        .splitlines()
    )

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

    sequence = re.sub(
        r"\s+",
        "",
        sequence
    )

    sequence = sequence.upper()

    if len(sequence) == 0:

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
            ", ".join(invalid)
            +
            "\n\nAllowed amino acids:\n"
            +
            "ACDEFGHIKLMNPQRSTVWYX"
        )

    return sequence


# ==================================================================================================
# 19. ESM-2 RESIDUE EMBEDDING
# ==================================================================================================

@torch.inference_mode()
def get_residue_embeddings(
    sequence,
    tokenizer,
    esm_model
):

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

    outputs = esm_model(
        **encoded
    )

    hidden = (
        outputs.last_hidden_state
    )

    # ESM-2 output includes special tokens.
    #
    # For a single sequence:
    #
    # [1, sequence_length + special_tokens, 1280]
    #
    # Remove BOS and EOS where present.

    if hidden.dim() != 3:

        raise RuntimeError(
            "Unexpected ESM-2 hidden-state shape: "
            f"{tuple(hidden.shape)}"
        )

    residue_embeddings = (
        hidden[
            0,
            1:-1,
            :
        ]
    )

    # Expected:
    #
    # [residues, 1280]

    if residue_embeddings.dim() != 2:

        raise RuntimeError(
            "Residue embedding tensor must be 2-D. "
            f"Received {tuple(residue_embeddings.shape)}"
        )

    if residue_embeddings.size(
        -1
    ) != ESM2_DIMENSION:

        raise RuntimeError(
            "Unexpected ESM-2 dimension: "
            f"{residue_embeddings.size(-1)}"
        )

    return (
        residue_embeddings
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )


# ==================================================================================================
# 20. RESIDUE → 2560-D TOKENS
# ==================================================================================================

def residue_embeddings_to_tokens(
    residue_embeddings
):

    residue_embeddings = np.asarray(
        residue_embeddings,
        dtype=np.float32
    )

    if residue_embeddings.ndim != 2:

        raise ValueError(
            "Residue embeddings must have shape "
            "[residues, 1280]. "
            f"Received {residue_embeddings.shape}"
        )

    if residue_embeddings.shape[
        1
    ] != ESM2_DIMENSION:

        raise ValueError(
            f"Expected ESM-2 dimension "
            f"{ESM2_DIMENSION}; "
            f"received {residue_embeddings.shape[1]}"
        )

    residue_count = (
        residue_embeddings.shape[0]
    )

    # COMPLETE chunks only.
    #
    # No partial chunk is included.

    if residue_count < CHUNK_SIZE:

        raise ValueError(
            f"Protein contains only {residue_count} residues. "
            f"At least {CHUNK_SIZE} residues are required."
        )

    n_complete_tokens = (
        residue_count
        // CHUNK_STRIDE
    )

    n_complete_tokens = min(
        n_complete_tokens,
        TOKEN_LENGTH
    )

    tokens = []

    for token_index in range(
        n_complete_tokens
    ):

        start = (
            token_index
            *
            CHUNK_STRIDE
        )

        end = (
            start
            +
            CHUNK_SIZE
        )

        if end > residue_count:

            break

        chunk = (
            residue_embeddings[
                start:end,
                :
            ]
        )

        # ------------------------------------------------------------------------------------------
        # Current benchmark representation:
        #
        # mean pooling + max pooling
        #
        # 1280 + 1280 = 2560
        # ------------------------------------------------------------------------------------------

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
            "No complete 48-aa tokens could be constructed."
        )

    tokens = np.stack(
        tokens
    ).astype(
        np.float32
    )

    if tokens.shape[
        1
    ] != INPUT_DIM:

        raise RuntimeError(
            "Token feature dimension mismatch. "
            f"Expected {INPUT_DIM}; "
            f"received {tokens.shape[1]}"
        )

    return tokens


# ==================================================================================================
# 21. PAD / TRUNCATE TO 91 TOKENS
# ==================================================================================================

def pad_or_truncate_tokens(
    tokens
):

    tokens = np.asarray(
        tokens,
        dtype=np.float32
    )

    if tokens.ndim != 2:

        raise ValueError(
            "Token matrix must be 2-D "
            "[tokens, 2560]. "
            f"Received {tokens.shape}"
        )

    if tokens.shape[
        1
    ] != INPUT_DIM:

        raise ValueError(
            f"Expected {INPUT_DIM} token features; "
            f"received {tokens.shape[1]}"
        )

    n_tokens = (
        tokens.shape[0]
    )

    if n_tokens >= TOKEN_LENGTH:

        final_tokens = (
            tokens[
                :TOKEN_LENGTH,
                :
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
            :n_tokens,
            :
        ] = tokens

    return final_tokens


# ==================================================================================================
# 22. APPLY BENCHMARK TRAIN-ONLY STANDARDIZATION
# ==================================================================================================

def standardize_benchmark_tokens(
    tokens
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
            "Expected benchmark token matrix "
            f"({TOKEN_LENGTH}, {INPUT_DIM}); "
            f"received {tokens.shape}"
        )

    standardized = (
        tokens
        -
        TRAIN_MEAN
    ) / TRAIN_STD

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
            "Standardized MODEL-09 input contains "
            "NaN or infinite values."
        )

    return standardized


# ==================================================================================================
# 23. CRITICAL MODEL INPUT SHAPE FUNCTION
# ==================================================================================================

def prepare_model_input(
    standardized_tokens
):

    x = np.asarray(
        standardized_tokens,
        dtype=np.float32
    )

    # ----------------------------------------------------------------------------------------------
    # REQUIRED UNBATCHED REPRESENTATION:
    #
    # [91, 2560]
    # ----------------------------------------------------------------------------------------------

    if x.shape != (
        TOKEN_LENGTH,
        INPUT_DIM
    ):

        raise RuntimeError(
            "Expected unbatched MODEL-09 representation "
            f"({TOKEN_LENGTH}, {INPUT_DIM}); "
            f"received {x.shape}"
        )

    # ----------------------------------------------------------------------------------------------
    # ADD EXACTLY ONE BATCH DIMENSION:
    #
    # [91, 2560]
    #       ↓
    # [1, 91, 2560]
    #
    # This is the shape MultiheadAttention expects.
    # ----------------------------------------------------------------------------------------------

    x = np.expand_dims(
        x,
        axis=0
    )

    if x.shape != (
        1,
        TOKEN_LENGTH,
        INPUT_DIM
    ):

        raise RuntimeError(
            "MODEL-09 batch construction produced "
            f"incorrect shape: {x.shape}"
        )

    tensor = torch.from_numpy(
        x
    ).to(
        DEVICE
    )

    if tensor.dim() != 3:

        raise RuntimeError(
            "CRITICAL: MODEL-09 tensor must be 3-D. "
            f"Received {tensor.dim()}-D tensor."
        )

    return tensor


# ==================================================================================================
# 24. COMPLETE MODEL-09 PREDICTION
# ==================================================================================================

@torch.inference_mode()
def predict_model09(
    model,
    sequence,
    tokenizer,
    esm_model
):

    # ----------------------------------------------------------------------------------------------
    # STEP 1 — NORMALIZE / VALIDATE
    # ----------------------------------------------------------------------------------------------

    sequence = normalize_protein_sequence(
        sequence
    )

    # ----------------------------------------------------------------------------------------------
    # STEP 2 — ESM-2
    # ----------------------------------------------------------------------------------------------

    residue_embeddings = (
        get_residue_embeddings(
            sequence,
            tokenizer,
            esm_model
        )
    )

    # ----------------------------------------------------------------------------------------------
    # STEP 3 — RESIDUE → TOKENS
    # ----------------------------------------------------------------------------------------------

    raw_tokens = (
        residue_embeddings_to_tokens(
            residue_embeddings
        )
    )

    raw_token_count = (
        raw_tokens.shape[0]
    )

    # ----------------------------------------------------------------------------------------------
    # STEP 4 — 91 TOKENS
    # ----------------------------------------------------------------------------------------------

    tokens = (
        pad_or_truncate_tokens(
            raw_tokens
        )
    )

    # ----------------------------------------------------------------------------------------------
    # STEP 5 — TRAIN-ONLY STANDARDIZATION
    # ----------------------------------------------------------------------------------------------

    standardized_tokens = (
        standardize_benchmark_tokens(
            tokens
        )
    )

    # ----------------------------------------------------------------------------------------------
    # STEP 6 — EXACT MODEL INPUT
    #
    # [1, 91, 2560]
    # ----------------------------------------------------------------------------------------------

    model_input = (
        prepare_model_input(
            standardized_tokens
        )
    )

    # ----------------------------------------------------------------------------------------------
    # FINAL SHAPE ASSERTION
    # ----------------------------------------------------------------------------------------------

    assert model_input.dim() == 3

    assert model_input.shape == (
        1,
        TOKEN_LENGTH,
        INPUT_DIM
    )

    # ----------------------------------------------------------------------------------------------
    # STEP 7 — MODEL-09
    # ----------------------------------------------------------------------------------------------

    logits, attention = model(
        model_input,
        training_noise=False
    )

    if logits.dim() != 1:

        raise RuntimeError(
            "Unexpected MODEL-09 logits shape: "
            f"{tuple(logits.shape)}"
        )

    probability = (
        torch.sigmoid(
            logits
        )[0]
        .detach()
        .cpu()
        .item()
    )

    prediction = int(
        probability
        >=
        FROZEN_THRESHOLD
    )

    return {

        "sequence": sequence,

        "sequence_length":
            len(sequence),

        "raw_token_count":
            raw_token_count,

        "final_token_count":
            TOKEN_LENGTH,

        "probability":
            probability,

        "threshold":
            FROZEN_THRESHOLD,

        "prediction":
            prediction,

        "label":
            (
                "RECOMBINANT"
                if prediction == 1
                else
                "NON-RECOMBINANT"
            ),

        "attention":
            attention[
                0
            ]
            .detach()
            .cpu()
            .numpy()

    }


# ==================================================================================================
# 25. LOAD DEPLOYMENT MODELS
# ==================================================================================================

try:

    model = load_model()

except Exception as exc:

    st.error(
        "MODEL-09 could not be initialized from the checkpoint."
    )

    st.exception(exc)

    st.stop()


# ==================================================================================================
# 26. LOAD ESM-2
# ==================================================================================================

try:

    tokenizer, esm_model = load_esm()

except Exception as exc:

    st.error(
        "ESM-2 could not be initialized."
    )

    st.exception(exc)

    st.stop()


# ==================================================================================================
# 27. HEADER
# ==================================================================================================

st.title(
    "🧬 MODEL-09 HIV-1 Recombinant Classifier"
)

st.caption(
    "Current 9-model benchmark deployment"
)


# ==================================================================================================
# 28. DEPLOYMENT STATUS
# ==================================================================================================

with st.expander(
    "Deployment configuration",
    expanded=False
):

    st.write(
        f"**Repository root:** `{PROJECT_ROOT}`"
    )

    st.write(
        f"**Artifacts:** `{ARTIFACT_DIR}`"
    )

    st.write(
        f"**Device:** `{DEVICE}`"
    )

    st.write(
        f"**ESM-2:** `{ESM_MODEL_NAME}`"
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
        f"**Frozen threshold:** `{FROZEN_THRESHOLD:.10f}`"
    )

    st.write(
        f"**Checkpoint:** `{CHECKPOINT_PATH}`"
    )

    st.success(
        "✓ MODEL-09 deployment artifacts loaded"
    )


# ==================================================================================================
# 29. INPUT
# ==================================================================================================

st.subheader(
    "Enter HIV-1 protein sequence"
)

sequence_input = st.text_area(

    "Protein sequence",

    height=220,

    placeholder=(
        "Paste an amino-acid sequence here..."
    ),

    help=(
        "Standard amino acids plus X are accepted. "
        "FASTA headers are also accepted."
    )
)


# ==================================================================================================
# 30. EXAMPLE
# ==================================================================================================

with st.expander(
    "Example input"
):

    st.code(
        "MRVMGTQKNYSLLWRWGIMIFGILMACSANNLWVTVYYGVPVWKEAETTLFCASDAKAQDPEVHNVWATHACVPTDPSP"
    )


# ==================================================================================================
# 31. PREDICT BUTTON
# ==================================================================================================

predict_button = st.button(
    "🔬 Predict recombinant status",
    type="primary",
    use_container_width=True
)


# ==================================================================================================
# 32. PREDICTION
# ==================================================================================================

if predict_button:

    if not sequence_input.strip():

        st.warning(
            "Please enter a protein sequence."
        )

        st.stop()

    try:

        start_time = time.time()

        with st.spinner(
            "Running ESM-2 and MODEL-09..."
        ):

            result = predict_model09(

                model,

                sequence_input,

                tokenizer,

                esm_model

            )

        elapsed = (
            time.time()
            -
            start_time
        )

        st.success(
            "Prediction completed."
        )

        st.divider()

        # ------------------------------------------------------------------------------------------
        # RESULT
        # ------------------------------------------------------------------------------------------

        if result[
            "prediction"
        ] == 1:

            st.error(
                "## RECOMBINANT"
            )

        else:

            st.success(
                "## NON-RECOMBINANT"
            )

        # ------------------------------------------------------------------------------------------
        # PROBABILITY
        # ------------------------------------------------------------------------------------------

        st.metric(
            "Recombinant Probability",
            f"{result['probability']:.8f}"
        )

        st.metric(
            "Frozen Decision Threshold",
            f"{result['threshold']:.8f}"
        )

        # ------------------------------------------------------------------------------------------
        # INPUT / TOKEN INFORMATION
        # ------------------------------------------------------------------------------------------

        col1, col2, col3 = st.columns(
            3
        )

        with col1:

            st.metric(
                "Protein Length",
                f"{result['sequence_length']} aa"
            )

        with col2:

            st.metric(
                "Complete 48-aa Tokens",
                result[
                    "raw_token_count"
                ]
            )

        with col3:

            st.metric(
                "MODEL-09 Tokens",
                result[
                    "final_token_count"
                ]
            )

        st.caption(
            f"Inference time: {elapsed:.2f} seconds"
        )

        # ------------------------------------------------------------------------------------------
        # PROBABILITY BAR
        # ------------------------------------------------------------------------------------------

        st.progress(
            min(
                max(
                    result["probability"],
                    0.0
                ),
                1.0
            )
        )

        # ------------------------------------------------------------------------------------------
        # TECHNICAL DETAILS
        # ------------------------------------------------------------------------------------------

        with st.expander(
            "Technical prediction details"
        ):

            st.write(
                "### Current benchmark pipeline"
            )

            st.code(
                "\n".join(
                    [
                        "RAW PROTEIN SEQUENCE",
                        "↓",
                        "ESM-2 t33 650M",
                        "↓",
                        "1280-D residue embeddings",
                        "↓",
                        "complete non-overlapping 48-aa chunks",
                        "↓",
                        "mean + max pooling",
                        "↓",
                        "2560-D tokens",
                        "↓",
                        "pad / truncate to 91 tokens",
                        "↓",
                        "benchmark TRAIN-ONLY standardization",
                        "↓",
                        "MODEL-09",
                        "↓",
                        "sigmoid probability",
                        "↓",
                        f"frozen threshold = {FROZEN_THRESHOLD:.10f}",
                        "↓",
                        "classification"
                    ]
                ),
                language="text"
            )

            st.write(
                f"**Input tensor shape:** "
                f"`[1, {TOKEN_LENGTH}, {INPUT_DIM}]`"
            )

            st.write(
                "**Attention tensor:** "
                f"`[1, {TOKEN_LENGTH}, {MODEL_DIM}]`"
            )

            st.write(
                "**X accepted:** Yes"
            )

            st.write(
                "**Benchmark standardization:** "
                "TRAIN ONLY"
            )

        # ------------------------------------------------------------------------------------------
        # ATTENTION
        # ------------------------------------------------------------------------------------------

        attention = result[
            "attention"
        ]

        if (
            attention is not None
            and
            len(attention) == TOKEN_LENGTH
        ):

            st.subheader(
                "MODEL-09 attention over tokens"
            )

            st.bar_chart(
                {
                    "Attention":
                        attention
                }
            )

            st.caption(
                "Attention weights correspond to the 91-token "
                "MODEL-09 representation."
            )

    except ValueError as exc:

        st.error(
            "Invalid protein sequence."
        )

        st.code(
            str(exc),
            language="text"
        )

    except RuntimeError as exc:

        st.error(
            "MODEL-09 prediction failed."
        )

        st.code(
            str(exc),
            language="text"
        )

    except Exception as exc:

        st.error(
            "Prediction failed."
        )

        st.exception(exc)


# ==================================================================================================
# 33. FOOTER
# ==================================================================================================

st.divider()

st.caption(
    "MODEL-09 — Current 9-model benchmark | "
    "ESM-2 650M → 48-aa complete chunks → "
    "mean+max 2560-D tokens → 91 tokens → "
    "train-only standardization → attention transformer"
)
