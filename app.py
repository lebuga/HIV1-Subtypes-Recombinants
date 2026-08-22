# ================================================================================================
# MODEL-09 HIV-1 RECOMBINANT CLASSIFIER
# CURRENT 9-MODEL BENCHMARK DEPLOYMENT
#
# AUTHORITATIVE DEPLOYMENT CONTRACT
#
# INPUT:
#     Raw HIV-1 protein amino-acid sequence
#
# PIPELINE:
#     raw protein sequence
#          ↓
#     ESM-2 residue embeddings
#          ↓
#     complete non-overlapping 48-aa chunks
#          ↓
#     mean + max pooling
#          ↓
#     2560-D token representation
#          ↓
#     pad/truncate to 91 tokens
#          ↓
#     TRAIN-ONLY standardization
#          ↓
#     MODEL-09
#          ↓
#     sigmoid probability
#          ↓
#     frozen validation threshold
#          ↓
#     recombinant / non-recombinant
#
# CURRENT BENCHMARK:
#     ESM dimension       = 1280
#     Chunk size          = 48
#     Chunk stride        = 48
#     Token dimension     = 2560
#     Token length       = 91
#     Model dimension    = 96
#     Attention heads    = 4
#
# REQUIRED ARTIFACTS:
#     MODEL-09_Bidirectional_Attention_Transformer_Encoder.pt
#     MODEL-09_BENCHMARK_TRAIN_MEAN.npy
#     MODEL-09_BENCHMARK_TRAIN_STD.npy
#     MODEL-09_BENCHMARK_FROZEN_THRESHOLD.txt
#
# IMPORTANT:
#     Artifact files are searched recursively.
#
# CRITICAL FIX:
#     Training mean/std are normalized to [1, 2560], NOT [1, 1, 2560].
#
#     Correct:
#         tokens       = [91, 2560]
#         train_mean   = [1, 2560]
#         train_std    = [1, 2560]
#         standardized = [91, 2560]
#         model_input  = [1, 91, 2560]
#
#     Incorrect:
#         tokens       = [91, 2560]
#         train_mean   = [1, 1, 2560]
#         standardized = [1, 91, 2560]
#
# ================================================================================================


# ================================================================================================
# 1. IMPORTS
# ================================================================================================

from pathlib import Path
import re
import traceback

import numpy as np
import streamlit as st

import torch
import torch.nn as nn


# ================================================================================================
# 2. STREAMLIT PAGE
# ================================================================================================

st.set_page_config(
    page_title="MODEL-09 HIV-1 Recombinant Classifier",
    page_icon="🧬",
    layout="centered"
)


# ================================================================================================
# 3. CONSTANTS
# ================================================================================================

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

CHECKPOINT_NAME = (
    "MODEL-09_Bidirectional_Attention_Transformer_Encoder.pt"
)

MEAN_NAME = (
    "MODEL-09_BENCHMARK_TRAIN_MEAN.npy"
)

STD_NAME = (
    "MODEL-09_BENCHMARK_TRAIN_STD.npy"
)

THRESHOLD_NAME = (
    "MODEL-09_BENCHMARK_FROZEN_THRESHOLD.txt"
)

ALLOWED_AMINO_ACIDS = set(
    "ACDEFGHIKLMNPQRSTVWY"
)


# ================================================================================================
# 4. REPOSITORY ROOT
# ================================================================================================

def locate_repository_root():

    candidates = []

    try:
        candidates.append(
            Path.cwd()
        )
    except Exception:
        pass

    try:
        candidates.append(
            Path(__file__).resolve().parent
        )
    except Exception:
        pass

    candidates.append(
        Path(
            "/mount/src/hiv1-subtypes-recombinants"
        )
    )

    seen = set()

    for candidate in candidates:

        try:
            candidate = candidate.resolve()
        except Exception:
            continue

        if candidate in seen:
            continue

        seen.add(candidate)

        if (
            candidate.exists()
            and (
                (candidate / "app.py").exists()
                or (candidate / ".git").exists()
                or (candidate / "requirements.txt").exists()
            )
        ):

            return candidate

    return Path.cwd().resolve()


PROJECT_ROOT = locate_repository_root()


# ================================================================================================
# 5. ARTIFACT SEARCH
# ================================================================================================

@st.cache_resource
def locate_artifacts():

    required_names = {
        "checkpoint": CHECKPOINT_NAME,
        "mean": MEAN_NAME,
        "std": STD_NAME,
        "threshold": THRESHOLD_NAME
    }

    found = {}

    search_roots = [
        PROJECT_ROOT,
        PROJECT_ROOT / "artifacts",
        Path.cwd(),
        Path("/mount/src/hiv1-subtypes-recombinants")
    ]

    unique_roots = []

    for root in search_roots:

        try:
            root = root.resolve()
        except Exception:
            continue

        if root not in unique_roots:
            unique_roots.append(root)

    # ------------------------------------------------------------------
    # Exact locations first
    # ------------------------------------------------------------------

    for root in unique_roots:

        if not root.exists():
            continue

        for key, filename in required_names.items():

            candidate = root / filename

            if candidate.is_file():
                found[key] = candidate

            candidate = (
                root
                / "artifacts"
                / filename
            )

            if candidate.is_file():
                found[key] = candidate

    # ------------------------------------------------------------------
    # Recursive search
    # ------------------------------------------------------------------

    for root in unique_roots:

        if not root.exists():
            continue

        for key, filename in required_names.items():

            if key in found:
                continue

            try:

                matches = list(
                    root.rglob(filename)
                )

                if matches:

                    matches.sort(
                        key=lambda p: (
                            len(p.parts),
                            str(p)
                        )
                    )

                    found[key] = matches[0]

            except Exception:
                pass

    return found


ARTIFACTS = locate_artifacts()


CHECKPOINT_PATH = ARTIFACTS.get(
    "checkpoint"
)

TRAIN_MEAN_PATH = ARTIFACTS.get(
    "mean"
)

TRAIN_STD_PATH = ARTIFACTS.get(
    "std"
)

THRESHOLD_PATH = ARTIFACTS.get(
    "threshold"
)


# ================================================================================================
# 6. ARTIFACT STATUS
# ================================================================================================

def artifact_status():

    required = {
        "MODEL-09 checkpoint":
            CHECKPOINT_PATH,

        "Training mean":
            TRAIN_MEAN_PATH,

        "Training std":
            TRAIN_STD_PATH,

        "Frozen threshold":
            THRESHOLD_PATH
    }

    missing = []

    for label, path in required.items():

        if path is None:
            missing.append(label)

    return required, missing


REQUIRED_ARTIFACTS, MISSING_ARTIFACTS = (
    artifact_status()
)


# ================================================================================================
# 7. MODEL ARCHITECTURE
# ================================================================================================

class LocalAttentionBlock(nn.Module):

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

    def forward(self, x):

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
                self.norm2(x)
            )
        )

        return x


class GlobalAttentionBlock(
    LocalAttentionBlock
):

    pass


class AttentionPooling(nn.Module):

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

    def forward(self, x):

        scores = self.score(
            x
        ).squeeze(-1)

        weights = torch.softmax(
            scores,
            dim=1
        )

        pooled = torch.sum(
            x
            * weights.unsqueeze(-1),
            dim=1
        )

        return (
            pooled,
            weights
        )


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

        # ------------------------------------------------------------
        # Normalize input dimensionality
        #
        # Accepted:
        #
        # [91, 2560]
        # [batch, 91, 2560]
        #
        # Accidental singleton 4-D dimensions are also handled.
        # ------------------------------------------------------------

        if x.ndim == 2:

            x = x.unsqueeze(0)

        elif x.ndim == 3:

            pass

        elif x.ndim == 4:

            if x.shape[1] == 1:

                x = x.squeeze(1)

            elif x.shape[0] == 1:

                x = x.squeeze(0)

            else:

                raise ValueError(
                    "MODEL-09 received unsupported 4-D "
                    f"tensor shape: {tuple(x.shape)}"
                )

        else:

            raise ValueError(
                "MODEL-09 input must be 2-D or 3-D. "
                f"Received shape: {tuple(x.shape)}"
            )

        if x.ndim != 3:

            raise ValueError(
                "MODEL-09 input could not be normalized "
                "to [batch, tokens, features]. "
                f"Current shape: {tuple(x.shape)}"
            )

        # ------------------------------------------------------------
        # Feature dimension
        # ------------------------------------------------------------

        if x.shape[-1] != INPUT_DIM:

            raise ValueError(
                "MODEL-09 feature dimension mismatch. "
                f"Expected {INPUT_DIM}, "
                f"received {x.shape[-1]}."
            )

        # ------------------------------------------------------------
        # Token length protection
        # ------------------------------------------------------------

        if x.shape[1] > TOKEN_LENGTH:

            x = x[
                :,
                :TOKEN_LENGTH,
                :
            ]

        elif x.shape[1] < TOKEN_LENGTH:

            padding = torch.zeros(
                x.shape[0],
                TOKEN_LENGTH - x.shape[1],
                INPUT_DIM,
                dtype=x.dtype,
                device=x.device
            )

            x = torch.cat(
                [
                    x,
                    padding
                ],
                dim=1
            )

        # ------------------------------------------------------------
        # Input projection
        # ------------------------------------------------------------

        x = self.input_projection(
            x
        )

        # ------------------------------------------------------------
        # Representation noise
        # ------------------------------------------------------------

        if (
            self.training
            and training_noise
            and REPRESENTATION_NOISE > 0
        ):

            x = (
                x
                +
                torch.randn_like(x)
                * REPRESENTATION_NOISE
            )

        # ------------------------------------------------------------
        # Positional embedding
        # ------------------------------------------------------------

        T = x.size(1)

        x = (
            x
            +
            self.position_embedding[
                :,
                :T,
                :
            ]
        )

        # ------------------------------------------------------------
        # Attention blocks
        # ------------------------------------------------------------

        x = self.local_attention(
            x
        )

        x = self.global_attention(
            x
        )

        # ------------------------------------------------------------
        # Attention pooling
        # ------------------------------------------------------------

        pooled, attention = (
            self.pool(x)
        )

        # ------------------------------------------------------------
        # Classification
        # ------------------------------------------------------------

        logits = self.classifier(
            pooled
        ).squeeze(-1)

        return (
            logits,
            attention
        )


# ================================================================================================
# 8. LOAD ESM-2
# ================================================================================================

@st.cache_resource
def load_esm_model():

    try:

        from transformers import (
            AutoTokenizer,
            AutoModel
        )

    except ImportError as exc:

        raise RuntimeError(
            "transformers is not installed. "
            "Add transformers to requirements.txt."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        ESM_MODEL_NAME
    )

    esm_model = AutoModel.from_pretrained(
        ESM_MODEL_NAME
    )

    esm_model.eval()

    for parameter in esm_model.parameters():

        parameter.requires_grad = False

    return (
        tokenizer,
        esm_model
    )


# ================================================================================================
# 9. LOAD MODEL-09 ARTIFACTS
# ================================================================================================

@st.cache_resource
def load_model09():

    if CHECKPOINT_PATH is None:

        raise FileNotFoundError(
            "MODEL-09 checkpoint was not found."
        )

    if TRAIN_MEAN_PATH is None:

        raise FileNotFoundError(
            "Training mean artifact was not found."
        )

    if TRAIN_STD_PATH is None:

        raise FileNotFoundError(
            "Training standard deviation artifact was not found."
        )

    if THRESHOLD_PATH is None:

        raise FileNotFoundError(
            "Frozen threshold artifact was not found."
        )

    # ------------------------------------------------------------
    # CHECKPOINT
    # ------------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False
    )

    if (
        isinstance(checkpoint, dict)
        and
        "model_state_dict" in checkpoint
    ):

        state_dict = (
            checkpoint[
                "model_state_dict"
            ]
        )

    else:

        state_dict = checkpoint

    model = (
        BidirectionalAttentionTransformerEncoder()
    )

    # ------------------------------------------------------------
    # STRICT CHECKPOINT VERIFICATION
    # ------------------------------------------------------------

    try:

        model.load_state_dict(
            state_dict,
            strict=True
        )

    except RuntimeError as exc:

        raise RuntimeError(
            "MODEL-09 checkpoint architecture does not "
            "match the deployment architecture.\n\n"
            f"Checkpoint: {CHECKPOINT_PATH}\n\n"
            f"Original error:\n{exc}"
        ) from exc

    model.eval()

    # ------------------------------------------------------------
    # TRAINING MEAN
    # ------------------------------------------------------------

    train_mean = np.load(
        TRAIN_MEAN_PATH
    ).astype(
        np.float32
    )

    # ------------------------------------------------------------
    # TRAINING STD
    # ------------------------------------------------------------

    train_std = np.load(
        TRAIN_STD_PATH
    ).astype(
        np.float32
    )

    # ------------------------------------------------------------
    # CRITICAL SHAPE FIX
    #
    # Final required shape:
    #
    #     [1, 2560]
    #
    # This guarantees:
    #
    #     [91,2560] - [1,2560]
    #
    # remains:
    #
    #     [91,2560]
    #
    # ------------------------------------------------------------

    if train_mean.size != INPUT_DIM:

        raise ValueError(
            "Training mean contains the wrong number "
            f"of values. Expected {INPUT_DIM}, "
            f"received {train_mean.size}. "
            f"Original shape: {train_mean.shape}"
        )

    if train_std.size != INPUT_DIM:

        raise ValueError(
            "Training std contains the wrong number "
            f"of values. Expected {INPUT_DIM}, "
            f"received {train_std.size}. "
            f"Original shape: {train_std.shape}"
        )

    train_mean = train_mean.reshape(
        1,
        INPUT_DIM
    )

    train_std = train_std.reshape(
        1,
        INPUT_DIM
    )

    # ------------------------------------------------------------
    # VALIDATE TRAINING STATISTICS
    # ------------------------------------------------------------

    if train_mean.shape != (
        1,
        INPUT_DIM
    ):

        raise RuntimeError(
            "Internal training mean shape error: "
            f"{train_mean.shape}"
        )

    if train_std.shape != (
        1,
        INPUT_DIM
    ):

        raise RuntimeError(
            "Internal training std shape error: "
            f"{train_std.shape}"
        )

    if not np.all(
        np.isfinite(
            train_mean
        )
    ):

        raise ValueError(
            "Training mean contains non-finite values."
        )

    if not np.all(
        np.isfinite(
            train_std
        )
    ):

        raise ValueError(
            "Training std contains non-finite values."
        )

    if np.any(
        train_std <= 0
    ):

        raise ValueError(
            "Training std contains zero or negative values."
        )

    # ------------------------------------------------------------
    # FROZEN THRESHOLD
    # ------------------------------------------------------------

    threshold_text = (
        THRESHOLD_PATH
        .read_text(
            encoding="utf-8"
        )
        .strip()
    )

    threshold_match = re.search(
        r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?",
        threshold_text
    )

    if threshold_match is None:

        raise ValueError(
            "Could not parse frozen threshold from:\n"
            f"{THRESHOLD_PATH}\n"
            f"Contents: {threshold_text!r}"
        )

    threshold = float(
        threshold_match.group(0)
    )

    if not (
        0.0
        <= threshold
        <= 1.0
    ):

        raise ValueError(
            "Frozen threshold must be between 0 and 1. "
            f"Received {threshold}."
        )

    return (
        model,
        train_mean,
        train_std,
        threshold
    )


# ================================================================================================
# 10. PROTEIN SEQUENCE CLEANING / VALIDATION
# ================================================================================================

def normalize_sequence(
    raw_sequence
):

    if raw_sequence is None:

        raise ValueError(
            "No protein sequence was supplied."
        )

    sequence = str(
        raw_sequence
    )

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

    sequence = re.sub(
        r"\s+",
        "",
        sequence
    )

    sequence = sequence.upper()

    if len(sequence) == 0:

        raise ValueError(
            "The protein sequence is empty."
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
            "".join(
                sorted(
                    ALLOWED_AMINO_ACIDS
                )
            )
        )

    return sequence


# ================================================================================================
# 11. ESM-2 RESIDUE EMBEDDINGS
# ================================================================================================

def get_residue_embeddings(
    sequence,
    tokenizer,
    esm_model
):

    # ------------------------------------------------------------
    # ESM-2 WINDOWING
    #
    # Window = 1024 aa
    # Overlap = 128 aa
    # Stride = 896 aa
    # ------------------------------------------------------------

    WINDOW_SIZE = 1024

    OVERLAP = 128

    STRIDE = (
        WINDOW_SIZE
        -
        OVERLAP
    )

    sequence_length = len(
        sequence
    )

    accumulator = np.zeros(
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

    esm_model.eval()

    with torch.no_grad():

        start = 0

        while start < sequence_length:

            end = min(
                start + WINDOW_SIZE,
                sequence_length
            )

            fragment = (
                sequence[
                    start:end
                ]
            )

            inputs = tokenizer(
                fragment,
                return_tensors="pt",
                add_special_tokens=True
            )

            outputs = esm_model(
                **inputs
            )

            hidden = (
                outputs
                .last_hidden_state
            )

            # Remove BOS and EOS
            residue_hidden = (
                hidden[
                    0,
                    1:-1,
                    :
                ]
            )

            residue_hidden = (
                residue_hidden
                .cpu()
                .float()
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
                    "ESM-2 residue length mismatch. "
                    f"Expected {expected_length}, "
                    f"received {residue_hidden.shape[0]}."
                )

            accumulator[
                start:end
            ] += residue_hidden

            counts[
                start:end
            ] += 1.0

            if end >= sequence_length:
                break

            start += STRIDE

    counts[
        counts == 0
    ] = 1.0

    residue_embeddings = (
        accumulator
        /
        counts[:, None]
    )

    expected_residue_shape = (
        sequence_length,
        ESM2_DIMENSION
    )

    if (
        residue_embeddings.shape
        !=
        expected_residue_shape
    ):

        raise RuntimeError(
            "Unexpected residue embedding shape: "
            f"{residue_embeddings.shape}. "
            f"Expected {expected_residue_shape}."
        )

    return residue_embeddings


# ================================================================================================
# 12. RESIDUE → 2560-D TOKENS
# ================================================================================================

def residue_to_tokens(
    residue_embeddings
):

    if residue_embeddings.ndim != 2:

        raise ValueError(
            "Residue embeddings must be 2-D. "
            f"Received {residue_embeddings.shape}"
        )

    if (
        residue_embeddings.shape[1]
        !=
        ESM2_DIMENSION
    ):

        raise ValueError(
            "Residue embedding dimension mismatch. "
            f"Expected {ESM2_DIMENSION}, "
            f"received {residue_embeddings.shape[1]}."
        )

    sequence_length = (
        residue_embeddings.shape[0]
    )

    # ------------------------------------------------------------
    # COMPLETE 48-AA CHUNKS ONLY
    # ------------------------------------------------------------

    n_complete = (
        sequence_length
        //
        CHUNK_SIZE
    )

    if n_complete < 1:

        raise ValueError(
            f"Protein is too short. "
            f"At least {CHUNK_SIZE} residues are required."
        )

    usable_length = (
        n_complete
        *
        CHUNK_SIZE
    )

    residues = (
        residue_embeddings[
            :usable_length
        ]
    )

    chunks = residues.reshape(
        n_complete,
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

    expected_shape = (
        n_complete,
        INPUT_DIM
    )

    if tokens.shape != expected_shape:

        raise RuntimeError(
            "Token representation shape mismatch. "
            f"Expected {expected_shape}, "
            f"received {tokens.shape}."
        )

    return tokens.astype(
        np.float32
    )


# ================================================================================================
# 13. PAD / TRUNCATE TO EXACTLY 91 TOKENS
# ================================================================================================

def fix_token_length(
    tokens
):

    if tokens.ndim != 2:

        raise ValueError(
            "Tokens must be 2-D. "
            f"Received {tokens.shape}"
        )

    if (
        tokens.shape[1]
        !=
        INPUT_DIM
    ):

        raise ValueError(
            "Token feature dimension mismatch. "
            f"Expected {INPUT_DIM}, "
            f"received {tokens.shape[1]}."
        )

    raw_tokens = (
        tokens.shape[0]
    )

    # ------------------------------------------------------------
    # Truncate
    # ------------------------------------------------------------

    if raw_tokens >= TOKEN_LENGTH:

        final_tokens = (
            tokens[
                :TOKEN_LENGTH
            ]
        )

    # ------------------------------------------------------------
    # Pad
    # ------------------------------------------------------------

    else:

        padding = np.zeros(
            (
                TOKEN_LENGTH
                -
                raw_tokens,
                INPUT_DIM
            ),
            dtype=np.float32
        )

        final_tokens = np.concatenate(
            [
                tokens,
                padding
            ],
            axis=0
        )

    expected_shape = (
        TOKEN_LENGTH,
        INPUT_DIM
    )

    if final_tokens.shape != expected_shape:

        raise RuntimeError(
            "Final token representation has "
            f"unexpected shape: {final_tokens.shape}. "
            f"Expected {expected_shape}."
        )

    return final_tokens.astype(
        np.float32
    )


# ================================================================================================
# 14. TRAIN-ONLY STANDARDIZATION
# ================================================================================================

def standardize_tokens(
    tokens,
    train_mean,
    train_std
):

    tokens = np.asarray(
        tokens,
        dtype=np.float32
    )

    # ------------------------------------------------------------
    # Input must be:
    #
    # [91, 2560]
    # ------------------------------------------------------------

    expected_token_shape = (
        TOKEN_LENGTH,
        INPUT_DIM
    )

    if tokens.shape != expected_token_shape:

        raise ValueError(
            "Unexpected token matrix shape before "
            f"standardization: {tokens.shape}. "
            f"Expected {expected_token_shape}."
        )

    train_mean = np.asarray(
        train_mean,
        dtype=np.float32
    )

    train_std = np.asarray(
        train_std,
        dtype=np.float32
    )

    # ------------------------------------------------------------
    # Statistics must contain exactly 2560 values.
    # ------------------------------------------------------------

    if train_mean.size != INPUT_DIM:

        raise ValueError(
            "Training mean does not contain exactly "
            f"{INPUT_DIM} values. "
            f"Received shape: {train_mean.shape}."
        )

    if train_std.size != INPUT_DIM:

        raise ValueError(
            "Training std does not contain exactly "
            f"{INPUT_DIM} values. "
            f"Received shape: {train_std.shape}."
        )

    # ------------------------------------------------------------
    # CRITICAL:
    #
    # Always force:
    #
    #     [2560]
    #
    # to:
    #
    #     [1,2560]
    #
    # NOT:
    #
    #     [1,1,2560]
    #
    # ------------------------------------------------------------

    train_mean = train_mean.reshape(
        1,
        INPUT_DIM
    )

    train_std = train_std.reshape(
        1,
        INPUT_DIM
    )

    if np.any(
        train_std <= 0
    ):

        raise ValueError(
            "Training std contains zero or negative values."
        )

    # ------------------------------------------------------------
    # Standardization
    #
    # [91,2560]
    #      -
    # [1,2560]
    #      =
    # [91,2560]
    # ------------------------------------------------------------

    standardized = (
        tokens
        -
        train_mean
    ) / train_std

    if standardized.shape != expected_token_shape:

        raise RuntimeError(
            "Standardization unexpectedly changed the "
            f"MODEL-09 shape. "
            f"Received {standardized.shape}; "
            f"expected {expected_token_shape}."
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

    return standardized.astype(
        np.float32
    )


# ================================================================================================
# 15. MODEL-09 PREDICTION
# ================================================================================================

def predict_model09(
    model,
    sequence,
    tokenizer,
    esm_model,
    train_mean,
    train_std,
    frozen_threshold
):

    # ------------------------------------------------------------
    # 1. NORMALIZE SEQUENCE
    # ------------------------------------------------------------

    sequence = normalize_sequence(
        sequence
    )

    # ------------------------------------------------------------
    # 2. ESM-2 RESIDUE EMBEDDINGS
    # ------------------------------------------------------------

    residue_embeddings = (
        get_residue_embeddings(
            sequence,
            tokenizer,
            esm_model
        )
    )

    # ------------------------------------------------------------
    # 3. RESIDUE → 2560-D TOKENS
    # ------------------------------------------------------------

    raw_tokens = (
        residue_to_tokens(
            residue_embeddings
        )
    )

    raw_token_count = (
        raw_tokens.shape[0]
    )

    # ------------------------------------------------------------
    # 4. PAD / TRUNCATE → 91 TOKENS
    # ------------------------------------------------------------

    final_tokens = (
        fix_token_length(
            raw_tokens
        )
    )

    # ------------------------------------------------------------
    # 5. TRAIN-ONLY STANDARDIZATION
    #
    # Must remain [91,2560]
    # ------------------------------------------------------------

    standardized = (
        standardize_tokens(
            final_tokens,
            train_mean,
            train_std
        )
    )

    if standardized.shape != (
        TOKEN_LENGTH,
        INPUT_DIM
    ):

        raise RuntimeError(
            "Unexpected standardized MODEL-09 shape: "
            f"{standardized.shape}. "
            f"Expected "
            f"{(TOKEN_LENGTH, INPUT_DIM)}."
        )

    # ------------------------------------------------------------
    # 6. CONVERT TO TORCH
    #
    # Before batch:
    #
    #     [91,2560]
    #
    # ------------------------------------------------------------

    model_input = torch.from_numpy(
        standardized
    ).float()

    if model_input.ndim != 2:

        raise RuntimeError(
            "Unexpected pre-batch MODEL-09 shape: "
            f"{tuple(model_input.shape)}. "
            "Expected (91,2560)."
        )

    if tuple(
        model_input.shape
    ) != (
        TOKEN_LENGTH,
        INPUT_DIM
    ):

        raise RuntimeError(
            "Unexpected pre-batch MODEL-09 shape: "
            f"{tuple(model_input.shape)}. "
            f"Expected "
            f"{(TOKEN_LENGTH, INPUT_DIM)}."
        )

    # ------------------------------------------------------------
    # 7. ADD EXACTLY ONE BATCH DIMENSION
    #
    #     [91,2560]
    #          ↓
    #     [1,91,2560]
    # ------------------------------------------------------------

    model_input = (
        model_input
        .unsqueeze(0)
    )

    expected_model_shape = (
        1,
        TOKEN_LENGTH,
        INPUT_DIM
    )

    if tuple(
        model_input.shape
    ) != expected_model_shape:

        raise RuntimeError(
            "MODEL-09 input shape mismatch. "
            f"Expected {expected_model_shape}, "
            f"received {tuple(model_input.shape)}."
        )

    # ------------------------------------------------------------
    # 8. MODEL INFERENCE
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # 9. FROZEN THRESHOLD
    # ------------------------------------------------------------

    prediction = int(
        probability
        >=
        frozen_threshold
    )

    if prediction == 1:

        label = (
            "RECOMBINANT"
        )

    else:

        label = (
            "NON-RECOMBINANT"
        )

    # ------------------------------------------------------------
    # 10. RETURN
    # ------------------------------------------------------------

    return {
        "sequence": sequence,

        "length": len(
            sequence
        ),

        "raw_token_count": int(
            raw_token_count
        ),

        "final_token_count": int(
            TOKEN_LENGTH
        ),

        "probability": probability,

        "threshold": float(
            frozen_threshold
        ),

        "prediction": prediction,

        "label": label,

        "attention": (
            attention
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )
    }


# ================================================================================================
# 16. HEADER
# ================================================================================================

st.title(
    "🧬 MODEL-09 HIV-1 Recombinant Classifier"
)

st.caption(
    "Current 9-model benchmark deployment"
)


# ================================================================================================
# 17. ARTIFACT DIAGNOSTICS
# ================================================================================================

if MISSING_ARTIFACTS:

    st.error(
        "MODEL-09 could not be initialized."
    )

    st.code(
        "\n".join(
            [
                "Required MODEL-09 deployment artifacts "
                "were not found.",
                "",
                "Repository root detected:",
                str(PROJECT_ROOT),
                "",
                "Artifact search status:",
                "",
                "MODEL-09 checkpoint: "
                +
                (
                    str(CHECKPOINT_PATH)
                    if CHECKPOINT_PATH
                    else
                    "MISSING"
                ),
                "",
                "Training mean: "
                +
                (
                    str(TRAIN_MEAN_PATH)
                    if TRAIN_MEAN_PATH
                    else
                    "MISSING"
                ),
                "",
                "Training std: "
                +
                (
                    str(TRAIN_STD_PATH)
                    if TRAIN_STD_PATH
                    else
                    "MISSING"
                ),
                "",
                "Frozen threshold: "
                +
                (
                    str(THRESHOLD_PATH)
                    if THRESHOLD_PATH
                    else
                    "MISSING"
                ),
                "",
                "Expected filenames:",
                CHECKPOINT_NAME,
                MEAN_NAME,
                STD_NAME,
                THRESHOLD_NAME
            ]
        ),
        language="text"
    )

    st.warning(
        "Commit all four MODEL-09 artifacts to the GitHub "
        "repository used by Streamlit Cloud."
    )

    st.stop()


# ================================================================================================
# 18. LOAD MODEL-09
# ================================================================================================

try:

    (
        MODEL09,
        TRAIN_MEAN,
        TRAIN_STD,
        FROZEN_THRESHOLD
    ) = load_model09()

except Exception:

    st.error(
        "MODEL-09 could not be initialized."
    )

    st.code(
        traceback.format_exc(),
        language="text"
    )

    st.stop()


# ================================================================================================
# 19. LOAD ESM-2
# ================================================================================================

try:

    with st.spinner(
        "Loading ESM-2 model..."
    ):

        (
            TOKENIZER,
            ESM_MODEL
        ) = load_esm_model()

except Exception:

    st.error(
        "ESM-2 could not be loaded."
    )

    st.code(
        traceback.format_exc(),
        language="text"
    )

    st.stop()


# ================================================================================================
# 20. DEPLOYMENT STATUS
# ================================================================================================

st.success(
    "MODEL-09 initialized successfully."
)

with st.expander(
    "Deployment diagnostics"
):

    st.write(
        "Repository root:",
        str(PROJECT_ROOT)
    )

    st.write(
        "Checkpoint:",
        str(CHECKPOINT_PATH)
    )

    st.write(
        "Training mean:",
        str(TRAIN_MEAN_PATH)
    )

    st.write(
        "Training std:",
        str(TRAIN_STD_PATH)
    )

    st.write(
        "Frozen threshold:",
        f"{FROZEN_THRESHOLD:.10f}"
    )

    st.write(
        "Training mean runtime shape:",
        str(TRAIN_MEAN.shape)
    )

    st.write(
        "Training std runtime shape:",
        str(TRAIN_STD.shape)
    )

    st.write(
        "ESM-2 model:",
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
        "Token length:",
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


# ================================================================================================
# 21. INPUT
# ================================================================================================

st.subheader(
    "Enter HIV-1 protein sequence"
)

st.write(
    "Paste an amino-acid protein sequence below. "
    "FASTA headers beginning with '>' are automatically ignored."
)

sequence_input = st.text_area(
    "Protein sequence",
    height=250,
    placeholder=(
        "Paste HIV-1 protein sequence here..."
    )
)


# ================================================================================================
# 22. EXAMPLE
# ================================================================================================

with st.expander(
    "Example input"
):

    st.code(
        "MRVMGTQKNYSLLWRWGIMIFGILMACSANNLWVTVYYGVPVW",
        language="text"
    )

    st.caption(
        "Use a complete protein sequence for meaningful prediction."
    )


# ================================================================================================
# 23. PREDICTION
# ================================================================================================

if st.button(
    "🔬 Predict recombinant status",
    type="primary",
    use_container_width=True
):

    if not sequence_input.strip():

        st.warning(
            "Please enter a protein sequence."
        )

        st.stop()

    try:

        with st.spinner(
            "Running ESM-2 → tokenization → "
            "standardization → MODEL-09..."
        ):

            result = predict_model09(
                MODEL09,
                sequence_input,
                TOKENIZER,
                ESM_MODEL,
                TRAIN_MEAN,
                TRAIN_STD,
                FROZEN_THRESHOLD
            )

        # --------------------------------------------------------
        # RESULT
        # --------------------------------------------------------

        st.subheader(
            "MODEL-09 Prediction"
        )

        if result["prediction"] == 1:

            st.error(
                "## RECOMBINANT"
            )

        else:

            st.success(
                "## NON-RECOMBINANT"
            )

        # --------------------------------------------------------
        # PROBABILITY
        # --------------------------------------------------------

        st.metric(
            "Recombinant Probability",
            f"{result['probability']:.8f}"
        )

        # --------------------------------------------------------
        # THRESHOLD
        # --------------------------------------------------------

        st.metric(
            "Frozen Validation Threshold",
            f"{result['threshold']:.8f}"
        )

        # --------------------------------------------------------
        # INPUT INFORMATION
        # --------------------------------------------------------

        st.write(
            f"**Protein length:** "
            f"{result['length']} amino acids"
        )

        st.write(
            f"**Raw complete 48-aa tokens:** "
            f"{result['raw_token_count']}"
        )

        st.write(
            f"**Final MODEL-09 tokens:** "
            f"{result['final_token_count']}"
        )

        st.write(
            "**Final model input shape:** "
            "`(1, 91, 2560)`"
        )

        st.write(
            "**Representation:** "
            "ESM-2 residue embeddings → complete "
            "48-aa chunks → mean + max → 2560-D "
            "tokens → 91-token representation → "
            "train-only standardization"
        )

        # --------------------------------------------------------
        # ATTENTION
        # --------------------------------------------------------

        attention = result[
            "attention"
        ]

        if attention.size > 0:

            st.subheader(
                "MODEL-09 Attention Summary"
            )

            valid_tokens = min(
                result["raw_token_count"],
                TOKEN_LENGTH
            )

            valid_attention = (
                attention[
                    :valid_tokens
                ]
            )

            if valid_attention.size > 0:

                st.bar_chart(
                    valid_attention
                )

                st.caption(
                    "Relative MODEL-09 attention weight "
                    "across complete 48-aa protein tokens."
                )

    except ValueError as exc:

        st.error(
            "Invalid protein sequence."
        )

        st.code(
            str(exc),
            language="text"
        )

    except Exception:

        st.error(
            "Prediction failed."
        )

        st.code(
            traceback.format_exc(),
            language="text"
        )


# ================================================================================================
# 24. FOOTER
# ================================================================================================

st.divider()

st.caption(
    "MODEL-09 — Current 9-model benchmark deployment"
)

st.caption(
    "ESM-2 t33 650M • 1280-D residue embeddings • "
    "48-aa complete chunks • 2560-D mean+max tokens • "
    "91-token representation • train-only standardization"
)
