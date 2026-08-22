# =============================================================================
# MODEL-09 HIV-1 RECOMBINANT CLASSIFIER
# CURRENT 9-MODEL BENCHMARK DEPLOYMENT
#
# AUTHORITATIVE DEPLOYMENT ARTIFACTS:
#
# artifacts/
# ├── MODEL-09_Bidirectional_Attention_Transformer_Encoder.pt
# ├── MODEL-09_BENCHMARK_TRAIN_MEAN.npy
# ├── MODEL-09_BENCHMARK_TRAIN_STD.npy
# └── MODEL-09_BENCHMARK_FROZEN_THRESHOLD.txt
#
# REPRESENTATION:
#
# Raw protein sequence
#       ↓
# ESM-2 t33 650M
#       ↓
# residue embeddings: 1280-D
#       ↓
# complete 48-aa chunks
#       ↓
# mean + max
#       ↓
# 2560-D tokens
#       ↓
# pad to 91 tokens
#       ↓
# TRAIN-ONLY standardization
#       ↓
# MODEL-09
#       ↓
# frozen validation threshold
#       ↓
# Non-Recombinant / Recombinant
#
# IMPORTANT:
# The LocalAttentionBlock and GlobalAttentionBlock below MUST match
# the architecture used when the benchmark checkpoint was trained.
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

try:
    from transformers import AutoTokenizer, EsmModel
except Exception as e:
    AutoTokenizer = None
    EsmModel = None
    TRANSFORMERS_IMPORT_ERROR = e


# =============================================================================
# 2. PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="MODEL-09 HIV-1 Recombinant Classifier",
    page_icon="🧬",
    layout="wide"
)


# =============================================================================
# 3. PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

ARTIFACTS_DIR = (
    PROJECT_ROOT
    / "artifacts"
)


MODEL_CHECKPOINT = (
    ARTIFACTS_DIR
    / "MODEL-09_Bidirectional_Attention_Transformer_Encoder.pt"
)

TRAIN_MEAN_PATH = (
    ARTIFACTS_DIR
    / "MODEL-09_BENCHMARK_TRAIN_MEAN.npy"
)

TRAIN_STD_PATH = (
    ARTIFACTS_DIR
    / "MODEL-09_BENCHMARK_TRAIN_STD.npy"
)

FROZEN_THRESHOLD_PATH = (
    ARTIFACTS_DIR
    / "MODEL-09_BENCHMARK_FROZEN_THRESHOLD.txt"
)


# =============================================================================
# 4. MODEL / REPRESENTATION CONFIGURATION
# =============================================================================

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

# CRITICAL:
# Recovered from the checkpoint parameter count:
#
# MODEL-09 total parameters = 418,882
#
# This requires:
#
# FF_DIM = 192
#
# Do not change this.
FF_DIM = 192


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =============================================================================
# 5. APPLICATION HEADER
# =============================================================================

st.title(
    "🧬 MODEL-09 HIV-1 Recombinant Classifier"
)

st.caption(
    "Current 9-model benchmark deployment"
)

st.write(
    "MODEL-09 uses residue-level ESM-2 embeddings, "
    "complete 48-residue tokens, a 2560-D mean+max representation, "
    "91-token padding, train-only standardization, and the frozen "
    "benchmark decision threshold."
)


# =============================================================================
# 6. ARTIFACT VERIFICATION
# =============================================================================

def verify_artifacts():

    required = {
        "MODEL-09 checkpoint":
            MODEL_CHECKPOINT,

        "Training mean":
            TRAIN_MEAN_PATH,

        "Training std":
            TRAIN_STD_PATH,

        "Frozen threshold":
            FROZEN_THRESHOLD_PATH
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
            + "\n".join(missing)
            + "\n\nExpected directory:\n"
            + str(ARTIFACTS_DIR)
        )

    return required


# =============================================================================
# 7. LOAD FROZEN STANDARDIZATION
# =============================================================================

@st.cache_data(show_spinner=False)
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
            f"{mean.shape}. Expected "
            f"(1, 1, {INPUT_DIM})."
        )

    if std.shape != (
        1,
        1,
        INPUT_DIM
    ):

        raise RuntimeError(
            "Invalid training std shape: "
            f"{std.shape}. Expected "
            f"(1, 1, {INPUT_DIM})."
        )

    if not np.all(
        np.isfinite(mean)
    ):

        raise RuntimeError(
            "Training mean contains non-finite values."
        )

    if not np.all(
        np.isfinite(std)
    ):

        raise RuntimeError(
            "Training std contains non-finite values."
        )

    if np.any(
        std <= 0
    ):

        raise RuntimeError(
            "Training std contains zero or negative values."
        )

    return mean, std


# =============================================================================
# 8. LOAD FROZEN THRESHOLD
# =============================================================================

@st.cache_data(show_spinner=False)
def load_frozen_threshold():

    text = (
        FROZEN_THRESHOLD_PATH
        .read_text(
            encoding="utf-8"
        )
        .strip()
    )

    # Allow files such as:
    #
    # 0.7100000000
    #
    # or:
    #
    # Frozen validation threshold: 0.7100000000

    match = re.search(
        r"(?<!\d)(0(?:\.\d+)?|1(?:\.0+)?)(?!\d)",
        text
    )

    if match is None:

        raise RuntimeError(
            "Could not parse frozen threshold from:\n"
            + str(FROZEN_THRESHOLD_PATH)
        )

    threshold = float(
        match.group(1)
    )

    if not (
        0.0
        <= threshold
        <= 1.0
    ):

        raise RuntimeError(
            f"Invalid frozen threshold: {threshold}"
        )

    return threshold


# =============================================================================
# 9. LOCAL ATTENTION BLOCK
#
# THIS IS THE IMPORTANT CHECKPOINT-COMPATIBLE VERSION.
#
# State-dict keys expected by the checkpoint:
#
# local_attention.norm1.*
# local_attention.norm2.*
# local_attention.ff.0.*
# local_attention.ff.3.*
#
# and equivalent global_attention keys.
# =============================================================================

class LocalAttentionBlock(
    nn.Module
):

    def __init__(
        self,
        dim=MODEL_DIM,
        heads=ATTENTION_HEADS
    ):

        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=ATTENTION_DROPOUT,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(
            dim
        )

        self.norm2 = nn.LayerNorm(
            dim
        )

        self.ff = nn.Sequential(

            nn.Linear(
                dim,
                FF_DIM
            ),

            nn.GELU(),

            nn.Dropout(
                BASE_DROPOUT
            ),

            nn.Linear(
                FF_DIM,
                dim
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
        # = [B, T, 96]

        if x.ndim != 3:

            raise RuntimeError(
                "LocalAttentionBlock expected a 3-D tensor "
                f"[B,T,D], received shape {tuple(x.shape)}."
            )

        z = self.norm1(
            x
        )

        attn_output, _ = (
            self.attn(
                z,
                z,
                z,
                need_weights=False
            )
        )

        x = (
            x
            + attn_output
        )

        z = self.norm2(
            x
        )

        x = (
            x
            + self.ff(z)
        )

        return x


# =============================================================================
# 10. GLOBAL ATTENTION BLOCK
#
# The original benchmark used the same block structure.
# The separate class is retained because the checkpoint contains:
#
# global_attention.norm1
# global_attention.norm2
# global_attention.ff
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

        if x.ndim != 3:

            raise RuntimeError(
                "AttentionPooling expected "
                "[B,T,D], received "
                f"{tuple(x.shape)}"
            )

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
            * weights.unsqueeze(-1),
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
            LocalAttentionBlock(
                dim=model_dim,
                heads=heads
            )
        )

        self.global_attention = (
            GlobalAttentionBlock(
                dim=model_dim,
                heads=heads
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

        # ---------------------------------------------------------
        # REQUIRED INPUT:
        #
        # [B, 91, 2560]
        # ---------------------------------------------------------

        if x.ndim != 3:

            raise RuntimeError(
                "MODEL-09 expected a 3-D input tensor "
                "[batch, tokens, features]. "
                f"Received {x.ndim}-D tensor with shape "
                f"{tuple(x.shape)}."
            )

        if x.size(-1) != INPUT_DIM:

            raise RuntimeError(
                "MODEL-09 expected input dimension "
                f"{INPUT_DIM}, received {x.size(-1)}."
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
                * REPRESENTATION_NOISE
            )

        T = x.size(1)

        if T > TOKEN_LENGTH:

            raise RuntimeError(
                f"Input contains {T} tokens, but MODEL-09 "
                f"supports at most {TOKEN_LENGTH}."
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
# 13. CHECKPOINT STATE-DICT EXTRACTION
# =============================================================================

def extract_state_dict(
    checkpoint
):

    if isinstance(
        checkpoint,
        dict
    ):

        if (
            "model_state_dict"
            in checkpoint
        ):

            return checkpoint[
                "model_state_dict"
            ]

        if (
            "state_dict"
            in checkpoint
        ):

            return checkpoint[
                "state_dict"
            ]

    if isinstance(
        checkpoint,
        dict
    ):

        # A raw state_dict is also a dictionary.
        tensor_values = [

            isinstance(
                value,
                torch.Tensor
            )

            for value in checkpoint.values()

        ]

        if (
            len(tensor_values) > 0
            and all(tensor_values)
        ):

            return checkpoint

    raise RuntimeError(
        "Could not locate model state_dict inside checkpoint."
    )


# =============================================================================
# 14. MODEL-09 CHECKPOINT VERIFICATION
# =============================================================================

def verify_checkpoint_architecture(
    state_dict
):

    expected_keys = [

        "local_attention.norm1.weight",
        "local_attention.norm1.bias",
        "local_attention.norm2.weight",
        "local_attention.norm2.bias",

        "local_attention.ff.0.weight",
        "local_attention.ff.0.bias",
        "local_attention.ff.3.weight",
        "local_attention.ff.3.bias",

        "global_attention.norm1.weight",
        "global_attention.norm1.bias",
        "global_attention.norm2.weight",
        "global_attention.norm2.bias",

        "global_attention.ff.0.weight",
        "global_attention.ff.0.bias",
        "global_attention.ff.3.weight",
        "global_attention.ff.3.bias"
    ]

    missing = [

        key
        for key in expected_keys
        if key not in state_dict
    ]

    if missing:

        raise RuntimeError(
            "Checkpoint architecture does not match the "
            "current MODEL-09 deployment architecture.\n\n"
            "Missing checkpoint keys:\n"
            + "\n".join(missing)
        )

    # Verify FF dimension directly.

    ff_weight = state_dict[
        "local_attention.ff.0.weight"
    ]

    if tuple(
        ff_weight.shape
    ) != (
        FF_DIM,
        MODEL_DIM
    ):

        raise RuntimeError(
            "Unexpected MODEL-09 feed-forward dimension.\n"
            f"Checkpoint shape: {tuple(ff_weight.shape)}\n"
            f"Expected: ({FF_DIM}, {MODEL_DIM})"
        )


# =============================================================================
# 15. LOAD MODEL
# =============================================================================

@st.cache_resource(show_spinner=True)
def load_model():

    verify_artifacts()

    checkpoint = torch.load(
        MODEL_CHECKPOINT,
        map_location="cpu",
        weights_only=False
    )

    state_dict = extract_state_dict(
        checkpoint
    )

    verify_checkpoint_architecture(
        state_dict
    )

    model = (
        BidirectionalAttentionTransformerEncoder()
    )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    # -------------------------------------------------------------------------
    # Parameter-count verification
    # -------------------------------------------------------------------------

    parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    expected_parameter_count = 418882

    if parameter_count != (
        expected_parameter_count
    ):

        raise RuntimeError(
            "MODEL-09 parameter-count mismatch.\n"
            f"Loaded model: {parameter_count:,}\n"
            f"Expected: {expected_parameter_count:,}"
        )

    return model


# =============================================================================
# 16. ESM-2 MODEL LOADING
# =============================================================================

@st.cache_resource(show_spinner=True)
def load_esm2():

    if (
        AutoTokenizer is None
        or EsmModel is None
    ):

        raise RuntimeError(
            "transformers could not be imported.\n\n"
            f"Original error: {TRANSFORMERS_IMPORT_ERROR}"
        )

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
# 17. PROTEIN SEQUENCE CLEANING / VALIDATION
# =============================================================================

ALLOWED_AMINO_ACIDS = set(
    "ACDEFGHIKLMNPQRSTVWY"
)


def clean_sequence(
    sequence
):

    if sequence is None:

        raise ValueError(
            "No protein sequence was supplied."
        )

    sequence = str(
        sequence
    )

    # Remove FASTA header if present.

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
            + ", ".join(invalid)
            + "\n\nAllowed amino acids:\n"
            + "".join(
                sorted(
                    ALLOWED_AMINO_ACIDS
                )
            )
            + "\n\n"
            "Important: X represents an unknown residue and "
            "is not accepted by this deployment because the "
            "benchmark ESM-2 representation was constructed "
            "from valid amino-acid sequences."
        )

    return sequence


# =============================================================================
# 18. ESM-2 RESIDUE EMBEDDING
# =============================================================================

def extract_residue_embeddings(
    sequence,
    tokenizer,
    esm_model
):

    # ESM-2 has a practical sequence-length limitation.
    #
    # For long HIV-1 proteins, process the sequence in overlapping
    # windows and reconstruct residue embeddings.
    #
    # This mirrors the benchmark's long-protein strategy.

    MAX_ESM_LENGTH = 1022

    OVERLAP = 128

    STRIDE = (
        MAX_ESM_LENGTH
        -
        OVERLAP
    )

    sequence_length = len(
        sequence
    )

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

    start = 0

    while start < sequence_length:

        end = min(
            start
            + MAX_ESM_LENGTH,
            sequence_length
        )

        fragment = sequence[
            start:end
        ]

        inputs = tokenizer(
            fragment,
            return_tensors="pt",
            add_special_tokens=True
        )

        inputs = {
            key: value.to(
                DEVICE
            )
            for key, value
            in inputs.items()
        }

        with torch.no_grad():

            outputs = esm_model(
                **inputs
            )

        hidden = (
            outputs
            .last_hidden_state
        )

        # Remove BOS and EOS.

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
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        expected_length = (
            end
            - start
        )

        if residue_hidden.shape[0] != (
            expected_length
        ):

            raise RuntimeError(
                "ESM-2 residue count mismatch.\n"
                f"Expected: {expected_length}\n"
                f"Received: {residue_hidden.shape[0]}"
            )

        accumulated[
            start:end
        ] += residue_hidden

        counts[
            start:end
        ] += 1.0

        if end >= sequence_length:

            break

        start += STRIDE

    if np.any(
        counts == 0
    ):

        raise RuntimeError(
            "Some residues did not receive an ESM-2 embedding."
        )

    accumulated /= (
        counts[:, None]
    )

    return accumulated


# =============================================================================
# 19. RESIDUE → 2560-D COMPLETE 48-AA TOKENS
# =============================================================================

def residue_to_tokens(
    residue_embeddings
):

    if residue_embeddings.ndim != 2:

        raise ValueError(
            "Residue embeddings must be 2-D "
            "[residues, 1280]. Received "
            f"{residue_embeddings.shape}"
        )

    if residue_embeddings.shape[1] != (
        ESM2_DIMENSION
    ):

        raise ValueError(
            "Expected residue embedding dimension "
            f"{ESM2_DIMENSION}, received "
            f"{residue_embeddings.shape[1]}"
        )

    number_of_complete_tokens = (
        residue_embeddings.shape[0]
        //
        CHUNK_SIZE
    )

    if number_of_complete_tokens < 1:

        raise ValueError(
            f"Protein is too short. At least "
            f"{CHUNK_SIZE} residues are required."
        )

    usable_length = (
        number_of_complete_tokens
        *
        CHUNK_SIZE
    )

    residues = (
        residue_embeddings[
            :usable_length
        ]
    )

    chunks = (
        residues
        .reshape(
            number_of_complete_tokens,
            CHUNK_SIZE,
            ESM2_DIMENSION
        )
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
            "Token dimension mismatch.\n"
            f"Received: {tokens.shape[1]}\n"
            f"Expected: {INPUT_DIM}"
        )

    return tokens.astype(
        np.float32
    )


# =============================================================================
# 20. PAD / TRUNCATE TO 91 TOKENS
# =============================================================================

def pad_tokens(
    tokens
):

    if tokens.ndim != 2:

        raise ValueError(
            "Tokens must be 2-D [tokens, 2560]."
        )

    if tokens.shape[1] != (
        INPUT_DIM
    ):

        raise ValueError(
            f"Expected token dimension {INPUT_DIM}."
        )

    number_of_tokens = (
        tokens.shape[0]
    )

    if number_of_tokens > (
        TOKEN_LENGTH
    ):

        # The benchmark representation is fixed to 91 tokens.
        #
        # For deployment, retain the first 91 complete tokens.

        tokens = tokens[
            :TOKEN_LENGTH
        ]

        number_of_tokens = (
            TOKEN_LENGTH
        )

    output = np.zeros(
        (
            TOKEN_LENGTH,
            INPUT_DIM
        ),
        dtype=np.float32
    )

    output[
        :number_of_tokens
    ] = tokens

    return output


# =============================================================================
# 21. TRAIN-ONLY STANDARDIZATION
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

    standardized = (
        np.asarray(
            standardized,
            dtype=np.float32
        )
    )

    if not np.all(
        np.isfinite(
            standardized
        )
    ):

        raise RuntimeError(
            "Standardized deployment representation "
            "contains non-finite values."
        )

    return standardized


# =============================================================================
# 22. FIXED-SHAPE MODEL INPUT
#
# IMPORTANT:
# This explicitly creates:
#
# [91, 2560]
#       ↓
# [1, 91, 2560]
#
# and NEVER:
#
# [1, 1, 91, 2560]
#
# This fixes the previous 4-D MultiheadAttention error.
# =============================================================================

def create_model_input(
    standardized_tokens
):

    array = np.asarray(
        standardized_tokens,
        dtype=np.float32
    )

    if array.shape != (
        TOKEN_LENGTH,
        INPUT_DIM
    ):

        raise RuntimeError(
            "Deployment representation has wrong shape.\n"
            f"Received: {array.shape}\n"
            f"Expected: ({TOKEN_LENGTH}, {INPUT_DIM})"
        )

    tensor = torch.from_numpy(
        array
    )

    # Exactly ONE batch dimension.

    tensor = tensor.unsqueeze(
        0
    )

    if tensor.ndim != 3:

        raise RuntimeError(
            "MODEL-09 input construction failed.\n"
            f"Received tensor shape: {tuple(tensor.shape)}"
        )

    if tensor.shape != (
        1,
        TOKEN_LENGTH,
        INPUT_DIM
    ):

        raise RuntimeError(
            "Unexpected MODEL-09 input shape.\n"
            f"Received: {tuple(tensor.shape)}\n"
            f"Expected: (1, {TOKEN_LENGTH}, {INPUT_DIM})"
        )

    return tensor.to(
        DEVICE
    )


# =============================================================================
# 23. FULL MODEL-09 PREDICTION
# =============================================================================

def predict_model09(
    model,
    sequence,
    train_mean,
    train_std,
    frozen_threshold
):

    sequence = clean_sequence(
        sequence
    )

    # -------------------------------------------------------------------------
    # ESM-2
    # -------------------------------------------------------------------------

    tokenizer, esm_model = (
        load_esm2()
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

    # -------------------------------------------------------------------------
    # Tokenization
    # -------------------------------------------------------------------------

    with st.spinner(
        "Constructing 48-residue tokens..."
    ):

        raw_tokens = (
            residue_to_tokens(
                residue_embeddings
            )
        )

        token_matrix = (
            pad_tokens(
                raw_tokens
            )
        )

    # -------------------------------------------------------------------------
    # Standardization
    # -------------------------------------------------------------------------

    standardized = (
        standardize_tokens(
            token_matrix,
            train_mean,
            train_std
        )
    )

    # -------------------------------------------------------------------------
    # MODEL INPUT
    # -------------------------------------------------------------------------

    model_input = (
        create_model_input(
            standardized
        )
    )

    # -------------------------------------------------------------------------
    # MODEL-09
    # -------------------------------------------------------------------------

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
        >= frozen_threshold
    )

    return {

        "sequence_length":
            len(sequence),

        "raw_token_count":
            raw_tokens.shape[0],

        "final_token_count":
            TOKEN_LENGTH,

        "residue_shape":
            residue_embeddings.shape,

        "token_shape":
            token_matrix.shape,

        "model_input_shape":
            tuple(
                model_input.shape
            ),

        "probability":
            probability,

        "threshold":
            frozen_threshold,

        "prediction":
            prediction,

        "attention":
            attention
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
    }


# =============================================================================
# 24. STARTUP INITIALIZATION
# =============================================================================

try:

    verify_artifacts()

    train_mean, train_std = (
        load_standardization()
    )

    FROZEN_THRESHOLD = (
        load_frozen_threshold()
    )

    model = load_model()

    st.success(
        "✓ MODEL-09 initialized successfully"
    )

except Exception as e:

    st.error(
        "MODEL-09 could not be initialized."
    )

    st.code(
        str(e),
        language="text"
    )

    st.stop()


# =============================================================================
# 25. SIDEBAR — DEPLOYMENT INFORMATION
# =============================================================================

with st.sidebar:

    st.header(
        "Deployment"
    )

    st.write(
        f"**Device:** `{DEVICE}`"
    )

    st.write(
        f"**ESM-2 dimension:** `{ESM2_DIMENSION}`"
    )

    st.write(
        f"**Chunk size:** `{CHUNK_SIZE}`"
    )

    st.write(
        f"**Token dimension:** `{INPUT_DIM}`"
    )

    st.write(
        f"**Token length:** `{TOKEN_LENGTH}`"
    )

    st.write(
        f"**MODEL dimension:** `{MODEL_DIM}`"
    )

    st.write(
        f"**Attention heads:** `{ATTENTION_HEADS}`"
    )

    st.write(
        f"**FF dimension:** `{FF_DIM}`"
    )

    st.write(
        f"**Frozen threshold:** `{FROZEN_THRESHOLD:.10f}`"
    )

    st.divider()

    st.caption(
        "Current 9-model benchmark"
    )


# =============================================================================
# 26. INPUT
# =============================================================================

st.subheader(
    "Protein sequence"
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
# 27. PREDICTION BUTTON
# =============================================================================

predict_button = st.button(
    "🧬 Predict recombinant status",
    type="primary",
    use_container_width=True
)


# =============================================================================
# 28. RUN PREDICTION
# =============================================================================

if predict_button:

    try:

        result = predict_model09(

            model,

            sequence_input,

            train_mean,

            train_std,

            FROZEN_THRESHOLD

        )

        st.divider()

        st.subheader(
            "MODEL-09 Prediction"
        )

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

        probability = (
            result[
                "probability"
            ]
        )

        st.metric(
            "Recombinant Probability",
            f"{probability:.8f}"
        )

        st.metric(
            "Frozen Decision Threshold",
            f"{result['threshold']:.8f}"
        )

        st.write(
            f"**Input sequence length:** "
            f"{result['sequence_length']} residues"
        )

        st.write(
            f"**Complete 48-aa tokens:** "
            f"{result['raw_token_count']}"
        )

        st.write(
            f"**Final MODEL-09 representation:** "
            f"{result['final_token_count']} × "
            f"{INPUT_DIM}"
        )

        st.write(
            f"**MODEL-09 input tensor:** "
            f"`{result['model_input_shape']}`"
        )

        # ---------------------------------------------------------------------
        # Attention visualization
        # ---------------------------------------------------------------------

        st.subheader(
            "Token attention"
        )

        attention = result[
            "attention"
        ]

        st.bar_chart(
            attention
        )

    except ValueError as e:

        st.error(
            "Invalid protein sequence."
        )

        st.code(
            str(e),
            language="text"
        )

    except Exception as e:

        st.error(
            "Prediction failed."
        )

        st.exception(
            e
        )


# =============================================================================
# 29. FOOTER
# =============================================================================

st.divider()

st.caption(
    "MODEL-09 | Current 9-model benchmark | "
    "ESM-2 t33 650M → 48-aa mean+max tokens → "
    "91-token representation → train-only standardization → "
    "Bidirectional Attention Transformer Encoder"
)
