import os
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


# =============================================================================
# MODEL-09 — AUTHORITATIVE WHOLE-PROTEIN DEPLOYMENT
# =============================================================================
#
# RAW HIV-1 PROTEIN
#        ↓
# ESM-2 t33 650M
#        ↓
# ESM-2 chunks: 1024 aa, stride 896 aa
#        ↓
# Residue embeddings: N × 1280
#        ↓
# MODEL-09 chunks: 48 aa, stride 48 aa
#        ↓
# Mean + STD
#        ↓
# 2560-D tokens
#        ↓
# Pad / truncate to 91 tokens
#        ↓
# Frozen training standardization
#        ↓
# MODEL-09 FP32 teacher
#        ↓
# Recombinant probability
#
# NO TRAINING IS PERFORMED.
# =============================================================================


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"


# -----------------------------------------------------------------------------
# ESM-2 RESIDUE EMBEDDING CONFIGURATION
# -----------------------------------------------------------------------------

ESM_DIM = 1280

ESM_CHUNK_SIZE = 1024

ESM_CHUNK_OVERLAP = 128

ESM_CHUNK_STRIDE = (
    ESM_CHUNK_SIZE - ESM_CHUNK_OVERLAP
)


# -----------------------------------------------------------------------------
# MODEL-09 REPRESENTATION CONFIGURATION
# -----------------------------------------------------------------------------

REP_CHUNK_SIZE = 48

REP_CHUNK_STRIDE = 48

TOKEN_FEATURE_DIM = 2560

TOKEN_LENGTH = 91


# -----------------------------------------------------------------------------
# MODEL-09 ARCHITECTURE
# -----------------------------------------------------------------------------

MODEL_DIM = 96

ATTENTION_HEADS = 4

ATTENTION_DROPOUT = 0.25

BASE_DROPOUT = 0.30


# -----------------------------------------------------------------------------
# DEPLOYMENT THRESHOLD
# -----------------------------------------------------------------------------

# We intentionally do NOT use the historical 0.88 threshold.

DEFAULT_THRESHOLD = 0.50


# -----------------------------------------------------------------------------
# DEVICE
# -----------------------------------------------------------------------------

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =============================================================================
# 2. FILE PATHS
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
# 3. PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="MODEL-09 HIV-1 Recombinant Classifier",
    page_icon="🧬",
    layout="wide"
)


# =============================================================================
# 4. PAGE HEADER
# =============================================================================

st.title(
    "MODEL-09 HIV-1 Recombinant Classifier"
)

st.write(
    """
    Whole-protein HIV-1 recombinant-status prediction using the
    original FP32 MODEL-09 attention-based teacher model.
    """
)

st.caption(
    "Inference only — no model training is performed."
)


# =============================================================================
# 5. LOCAL ATTENTION
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
# 6. GLOBAL ATTENTION
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
# 8. BIDIRECTIONAL ATTENTION TRANSFORMER
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
                "training_noise is not allowed during deployment."
            )


        T = x.size(1)


        if T > TOKEN_LENGTH:

            raise RuntimeError(
                f"Input contains {T} tokens, "
                f"but MODEL-09 supports at most "
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
# 9. LOAD TRAINING STANDARDIZATION
# =============================================================================

@st.cache_resource
def load_standardization():

    if not os.path.isfile(
        TRAIN_MEAN_PATH
    ):

        raise FileNotFoundError(
            "Missing MODEL-09 training mean:\n"
            f"{TRAIN_MEAN_PATH}"
        )


    if not os.path.isfile(
        TRAIN_STD_PATH
    ):

        raise FileNotFoundError(
            "Missing MODEL-09 training std:\n"
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


    expected_shape = (
        TOKEN_FEATURE_DIM,
    )


    if train_mean.shape != expected_shape:

        raise RuntimeError(
            f"TRAIN_MEAN shape is "
            f"{train_mean.shape}; "
            f"expected {expected_shape}."
        )


    if train_std.shape != expected_shape:

        raise RuntimeError(
            f"TRAIN_STD shape is "
            f"{train_std.shape}; "
            f"expected {expected_shape}."
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


    # Protect against zero standard deviations.

    train_std = np.where(
        train_std < 1e-8,
        1.0,
        train_std
    ).astype(
        np.float32
    )


    return (
        train_mean,
        train_std
    )


# =============================================================================
# 10. LOAD MODEL-09 TEACHER
# =============================================================================

@st.cache_resource
def load_teacher():

    if not os.path.isfile(
        TEACHER_PATH
    ):

        raise FileNotFoundError(
            "Missing MODEL-09 teacher checkpoint:\n"
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


    if "model_state_dict" not in checkpoint:

        raise RuntimeError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )


    # Verify known checkpoint metadata.

    expected_metadata = {

        "esm2_dimension":
            ESM_DIM,

        "chunk_size":
            REP_CHUNK_SIZE,

        "chunk_stride":
            REP_CHUNK_STRIDE,

        "input_dim":
            TOKEN_FEATURE_DIM,

        "token_length":
            TOKEN_LENGTH,

        "model_dim":
            MODEL_DIM,

        "attention_heads":
            ATTENTION_HEADS
    }


    mismatches = []


    for key, expected in expected_metadata.items():

        if key in checkpoint:

            actual = checkpoint[key]

            if actual != expected:

                mismatches.append(
                    f"{key}: "
                    f"checkpoint={actual}, "
                    f"deployment={expected}"
                )


    if mismatches:

        raise RuntimeError(
            "MODEL-09 checkpoint metadata mismatch:\n\n"
            +
            "\n".join(
                mismatches
            )
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
            "MODEL-09 checkpoint did not load exactly.\n\n"
            f"Missing keys: {missing}\n\n"
            f"Unexpected keys: {unexpected}"
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
# 12. SEQUENCE CLEANING
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
# 13. SEQUENCE VALIDATION
# =============================================================================

def validate_sequence(
    sequence
):

    allowed = set(
        "ACDEFGHIKLMNPQRSTVWY"
    )


    if len(sequence) == 0:

        return (
            False,
            "The sequence is empty."
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
            ", ".join(
                invalid
            )
        )


    return (
        True,
        ""
    )


# =============================================================================
# 14. ESM-2 SINGLE CHUNK
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
        max_length=ESM_CHUNK_SIZE + 2
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


    hidden = outputs.last_hidden_state


    # Remove BOS/EOS special tokens.

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
# 15. WHOLE-PROTEIN ESM-2 RESIDUE EMBEDDINGS
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
            ESM_CHUNK_STRIDE
        )
    )


    total_chunks = len(
        starts
    )


    progress = st.progress(
        0,
        text="Computing ESM-2 residue embeddings..."
    )


    for i, start in enumerate(
        starts,
        1
    ):

        end = min(
            start + ESM_CHUNK_SIZE,
            sequence_length
        )


        sequence_chunk = sequence[
            start:end
        ]


        embedding = embed_esm_chunk(
            sequence_chunk,
            tokenizer,
            esm_model
        )


        expected_shape = (
            end - start,
            ESM_DIM
        )


        if embedding.shape != expected_shape:

            raise RuntimeError(
                "Unexpected ESM-2 embedding shape: "
                f"{embedding.shape}; "
                f"expected {expected_shape}."
            )


        embedding_sum[
            start:end
        ] += embedding


        embedding_count[
            start:end
        ] += 1.0


        progress.progress(
            i / total_chunks,
            text=(
                f"ESM-2 chunk {i}/{total_chunks}"
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


    expected_final_shape = (
        sequence_length,
        ESM_DIM
    )


    if residue_embeddings.shape != expected_final_shape:

        raise RuntimeError(
            "Final residue embedding shape is "
            f"{residue_embeddings.shape}; "
            f"expected {expected_final_shape}."
        )


    return residue_embeddings


# =============================================================================
# 16. BUILD MODEL-09 REPRESENTATION
# =============================================================================

def build_model09_representation(
    residue_embeddings
):

    sequence_length = (
        residue_embeddings.shape[0]
    )


    if residue_embeddings.ndim != 2:

        raise RuntimeError(
            "Residue embeddings must be 2-D."
        )


    if residue_embeddings.shape[1] != ESM_DIM:

        raise RuntimeError(
            f"Residue embedding dimension is "
            f"{residue_embeddings.shape[1]}; "
            f"expected {ESM_DIM}."
        )


    tokens = []


    for start in range(
        0,
        sequence_length,
        REP_CHUNK_STRIDE
    ):

        end = min(
            start + REP_CHUNK_SIZE,
            sequence_length
        )


        chunk = residue_embeddings[
            start:end
        ]


        mean = chunk.mean(
            axis=0,
            dtype=np.float32
        )


        std = chunk.std(
            axis=0,
            ddof=0,
            dtype=np.float32
        )


        token = np.concatenate(
            [
                mean,
                std
            ]
        ).astype(
            np.float32
        )


        tokens.append(
            token
        )


    if len(tokens) == 0:

        raise RuntimeError(
            "No MODEL-09 tokens were created."
        )


    tokens = np.stack(
        tokens,
        axis=0
    )


    raw_token_count = (
        tokens.shape[0]
    )


    # MODEL-09 requires exactly 91 tokens.

    if raw_token_count > TOKEN_LENGTH:

        tokens = tokens[
            :TOKEN_LENGTH
        ]


    elif raw_token_count < TOKEN_LENGTH:

        padding = np.zeros(
            (
                TOKEN_LENGTH - raw_token_count,
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


    expected_shape = (
        TOKEN_LENGTH,
        TOKEN_FEATURE_DIM
    )


    if tokens.shape != expected_shape:

        raise RuntimeError(
            "MODEL-09 representation shape is "
            f"{tokens.shape}; "
            f"expected {expected_shape}."
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

    expected_shape = (
        TOKEN_LENGTH,
        TOKEN_FEATURE_DIM
    )


    if tokens.shape != expected_shape:

        raise RuntimeError(
            f"Unexpected token shape: "
            f"{tokens.shape}; "
            f"expected {expected_shape}."
        )


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
            "Standardized representation "
            "contains non-finite values."
        )


    return standardized.astype(
        np.float32
    )


# =============================================================================
# 18. PREDICTION
# =============================================================================

@torch.inference_mode()
def predict(
    standardized_tokens,
    teacher,
    threshold
):

    expected_shape = (
        TOKEN_LENGTH,
        TOKEN_FEATURE_DIM
    )


    if standardized_tokens.shape != expected_shape:

        raise RuntimeError(
            "Invalid MODEL-09 input shape: "
            f"{standardized_tokens.shape}; "
            f"expected {expected_shape}."
        )


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

        prediction = "RECOMBINANT"

    else:

        prediction = "NON-RECOMBINANT"


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
# 19. SIDEBAR
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
        f"ESM chunk size: `{ESM_CHUNK_SIZE}` aa"
    )


    st.write(
        f"ESM overlap: `{ESM_CHUNK_OVERLAP}` aa"
    )


    st.write(
        f"ESM stride: `{ESM_CHUNK_STRIDE}` aa"
    )


    st.write(
        f"MODEL-09 chunk size: `{REP_CHUNK_SIZE}` aa"
    )


    st.write(
        f"MODEL-09 stride: `{REP_CHUNK_STRIDE}` aa"
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


    threshold = st.slider(
        "Classification threshold",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_THRESHOLD,
        step=0.01
    )


    st.caption(
        "Default threshold = 0.50. "
        "Historical 0.88 threshold is not used."
    )


# =============================================================================
# 20. LOAD DEPLOYMENT ARTIFACTS
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


except Exception as error:

    st.error(
        "Failed to load MODEL-09 deployment artifacts."
    )


    st.exception(
        error
    )


    st.stop()


# =============================================================================
# 21. SEQUENCE INPUT
# =============================================================================

st.subheader(
    "Enter HIV-1 protein sequence"
)


sequence_input = st.text_area(
    "Raw amino-acid sequence",
    height=250,
    placeholder=(
        "Paste a complete HIV-1 protein sequence here..."
    )
)


run_prediction = st.button(
    "Run MODEL-09 Prediction",
    type="primary"
)


# =============================================================================
# 22. RUN PREDICTION
# =============================================================================

if run_prediction:

    sequence = clean_sequence(
        sequence_input
    )


    valid, error_message = (
        validate_sequence(
            sequence
        )
    )


    if not valid:

        st.error(
            error_message
        )

        st.stop()


    st.info(
        f"Input sequence length: "
        f"{len(sequence):,} amino acids"
    )


    try:

        # -----------------------------------------------------------------
        # STEP 1 — ESM-2
        # -----------------------------------------------------------------

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
            "ESM-2 residue embeddings generated: "
            f"{residue_embeddings.shape}"
        )


        # -----------------------------------------------------------------
        # STEP 2 — MODEL-09 REPRESENTATION
        # -----------------------------------------------------------------

        with st.spinner(
            "Building MODEL-09 48-aa representation..."
        ):

            tokens, raw_token_count = (
                build_model09_representation(
                    residue_embeddings
                )
            )


        st.success(
            f"Raw MODEL-09 tokens: "
            f"{raw_token_count} | "
            f"Final input: {tokens.shape}"
        )


        # -----------------------------------------------------------------
        # STEP 3 — STANDARDIZATION
        # -----------------------------------------------------------------

        with st.spinner(
            "Applying frozen training standardization..."
        ):

            standardized = standardize(
                tokens,
                train_mean,
                train_std
            )


        # -----------------------------------------------------------------
        # STEP 4 — TEACHER
        # -----------------------------------------------------------------

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


        # -----------------------------------------------------------------
        # STEP 5 — RESULTS
        # -----------------------------------------------------------------

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


        # -----------------------------------------------------------------
        # STEP 6 — TECHNICAL DETAILS
        # -----------------------------------------------------------------

        with st.expander(
            "Technical details"
        ):

            st.write(
                f"Input sequence length: "
                f"{len(sequence)} aa"
            )


            st.write(
                f"ESM-2 model: "
                f"`{MODEL_NAME}`"
            )


            st.write(
                f"ESM-2 residue embedding shape: "
                f"`{residue_embeddings.shape}`"
            )


            st.write(
                f"ESM-2 chunk size: "
                f"`{ESM_CHUNK_SIZE}` aa"
            )


            st.write(
                f"ESM-2 overlap: "
                f"`{ESM_CHUNK_OVERLAP}` aa"
            )


            st.write(
                f"ESM-2 stride: "
                f"`{ESM_CHUNK_STRIDE}` aa"
            )


            st.write(
                f"MODEL-09 representation chunk size: "
                f"`{REP_CHUNK_SIZE}` aa"
            )


            st.write(
                f"MODEL-09 representation stride: "
                f"`{REP_CHUNK_STRIDE}` aa"
            )


            st.write(
                f"Raw MODEL-09 token count: "
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
                f"Device: `{DEVICE}`"
            )


            st.write(
                "MODEL-09 parameters: "
                f"{sum(p.numel() for p in teacher.parameters()):,}"
            )


            st.write(
                f"Training mean shape: "
                f"`{train_mean.shape}`"
            )


            st.write(
                f"Training std shape: "
                f"`{train_std.shape}`"
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


        # -----------------------------------------------------------------
        # STEP 7 — ATTENTION
        # -----------------------------------------------------------------

        with st.expander(
            "MODEL-09 attention weights"
        ):

            st.line_chart(
                attention
            )


    except Exception as error:

        st.error(
            "MODEL-09 prediction failed."
        )


        st.exception(
            error
        )
