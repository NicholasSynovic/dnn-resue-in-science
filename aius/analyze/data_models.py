import mdformat
from pandas import DataFrame
from pydantic import BaseModel


class Document(BaseModel):
    doi: str
    content: str


class ModelResponse(BaseModel):
    doi: str
    system_prompt: str
    user_prompt: str
    model_response: str
    model_reasoning: str
    compute_time_seconds: float

    @property
    def to_df(self) -> DataFrame:
        return DataFrame(
            data={
                "doi": [self.doi],
                "system_prompt": [self.system_prompt],
                "user_prompt": [self.user_prompt],
                "model_response": [self.model_response],
                "model_reasoning": [self.model_reasoning],
                "compute_time_seconds": [self.compute_time_seconds],
            }
        )


class COSTAR_SystemPrompt(BaseModel):  # noqa: D101, N801
    tag: str
    context: str
    objective: str
    response: str
    style: str = "Responses must be strictly machine-readable JSON. No natural language, commentary, or formatting beyond the JSON object is permitted."
    tone: str = "Neutral, objective, and machine-like."
    audience: str = "The audience is a machine system that parses JSON. Human readability is irrelevant."  # noqa: E501

    def create_prompt(self) -> str:  # noqa: D102
        return mdformat.text(
            options={"wrap": 80},
            md=f"""
## Context
{self.context}

## Objective
{self.objective}

## Style
{self.style}

## Tone
{self.tone}

## Audience
{self.audience}

## Response
{self.response}
""",
        )


__model_json_object: str = """```json
{
    "text_models": ["ALBERT", "Apertus", "Arcee", "Bamba", "BART", "BARThez", "BARTpho", "BERT", "BertGeneration", "BertJapanese", "BERTweet", "BigBird", "BigBirdPegasus", "BioGpt", "BitNet", "Blenderbot", "Blenderbot Small", "BLOOM", "BLT", "BORT", "ByT5", "CamemBERT", "CANINE", "CodeGen", "CodeLlama", "Cohere", "Cohere2", "ConvBERT", "CPM", "CPMANT", "CTRL", "DBRX", "DeBERTa", "DeBERTa-v2", "DeepSeek-V2", "DeepSeek-V3", "DialoGPT", "DiffLlama", "DistilBERT", "Doge", "dots1", "DPR", "ELECTRA", "Encoder Decoder Models", "ERNIE", "Ernie4_5", "Ernie4_5_MoE", "ErnieM", "ESM", "EXAONE-4.0", "Falcon", "Falcon3", "FalconH1", "FalconMamba", "FLAN-T5", "FLAN-UL2", "FlauBERT", "FlexOlmo", "FNet", "FSMT", "Funnel Transformer", "Fuyu", "Gemma", "Gemma2", "GLM", "glm4", "glm4_moe", "GPT", "GPT Neo", "GPT NeoX", "GPT NeoX Japanese", "GPT-J", "GPT2", "GPTBigCode", "GptOss", "GPTSAN Japanese", "GPTSw3", "Granite", "GraniteMoe", "GraniteMoeHybrid", "GraniteMoeShared", "Helium", "HerBERT", "HGNet-V2", "HunYuanDenseV1", "HunYuanMoEV1", "I-BERT", "Jamba", "JetMoe", "Jukebox", "LED", "LFM2", "LLaMA", "Llama2", "Llama3", "LongCatFlash", "Longformer", "LongT5", "LUKE", "M2M100", "MADLAD-400", "Mamba", "Mamba2", "MarianMT", "MarkupLM", "MBart and MBart-50", "MEGA", "MegatronBERT", "MegatronGPT2", "MiniMax", "Ministral", "Mistral", "Mixtral", "mLUKE", "MobileBERT", "ModernBert", "ModernBERTDecoder", "MPNet", "MPT", "MRA", "MT5", "MVP", "myt5", "Nemotron", "NEZHA", "NLLB", "NLLB-MoE", "Nyströmformer", "OLMo", "OLMo2", "Olmo3", "OLMoE", "Open-Llama", "OPT", "Pegasus", "PEGASUS-X", "Persimmon", "Phi", "Phi-3", "PhiMoE", "PhoBERT", "PLBart", "ProphetNet", "QDQBert", "Qwen2", "Qwen2MoE", "Qwen3", "Qwen3MoE", "Qwen3Next", "RAG", "REALM", "RecurrentGemma", "Reformer", "RemBERT", "RetriBERT", "RoBERTa", "RoBERTa-PreLayerNorm", "RoCBert", "RoFormer", "RWKV", "Seed-Oss", "Splinter", "SqueezeBERT", "StableLm", "Starcoder2", "SwitchTransformers", "T5", "T5Gemma", "T5v1.1", "TAPEX", "Transformer XL", "UL2", "UMT5", "VaultGemma", "X-MOD", "XGLM", "XLM", "XLM-ProphetNet", "XLM-RoBERTa", "XLM-RoBERTa-XL", "XLM-V", "XLNet", "xLSTM", "YOSO", "Zamba", "Zamba2"],
    "vision_models": ["Aimv2", "BEiT", "BiT", "Conditional DETR", "ConvNeXT", "ConvNeXTV2", "CvT", "D-FINE", "DAB-DETR", "Deformable DETR", "DeiT", "Depth Anything", "Depth Anything V2", "DepthPro", "DETA", "DETR", "DiNAT", "DINOV2", "DINOv2 with Registers", "DINOv3", "DiT", "DPT", "EfficientFormer", "EfficientLoFTR", "EfficientNet", "EoMT", "FocalNet", "GLPN", "HGNet-V2", "Hiera", "I-JEPA", "ImageGPT", "LeViT", "LightGlue", "Mask2Former", "MaskFormer", "MLCD", "MobileNetV1", "MobileNetV2", "MobileViT", "MobileViTV2", "NAT", "PoolFormer", "Prompt Depth Anything", "Pyramid Vision Transformer (PVT)", "Pyramid Vision Transformer v2 (PVTv2)", "RegNet", "ResNet", "RT-DETR", "RT-DETRv2", "SAM2", "SegFormer", "SegGpt", "Segment Anything", "Segment Anything High Quality", "SuperGlue", "SuperPoint", "SwiftFormer", "Swin Transformer", "Swin Transformer V2", "Swin2SR", "Table Transformer", "TextNet", "Timm Wrapper", "UperNet", "VAN", "Vision Transformer (ViT)", "ViT Hybrid", "ViTDet", "ViTMAE", "ViTMatte", "ViTMSN", "ViTPose", "YOLOS", "ZoeDepth"],
    "audio_models": ["Audio Spectrogram Transformer", "Bark", "CLAP", "CSM", "dac", "Dia", "EnCodec", "FastSpeech2Conformer", "GraniteSpeech", "Hubert", "Kyutai Speech-To-Text", "MCTCT", "Mimi", "MMS", "Moonshine", "Moshi", "MusicGen", "MusicGen Melody", "Parakeet", "Pop2Piano", "Seamless-M4T", "SeamlessM4T-v2", "SEW", "SEW-D", "Speech2Text", "Speech2Text2", "SpeechT5", "UniSpeech", "UniSpeech-SAT", "UnivNet", "VITS", "Wav2Vec2", "Wav2Vec2-BERT", "Wav2Vec2-Conformer", "Wav2Vec2Phoneme", "WavLM", "Whisper", "X-Codec", "XLS-R", "XLSR-Wav2Vec2"],
    "video_models": ["SAM2 Video", "TimeSformer", "V-JEPA 2", "VideoMAE", "ViViT"],
    "multimodal_models": ["ALIGN", "AltCLIP", "Aria", "AyaVision", "BLIP", "BLIP-2", "BridgeTower", "BROS", "Chameleon", "Chinese-CLIP", "CLIP", "CLIPSeg", "CLVP", "Cohere2Vision", "ColPali", "ColQwen2", "Data2Vec", "DeepseekVL", "DeepseekVLHybrid", "DePlot", "Donut", "EdgeTAM", "EdgeTamVideo", "Emu3", "Evolla", "FLAVA", "Florence2", "Gemma3", "Gemma3n", "GIT", "glm4v", "glm4v_moe", "GOT-OCR2", "GraniteVision", "Grounding DINO", "GroupViT", "IDEFICS", "Idefics2", "Idefics3", "InstructBLIP", "InstructBlipVideo", "InternVL", "Janus", "KOSMOS-2", "KOSMOS-2.5", "LayoutLM", "LayoutLMV2", "LayoutLMV3", "LayoutXLM", "LFM2-VL", "LiLT", "Llama4", "LLaVA", "LLaVA-NeXT", "LLaVa-NeXT-Video", "LLaVA-Onevision", "LXMERT", "MatCha", "MetaCLIP 2", "MGP-STR", "Mistral3", "mllama", "MM Grounding DINO", "Nougat", "OmDet-Turbo", "OneFormer", "Ovis2", "OWL-ViT", "OWLv2", "PaliGemma", "Perceiver", "PerceptionLM", "Phi4 Multimodal", "Pix2Struct", "Pixtral", "Qwen2.5-Omni", "Qwen2.5-VL", "Qwen2Audio", "Qwen2VL", "Qwen3-Omni-MoE", "Qwen3VL", "Qwen3VLMoe", "ShieldGemma2", "SigLIP", "SigLIP2", "SmolLM3", "SmolVLM", "Speech Encoder Decoder Models", "TAPAS", "TrOCR", "TVLT", "TVP", "UDOP", "VideoLlava", "ViLT", "VipLlava", "Vision Encoder Decoder Models", "Vision Text Dual Encoder", "VisualBERT", "Voxtral", "X-CLIP"],
    "reinforcement_learning_models": ["Decision Transformer", "Trajectory Transformer"],
    "time_series_models": ["Autoformer", "Informer", "PatchTSMixer", "PatchTST", "Time Series Transformer", "TimesFM"],
    "graph_models": ["Graphormer"],
    "edge_cases": ["SignalP", "DeepMedic", "SecretomeP", "DeepLoc", "DeepLocPro"]
}
```"""  # noqa: E501

__reuse_key_words: str = """
```json
{
    "conceptual_reuse": ["Tensorflow", "PyTorch", "JAX", "FLAX", "reproduce", "reimplementation", "recreate"],
    "adaptation_reuse": ["fine-tuning", "knowledge distillation", "parameter efficient fine tuning (PEFT)", "low-rank adaptation (LoRA)"],
    "deployment_reuse": ["vLLM", "inference server", "quantize", "ONNX", "MLIR", "LiteRT", "ExecuTorch", "TorchScript"],
}
```"""  # noqa: E501

USES_DL_PROMPT: COSTAR_SystemPrompt = COSTAR_SystemPrompt(
    tag="uses_dl",
    context="You are an AI model integrated into an automated pipeline that processes academic natural science papers into a machine-readable format. Your sole responsibility is to evaluate the paper’s content and determine whether the author uses deep learning models or methods. Your response will be consumed by downstream systems that require structured JSON.",
    objective="""Your task is to output only a JSON object containing key-value pairs, where:

- The value for the "result" key is a boolean based on whether the input text uses deep learning models or methods, and
- The key "prose" value is the most salient excerpt from the paper that shows concrete evidence of deep learning usage, or `null` if no deep learning methods are used.

No explanations or extra output are allowed.""",  # noqa: E501
    response="""Return only a JSON object of the form:

```json
{
    "result": boolean,
    "prose": string | null
}
```

Nothing else should ever be returned.""",
)

USES_PTMS_PROMPT: COSTAR_SystemPrompt = COSTAR_SystemPrompt(
    tag="uses_ptms",
    context="You are an AI model integrated into an automated pipeline that processes academic natural science papers into a machine-readable format. Your sole responsibility is to evaluate the paper's content and determine whether the authors use pre-trained deep learning models (PTMs). Your response will be consumed by downstream systems that require a structured JSON format.",  # noqa: E501
    objective="""Your task is to output only a JSON object containing key-value pairs, where:

- The value for the "result" key is a boolean based on whether the input text indicates the use of pre-trained deep learning models, and
- The key "prose" value is the most salient excerpt from the paper that shows concrete evidence of pre-trained model usage, or an `null` if no pre-trained deep learning models are used.

No explanations or extra output are allowed.""",  # noqa: E501
    response="""Return only a JSON object of the form:

```json
{
    "result": boolean,
    "prose": string | null
}
```

Nothing else should ever be returned.""",
)

IDENTIFY_PTMS_PROMPT: COSTAR_SystemPrompt = COSTAR_SystemPrompt(
    tag="identify_ptms",
    context=f"""You are an AI model integrated into an automated pipeline that processes academic natural science papers into a machine-readable format. Your sole responsibility is to evaluate the paper's content and determine what pre-trained deep learning models the authors use. Your response will be consumed by downstream systems that require a structured JSON format.

The following JSON object contains a reference list of known pre-trained deep learning models categorized by modality. Use this as a guide, but identify any PTM mentioned in the paper even if it does not appear on this list:

{__model_json_object}

However, not all papers use these models. In the event that the paper does not use one of these PTMs, identify the PTM.
""",  # noqa: E501
    objective="""Your task is to output only an array of JSON objects containing key-value pairs, where:

- The key "model" value is a string of the pre-trained deep learning model's name as it appears in the source text, and
- The key "prose" value is the most salient excerpt from the paper that shows concrete evidence of the pre-trained model's usage.

No explanations or extra output are allowed.""",  # noqa: E501
    response="""Return only an array of JSON objects of the form:

```json
[
    {
        "model": string,
        "prose": string
    },

    ...
]
```

Nothing else should ever be returned.""",
)

IDENTIFY_PTM_REUSE_PROMPT: COSTAR_SystemPrompt = COSTAR_SystemPrompt(
    tag="identify_ptm_reuse",
    context=f"""You are an AI model integrated into an automated pipeline that processes academic natural science papers into a machine-readable format. Your sole responsibility is to evaluate the paper's content and determine the form and classification of reuse of pre-trained deep learning models that the authors use in their methodology. Your response will be consumed by downstream systems that require a structured JSON format.

The following JSON object contains a reference list of known pre-trained deep learning models categorized by modality. Use this as a guide, but identify any PTM mentioned in the paper even if it does not appear on this list:

{__model_json_object}

Pre-trained deep learning reuse methods can be classified as one of the following patterns:

- Conceptual Reuse: Replicate and reengineer the algorithms, model architectures, or other concepts described in academic literature and similar sources, integrating the replication into new projects. An engineer might do this because of licensing issues or if they are required to use a particular deep learning framework, but are reusing ideas previously realized in another deep learning framework. This paradigm is related to Sommerville's notion of abstraction reuse, where an engineer reuses knowledge but not code directly. This paradigm is also related to reproducibility in the scientific sense, since an engineer independently confirms the reported results of a proposed technique.
- Adaptation Reuse: Leverage existing pre-trained deep learning models and adapt them to solve different learning tasks. An engineer might do this using several techniques, such as transfer learning or knowledge distillation. This form of reuse is suitable if a publicly available implementation of an appropriate model (such as a pre-trained deep learning model) is available. This paradigm is related to Sommerville's notion of object/component reuse, since an engineer must identify existing models suited for a purpose and then customize them for a different task.
- Deployment Reuse: Convert and deploy pre-trained deep learning models in different computational environments and frameworks. This pattern of reuse is suitable if there is a perfect fit for the engineer's needs, viz., a pre-trained deep learning model trained on the engineer's desired task (e.g., demonstrating proof of concept in a hackathon). This paradigm is related to Sommerville's notion of system reuse, since an engineer is reusing an entire model (including its training) and deploying it in the appropriate context. Deployment often requires the conversion of a pre-trained deep learning model from one representation to another, followed by compilation to optimize it for hardware.

Reusing pre-trained deep learning models can be implemented with many methods, but the following are some key words commonly associated with each reuse classification that you can reference in your analysis:

{__reuse_key_words}""",  # noqa: E501
    objective="""Your task is to output only an array of JSON objects containing key-value pairs, where:

- The key "model" value is a string of the pre-trained deep learning model's name as stated in the prose,
- The key "form" value is a string of a brief description of the specific reuse action (e.g., "Fine-tuned on 10k protein sequences"),
- The key "classification" value is a string of either "conceptual_reuse", "adaptation_reuse", or "deployment_reuse" that best classifies the pre-trained deep learning model reuse form, and
- The key "prose" value is the most salient excerpt from the paper providing concrete evidence of this specific reuse pattern.

No explanations or extra output are allowed.""",  # noqa: E501
    response="""Return only an array of JSON objects of the form:

```json
[
    {
        "model": string,
        "form": string,
        "classification": string,
        "prose": string
    },

    ...
]
```

Nothing else should ever be returned.""",
)

IDENTIFY_PTM_IMPACT_IN_SCIENTIFIC_PROCESS: COSTAR_SystemPrompt = COSTAR_SystemPrompt(
    tag="identify_ptm_impact",
    context=f"""You are an AI model integrated into an automated pipeline that processes academic natural science papers into a machine-readable format. Your sole responsibility is to evaluate the paper's content and determine which step of the scientific method they serve. Your response will be consumed by downstream systems that require a structured JSON format.

Use this as a guide, but identify any PTM mentioned in the paper even if it does not appear on this list:

{__model_json_object}

The scientific method is made up of several steps. These include:

- "Observation": The first involves making an identification of a phenomenon in the natural world,
- "Hypothesis": The second involves creating a testable explanation,
- "Background": The third requires conducting a background review and identifying relevant previous works,
- "Test": The fourth step is to experiment and challenge the hypothesis, and
- "Analysis": The fifth step reviews the data generated from the test and iterates upon challenging the hypothesis.""",  # noqa: E501
    objective="""Your task is to output only an array of JSON objects containing key-value pairs, where:

- The key "model" value is a string of the pre-trained deep learning model's name as stated in the prose,
- The key "step" value is the most probable step in the scientific process that the pre-trained deep learning model is written for, and
- The key "prose" value is the most salient excerpt from the paper that shows concrete evidence of the pre-trained model's usage.

No explanations or extra output are allowed.""",  # noqa: E501
    response="""Return only an array of JSON objects of the form:

```json
[
    {
        "model": string,
        "step": string,
        "prose": string
    },
    ...
]
```

Nothing else should ever be returned.""",
)
