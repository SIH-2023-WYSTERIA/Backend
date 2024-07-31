import os
import shutil

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import torch
from pyannote.audio import Pipeline
from torchaudio import functional as F
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_read
import gc

import torch
import time


device = "cuda" if torch.cuda.is_available() else "cpu"

from pyannote.audio import Pipeline

diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization@2.1",
    use_auth_token="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",
)
import torch
import onnxruntime as rt
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = rt.InferenceSession("yolo.onnx", providers=providers)
from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.onnxruntime import ORTQuantizer, ORTModelForSequenceClassification
from optimum.onnxruntime.configuration import AutoQuantizationConfig

from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
)
from datasets import Audio, load_dataset, Features, Value, ClassLabel

processor = WhisperProcessor.from_pretrained("Venkatesh4342/whisper-small-en-hi")
forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="english", task="transcribe"
)

tokenizer = WhisperTokenizer.from_pretrained(
    "Venkatesh4342/whisper-small-en-hi", language="english", task="transcribe"
)
model = WhisperForConditionalGeneration.from_pretrained(
    "Venkatesh4342/whisper-small-en-hi"
).to(device)
feature_extractor = WhisperFeatureExtractor.from_pretrained(
    "Venkatesh4342/whisper-small-en-hi"
)

asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model,
    feature_extractor=feature_extractor,
    tokenizer=tokenizer,
    device=device,
    generate_kwargs={"forced_decoder_ids": forced_decoder_ids},
)

sampling_rate = asr_pipeline.feature_extractor.sampling_rate


def preprocess(inputs):
    if isinstance(inputs, str):
        if inputs.startswith("http://") or inputs.startswith("https://"):
            inputs = requests.get(inputs).content
        else:
            with open(inputs, "rb") as f:
                inputs = f.read()

    if isinstance(inputs, bytes):
        inputs = ffmpeg_read(inputs, sampling_rate)

    if isinstance(inputs, dict):
        if not ("sampling_rate" in inputs and ("raw" in inputs or "array" in inputs)):
            raise ValueError(
                "When passing a dictionary to ASRDiarizePipeline, the dict needs to contain a "
                '"raw" key containing the numpy array representing the audio and a "sampling_rate" key, '
                "containing the sampling_rate associated with that array"
            )

        _inputs = inputs.pop("raw", None)
        if _inputs is None:
            inputs.pop("path", None)
            _inputs = inputs.pop("array", None)
        in_sampling_rate = inputs.pop("sampling_rate")
        inputs = _inputs
        if in_sampling_rate != sampling_rate:
            inputs = F.resample(
                torch.from_numpy(inputs), in_sampling_rate, sampling_rate
            ).numpy()

    if not isinstance(inputs, np.ndarray):
        raise ValueError(f"We expect a numpy ndarray as input, got `{type(inputs)}`")
    if len(inputs.shape) != 1:
        raise ValueError(
            "We expect a single channel audio input for ASRDiarizePipeline"
        )

    # diarization model expects float32 torch tensor of shape `(channels, seq_len)`
    diarizer_inputs = torch.from_numpy(inputs).float()
    diarizer_inputs = diarizer_inputs.unsqueeze(0)

    return inputs, diarizer_inputs


def speech2text_pipeline(input, diarization_pipeline, asr_pipeline):
    inputs, diarizer_inputs = preprocess(input)
    diarization_pipeline.to(torch.device(device))

    diarizer_output = diarization_pipeline(
        {"waveform": diarizer_inputs, "sample_rate": sampling_rate}, num_speakers=2
    )
    segments = []
    for segment, track, label in diarizer_output.itertracks(yield_label=True):
        segments.append(
            {
                "segment": {"start": segment.start, "end": segment.end},
                "track": track,
                "label": label,
            }
        )

    new_segments = []
    prev_segment = cur_segment = segments[0]

    for i in range(1, len(segments)):
        cur_segment = segments[i]
        if cur_segment["label"] != prev_segment["label"] and i < len(segments):
            new_segments.append(
                {
                    "segment": {
                        "start": prev_segment["segment"]["start"],
                        "end": cur_segment["segment"]["start"],
                    },
                    "speaker": prev_segment["label"],
                }
            )
            prev_segment = segments[i]

    new_segments.append(
        {
            "segment": {
                "start": prev_segment["segment"]["start"],
                "end": cur_segment["segment"]["end"],
            },
            "speaker": prev_segment["label"],
        }
    )

    asr_out = asr_pipeline(
        {"array": inputs, "sampling_rate": sampling_rate},
        chunk_length_s=30,
        return_timestamps=True,
    )

    transcript = asr_out["chunks"]
    end_timestamps = np.array([chunk["timestamp"][-1] for chunk in transcript])
    if end_timestamps[-1] == None:
        end_timestamps[-1] = new_segments[-1]["segment"]["end"]

    group_by_speaker = True

    segmented_preds = []
    for segment in new_segments:
        end_time = segment["segment"]["end"]
        try:
            upto_idx = np.argmin(np.abs(end_timestamps - end_time))
        except ValueError:
            continue
        if group_by_speaker:
            segmented_preds.append(
                {
                    "speaker": segment["speaker"],
                    "text": "".join(
                        [chunk["text"] for chunk in transcript[: upto_idx + 1]]
                    ),
                    "timestamp": (
                        transcript[0]["timestamp"][0],
                        transcript[upto_idx]["timestamp"][1],
                    ),
                }
            )
        else:
            for i in range(upto_idx + 1):
                segmented_preds.append({"speaker": segment["speaker"], **transcript[i]})
        transcript = transcript[upto_idx + 1 :]
        end_timestamps = end_timestamps[upto_idx + 1 :]

    conv = []
    for i, sub in enumerate(segmented_preds):
        if i % 2 == 0:
            conv.append(f"helpdesk:{sub['text']}")
        else:
            conv.append(f"customer:{sub['text']}")

    return conv


model_name = "Venkatesh4342/bart-samsum"


def summerization_pipeline(s2t_output, model_name):
    pipe = pipeline("summarization", model=model_name, device=device)
    pipe_out = pipe("\n".join(s2t_output))
    return pipe_out[0]["summary_text"]


quant_model = ORTModelForSequenceClassification.from_pretrained("Venkatesh4342/quantized-helpdesk")
quant_tokenizer = AutoTokenizer.from_pretrained("Venkatesh4342/quantized-helpdesk")


def classification(summerized_op, quant_model, quant_tokenizer):
    pipe = pipeline(
        "text-classification",
        model=quant_model,
        tokenizer=quant_tokenizer,
        device=device,
    )
    res = pipe(summerized_op)
    return res


def Model_Inference(file):
    start = time.time()
    data = {}
    speech2text_output = speech2text_pipeline(file, diarization_pipeline, asr_pipeline)
    summerized_output = summerization_pipeline(speech2text_output, model_name)
    sentiment = classification(summerized_output, quant_model, quant_tokenizer)

    data.update(
        {
            "conversation_transcript":speech2text_output,
            "summary": summerized_output,
            "sentiment": sentiment[0]["label"],
            "score": sentiment[0]["score"],
        }
    )
    end = time.time()
    print(end - start)
    torch.cuda.empty_cache()
    gc.collect()
    return data


def finetune(filepath):
    df = pd.read_csv(filepath, encoding="utf-8")
    train, validation = train_test_split(
        df, test_size=0.09, random_state=42, stratify=df["label"]
    )
    train.to_csv("train_help.csv", index=False)
    validation.to_csv("val_help.csv", index=False)

    class_names = ["Negative", "Neutral", "Positive"]
    ft = Features(
        {
            "text": Value(dtype="string", id=None),
            "label": ClassLabel(num_classes=3, names=class_names),
        }
    )
    dataset = load_dataset(
        "csv",
        data_files={"train": "train_help.csv", "validation": "val_help.csv"},
        features=ft,
    )

    model_ckpt = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True)

    helpdesk_encoded = dataset.map(tokenize, batched=True, batch_size=None)
    helpdesk_encoded.set_format(
        "torch", columns=["input_ids", "attention_mask", "label"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_labels = 3
    model = AutoModelForSequenceClassification.from_pretrained(
        model_ckpt, num_labels=num_labels
    ).to(device)

    config = AutoConfig.from_pretrained(
        model_ckpt,
        num_labels=len(class_names),
        id2label={i: label for i, label in enumerate(class_names)},
        label2id={label: i for i, label in enumerate(class_names)},
    )

    model.config = config

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        f1 = f1_score(labels, preds, average="weighted")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1}

    batch_size = 2
    logging_steps = len(helpdesk_encoded["train"]) // batch_size
    model_name = "Venkatesh4342/distilbert-helpdesk-sentiment"
    training_args = TrainingArguments(
        output_dir=model_name,
        num_train_epochs=6,
        learning_rate=2e-5,
        evaluation_strategy="steps",
        eval_steps=100,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        logging_steps=logging_steps,
        gradient_checkpointing=True,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=helpdesk_encoded["train"],
        eval_dataset=helpdesk_encoded["validation"],
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model("fine_tuned_model")

    onnx_model = ORTModelForSequenceClassification.from_pretrained(
        "fine_tuned_model", export=True
    )
    quantizer = ORTQuantizer.from_pretrained(onnx_model)
    dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    model_quantized_path = quantizer.quantize(
        save_dir="model",
        quantization_config=dqconfig,
    )
    from huggingface_hub import HfApi
    api = HfApi()

    api.upload_folder(
        folder_path="model",
        repo_id="Venkatesh4342/quantized-helpdesk",
        repo_type="model"
    )
    time.sleep(15)
    shutil.rmtree("fine_tuned_model")
    os.remove("train_help.csv")
    os.remove("val_help.csv")

