from typing import List, Optional, Union

import numpy as np
import requests
import torch
from pyannote.audio import Pipeline
from torchaudio import functional as F
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_read
import torchaudio
import io

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

from pyannote.audio import Pipeline

diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization@2.1",
    use_auth_token="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",
)

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
)
from datasets import Audio, load_dataset

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


from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

quant_model = ORTModelForSequenceClassification.from_pretrained("services/models/model")
quant_tokenizer = AutoTokenizer.from_pretrained(
    "Venkatesh4342/distilbert-helpdesk-sentiment"
)


def classification(summerized_op, quant_model, quant_tokenizer):
    pipe = pipeline(
        "text-classification", model=quant_model, tokenizer=quant_tokenizer, device=0
    )
    res = pipe(summerized_op)
    return res


def Model_Inference(file):
    data = {}
    speech2text_output = speech2text_pipeline(file, diarization_pipeline, asr_pipeline)
    summerized_output = summerization_pipeline(speech2text_output, model_name)
    sentiment = classification(summerized_output, quant_model, quant_tokenizer)

    data.update(
        {
            "summary": summerized_output,
            "sentiment": sentiment[0]["label"],
            "score": sentiment[0]["score"],
        }
    )
    return data
