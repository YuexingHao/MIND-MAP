#!/usr/bin/env python3
"""
Generate an answer summary CSV given a set of MCQ questions, with automatic resume support.
The input CSV should have the following columns:
    - question: the question text
    - options: the options for the question, separated by newlines
    - correct_answer: the correct answer for the question (A-J)

The output CSV will have the following columns:
    - QA_ID: the question ID (e.g. "Merge Q1")
    - Extracted_Answer: the answer extracted by the model
    - Raw_Response: the raw response from the model

Example:
    python generate_mcq_answers.py \
        --csv_path merged_llm_4k_questions.csv \
        --model_id meta-llama/Llama-3.1-70B-Instruct \
        --dtype float16 \
        --load_in_8bit \
        --device_map cuda:0 \
        --output_dir outputs/llama70b_answers \
        --start_id 1 \
        --end_id 700 \
        --flush_every 25 > logs/answers/llama70b_answers_1_700.log 2>&1 &

    python generate_mcq_answers.py \
        --csv_path merged_llm_4k_questions.csv \
        --model_id meta-llama/Llama-3.1-70B-Instruct \
        --dtype float16 \
        --load_in_8bit \
        --device_map cuda:0 \
        --output_dir outputs/llama70b_answers \
        --start_id 701 \
        --flush_every 25 > logs/answers/llama70b_answers_701_end.log 2>&1 &
"""
import argparse
import os
import re
from pathlib import Path
from typing import List, Set
import ast

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# --------------------------------------------------------------------------- #
#                               argument parsing                              #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM attribution generator")
    parser.add_argument(
        "--csv_path",
        default="merged_llm_4k_questions.csv",
        help="Input CSV with at least columns `context` and `question`",
    )
    parser.add_argument(
        "--model_id",
        default="meta-llama/Llama-3.1-70B-Instruct",
        help="🤗 model repo or local path",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Torch dtype to load the model with (ignored if --load_in_8bit is set)",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load the model with 8-bit weight-only quantization (bitsandbytes).",
    )
    parser.add_argument(
        "--device_map",
        default="auto",
        help=(
            "Device placement strategy.  Common values:  "
            "`auto` (HF will shard), "
            "`cuda:0` (entire model on a single GPU), "
            "`balanced_low_0`, `sequential`, or `cpu`.\n"
            "See https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained"  # noqa: E501
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="llama70b_attribution_scores",
        help="Folder where per-question CSVs and the running summary are saved",
    )
    parser.add_argument(
        "--flush_every",
        type=int,
        default=25,
        help="Flush buffered results to disk every N rows (safer restarts)",
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=3,
        help="Maximum number of attempts to extract an answer from the model",
    )
    parser.add_argument(
        "--start_id",
        type=int,
        default=1,
        help="1-based index of the first QA row to process (default: 1)",
    )
    parser.add_argument(
        "--end_id",
        type=int,
        default=None,
        help="1-based index of the last QA row to process, inclusive "
             "(omit to run through the end of the CSV)",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
#                              helper functions                               #
# --------------------------------------------------------------------------- #
_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def load_model_and_tokenizer(
    model_id: str,
    dtype: str,
    device_map: str,
    load_in_8bit: bool,
):
    """
    Load model & tokenizer, optionally with 8-bit weight-only quantisation.
    """
    model_kwargs = {"device_map": device_map}

    if load_in_8bit:
        # weight-only Int8 quantisation (bitsandbytes)
        bnb_cfg = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,       # HF defaults
            llm_int8_has_fp16_weight=False,
        )
        model_kwargs["quantization_config"] = bnb_cfg
        # keep matmuls in BF16/FP16 so we still pass torch_dtype
        model_kwargs["torch_dtype"] = _DTYPE_MAP.get(dtype, torch.bfloat16)
    else:
        model_kwargs["torch_dtype"] = _DTYPE_MAP[dtype]

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


def extract_answer(raw_response: str) -> str:
    """
    Extract the answer choice letter (A-J) from the model's textual reply.

    The model is *supposed* to wrap the answer like:
        <answer>Option B</answer>

    But we also try a couple of fall-back patterns, just in case.
    Returns
    -------
    letter : str
        The uppercase letter (A-J).

    Raises
    ------
    ValueError
        If no valid answer letter can be found.
    """

    patterns = [
        # Match the answer: <LETTER>
        r"Answer:\s*([A-J])",
        # That could fail if the model doesn't wrap the answer in <answer> tags
        # So we try a couple of fall-back patterns, just in case
        # 1) Well-formed tag, allowing optional brackets: <answer>Option [D]</answer>
        r"<answer>\s*Option\s*\[?\s*([A-J])\s*\]?\s*</answer>",
        # 2) Tag present but missing the "Option" word, allowing brackets: <answer>[D]</answer>
        r"<answer>\s*\[?\s*([A-J])\s*\]?\s*</answer>",
        # 3) Loose “Option X” anywhere, with optional brackets: “Option [D]”
        r"Option\s*\[?\s*([A-J])\s*\]?",
        # Option (C)   or   Option (F)</answer>
        r"Option\s*\(?\s*([A-J])\s*\)?",
        # answer is: $\boxed{I}$      (handles optional $ … $)
        r"answer\s+is\s*:?\s*\$?\\boxed\{\s*([A-J])\s*\}\$?",
        # The best answer is B.   Best answer is: C
        r"\bbest\s+answer\s+is\s*([A-J])\b",
        # The correct answer is: (E) …
        r"\bcorrect\s+answer\s+(?:is|:)\s*\(?\s*([A-J])\s*\)?",
    ]

    for pat in patterns:
        m = re.search(pat, raw_response, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).upper()

    # If we get here, nothing matched
    raise ValueError(
        "Could not extract an answer letter (A-J) from the model's response:\n"
        f"{raw_response[:200]}…"
    )


def flush_buffer(
    rows: List[pd.DataFrame], summary_path: Path, header_if_new: bool
) -> None:
    """Append accumulated results to the running summary CSV on disk."""
    if not rows:
        return
    df_out = pd.concat(rows, ignore_index=True)
    df_out.to_csv(
        summary_path,
        mode="a",
        header=header_if_new,
        index=False,
    )
    rows.clear()


# --------------------------------------------------------------------------- #
#                                    main                                     #
# --------------------------------------------------------------------------- #
def main() -> None:
    """
    Answer MCQs with an LLM, retrying up to `args.max_attempts` times
    when we fail to extract a letter (A-D) from the model's reply.

    ── Behaviour ─────────────────────────────────────────────────────────
    • If extraction succeeds on any attempt, we keep that answer and
      move on.
    • After  `args.max_attempts`  unsuccessful tries, we record None
    in the Extracted_Answer column.
    • The script can be resumed; already-processed QA_IDs are skipped.
    """
    args = parse_args()                                      # ← assumes --max-attempts in CLI
    os.makedirs(args.output_dir, exist_ok=True)
    # Give each run its own summary file
    range_tag = f"{args.start_id}_{args.end_id}"
    summary_path = Path(args.output_dir) / f"answers_summary_{range_tag}.csv"

    # ─────────────────────────── Model
    model, tokenizer = load_model_and_tokenizer(
        args.model_id, args.dtype, args.device_map, args.load_in_8bit
    )

    # ─────────────────────────── Data
    df = pd.read_csv(args.csv_path)

    # --------------------------- Optional range filtering (1-based, inclusive)
    #    e.g. --start_id 1 --end_id 2000  ➜  processes Merge Q1 … Merge Q2000
    #    Omitting --end_id means "process until the end of the CSV".
    start_id = max(args.start_id, 1)
    end_id = args.end_id if args.end_id is not None else len(df)
    df = df.iloc[start_id - 1 : end_id]

    print(
        f"📄 Loaded {len(df):,} rows from {args.csv_path} "
        f"(processing IDs {start_id}-{end_id})"
    )

    # Change column names if not already as expected (New_Sentences -> context, question_options -> question)
    if "New_Sentences" in df.columns:
        df.rename(columns={"New_Sentences": "context", "question_options": "question"}, inplace=True)
        print("Renamed columns to `context` and `question`.")
    elif "context" not in df.columns or "question" not in df.columns:
        raise ValueError("Input CSV must have columns `context` and `question`")
    else:
        print("Columns already as expected.")

    processed_ids: Set[str] = set()
    if summary_path.exists():
        processed_ids = set(pd.read_csv(summary_path)["QA_ID"])
        print(f"🔄 Resuming — {len(processed_ids):,} rows already done.")

    buffer: List[pd.DataFrame] = []
    header_written = summary_path.exists()

    # ─────────────────────────── Main loop
    for idx, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Answering MCQs",
        unit="qa",
    ):
        qa_id = f"Merge Q{idx + 1}"
        if qa_id in processed_ids:
            continue  # already done

        # handle missing values (e.g. some questions are missing context)
        context_text = row["context"] if isinstance(row["context"], str) else ""
        raw_context = row["context"]
        context_text = ""
        # if the context is a string representation of a list, join the list into a string
        if isinstance(raw_context, str):
            try:
                # Check if it's a string representation of a list
                context_list = ast.literal_eval(raw_context)
                if isinstance(context_list, list):
                    context_text = ' '.join(context_list)
                else:
                    context_text = raw_context  # Just use as-is if not a list
            except (ValueError, SyntaxError):
                context_text = raw_context  # Fallback if it's not parsable

        question = row["question"].strip() if isinstance(row["question"], str) else ""

        if "meta-llama" in args.model_id:
            prompt = (
                "You are given some context and a multiple-choice question. "
                "Select the most appropriate answer from the options provided.\n\n"
                f"{context_text}\n\n{question}"
                "Provide your response in the following format:\n<answer>Option [letter]</answer>"
            )
        else:
            prompt = (
                "You are a clinical reasoning assistant. You will receive a patient case summary "
                "and a multiple-choice question.\n\n"
                f"Patient Context:\n{context_text}\n\n"
                f"Question and Options:\n{question}\n\n"
                "Please select the single most appropriate answer. Respond only in the following format:\n\n"
                "Answer: <LETTER>"
            )

        try:
            answer_letter = None
            raw_response = ""  # keep the last response (useful for inspection)

            for attempt in range(1, args.max_attempts + 1):
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
                prompt_len = inputs["input_ids"].shape[-1]        # tokens that belong to the prompt

                deterministic = (attempt == 1)          # first pass → greedy

                gen_kwargs = dict(
                    max_new_tokens=2048,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=not deterministic,
                )
                if not deterministic:                   # only matters when we sample
                    gen_kwargs["temperature"] = 0.7

                with torch.no_grad():
                    outputs = model.generate(**inputs, **gen_kwargs) # shape: (1, prompt_len + output_len)

                raw_response = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True) # shape: (output_len,)

                try:
                    answer_letter = extract_answer(raw_response)
                    break                               # success – stop retrying
                except ValueError:
                    continue                            # try again (if attempts remain)

            # If all attempts failed, answer_letter stays None
            if answer_letter is None:
                print(f"⚠️  {qa_id}: reached max attempts with no valid answer.")

            # ----- Store the result
            result = pd.DataFrame(
                {
                    "QA_ID": [qa_id],
                    "Extracted_Answer": [answer_letter],
                    "Raw_Response": [raw_response],
                }
            )

            buffer.append(result)
            # print(f"✅ {qa_id}  (answer={answer_letter})")

            # ----- Periodic flush
            if len(buffer) >= args.flush_every:
                flush_buffer(buffer, summary_path, not header_written)
                header_written = True

        except Exception as exc:
            print(f"❌ Error processing {qa_id}: {exc}")

    # ─────────────────────────── Finish up
    flush_buffer(buffer, summary_path, not header_written)
    print("🎉 Finished!")


if __name__ == "__main__":
    main()
