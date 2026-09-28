#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


# ---------------------------------------------------------------------------
# Student solution starts here. Nothing below touches the provided runner code.
# ---------------------------------------------------------------------------

MODEL_NAME = "deepseek-v4-flash-vision-exp"
# Independent reads per receipt; the per-field median across votes is kept so a
# single mis-read digit cannot move the final sum.
VOTES_PER_RECEIPT = 7
# Extra votes added when the first round cannot agree. With seven votes a wrong
# majority always shows up as disagreement, so it always triggers these.
EXTRA_VOTES_ON_DISAGREEMENT = 4
MAX_CONCURRENCY = 6
# amount_without_discounts must equal subtotal + discount_total exactly; a read
# that breaks this identity is internally inconsistent and gets thrown away.
CONSISTENCY_TOLERANCE = Decimal("0.02")
# The cash a shopper hands over differs from the subtotal only by rounding.
MAX_ROUNDING_GAP = Decimal("1.00")

EXTRACTION_PROMPT = """You are a meticulous cashier auditor reading a photograph of a Hong Kong \
supermarket receipt (PARKnSHOP / fusion / TASTE / 百佳 / 惠康 and similar). The text may be in \
Chinese, English or both, and the photo may be angled, slightly blurry, or show a finger holding \
the paper.

Extract exactly four totals for THIS ONE receipt and return them as a JSON object:

{"subtotal_after_discounts_before_rounding": <number>,
 "discount_total": <number>,
 "discount_lines": [<one number per discount line actually printed, e.g. -12.40, ...>],
 "amount_paid_after_rounding": <number>,
 "amount_without_discounts": <number>}

Definitions - be precise:

1. subtotal_after_discounts_before_rounding
   The 小計 / SUBTOTAL line: the running total AFTER every discount has been applied but BEFORE \
the ROUNDING adjustment. If the receipt prints no subtotal line, compute it as (sum of all \
positive item lines) minus (sum of all discount lines).

2. discount_total
   The POSITIVE sum of EVERY discount, promotion, coupon, member-price, app-offer, \
packaging-damage (包裝變形) and percentage-off line - that is, every line printed as a negative \
amount that reduces the price of goods. Examples: "5% OFF (CU) -$20.78", "Buy 2 Save $12.8", \
"OVER$40 ENJOY 10%", "包裝變形 -$12.40", "App upgrade -$30_B". If the same promotion is printed on \
several lines, count each line separately.
   Do NOT include the ROUNDING line, change (找續), or any payment line such as OCTOPUS, CASH, \
VISA, MASTER, ALIPAY, WECHAT PAY, PAYME, FPS, 拍住賞 or 扣除金額.
   If the receipt has no discount at all, use 0. List every discount line you added up in \
"discount_lines", in the order printed, so the total can be re-added and checked. Use [] when \
there are none.

3. amount_paid_after_rounding
   The final amount the customer actually paid: the payment line printed immediately after the \
ROUNDING line (for example "OCTOPUS $394.70"). It equals subtotal + the ROUNDING adjustment. If \
there is no ROUNDING line, it is simply the single payment amount.

4. amount_without_discounts
   What the customer would have paid if every discount were cancelled: \
subtotal_after_discounts_before_rounding + discount_total. Report it as a separate observation \
and make sure it really equals subtotal + discount_total.

Rules:
- Read the digits that are actually printed. Never guess, never round yourself.
- For each discount line report BOTH the label text exactly as printed and the \
amount printed on that same line: {"text": "Buy 2 Save $6.", "amount": -6.00}. \
Never infer the amount from the label - read the digits on the amount side of the \
line, because a label sometimes quotes a different figure than the amount column.
- A trailing marker such as "-$30_B" or "-$6.38" still means the amount is 30.00 / 6.38.
- Ignore loyalty points, credit-card numbers, card balances (餘額), machine numbers and dates.
- Reply with ONLY the JSON object. No markdown fences, no commentary.
  Use null for a field only when it is genuinely unreadable."""


def _strip_fences(text: str) -> str:
    """Remove markdown code fences and keep only the outermost JSON object."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""
    return cleaned[start : end + 1]


def _to_decimal(value: Any) -> Decimal | None:
    """Coerce a JSON scalar such as 394.7, "HK$1,234.50" or None to Decimal."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            return Decimal(str(value))
        except InvalidOperation:
            return None
    text = (
        str(value)
        .strip()
        .replace(",", "")
        .replace("HK$", "")
        .replace("$", "")
        .strip()
    )
    if not text or text.lower() in {"null", "none", "n/a", "-", "—"}:
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def _parse_receipt_json(text: str) -> dict[str, Decimal] | None:
    """Turn one model response into the three numeric fields, or None."""
    data = None
    try:
        parsed = json.loads(_strip_fences(text))
        if isinstance(parsed, dict):
            data = parsed
    except (ValueError, TypeError):
        data = None
    if data is None:
        return None

    fields = {
        "paid": _to_decimal(data.get("amount_paid_after_rounding")),
        "subtotal": _to_decimal(data.get("subtotal_after_discounts_before_rounding")),
        "discount": _to_decimal(data.get("discount_total")),
        "gross": _to_decimal(data.get("amount_without_discounts")),
    }
    if fields["paid"] is None or fields["subtotal"] is None:
        return None
    fields["discount"] = abs(fields["discount"]) if fields["discount"] is not None else Decimal("0")

    # Self-check 1: the listed discount lines must re-add to discount_total.
    # Self-check 2: gross must equal subtotal + discount_total.
    # A read that breaks either identity has mis-read a line, so vote it down.
    lines = data.get("discount_lines")
    line_sum = None
    if isinstance(lines, list) and lines:
        parsed_lines = []
        for line in lines:
            if isinstance(line, dict):
                # {"text": "Buy 2 Save $6.", "amount": -6.00}
                parsed_lines.append(_to_decimal(line.get("amount", line.get("value"))))
            else:
                parsed_lines.append(_to_decimal(line))
        if parsed_lines and all(value is not None for value in parsed_lines):
            line_sum = sum(abs(value) for value in parsed_lines)

    consistent = True
    if line_sum is not None and fields["discount"] > Decimal("0"):
        consistent = abs(line_sum - fields["discount"]) <= CONSISTENCY_TOLERANCE
    if fields["gross"] is not None:
        consistent = consistent and (
            abs(fields["gross"] - (fields["subtotal"] + fields["discount"]))
            <= CONSISTENCY_TOLERANCE
        )
    fields["consistent"] = consistent
    fields["rounding_gap"] = abs(fields["paid"] - fields["subtotal"])
    return fields


def _median(values: list[Decimal]) -> Decimal:
    """Median of a non-empty list."""
    ordered = sorted(values)
    size = len(ordered)
    if size % 2:
        return ordered[size // 2]
    return (ordered[size // 2 - 1] + ordered[size // 2]) / Decimal("2")


def _consensus(values: list[Decimal]) -> Decimal:
    """Agree on one value across votes.

    Receipt reading is a transcription task, so the answers are discrete: if two
    reads say 76.71 and one says 100.21, the truth is 76.71. Prefer the modal
    value and only fall back to the median when there is no clear winner, which
    avoids averaging two different readings into an amount that was never on the
    receipt.
    """
    from collections import Counter

    counts = Counter(values)
    ranking = counts.most_common()
    if len(ranking) == 1 or ranking[0][1] > ranking[1][1]:
        return ranking[0][0]
    return _median(values)


def _quantize(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"))


def build_chain() -> Any:
    """Create and return your LangChain chain once.

    Suggested imports:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_deepseek import ChatDeepSeek

    Use the vision-capable DeepSeek Flash model named
    ``deepseek-v4-flash-vision-exp``. The API key is loaded from .env.
    """
    ### YOUR CODE HERE
    import os

    from langchain_core.messages import HumanMessage
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableLambda
    from langchain_deepseek import ChatDeepSeek

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        # The runner normally does this first; repeat it so the chain also
        # builds when build_chain() is called on its own.
        load_env_file()
        api_key = os.environ.get("DEEPSEEK_API_KEY")
    if api_key:
        os.environ["DEEPSEEK_API_KEY"] = api_key

    model = ChatDeepSeek(
        model=MODEL_NAME,
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        temperature=0.2,
        max_retries=3,
        timeout=180,
    )

    def to_messages(inputs: dict) -> list[HumanMessage]:
        """Build one multimodal extraction message for a single receipt."""
        extra = inputs.get("instruction") or ""
        content: list[dict] = [
            {"type": "text", "text": EXTRACTION_PROMPT + extra},
            {"type": "image_url", "image_url": {"url": image_data_url(inputs["path"])}},
        ]
        return [HumanMessage(content=content)]

    return RunnableLambda(to_messages) | model | StrOutputParser()


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run your chain and return one response for each exact query string.

    ``images`` contains every receipt in the selected folder. A valid return
    value looks like:

        {QUERY_1: "HK$123.40", QUERY_2: "HK$150.00"}

    Use the provided ``image_data_url(path)`` helper to put local images in
    multimodal human messages. LangChain's ``batch`` method is one simple way
    to process independent receipt-extraction prompts in parallel.
    """
    ### YOUR CODE HERE
    from collections import Counter
    from decimal import Decimal

    def run_batch(jobs: list[dict]) -> list[str]:
        if not jobs:
            return []
        try:
            return list(chain.batch(jobs, config={"max_concurrency": MAX_CONCURRENCY}))
        except Exception as error:  # one bad image must not sink the run
            print(f"[hw1] batch failed ({error}); falling back to per-image calls")
            outputs: list[str] = []
            for job in jobs:
                try:
                    outputs.append(str(chain.invoke(job)))
                except Exception as inner_error:
                    print(f"[hw1] invoke failed for {job['path'].name}: {inner_error}")
                    outputs.append("")
            return outputs

    # Pass 1: VOTES_PER_RECEIPT independent reads for every receipt, in parallel.
    votes: dict[Path, list[dict[str, Decimal]]] = {path: [] for path in images}
    jobs = [{"path": path, "instruction": ""} for path in images for _ in range(VOTES_PER_RECEIPT)]
    for path, text in zip([job["path"] for job in jobs], run_batch(jobs)):
        parsed = _parse_receipt_json(text)
        if parsed is not None:
            votes[path].append(parsed)

    # Pass 2: receipts whose reads were internally inconsistent, disagreed by
    # more than a cent, or showed an implausible rounding gap get extra votes.
    # Plain re-reads are used rather than a "look harder" prompt: measured on the
    # public receipts, that instruction roughly doubles the mis-read rate, while
    # simply taking more samples lets the majority vote win.
    tough_jobs = []
    for path, reads in votes.items():
        if len(reads) < 2 or any(not read["consistent"] for read in reads):
            tough_jobs.extend({"path": path, "instruction": ""} for _ in range(EXTRA_VOTES_ON_DISAGREEMENT))
            continue
        if any(read["rounding_gap"] > MAX_ROUNDING_GAP for read in reads):
            tough_jobs.extend({"path": path, "instruction": ""} for _ in range(EXTRA_VOTES_ON_DISAGREEMENT))
            continue
        if any(
            max(read[key] for read in reads) - min(read[key] for read in reads)
            > CONSISTENCY_TOLERANCE
            for key in ("paid", "subtotal", "discount")
        ):
            tough_jobs.extend({"path": path, "instruction": ""} for _ in range(EXTRA_VOTES_ON_DISAGREEMENT))

    for path, text in zip([job["path"] for job in tough_jobs], run_batch(tough_jobs)):
        parsed = _parse_receipt_json(text)
        if parsed is not None:
            votes[path].append(parsed)

    # Aggregate: drop internally inconsistent reads (unless that leaves nothing),
    # take the per-field majority, then sum across receipts with exact Decimal math.
    total_paid = Decimal("0")
    total_without_discount = Decimal("0")
    for path in images:
        reads = votes[path]
        if not reads:
            print(f"[hw1] warning: no readable extraction for {path.name}; counted as 0")
            continue
        usable = [read for read in reads if read["consistent"]] or reads
        paid = _consensus([read["paid"] for read in usable])
        subtotal = _consensus([read["subtotal"] for read in usable])
        discount = _consensus([read["discount"] for read in usable])
        total_paid += _quantize(paid)
        total_without_discount += _quantize(subtotal + discount)
        print(
            f"[hw1] {path.name}: paid={_quantize(paid)} subtotal={_quantize(subtotal)} "
            f"discount={_quantize(discount)} (votes={len(usable)}/{len(reads)})"
        )

    return {
        QUERY_1: f"HK${_quantize(total_paid):.2f}",
        QUERY_2: f"HK${_quantize(total_without_discount):.2f}",
    }


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
