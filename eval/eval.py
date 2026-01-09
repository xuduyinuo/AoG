import argparse
import json
from typing import List, Optional

from utils import (
    extract_ground_truth,
    extract_predictions,
    is_prediction_correct,
    read_output,
)
from utils import write_output


def evaluate_predictions(
    records: List[dict],
    treat_missing_gt_as_correct: bool = False,
    save_incorrect_path: Optional[str] = None,
    output_file: Optional[str] = None,
) -> None:
    total = len(records)
    if total == 0:
        print("No prediction records found.")
        return

    correct = 0
    evaluated = 0
    skipped_missing_gt = 0
    incorrect_details = []

    for record in records:
        ground_truth = extract_ground_truth(record)
        predictions = extract_predictions(record)

        # Skip or treat as correct when ground_truth is empty
        if not ground_truth:
            if treat_missing_gt_as_correct:
                correct += 1
                record["correct"] = True
            else:
                skipped_missing_gt += 1
                record["correct"] = False
            continue

        evaluated += 1
        is_corr = is_prediction_correct(predictions, ground_truth)
        if is_corr:
            correct += 1
            record["correct"] = True
        else:
            record["correct"] = False
            q_text = record.get("question") or record.get("RawQuestion") or record.get("ID")
            incorrect_details.append({
                "question": q_text,
                "ground_truth": ground_truth,
                "predictions": predictions,
            })

    # Compute accuracy based on evaluated records only (excluding skipped)
    # Write annotated records back to the file if an output path was provided
    if output_file is not None:
        try:
            write_output(output_file, records)
            print(f"Wrote annotated records with 'correct' flags to {output_file}.")
        except Exception as exc:
            print(f"Failed to write annotated output to {output_file}: {exc}")

    if evaluated == 0:
        print(f"Total records: {total}")
        print(f"Evaluated records: 0")
        print(f"Skipped (missing ground truth): {skipped_missing_gt}")
        print("No evaluable records with ground truth. Nothing to report.")
        return

    accuracy = correct / evaluated
    print(f"Total records: {total}")
    print(f"Evaluated records: {evaluated}")
    print(f"Skipped (missing ground truth): {skipped_missing_gt}")
    print(f"Correct predictions: {correct}")
    print(f"Incorrect predictions: {evaluated - correct}")
    print(f"Exact match accuracy: {accuracy:.4f}")

    if incorrect_details:
        print("\nIncorrect samples (all):")
        for idx, item in enumerate(incorrect_details, start=1):
            q = item.get("question")
            gt = ", ".join(item.get("ground_truth") or [])
            pr = ", ".join(item.get("predictions") or [])
            print(f"[{idx}] Question: {q}")
            print(f"     Ground truth: {gt}")
            print(f"     Prediction  : {pr}")

    # Optionally save incorrect samples to a JSON file
    if save_incorrect_path is not None:
        try:
            with open(save_incorrect_path, "w", encoding="utf-8") as f:
                json.dump(incorrect_details, f, ensure_ascii=False, indent=2)
            print(f"Saved incorrect samples to {save_incorrect_path} ({len(incorrect_details)} items).")
        except Exception as exc:
            print(f"Failed to save incorrect samples to {save_incorrect_path}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictions against ground-truth answers in a JSONL file.")
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to the prediction JSONL file produced by the model.",
    )
    parser.add_argument(
        "--treat-missing-gt-as-correct",
        action="store_true",
        help="When a record has empty ground_truth, treat it as correct instead of skipping.",
    )
    parser.add_argument(
        "--save-incorrect",
        type=str,
        default=None,
        help="Optional path to save all incorrect samples as a JSON file.",
    )
    args = parser.parse_args()

    records = read_output(args.output_file)
    evaluate_predictions(
        records,
        treat_missing_gt_as_correct=args.treat_missing_gt_as_correct,
        save_incorrect_path=args.save_incorrect,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()


