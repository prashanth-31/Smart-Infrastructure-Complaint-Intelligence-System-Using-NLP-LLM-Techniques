from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import spacy
from spacy.training import Example
from spacy.util import minibatch

from training.config import PATHS


def _load_annotations(path: Path) -> List[Tuple[str, List[Tuple[int, int, str]]]]:
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            sample = json.loads(line)
            text = sample["text"]
            entities = [(start, end, label) for start, end, label in sample.get("entities", [])]
            records.append((text, entities))
    return records


def _create_examples(nlp, records: Iterable[Tuple[str, List[Tuple[int, int, str]]]]) -> List[Example]:
    examples = []
    for text, entities in records:
        doc = nlp.make_doc(text)
        spans = [(start, end, label) for start, end, label in entities]
        example = Example.from_dict(doc, {"entities": spans})
        examples.append(example)
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune spaCy NER model")
    parser.add_argument("--annotations", default=str(PATHS.data_csv.parent / "ner_annotations.jsonl"), help="JSONL with text and entity spans")
    parser.add_argument("--base", default="en_core_web_sm", help="Base spaCy model to fine-tune")
    parser.add_argument("--n_iter", type=int, default=20, help="Training iterations")
    parser.add_argument("--output", default=str(PATHS.spacy_model_dir), help="Output directory for trained pipeline")
    args = parser.parse_args()

    annotations_path = Path(args.annotations)
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotations_path}")

    nlp = spacy.load(args.base)
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    records = _load_annotations(annotations_path)
    for _, entities in records:
        for _, _, label in entities:
            ner.add_label(label)

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.resume_training()
        examples = _create_examples(nlp, records)
        for itn in range(args.n_iter):
            losses = {}
            batches = minibatch(examples, size=8)
            for batch in batches:
                nlp.update(batch, sgd=optimizer, losses=losses)
            print(f"Iteration {itn+1}/{args.n_iter} - Losses: {losses}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(output_dir)
    print(f"Saved spaCy model to {output_dir}")


if __name__ == "__main__":
    main()
