import inspect
import logging
import os

import jsonlines
import torch
from transformers import Trainer
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.utils import is_peft_available

log = logging.getLogger(__name__)


class DynamicTrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        How the loss is computed by Trainer.
        By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        # === injection ===
        ids = inputs.pop("id", None)
        outputs = model(**inputs)
        if model.training:
            self._write_dynamics(
                logits=outputs.logits, labels=inputs["labels"], ids=ids
            )
        # === /injection ===

        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if is_peft_available() and isinstance(model, PeftModel):  # noqa
                model_name = unwrap_model(model.base_model)._get_name()
            else:
                model_name = unwrap_model(model)._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, \
                        the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named as label or label_ids, the default data collator handles that.
            self._signature_columns += list(
                set(["label", "label_ids", "id"] + self.label_names)
            )

    def _write_dynamics(
        self, logits: torch.Tensor, labels: torch.Tensor, ids: torch.Tensor
    ):
        output_dir = self.args.output_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        epoch_file_name = os.path.join(output_dir, "result.jsonl")
        ids = ids.tolist()
        logits = logits.tolist()
        labels = labels.tolist()

        with jsonlines.open(epoch_file_name, "a") as writer:
            for id_sample, _logits, label in zip(ids, logits, labels):
                writer.write(
                    {
                        "id_sample": id_sample,
                        "logits": _logits,
                        "gold": label,
                        "epoch": int(self.state.epoch),
                    }
                )
