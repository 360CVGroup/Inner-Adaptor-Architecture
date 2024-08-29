from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..iaa_arch import IAAMetaModel, IAAMetaForCausalLM



from .modeling_llama_iaa import LlamaModel, LlamaForCausalLM



class IAAConfig(LlamaConfig):
    model_type = "IAA"


class IAALlamaModel(IAAMetaModel, LlamaModel):
    config_class = IAAConfig

    def __init__(self, config: LlamaConfig):
        super(IAALlamaModel, self).__init__(config)


class IAALlamaForCausalLM(LlamaForCausalLM, IAAMetaForCausalLM):
    config_class = IAAConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)

        config._attn_implementation = "flash_attention_2"
        self.model = IAALlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.lm_head_condtion = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.lm_head_condtion_grounding = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.post_init()

    def get_model(self):
        return self.model


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        task_type = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images, task_type)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict = return_dict,
            task_type=task_type,
        )

        hidden_states = outputs[0]

        if task_type == "MM":
            logits = self.lm_head_condtion(hidden_states)
        elif task_type == "G":
            logits = self.lm_head_condtion_grounding(hidden_states)
        else:
            logits = self.lm_head(hidden_states)
        

        loss = None
        assert labels is None

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        
        # print(attention_mask)
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
                "task_type": kwargs.get("task_type", "Text"),
            }
        )
        return model_inputs

AutoConfig.register("IAA", IAAConfig)
AutoModelForCausalLM.register(IAAConfig, IAALlamaForCausalLM)





