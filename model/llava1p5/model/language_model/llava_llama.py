#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

from torch.nn import CrossEntropyLoss
import copy

class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    # def process_embeds(self, input_ids, inputs_embeds, correct_emb):
    #     # 1. 检查inputs_embeds和raw_input_ids第二个维度的差值是否为575
    #     assert inputs_embeds.shape[1] - input_ids.shape[1] == 575
        
    #     # 2. 检查raw_input_ids中的32000的数量是否与correct_emb的形状匹配
    #     count_32000 = (input_ids == 32000).sum(dim=1).tolist()
    #     assert all(count % 2 == 0 for count in count_32000)
        
    #     # for i, count in enumerate(count_32000):
    #     #     assert count == len(correct_emb[i]) * 2
        
    #     # 3. 找到32000的位置，将前1/2的位置替换为correct_emb中的对应值
    #     for batch_idx in range(input_ids.shape[0]):
    #         indices = torch.nonzero(input_ids[batch_idx] == 32000).flatten().tolist()
    #         num_32000 = len(indices)
            
    #         half_num_32000 = num_32000 // 2
    #         for idx, correct in zip(indices[:half_num_32000], correct_emb[batch_idx]):
    #             inputs_embeds[batch_idx, idx + 575, :] = correct
        
    #     return inputs_embeds
    def process_embeds(self, input_ids, inputs_embeds, correct_emb):
        # 1. 检查inputs_embeds和raw_input_ids第二个维度的差值是否为575
        assert inputs_embeds.shape[1] - input_ids.shape[1] == 575
        
        # 2. 检查raw_input_ids中的32000的数量是否与correct_emb的形状匹配
        count_32000 = (input_ids == 32000).sum(dim=1).tolist()
        assert all(count % 2 == 0 for count in count_32000)
        
        # for i, count in enumerate(count_32000):
        #     assert count == len(correct_emb[i]) * 2
        
        # 3. 找到32000的位置，将前1/2的位置替换为correct_emb中的对应值
        half_correct_num = 0
        for batch_idx in range(input_ids.shape[0]):
            indices = torch.nonzero(input_ids[batch_idx] == 32000).flatten().tolist()
            num_32000 = len(indices)
            
            half_num_32000 = num_32000 // 2
            for idx, correct in zip(indices[:half_num_32000], correct_emb[half_correct_num : half_correct_num + half_num_32000]):
                inputs_embeds[batch_idx, idx + 575, :] = correct
            half_correct_num += half_num_32000
        assert half_correct_num*2 == sum(count_32000)
        return inputs_embeds

    def process_embeds_inference(self, input_ids, inputs_embeds, correct_emb):
        # 1. 检查inputs_embeds和raw_input_ids第二个维度的差值是否为575
        assert inputs_embeds.shape[1] - input_ids.shape[1] == 575
                
        # 3. 找到32000的位置，将前1/2的位置替换为correct_emb中的对应值
        for batch_idx in range(input_ids.shape[0]):
            indices = torch.nonzero(input_ids[batch_idx] == 32000).flatten().tolist()
            indices = indices[:len(correct_emb)]
            
            for idx, correct in zip(indices, correct_emb):
                inputs_embeds[batch_idx, idx + 575, :] = correct
        return inputs_embeds
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        correct_emb: Optional[List[torch.FloatTensor]] = None,
        return_dict: Optional[bool] = None,
        if_inference: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        if correct_emb is not None:
            raw_input_ids  = copy.deepcopy(input_ids)

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if correct_emb is not None and if_inference is None:
            inputs_embeds = self.process_embeds(raw_input_ids, inputs_embeds, correct_emb)
        elif correct_emb is not None and if_inference is not None:
            inputs_embeds = self.process_embeds_inference(raw_input_ids, inputs_embeds, correct_emb)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        if self.training:
            output_hidden_states = outputs.hidden_states
        else:
            output_hidden_states = hidden_states

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=output_hidden_states, #outputs.hidden_states,
            attentions=outputs.attentions,
        )


        # return super().forward(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     past_key_values=past_key_values,
        #     inputs_embeds=inputs_embeds,
        #     labels=labels,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict
        # )

    # def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
    #     images = kwargs.pop("images", None)
    #     _inputs = super().prepare_inputs_for_generation(
    #         input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
    #     )
    #     if images is not None:
    #         _inputs['images'] = images
    #     return _inputs

    # def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, images=None, **kwargs):
    #     # images = kwargs.pop("images", None)
    #     _inputs = super().prepare_inputs_for_generation(
    #         input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
    #     )
    #     if images is not None:
    #         _inputs['images'] = images
    #     return _inputs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        images=None,
        correct_emb=None,
        if_inference=None,
        **kwargs
    ):
        past_key_values=None
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
                "images": images,
                "correct_emb":correct_emb,
                "if_inference":if_inference,
            }
        )
        return model_inputs

AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
