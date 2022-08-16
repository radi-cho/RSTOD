import copy

import torch
from torch import nn
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput

class T5WithSpan(T5ForConditionalGeneration):
    def __init__(self, config, num_span, consistency_task, max_seq_len, expected_vocab_size):
        super(T5WithSpan, self).__init__(config)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False

        self.resp_decoder = type(self.decoder)(decoder_config, self.shared)
        self.resp_lm_head = type(self.lm_head)(
            config.d_model, config.vocab_size, bias=False)

        self.max_seq_len = max_seq_len
        self.expected_vocab_size = expected_vocab_size

        if consistency_task:
            self.consistency_head = torch.nn.Linear(self.max_seq_len * expected_vocab_size, 1)

        self.dropout = nn.Dropout(config.dropout_rate)


    def initialize_additional_decoder(self):
        decoder_config = copy.deepcopy(self.config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False

        self.resp_decoder = type(self.decoder)(decoder_config, self.shared)
        self.resp_lm_head = type(self.lm_head)(
            self.config.d_model, self.config.vocab_size, bias=False)

        self.resp_decoder.load_state_dict(self.decoder.state_dict())
        self.resp_lm_head.load_state_dict(self.lm_head.state_dict())

    def initialize_weights(self, modules):
        for module in modules:
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def predict_span(self, encoder_hidden_states, attention_mask, span_labels=None):
        span_loss, pred_spans, span_logits = 0, None, None

        return span_loss, pred_spans, span_logits

    def prepare_inputs_for_generation(self, input_ids,
                                      past=None, attention_mask=None,
                                      use_cache=None, encoder_outputs=None,
                                      **kwargs):
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {"decoder_input_ids": input_ids,
                "past_key_values": past,
                "encoder_outputs": encoder_outputs,
                "attention_mask": attention_mask,
                "use_cache": use_cache,
                "decoder_type": kwargs.get("decoder_type")}

    def forward(self,
                input_ids=None,
                attention_mask=None,
                decoder_input_ids=None,
                encoder_outputs=None,
                past_key_values=None,
                inputs_embeds=None,
                decoder_inputs_embeds=None,
                span_labels=None,
                lm_labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                encoder_only=None,
                span_task=None,
                decoder_type=None,
                consistency_task=False,
                tau=1):

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        span_loss, pred_spans, span_logits = 0, None, None

        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           inputs_embeds=inputs_embeds,
                                           return_dict=return_dict)

            if return_dict:
                encoder_hidden_states = encoder_outputs.last_hidden_state
            else:
                encoder_hidden_states = encoder_outputs[0]

            hs = encoder_hidden_states * (self.model_dim ** -0.5)

            if span_task:
                span_loss, pred_spans, span_logits = self.predict_span(
                    hs, attention_mask, span_labels)

        else:
            if isinstance(encoder_outputs, tuple):
                encoder_hidden_states = encoder_outputs[0]
            else:
                encoder_hidden_states = encoder_outputs.last_hidden_state

        if encoder_only:
            return (span_loss, pred_spans, span_logits), encoder_outputs

        if lm_labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = self._shift_right(lm_labels)

        if decoder_type == "resp":
            decoder = self.resp_decoder
            lm_head = self.resp_lm_head
        else:
            decoder = self.decoder
            lm_head = self.lm_head

        if past_key_values is not None:
            assert lm_labels is None, "Decoder should not use cached key value states when training"
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        decoder_outputs = decoder(input_ids=decoder_input_ids,
                                  inputs_embeds=decoder_inputs_embeds,
                                  past_key_values=past_key_values,
                                  encoder_hidden_states=encoder_hidden_states,
                                  encoder_attention_mask=attention_mask,
                                  use_cache=use_cache,
                                  return_dict=return_dict)

        sequence_output = decoder_outputs[0]

        sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = lm_head(sequence_output)

        lm_loss = None
        if lm_labels is not None:
            lm_loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            lm_loss = lm_loss_fct(
                lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
        
        consistency_loss = None
        if consistency_task and not return_dict:
            batch_size = lm_labels.shape[0]

            if batch_size - 1 > 0:
                consistency_loss_fct = nn.BCEWithLogitsLoss()
                generated_hard = torch.nn.functional.gumbel_softmax(lm_logits, dim=2, tau=tau, hard=True)
                generated_soft = torch.nn.functional.gumbel_softmax(lm_logits, dim=2, tau=tau)
                generated = generated_hard - generated_soft.detach() + generated_soft

                padding = torch.nn.functional.one_hot(torch.zeros([generated.shape[0], self.max_seq_len - generated.shape[1]], dtype=torch.long, device=generated.device), num_classes=generated.shape[2])
                generated = torch.cat((generated, padding), 1)
                truth = torch.nn.functional.one_hot(lm_labels, num_classes=generated.shape[2]).to(dtype=torch.float)
                truth = torch.cat((truth, padding), 1)

                generated = generated.reshape(generated.shape[0], self.expected_vocab_size * self.max_seq_len)
                truth = truth.reshape(truth.shape[0], self.expected_vocab_size * self.max_seq_len)

                gen_logits = self.consistency_head(generated)
                gen_loss = consistency_loss_fct(gen_logits, torch.zeros([gen_logits.shape[0], 1], device=gen_logits.device))
                truth_logits = self.consistency_head(truth)
                truth_loss = consistency_loss_fct(truth_logits, torch.ones([truth_logits.shape[0], 1], device=truth_logits.device))

                consistency_loss = gen_loss + truth_loss
            else:
                consistency_loss = torch.tensor(0, dtype=torch.float).to(lm_labels.device)

        # Training
        if not return_dict:
            pred_lm = torch.argmax(lm_logits, dim=-1)
            outputs = (lm_loss, pred_lm,) + \
                (span_loss, pred_spans, span_logits, encoder_hidden_states, consistency_loss) + \
                    decoder_outputs[1:]

        # Inference
        else:
            outputs = Seq2SeqLMOutput(
                loss=lm_loss,
                logits=lm_logits,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs[1] if len(
                    encoder_outputs) > 1 else None,
                encoder_attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None)

        return outputs


class T5WithTokenSpan(T5WithSpan):
    def __init__(self, config, num_span, consistency_task, max_seq_len, expected_vocab_size):
        super(T5WithTokenSpan, self).__init__(config, num_span, consistency_task, max_seq_len, expected_vocab_size)

        self.num_span_labels = num_span + 2
        self.span_head = nn.Linear(config.d_model, self.num_span_labels)

        self.initialize_weights([self.span_head])

    def predict_span(self, encoder_hidden_states, attention_mask, span_labels=None):
        span_head = self.span_head.to(encoder_hidden_states.device)

        span_logits = span_head(encoder_hidden_states)

        pred_spans = torch.argmax(span_logits, dim=-1)

        span_loss = 0
        if span_labels is not None:
            span_loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            span_loss = span_loss_fct(
                span_logits.view(-1, self.num_span_labels), span_labels.view(-1))

        return span_loss, pred_spans, span_logits
