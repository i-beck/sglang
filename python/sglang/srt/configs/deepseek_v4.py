from typing import Dict, List

from transformers import PretrainedConfig


class DeepSeekV4Config(PretrainedConfig):
    model_type = "deepseek_ref"

    def __init__(
        self,
        architectures=None,
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=0,
        eos_token_id=1,
        ep_size=1,
        first_k_dense_replace=0,
        hidden_act="silu",
        hidden_size=4096,
        index_head_dim=128,
        index_n_heads=64,
        index_topk=512,
        initializer_range=0.02,
        intermediate_size=2048,
        kv_lora_rank=512,
        max_position_embeddings=65536,
        moe_intermediate_size=2048,
        moe_layer_freq=1,
        n_group=8,
        n_routed_experts=256,
        n_shared_experts=1,
        norm_topk_prob=True,
        num_attention_heads=64,
        num_experts_per_tok=6,
        num_hidden_layers=43,
        num_key_value_heads=1,
        q_lora_rank=1024,
        qk_nope_head_dim=448,
        qk_rope_head_dim=64,
        quantization_config=None,
        rms_norm_eps=1e-6,
        rope_scaling=None,
        rope_theta=10000,
        routed_scaling_factor=1.5,
        scoring_func="sqrtsoftplus",
        tie_word_embeddings=False,
        topk_group=8,
        topk_method="noaux_tc",
        use_cache=True,
        v_head_dim=512,
        vocab_size=129280,
        o_lora_rank=1024,
        o_groups=8,
        window_size=128,
        compress_rope_theta=40000,
        compress_ratios=None,
        n_hash_layers=3,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        **kwargs,
    ):
        self.architectures = architectures or []
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.ep_size = ep_size
        self.first_k_dense_replace = first_k_dense_replace
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.index_topk = index_topk
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.kv_lora_rank = kv_lora_rank
        self.max_position_embeddings = max_position_embeddings
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_layer_freq = moe_layer_freq
        self.n_group = n_group
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.norm_topk_prob = norm_topk_prob
        self.num_attention_heads = num_attention_heads
        self.num_experts_per_tok = num_experts_per_tok
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.quantization_config = quantization_config or {}
        self.rms_norm_eps = rms_norm_eps
        self.rope_scaling = rope_scaling or {}
        self.rope_theta = rope_theta
        self.routed_scaling_factor = routed_scaling_factor
        self.scoring_func = scoring_func
        self.topk_group = topk_group
        self.topk_method = topk_method
        self.use_cache = use_cache
        self.v_head_dim = v_head_dim
        self.vocab_size = vocab_size
        self.o_lora_rank = o_lora_rank
        self.o_groups = o_groups
        self.window_size = window_size
        self.compress_rope_theta = compress_rope_theta
        self.compress_ratios = compress_ratios or []
        self.n_hash_layers = n_hash_layers
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
