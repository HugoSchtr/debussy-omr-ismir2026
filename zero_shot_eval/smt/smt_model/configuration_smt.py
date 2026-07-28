from transformers import PretrainedConfig

class SMTConfig(PretrainedConfig):
    model_type = "SMT"

    def __init__(self, maxh=3508, maxw=2480, maxlen=1512, out_categories=2512, padding_token=0, 
                 in_channels=1, w2i={}, i2w={}, out_dir="out_smt", 
                 d_model=256, dim_ff=256, num_dec_layers=8, attn_heads=4,
                 use_flash_attn=False, _attn_implementation_internal=None, 
                 _experts_implementation_internal=None, **kwargs):
        super().__init__(**kwargs)
        
        self.architectures = ["SMT"]
        self.maxh = maxh
        self.maxw = maxw
        self.maxlen = maxlen
        self.out_categories = out_categories
        self.padding_token = padding_token
        self.in_channels = in_channels
        
        # Convert dictionary keys to integers (JSON stores them as strings)
        self.w2i = {k: int(v) if isinstance(v, str) and v.isdigit() else v 
                    for k, v in w2i.items()}
        self.i2w = {int(k) if isinstance(k, str) and k.lstrip('-').isdigit() else k: v 
                    for k, v in i2w.items()}
        
        self.out_dir = out_dir
        self.d_model = d_model
        self.dim_ff = dim_ff
        self.num_attn_heads = attn_heads
        self.num_dec_layers = num_dec_layers
        self.use_flash_attn = use_flash_attn
        self._attn_implementation_internal = _attn_implementation_internal
        self._experts_implementation_internal = _experts_implementation_internal

