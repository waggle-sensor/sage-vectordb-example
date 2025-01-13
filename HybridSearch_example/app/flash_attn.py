'''
For some reason Florence-2 requires flash_attention even with the CPU
this doesn't work because it wants CUDA :( 

So to get around it I just made a fake flash_attn.py script

Maybe too much of a hackish fix but it works
'''
import logging

#TODO: Not sure if I need this yet, but the example in Sage plugin used it
# def flash_attention(query, key, value, mask=None):
#   """Empty function to replace flash_attn.flash_attention"""
#   logging.debug("FlashAttention is not available, transformers might not work as expected.")
#   pass