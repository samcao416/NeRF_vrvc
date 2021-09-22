# encoding: utf-8

from .rfrender import RFRender

def build_model(cfg,camera_num=0,scale=None,shift=None,rotation = None,mirror = None):
    
    model = RFRender(cfg, camera_num)
    
    return model
