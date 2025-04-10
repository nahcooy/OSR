import timm
print([m for m in timm.list_models() if 'coatnet' in m])
