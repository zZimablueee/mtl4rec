import models
print("PLE_v6:", hasattr(models, "PLE_v6"))
print("PLE_v7:", hasattr(models, "PLE_v7"))
print("All models:", [x for x in dir(models) if not x.startswith("_")])